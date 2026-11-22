"""Feasibility prototype (scratch, not project code): closed-chain Clavel delta in MuJoCo.

Model A: full multibody. 3 hinged upper arms, 6 struts (universal joint at elbow), free platform,
         6 `connect` equality constraints (site1/site2) closing the loops.
Model B: lumped (Codourey-type) model. 3 hinged upper arms, platform on 3 slides (translation only),
         3 spatial tendons elbow->platform held at constant length by `equality/tendon`;
         strut mass lumped at elbow (fraction r) and platform (1-r), strut rotational inertia neglected.
Controller: projected inverse dynamics feedforward (computed on model B) + joint PD.
"""
from __future__ import annotations
import sys, time
import numpy as np
import mujoco
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kin import ik, jac, PHI

# geometry (m): IRB 360-1/1130-class, TU Gdansk ROS driver values (third party), see model-delta.md
RB, RA, L, LF, W, ZM = 0.200, 0.045, 0.350, 0.800, 0.100, -0.275
TCP_OFF = 0.05
# inertial assumptions (placeholders for the prototype)
M_ARM, C_ARM, I_ARM_ROTOR = 0.9, 0.15, 0.05   # kg, m (CoM from axis), kg m^2 reflected rotor+gear inertia (armature)
M_STRUT, M_PLAT, M_LOAD = 0.13, 0.6, 1.0
import os
DT = float(os.environ.get("DT", "0.0005"))
SOLREF = os.environ.get("SOLREF", "0.002 1")
SOLIMP = os.environ.get("SOLIMP", "0.9 0.95 0.001 0.5 2")


def tdir(ph):
    return np.array([-np.sin(ph), np.cos(ph), 0.0])


def udir(ph):
    return np.array([np.cos(ph), np.sin(ph), 0.0])


def arm_world(th, ph):
    """motor axis point, elbow point (world)"""
    P = np.array([RB * np.cos(ph), RB * np.sin(ph), ZM])
    E = P + L * (np.cos(th) * udir(ph) - np.sin(th) * np.array([0, 0, 1.0]))
    return P, E


def p_motor(pw):
    """platform centre world -> kin.py frame (relative to motor plane)"""
    return np.array([pw[0], pw[1], pw[2] - ZM])


def build_A(p0w):
    th0 = ik(p_motor(p0w), RB, RA, L, LF)
    bodies, eqs = [], []
    for i, ph in enumerate(PHI):
        P, E = arm_world(th0[i], ph)
        t = tdir(ph)
        com = P + C_ARM / L * (E - P)
        struts = []
        for k, sgn in enumerate((-1, 1)):
            Ek = E + sgn * W / 2 * t
            Jk = p0w + RA * udir(ph) + sgn * W / 2 * t
            d = Jk - Ek
            n = np.cross(t, d); n /= np.linalg.norm(n)
            struts.append(f"""
        <body name="s{i}{k}" pos="{Ek[0]-P[0]:.6f} {Ek[1]-P[1]:.6f} {Ek[2]-P[2]:.6f}">
          <joint name="s{i}{k}a" type="hinge" axis="{t[0]:.6f} {t[1]:.6f} 0" armature="1e-5"/>
          <joint name="s{i}{k}b" type="hinge" axis="{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}" armature="1e-5"/>
          <geom type="capsule" size="0.008" fromto="0 0 0 {d[0]:.6f} {d[1]:.6f} {d[2]:.6f}" mass="{M_STRUT}" rgba=".2 .2 .2 1"/>
          <site name="se{i}{k}" pos="{d[0]:.6f} {d[1]:.6f} {d[2]:.6f}"/>
        </body>""")
            eqs.append(f'<connect site1="se{i}{k}" site2="pj{i}{k}"/>')
        bodies.append(f"""
      <body name="arm{i}" pos="{P[0]:.6f} {P[1]:.6f} {P[2]:.6f}">
        <joint name="q{i}" type="hinge" axis="{t[0]:.6f} {t[1]:.6f} 0" armature="{I_ARM_ROTOR}"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 {E[0]-P[0]:.6f} {E[1]-P[1]:.6f} {E[2]-P[2]:.6f}" mass="0" rgba=".8 .4 .1 1"/>
        <inertial pos="{com[0]-P[0]:.6f} {com[1]-P[1]:.6f} {com[2]-P[2]:.6f}" mass="{M_ARM}" diaginertia="2e-3 2e-3 2e-3"/>
        {''.join(struts)}
      </body>""")
    psites = "".join(
        f'<site name="pj{i}{k}" pos="{(RA*udir(ph)+sgn*W/2*tdir(ph))[0]:.6f} {(RA*udir(ph)+sgn*W/2*tdir(ph))[1]:.6f} 0"/>'
        for i, ph in enumerate(PHI) for k, sgn in enumerate((-1, 1)))
    xml = f"""
<mujoco model="deltaA">
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" gravity="0 0 -9.81"/>
  <default><equality solref="{SOLREF}" solimp="{SOLIMP}"/></default>
  <worldbody>
    {''.join(bodies)}
    <body name="plat" pos="{p0w[0]:.6f} {p0w[1]:.6f} {p0w[2]:.6f}">
      <freejoint name="pf"/>
      <geom type="cylinder" size="0.06 0.01" mass="{M_PLAT}" rgba=".7 .7 .7 1"/>
      <geom type="sphere" size="0.02" pos="0 0 {-TCP_OFF}" mass="{M_LOAD}"/>
      {psites}
      <site name="tcp" pos="0 0 {-TCP_OFF}"/>
    </body>
  </worldbody>
  <equality>{''.join(eqs)}</equality>
  <actuator>{''.join(f'<motor name="m{i}" joint="q{i}"/>' for i in range(3))}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), th0


def build_B(p0w, r_elbow=0.5):
    th0 = ik(p_motor(p0w), RB, RA, L, LF)
    bodies, tendons, eqs = [], [], []
    for i, ph in enumerate(PHI):
        P, E = arm_world(th0[i], ph)
        com = P + C_ARM / L * (E - P)
        me = 2 * M_STRUT * r_elbow
        bodies.append(f"""
    <body name="arm{i}" pos="{P[0]:.6f} {P[1]:.6f} {P[2]:.6f}">
      <joint name="q{i}" type="hinge" axis="{tdir(ph)[0]:.6f} {tdir(ph)[1]:.6f} 0" armature="{I_ARM_ROTOR}"/>
      <inertial pos="{com[0]-P[0]:.6f} {com[1]-P[1]:.6f} {com[2]-P[2]:.6f}" mass="{M_ARM}" diaginertia="2e-3 2e-3 2e-3"/>
      <geom type="sphere" size="0.01" pos="{E[0]-P[0]:.6f} {E[1]-P[1]:.6f} {E[2]-P[2]:.6f}" mass="{me}"/>
      <site name="el{i}" pos="{E[0]-P[0]:.6f} {E[1]-P[1]:.6f} {E[2]-P[2]:.6f}"/>
    </body>""")
        tendons.append(f'<spatial name="f{i}"><site site="el{i}"/><site site="pa{i}"/></spatial>')
        eqs.append(f'<tendon tendon1="f{i}"/>')
    mp = M_PLAT + 6 * M_STRUT * (1 - r_elbow)
    psites = "".join(f'<site name="pa{i}" pos="{RA*np.cos(ph):.6f} {RA*np.sin(ph):.6f} 0"/>' for i, ph in enumerate(PHI))
    xml = f"""
<mujoco model="deltaB">
  <option timestep="{DT}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default><equality solref="{SOLREF}" solimp="{SOLIMP}"/></default>
  <worldbody>
    {''.join(bodies)}
    <body name="plat" pos="{p0w[0]:.6f} {p0w[1]:.6f} {p0w[2]:.6f}">
      <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/><joint name="pz" type="slide" axis="0 0 1"/>
      <geom type="sphere" size="0.02" mass="{mp}"/>
      <geom type="sphere" size="0.02" pos="0 0 {-TCP_OFF}" mass="{M_LOAD}"/>
      {psites}
      <site name="tcp" pos="0 0 {-TCP_OFF}"/>
    </body>
  </worldbody>
  <tendon>{''.join(tendons)}</tendon>
  <equality>{''.join(eqs)}</equality>
  <actuator>{''.join(f'<motor name="m{i}" joint="q{i}"/>' for i in range(3))}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), th0


def traj(p0w, dx, amax=150.0, vmax=10.0, jmax=1e4):
    import ruckig
    o = ruckig.Ruckig(1); inp = ruckig.InputParameter(1); tr = ruckig.Trajectory(1)
    inp.current_position = [0.0]; inp.target_position = [dx]
    inp.max_velocity = [vmax]; inp.max_acceleration = [amax]; inp.max_jerk = [jmax]
    o.calculate(inp, tr)
    def f(t):
        t = min(max(t, 0.0), tr.duration)
        p, v, a = tr.at_time(t)
        return (p0w + np.array([p[0], 0, 0]), np.array([v[0], 0, 0]), np.array([a[0], 0, 0]))
    return f, tr.duration


def joint_ref(pw, vw, aw):
    """theta, thetadot, thetaddot for a platform state (finite-difference on IK for acceleration)."""
    pm = p_motor(pw)
    th = ik(pm, RB, RA, L, LF)
    Jx, Jq = jac(pm, th, RB, RA, L, LF)
    thd = np.linalg.solve(Jq, Jx @ vw)
    h = 1e-5
    pm2 = pm + h * vw
    th2 = ik(pm2, RB, RA, L, LF)
    Jx2, Jq2 = jac(pm2, th2, RB, RA, L, LF)
    # thdd = Jq^-1 (Jx a + d/dt(Jx) v - d/dt(Jq) thd)
    dJx, dJq = (Jx2 - Jx) / h, (Jq2 - Jq) / h
    thdd = np.linalg.solve(Jq, Jx @ aw + dJx @ vw - dJq @ thd)
    return th, thd, thdd


def ff_B(mB, dB, pw, vw, aw):
    """projected inverse dynamics on model B: M qacc + c = S^T tau + Jeq^T lam."""
    th, thd, thdd = joint_ref(pw, vw, aw)
    p0 = mB.body("plat").pos.copy()  # slide joints measure displacement from body pos
    dB.qpos[:] = np.r_[th - th0B, pw - p0]
    dB.qvel[:] = np.r_[thd, vw]
    dB.qacc[:] = np.r_[thdd, aw]
    mujoco.mj_forward(mB, dB)  # to fill efc_J for equality rows
    neq = mB.neq
    Jeq = dB.efc_J.reshape(dB.nefc, mB.nv)[:dB.ne] if dB.nefc else np.zeros((0, mB.nv))
    dB.qacc[:] = np.r_[thdd, aw]
    mB.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    mujoco.mj_inverse(mB, dB)
    mB.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    q = dB.qfrc_inverse.copy()
    S = np.zeros((3, mB.nv)); S[0, 0] = S[1, 1] = S[2, 2] = 1
    A = np.vstack([S, Jeq]).T
    sol, *_ = np.linalg.lstsq(A, q, rcond=None)
    return sol[:3], th, thd


th0B = None


def run(model, th0, mB, dB, p0w, dx, kp=2e4, kd=300.0, label=""):
    d = mujoco.MjData(model)
    # model A: qpos0 is assembled (struts built at the IK configuration) so start from qpos0
    mujoco.mj_forward(model, d)
    f, T = traj(p0w, dx)
    tcp = model.site("tcp").id
    qadr = [model.jnt_qposadr[model.joint(f"q{i}").id] for i in range(3)]
    vadr = [model.jnt_dofadr[model.joint(f"q{i}").id] for i in range(3)]
    errs, taus, viol, accs, fb, tilt = [], [], [], [], [], [0.0]
    nsteps = int((T + 0.05) / DT)
    t0 = time.perf_counter()
    for k in range(nsteps):
        t = k * DT
        pw, vw, aw = f(t)
        tff, _, _ = ff_B(mB, dB, *f(t + DT / 2))   # feedforward at step midpoint (semi-implicit integration)
        _, thd_, _ = ff_B(mB, dB, pw, vw, aw)
        _, _, thdd_ = ff_B(mB, dB, *f(t - DT / 2))   # semi-implicit Euler: qvel_k is the mean velocity over [t-dt, t]
        q = d.qpos[qadr] + th0
        qd = d.qvel[vadr]
        tau = tff + kp * (thd_ - q) + kd * (thdd_ - qd)
        d.ctrl[:] = tau
        mujoco.mj_step(model, d)
        tcp_ref = pw + np.array([0, 0, -TCP_OFF])
        errs.append(np.linalg.norm(d.site_xpos[tcp] - tcp_ref))
        if k % 40 == 0 and os.environ.get("VERB"): print(f"   t={t*1000:6.1f}ms v={vw[0]:6.2f} a={aw[0]:7.1f} err_x={(d.site_xpos[tcp]-tcp_ref)[0]*1000:+.3f}mm err_z={(d.site_xpos[tcp]-tcp_ref)[2]*1000:+.3f}mm qerr={(thd_-q)*1000} mrad qderr={(thdd_-qd)}")
        taus.append(tau.copy()); fb.append(np.abs(tau - tff).max())
        if model.nq > 9:
            R = d.xmat[model.body("plat").id].reshape(3, 3); tilt.append(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        if model.neq:
            viol.append(np.abs(d.efc_pos[:d.ne]).max() if d.ne else 0)
    wall = time.perf_counter() - t0
    taus = np.array(taus)
    print(f"{label}: move {dx*1000:.0f} mm in {T*1000:.1f} ms | max TCP err {max(errs)*1000:.3f} mm | "
          f"max |tau| {np.abs(taus).max(axis=0).round(1)} N m | max eq viol {max(viol)*1e6 if viol else 0:.1f} um | max|tau-tau_ff| {max(fb):.2f} N m | "
          f"{nsteps/wall:.0f} steps/s | max platform tilt {max(tilt)*1e3:.3f} mrad")
    return taus


if __name__ == "__main__":
    p0w = np.array([-0.15, 0.0, -0.94])  # platform centre, TCP 0.99 m below base frame
    dx = 0.30
    for r in (0.5, 2 / 3):
        mB, th0B = build_B(p0w, r_elbow=r)
        dB = mujoco.MjData(mB)
        mA, th0A = build_A(p0w)
        print(f"--- lumped split r_elbow={r:.2f} used for feedforward ---")
        tB = run(mB, th0B, mB, dB, p0w, dx, label="model B (self-consistent)")
        tA = run(mA, th0A, mB, dB, p0w, dx, label="model A (full multibody)")
