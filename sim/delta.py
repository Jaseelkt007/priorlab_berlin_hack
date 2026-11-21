"""IRB 360-1/1130-class delta robot in MuJoCo (gate 4 of docs/MODEL_SPEC.md).

Design: docs/research/model-delta.md §1, §4, §5. Built from the feasibility prototype in research/proto_delta/.
- Model A (the plant): closed chain. 3 hinged upper arms, 6 struts on 2-hinge universal joints, a free
  platform, 6 `equality/connect` constraints with stiff settings (prototype: 0.044 mm tracking at 150 m/s²).
- Model B (the feedforward model): Codourey-type lumped model. Arms plus elbow point masses, the platform
  on slides, and 3 tendons of constant length.
- Controller: projected inverse dynamics on Model B (mj_inverse, then least squares against the
  equality-constraint Jacobian) plus joint PD. Uses the dt/2 timing correction for semi-implicit Euler.
Frames: world origin = robot base frame, z up, so the motors sit at z = −0.275 m and the TCP works
0.865–1.115 m below the base frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

PHI = np.deg2rad([0.0, 120.0, 240.0])


@dataclass
class DeltaParams:
    # geometry (model-delta.md table 5.1)
    r_b: float = 0.200        # base radius, centre to motor axis (ROS driver; ABB figure ≈ 0.185, sensitivity case)
    z_m: float = -0.275       # motor axis height in the base frame (ABB product spec)
    L: float = 0.366          # upper arm: ABB swing radius R (primary source). The ROS driver value 0.350 cannot fit
                              # ABB's envelope within the joint limits for any TCP offset (gate 4, V2); kept as a sensitivity case
    l: float = 0.800          # forearm strut
    r_p: float = 0.045        # platform joint radius
    w: float = 0.100          # parallelogram width
    tcp_off: float = 0.050    # TCP below the platform joint plane
    q_range: tuple = (-46.1, 100.0)  # deg, motor joint range (ROS URDF)
    # dynamics priors (scaled from Gnad et al. 2024, IRB 360-6/1600; calibrated in gate 4)
    m_arm: float = 0.70       # kg, arm as a rod; m·c = 0.10 kg·m with c below
    c_arm: float = 0.143      # m, arm CoM from the motor axis
    theta_total: float = 0.08  # kg·m², arm inertia about the motor axis incl. reflected drive inertia
    m_strut: float = 0.13
    m_plat: float = 0.60
    m_tool: float = 0.10      # payload incl. gripper, carried at the TCP (ABB cycle tables: 0.1 / 1.0 kg)
    fric_coulomb: float = 4.7  # N·m at the gearbox output (Gnad et al.); smooth law F_c·tanh(θ̇/v_s), applied by plant_friction()
    fric_vs: float = 0.01      # rad/s, smoothing velocity of the Coulomb term
    fric_viscous: float = 1.6  # N·m·s/rad
    tau_max: float = 123.9    # N·m at the gearbox output: not published; calibrated on ABB 25/305/25 @ 1 kg (V5)
    r_elbow: float = 0.5      # strut mass share lumped at the elbow in Model B
    r_plat: float | None = None  # strut mass share lumped at the platform (default 1 − r_elbow). Chosen by test V4.
    # numerics
    dt: float = 0.0005
    eq_solref: tuple = (0.001, 1.0)
    eq_solimp: tuple = (0.99, 0.999, 0.001)

    @property
    def armature(self) -> float:
        i_arm = self.m_arm * self.c_arm ** 2 + self.m_arm * self.L ** 2 / 12
        return max(self.theta_total - i_arm, 0.0)


# ------------------------------------------------------------------ kinematics (model-delta.md §1.2–1.4)
def ik(pm: np.ndarray, P: DeltaParams) -> np.ndarray | None:
    """Platform centre pm in the motor-plane frame (z < 0 below the motors) -> arm angles (rad, + = down)."""
    th = np.empty(3)
    for i, ph in enumerate(PHI):
        c, s = np.cos(ph), np.sin(ph)
        x, y, z = c * pm[0] + s * pm[1], -s * pm[0] + c * pm[1], pm[2]
        a = x + P.r_p - P.r_b
        K = (P.l ** 2 - P.L ** 2 - a ** 2 - y ** 2 - z ** 2) / (2 * P.L)
        R = np.hypot(a, z)
        if abs(K) > R:
            return None
        g = np.arctan2(z, -a)
        cand = [((g + sg * np.arccos(K / R) + np.pi) % (2 * np.pi)) - np.pi for sg in (1, -1)]
        th[i] = max(cand, key=np.cos)  # elbow-out branch
    return th


def jacobians(pm: np.ndarray, th: np.ndarray, P: DeltaParams):
    """J_x ṗ = J_q θ̇ (rows: forearm vectors)."""
    Jx, Jq = np.zeros((3, 3)), np.zeros((3, 3))
    for i, ph in enumerate(PHI):
        c, s = np.cos(ph), np.sin(ph)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        E = R @ np.array([P.r_b + P.L * np.cos(th[i]), 0, -P.L * np.sin(th[i])])
        d = pm + R @ np.array([P.r_p, 0, 0]) - E
        dE = R @ np.array([-P.L * np.sin(th[i]), 0, -P.L * np.cos(th[i])])
        Jx[i], Jq[i, i] = d, d @ dE
    return Jx, Jq


def to_motor(pw: np.ndarray, P: DeltaParams) -> np.ndarray:
    return np.array([pw[0], pw[1], pw[2] - P.z_m])


def joint_state(pw, vw, aw, P: DeltaParams, h: float = 1e-6):
    """θ, θ̇, θ̈ for a platform state (θ̈ via the time derivative of the Jacobians)."""
    pm = to_motor(pw, P)
    th = ik(pm, P)
    if th is None:
        raise ValueError(f"platform point {pw} is outside the workspace")
    Jx, Jq = jacobians(pm, th, P)
    thd = np.linalg.solve(Jq, Jx @ vw)
    pm2 = pm + h * np.asarray(vw)
    th2 = th + h * thd
    Jx2, Jq2 = jacobians(pm2, th2, P)
    thdd = np.linalg.solve(Jq, Jx @ aw + (Jx2 - Jx) / h @ vw - (Jq2 - Jq) / h @ thd)
    return th, thd, thdd


def tdir(ph):
    return np.array([-np.sin(ph), np.cos(ph), 0.0])


def udir(ph):
    return np.array([np.cos(ph), np.sin(ph), 0.0])


def arm_points(th: float, ph: float, P: DeltaParams):
    Pm = np.array([P.r_b * np.cos(ph), P.r_b * np.sin(ph), P.z_m])
    E = Pm + P.L * (np.cos(th) * udir(ph) - np.sin(th) * np.array([0, 0, 1.0]))
    return Pm, E


# ------------------------------------------------------------------ Model A (plant)
def build_model_a(P: DeltaParams, p0w: np.ndarray, extra_platform_xml: str = "", extra_world_xml: str = "",
                  extra_contact_xml: str = "", extra_actuator_xml: str = "", option_xml: str = ""):
    """Closed chain assembled at platform position p0w (world). Returns (model, θ0). Joint zero = assembly pose."""
    th0 = ik(to_motor(p0w, P), P)
    if th0 is None:
        raise ValueError("assembly pose outside the workspace")
    lo, hi = P.q_range
    arms, eqs = [], []
    for i, ph in enumerate(PHI):
        Pm, E = arm_points(th0[i], ph, P)
        t = tdir(ph)
        com = Pm + P.c_arm / P.L * (E - Pm)
        struts = []
        for k, sg in enumerate((-1, 1)):
            Ek = E + sg * P.w / 2 * t
            Jk = p0w + P.r_p * udir(ph) + sg * P.w / 2 * t
            d = Jk - Ek
            n = np.cross(t, d)
            n /= np.linalg.norm(n)
            struts.append(f"""
          <body name="s{i}{k}" pos="{Ek[0] - Pm[0]:.7f} {Ek[1] - Pm[1]:.7f} {Ek[2] - Pm[2]:.7f}">
            <joint name="s{i}{k}a" type="hinge" axis="{t[0]:.7f} {t[1]:.7f} 0" armature="1e-5"/>
            <joint name="s{i}{k}b" type="hinge" axis="{n[0]:.7f} {n[1]:.7f} {n[2]:.7f}" armature="1e-5"/>
            <geom type="capsule" size="0.006" fromto="0 0 0 {d[0]:.7f} {d[1]:.7f} {d[2]:.7f}" mass="{P.m_strut}" rgba=".25 .25 .28 1" contype="0" conaffinity="0"/>
            <site name="se{i}{k}" pos="{d[0]:.7f} {d[1]:.7f} {d[2]:.7f}"/>
          </body>""")
            eqs.append(f'<connect site1="se{i}{k}" site2="pj{i}{k}"/>')
        # joint range relative to the assembly pose (joint zero = θ0)
        r_lo, r_hi = np.deg2rad(lo) - th0[i], np.deg2rad(hi) - th0[i]
        arms.append(f"""
      <body name="arm{i}" pos="{Pm[0]:.7f} {Pm[1]:.7f} {Pm[2]:.7f}">
        <joint name="q{i}" type="hinge" axis="{t[0]:.7f} {t[1]:.7f} 0" armature="{P.armature:.6g}" damping="{P.fric_viscous}" range="{r_lo:.6f} {r_hi:.6f}" limited="true"/>
        <inertial pos="{com[0] - Pm[0]:.7f} {com[1] - Pm[1]:.7f} {com[2] - Pm[2]:.7f}" mass="{P.m_arm}" diaginertia="{P.m_arm * P.L ** 2 / 12:.6g} {P.m_arm * P.L ** 2 / 12:.6g} 1e-4"/>
        <geom type="capsule" size="0.018" fromto="0 0 0 {E[0] - Pm[0]:.7f} {E[1] - Pm[1]:.7f} {E[2] - Pm[2]:.7f}" mass="0" rgba=".93 .55 .14 1" contype="0" conaffinity="0"/>
        {''.join(struts)}
      </body>""")
    psites = "".join(
        f'<site name="pj{i}{k}" pos="{(P.r_p * udir(ph) + sg * P.w / 2 * tdir(ph))[0]:.7f} {(P.r_p * udir(ph) + sg * P.w / 2 * tdir(ph))[1]:.7f} 0"/>'
        for i, ph in enumerate(PHI) for k, sg in enumerate((-1, 1)))
    xml = f"""
<mujoco model="irb360_class_A">
  <compiler angle="radian"/>
  <option timestep="{P.dt}" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81" {option_xml}/>
  <default><equality solref="{P.eq_solref[0]} {P.eq_solref[1]}" solimp="{P.eq_solimp[0]} {P.eq_solimp[1]} {P.eq_solimp[2]}"/></default>
  <worldbody>
    <geom name="base" type="cylinder" size="0.26 0.02" pos="0 0 -0.02" rgba=".6 .62 .65 1" contype="0" conaffinity="0" mass="0"/>
    {''.join(arms)}
    <body name="plat" pos="{p0w[0]:.7f} {p0w[1]:.7f} {p0w[2]:.7f}">
      <freejoint name="pf"/>
      <inertial pos="0 0 0" mass="{P.m_plat}" diaginertia="1e-3 1e-3 2e-3"/>
      <geom type="cylinder" size="0.055 0.008" rgba=".75 .77 .8 1" contype="0" conaffinity="0" mass="0"/>
      {psites}
      <site name="tcp" pos="0 0 {-P.tcp_off}"/>
      {extra_platform_xml}
    </body>
    {extra_world_xml}
  </worldbody>
  <equality>{''.join(eqs)}</equality>
  <contact>{extra_contact_xml}</contact>
  <actuator>{''.join(f'<motor name="m{i}" joint="q{i}" ctrllimited="true" ctrlrange="{-P.tau_max} {P.tau_max}"/>' for i in range(3))}{extra_actuator_xml}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), th0


def tool_xml(P: DeltaParams) -> str:
    """Payload (incl. gripper) as a point-like mass at the TCP, used for cycle-time calibration."""
    return f'<body name="tool" pos="0 0 {-P.tcp_off}"><inertial pos="0 0 0" mass="{P.m_tool}" diaginertia="1e-5 1e-5 1e-5"/></body>'


# ------------------------------------------------------------------ Model B (feedforward)
def build_model_b(P: DeltaParams, p0w: np.ndarray):
    th0 = ik(to_motor(p0w, P), P)
    bodies, tendons, eqs = [], [], []
    for i, ph in enumerate(PHI):
        Pm, E = arm_points(th0[i], ph, P)
        com = Pm + P.c_arm / P.L * (E - Pm)
        bodies.append(f"""
    <body name="arm{i}" pos="{Pm[0]:.7f} {Pm[1]:.7f} {Pm[2]:.7f}">
      <joint name="q{i}" type="hinge" axis="{tdir(ph)[0]:.7f} {tdir(ph)[1]:.7f} 0" armature="{P.armature:.6g}"/>
      <inertial pos="{com[0] - Pm[0]:.7f} {com[1] - Pm[1]:.7f} {com[2] - Pm[2]:.7f}" mass="{P.m_arm}" diaginertia="{P.m_arm * P.L ** 2 / 12:.6g} {P.m_arm * P.L ** 2 / 12:.6g} 1e-4"/>
      <body name="el{i}" pos="{E[0] - Pm[0]:.7f} {E[1] - Pm[1]:.7f} {E[2] - Pm[2]:.7f}">
        <inertial pos="0 0 0" mass="{2 * P.m_strut * P.r_elbow}" diaginertia="1e-6 1e-6 1e-6"/>
        <site name="els{i}"/>
      </body>
    </body>""")
        tendons.append(f'<spatial name="f{i}"><site site="els{i}"/><site site="pa{i}"/></spatial>')
        eqs.append(f'<tendon tendon1="f{i}"/>')
    r_plat = (1 - P.r_elbow) if P.r_plat is None else P.r_plat
    m_plat = P.m_plat + 6 * P.m_strut * r_plat
    psites = "".join(f'<site name="pa{i}" pos="{P.r_p * np.cos(ph):.7f} {P.r_p * np.sin(ph):.7f} 0"/>' for i, ph in enumerate(PHI))
    xml = f"""
<mujoco model="irb360_class_B">
  <compiler angle="radian"/>
  <option timestep="{P.dt}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <worldbody>
    {''.join(bodies)}
    <body name="plat" pos="{p0w[0]:.7f} {p0w[1]:.7f} {p0w[2]:.7f}">
      <joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/><joint name="pz" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="{m_plat}" diaginertia="1e-6 1e-6 1e-6"/>
      <body name="tool" pos="0 0 {-P.tcp_off}"><inertial pos="0 0 0" mass="{P.m_tool}" diaginertia="1e-6 1e-6 1e-6"/></body>
      {psites}
    </body>
  </worldbody>
  <tendon>{''.join(tendons)}</tendon>
  <equality>{''.join(eqs)}</equality>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), th0


def plant_friction(m: mujoco.MjModel, d: mujoco.MjData, P: DeltaParams, arm_dof: np.ndarray) -> None:
    """Gearbox Coulomb friction of the identified law (Gnad et al.: Coulomb + viscous), applied to the plant's
    motor joints. Viscous friction is the joints' `damping` (integrated implicitly). Call before every mj_step."""
    d.qfrc_applied[arm_dof] = -P.fric_coulomb * np.tanh(d.qvel[arm_dof] / P.fric_vs)


def base_force_lumped(pw, vw, aw, P: DeltaParams, h: float = 1e-6) -> np.ndarray:
    """Dynamic base force Σ m_i a_i (weight excluded) of the lumped model: arms (CoM on the arm), elbow masses
    (r_elbow of the struts), platform + tool + the rest of the struts. Point accelerations on each arm come from
    r(θ): a = r'(θ) θ̈ + r''(θ) θ̇²."""
    th, thd, thdd = joint_state(pw, vw, aw, P)
    r_plat = (1 - P.r_elbow) if P.r_plat is None else P.r_plat
    F = (P.m_plat + P.m_tool + 6 * P.m_strut * r_plat) * np.asarray(aw, float)
    m_el = 2 * P.m_strut * P.r_elbow
    for i, ph in enumerate(PHI):
        def pts(t):
            Pm, E = arm_points(t, ph, P)
            return Pm + P.c_arm / P.L * (E - Pm), E
        (c0, e0), (cp, ep), (cm, em) = pts(th[i]), pts(th[i] + h), pts(th[i] - h)
        for m_i, r0, rp, rm in ((P.m_arm, c0, cp, cm), (m_el, e0, ep, em)):
            d1, d2 = (rp - rm) / (2 * h), (rp - 2 * r0 + rm) / h ** 2
            F = F + m_i * (d1 * thdd[i] + d2 * thd[i] ** 2)
    return F


class Feedforward:
    """Projected inverse dynamics on Model B: [Sᵀ J_eqᵀ][τ; λ] = M q̈ + c (least squares)."""

    def __init__(self, P: DeltaParams, p0w: np.ndarray):
        self.P = P
        self.m, self.th0 = build_model_b(P, p0w)
        self.d = mujoco.MjData(self.m)
        self.p0 = self.m.body("plat").pos.copy()
        self.S = np.zeros((3, self.m.nv))
        self.S[0, 0] = self.S[1, 1] = self.S[2, 2] = 1.0

    def torque(self, pw, vw, aw):
        th, thd, thdd = joint_state(pw, vw, aw, self.P)
        m, d = self.m, self.d
        d.qpos[:] = np.r_[th - self.th0, np.asarray(pw) - self.p0]
        d.qvel[:] = np.r_[thd, vw]
        mujoco.mj_forward(m, d)
        Jeq = d.efc_J.reshape(d.nefc, m.nv)[:d.ne].copy() if d.nefc else np.zeros((0, m.nv))
        d.qacc[:] = np.r_[thdd, aw]
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_inverse(m, d)
        m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        sol, *_ = np.linalg.lstsq(np.vstack([self.S, Jeq]).T, d.qfrc_inverse.copy(), rcond=None)
        return sol[:3], th, thd


class DeltaController:
    """τ = τ_ff (at t + dt/2) + Kp (θ_ref(t) − θ) + Kd (θ̇_ref(t − dt/2) − θ̇), clipped to ±τ_max.
    Friction (Coulomb + viscous at the gearbox output) is compensated in the feedforward."""

    def __init__(self, P: DeltaParams, model: mujoco.MjModel, p0w: np.ndarray, kp: float = 2e4, kd: float = 300.0):
        self.P, self.m = P, model
        self.ff = Feedforward(P, p0w)
        self.kp, self.kd = kp, kd
        self.qadr = np.array([model.jnt_qposadr[model.joint(f"q{i}").id] for i in range(3)])
        self.vadr = np.array([model.jnt_dofadr[model.joint(f"q{i}").id] for i in range(3)])
        self.th0 = ik(to_motor(p0w, P), P)
        self.last_tau = np.zeros(3)

    def use_table(self, tff_table: np.ndarray | None) -> None:
        """Use precomputed exact feedforward (FeedforwardA.precompute) instead of the lumped Model B."""
        self.tff_table = tff_table

    def torque(self, d: mujoco.MjData, ref, t: float) -> np.ndarray:
        """ref(t) -> (p, v, a) of the platform centre in world coordinates."""
        dt = self.P.dt
        table = getattr(self, "tff_table", None)
        if table is not None:
            tff = table[min(int(round(t / dt)), len(table) - 1)]
        else:
            tff, _, _ = self.ff.torque(*ref(t + dt / 2))
        th_ref, _, _ = joint_state(*ref(t), self.P)
        _, thd_ref, _ = joint_state(*ref(t - dt / 2), self.P)
        q = d.qpos[self.qadr] + self.th0
        qd = d.qvel[self.vadr]
        fric = self.P.fric_viscous * qd + self.P.fric_coulomb * np.tanh(thd_ref / self.P.fric_vs)
        fb = self.kp * (th_ref - q) + self.kd * (thd_ref - qd)
        tau = np.clip(tff + fric + fb, -self.P.tau_max, self.P.tau_max)
        self.last_tau, self.last_ff, self.last_fb = tau, tff + fric, fb
        return tau


class FeedforwardA:
    """Exact feedforward from the full closed chain (Model A): projected inverse dynamics with every strut DOF.
    Needed because the lumped (Codourey) model is off by ~10 % of peak torque at light payloads (gate 4, V4).
    Use precompute() once per path; the controller then reads τ_ff from the table."""

    def __init__(self, P: DeltaParams, p0w: np.ndarray, extra_platform_xml: str = ""):
        import dataclasses
        self.P = P
        # friction is compensated separately by the controller, so the feedforward model has none (no double count)
        Pf = dataclasses.replace(P, fric_coulomb=0.0, fric_viscous=0.0)
        self.m, self.th0 = build_model_a(Pf, p0w, extra_platform_xml=extra_platform_xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_dof = np.array([m.jnt_dofadr[m.joint(f"q{i}").id] for i in range(3)])
        self.arm_q = np.array([m.jnt_qposadr[m.joint(f"q{i}").id] for i in range(3)])
        self.strut_dof = np.array([m.jnt_dofadr[m.joint(f"s{i}{k}{ab}").id] for i in range(3) for k in range(2) for ab in "ab"])
        self.strut_q = np.array([m.jnt_qposadr[m.joint(f"s{i}{k}{ab}").id] for i in range(3) for k in range(2) for ab in "ab"])
        pf = m.joint("pf").id
        self.pq, self.pv = m.jnt_qposadr[pf], m.jnt_dofadr[pf]
        self.p0 = m.body("plat").pos.copy()
        self.qs = np.zeros(len(self.strut_q))
        self.S = np.zeros((3, m.nv))
        self.S[np.arange(3), self.arm_dof] = 1.0

    def _pose(self, pw):
        """Full qpos for a platform position: arms by IK, struts by Gauss–Newton on the loop residual."""
        m, d, P = self.m, self.d, self.P
        th = ik(to_motor(pw, P), P)
        d.qpos[:] = 0
        d.qpos[self.arm_q] = th - self.th0
        d.qpos[self.pq:self.pq + 3] = pw
        d.qpos[self.pq + 3:self.pq + 7] = [1, 0, 0, 0]
        d.qpos[self.strut_q] = self.qs
        for _ in range(8):
            mujoco.mj_forward(m, d)
            r = d.efc_pos[:d.ne].copy()
            if np.abs(r).max() < 1e-12:
                break
            J = d.efc_J.reshape(d.nefc, m.nv)[:d.ne][:, self.strut_dof]
            dq, *_ = np.linalg.lstsq(J, -r, rcond=None)
            d.qpos[self.strut_q] += dq
        self.qs = d.qpos[self.strut_q].copy()
        return th

    def _state(self, pw, vw):
        th, thd, _ = joint_state(pw, vw, np.zeros(3), self.P)
        self._pose(pw)
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        J = d.efc_J.reshape(d.nefc, m.nv)[:d.ne]
        qv = np.zeros(m.nv)
        qv[self.arm_dof] = thd
        qv[self.pv:self.pv + 3] = vw
        rhs = -J @ qv
        vs, *_ = np.linalg.lstsq(J[:, self.strut_dof], rhs, rcond=None)
        qv[self.strut_dof] = vs
        return d.qpos.copy(), qv

    def torque(self, ref, t: float, h: float = 1e-5):
        """τ_ff at time t for reference ref(t) -> (p, v, a)."""
        q0, v0 = self._state(*ref(t)[:2])
        _, vp = self._state(*ref(t + h)[:2])
        _, vm = self._state(*ref(t - h)[:2])
        m, d = self.m, self.d
        d.qpos[:], d.qvel[:] = q0, v0
        mujoco.mj_forward(m, d)
        Jeq = d.efc_J.reshape(d.nefc, m.nv)[:d.ne].copy()
        d.qacc[:] = (vp - vm) / (2 * h)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_inverse(m, d)
        m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        # the passive joint damping/friction of the arms is compensated separately by the controller
        qf = d.qfrc_inverse.copy()
        sol, *_ = np.linalg.lstsq(np.vstack([self.S, Jeq]).T, qf, rcond=None)
        return sol[:3]

    def precompute(self, ref, T: float, dt: float) -> np.ndarray:
        """τ_ff on the half-step grid t_k + dt/2, k = 0..N (what the controller needs)."""
        n = int(np.ceil(T / dt)) + 2
        return np.array([self.torque(ref, k * dt + dt / 2) for k in range(n)])
