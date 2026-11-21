"""Compliant suction cup in MuJoCo coupled to the vacuum pneumatics (gate 2 of docs/MODEL_SPEC.md).

Design: docs/research/model-suction.md §4.
- Lip: n segments, each a small body on a slide joint along the cup axis with a real spring
  (Dex-Net 3.0 ring of springs, p(θ) = k·δz(θ)). The joint range stops a segment at its free
  length, so an unloaded segment lifts off and opens a gap.
- Contacts: explicit lip/product pairs with margin = 0, gap = g_det (MuJoCo 4 semantics: force only
  below margin; inactive contacts reported up to margin + gap). Separations up to g_det are
  reported but carry no force, which gives us the gap per segment.
- Vacuum: F = Δp·A_eff applied with mj_applyFT along the sealed face's normal (pressure acts normal to
  the surface), through the cup centre, on the product; the opposite force on the cup body. Δp comes from sim.pneumatics.VacuumCircuit, whose leak is the sum of
  per-segment leaks at the measured gaps (+ roughness g0, porosity K_p).
- Effector: x/y/z slides, force-driven to track a reference exactly (delta arm not modelled here).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .pneumatics import EJECTORS, P_ATM, LipLeak, VacuumCircuit, VacuumSensor


@dataclass
class CupConfig:
    # cup (defaults: Schmalz SPB1 20, model-suction.md §2.1/§5)
    r_lip: float = 0.0107            # m, Ds/2
    n_seg: int = 16
    A_eff: float = 9.9 / 60e3        # m^2, F_suction(-0.6 bar)/60 kPa
    cup_mass: float = 2.2e-3         # kg (SPB1 20)
    k_ring: float = 4000.0           # N/m, total ring stiffness n·k_l (not published: swept in tests)
    zeta: float = 1.0                # lip damping ratio (critical)
    delta_max: float = 0.004         # m, lip compression range
    lip_r: float = 0.00075           # m, radius of the lip contact sphere
    g_det: float = 0.003             # m, contact detection band (margin = gap)
    # pneumatics
    ejector: str = "Si08-3"
    V_cup: float = 3.2e-6 + 1e-6     # m^3, SPB1 20 volume + short hose
    L_x: float = 1.0e-3
    C_d: float = 0.7
    g0: float = 0.0
    K_p: float = 0.0
    frozen_dp: float | None = None   # Pa; if set, the ODE is off and Δp is held constant
    # product
    size: tuple = (0.05, 0.05, 0.02)
    mass: float = 0.2
    com: tuple = (0.0, 0.0, 0.0)     # CoM offset from the box centre (m)
    mu: float = 0.6
    # numerics
    dt: float = 2e-4
    n_sub: int = 10
    solref: tuple = (0.002, 1.0)
    noslip: int = 10                 # MuJoCo noslip iterations: removes soft-constraint creep inside the friction cone
    limit_solref: tuple = (0.0005, 1.0)  # stiff lip end stop at free length (default 0.02 lets a lip stretch)


class CupCell:
    def __init__(self, cfg: CupConfig | None = None, seed: int = 0):
        self.cfg = c = cfg or CupConfig()
        sx, sy, sz = (s / 2 for s in c.size)
        m_l = c.cup_mass / c.n_seg
        k_l = c.k_ring / c.n_seg
        d_l = 2 * c.zeta * np.sqrt(k_l * m_l)
        lips, pairs = [], []
        for i in range(c.n_seg):
            th = 2 * np.pi * i / c.n_seg
            x, y = c.r_lip * np.cos(th), c.r_lip * np.sin(th)
            lips.append(f'''
        <body name="lip{i}" pos="{x:.6f} {y:.6f} 0">
          <joint name="lz{i}" type="slide" axis="0 0 1" range="0 {c.delta_max}" limited="true" solreflimit="{c.limit_solref[0]} {c.limit_solref[1]}" solimplimit="0.95 0.99 0.0001" stiffness="{k_l}" springref="0" damping="{d_l:.6g}" armature="0"/>
          <inertial pos="0 0 0" mass="{m_l:.6g}" diaginertia="1e-9 1e-9 1e-9"/>
          <geom name="lipg{i}" type="sphere" size="{c.lip_r}" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>
        </body>''')
            pairs.append(f'<pair geom1="lipg{i}" geom2="prod" condim="3" friction="{c.mu} {c.mu} 0.005 0.0001 0.0001" '
                         f'solref="{c.solref[0]} {c.solref[1]}" solimp="0.95 0.99 0.0005" margin="0" gap="{c.g_det}"/>')
        z0 = c.size[2] + c.lip_r + 0.001  # lips 1 mm above the product top at start
        xml = f"""
<mujoco model="cup">
  <option timestep="{c.dt}" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="{c.noslip}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="0.5 0.005 0.0001"/>
    <body name="eff" pos="0 0 {z0}">
      <joint name="ex" type="slide" axis="1 0 0"/>
      <joint name="ey" type="slide" axis="0 1 0"/>
      <joint name="ez" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0.02" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>
      <site name="cupc" pos="0 0 0"/>
      {''.join(lips)}
    </body>
    <body name="prod" pos="0 0 {sz}">
      <freejoint name="prod_free"/>
      <inertial pos="{c.com[0]} {c.com[1]} {c.com[2]}" mass="{c.mass}" diaginertia="{c.mass / 12 * (c.size[1]**2 + c.size[2]**2):.4e} {c.mass / 12 * (c.size[0]**2 + c.size[2]**2):.4e} {c.mass / 12 * (c.size[0]**2 + c.size[1]**2):.4e}"/>
      <geom name="prod" type="box" size="{sx} {sy} {sz}" rgba="0.85 0.66 0.36 1"/>
    </body>
  </worldbody>
  <contact>{''.join(pairs)}</contact>
  <actuator>
    <motor joint="ex" ctrlrange="-2000 2000"/><motor joint="ey" ctrlrange="-2000 2000"/><motor joint="ez" ctrlrange="-2000 2000"/>
  </actuator>
</mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.lip_geoms = np.array([m.geom(f"lipg{i}").id for i in range(c.n_seg)])
        self.prod_geom, self.prod_body, self.eff_body = m.geom("prod").id, m.body("prod").id, m.body("eff").id
        self.lip_qadr = np.array([m.jnt_qposadr[m.joint(f"lz{i}").id] for i in range(c.n_seg)])
        self.prod_qadr = m.jnt_qposadr[m.joint("prod_free").id]
        self.site = m.site("cupc").id
        self.leak = LipLeak(r_lip=c.r_lip, n_seg=c.n_seg, L_x=c.L_x, C_d=c.C_d, g0=c.g0, K_p=c.K_p)
        self.vac = VacuumCircuit(EJECTORS[c.ejector], V_cup=c.V_cup, n_sub=c.n_sub)
        self.vac.on = False
        self.sensor = VacuumSensor(rng=np.random.default_rng(seed))
        self.kp, self.kd, self.eff_mass = 4e5, 1.2e3, 0.5 + c.cup_mass
        self.t = 0.0
        self.gaps = np.full(c.n_seg, c.g_det)
        self.ext_force = np.zeros(3)   # extra force on the product at its CoM (tests: d'Alembert loads)
        self.ext_torque = np.zeros(3)
        mujoco.mj_forward(m, self.d)

    # ---------------------------------------------------------------- state readout
    def read_gaps(self) -> np.ndarray:
        """Per-segment separation (m): 0 when touching or compressed, g_det when out of the detection band."""
        gaps = np.full(self.cfg.n_seg, self.cfg.g_det)
        lg = {int(g): i for i, g in enumerate(self.lip_geoms)}
        d = self.d
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in lg and g2 == self.prod_geom:
                i = lg[g1]
            elif g2 in lg and g1 == self.prod_geom:
                i = lg[g2]
            else:
                continue
            gaps[i] = min(gaps[i], max(0.0, con.dist))
        return gaps

    @property
    def dp(self) -> float:
        return self.cfg.frozen_dp if self.cfg.frozen_dp is not None else self.vac.vacuum

    def lip_compression(self) -> np.ndarray:
        return self.d.qpos[self.lip_qadr].copy()

    def prod_rel(self):
        """Product position relative to the cup centre, and its tilt (deg)."""
        p = self.d.xpos[self.prod_body] - self.d.site_xpos[self.site]
        q = self.d.qpos[self.prod_qadr + 3:self.prod_qadr + 7]
        return p, np.degrees(2 * np.arccos(min(1.0, abs(q[0]))))

    # ---------------------------------------------------------------- stepping
    def step(self, ref) -> None:
        """One MuJoCo step: effector tracks ref = (p, v, a); vacuum force from Δp; pneumatics advanced."""
        c, d, m = self.cfg, self.d, self.m
        p, v, a = ref
        q, qd = d.qpos[:3], d.qvel[:3]
        F_vac = self.dp * c.A_eff
        d.ctrl[:3] = self.eff_mass * (np.asarray(a) + [0, 0, 9.81]) + self.kp * (np.asarray(p) - q) + self.kd * (np.asarray(v) - qd) \
            + np.array([0, 0, F_vac])  # feedforward for the vacuum reaction on the cup
        d.qfrc_applied[:] = 0
        sealed_any = np.any(self.gaps < c.g_det)
        if F_vac > 0 and sealed_any:
            point = d.site_xpos[self.site].copy()
            # pressure acts normal to the sealed face: the product's top-face normal (its local +z)
            axis = d.xmat[self.prod_body].reshape(3, 3)[:, 2].copy()
            mujoco.mj_applyFT(m, d, F_vac * axis, np.zeros(3), point, self.prod_body, d.qfrc_applied)
            mujoco.mj_applyFT(m, d, -F_vac * axis, np.zeros(3), point, self.eff_body, d.qfrc_applied)
        if np.any(self.ext_force) or np.any(self.ext_torque):
            mujoco.mj_applyFT(m, d, self.ext_force, self.ext_torque, d.xipos[self.prod_body].copy(), self.prod_body, d.qfrc_applied)
        mujoco.mj_step(m, d)
        self.t += c.dt
        self.gaps = self.read_gaps()
        if c.frozen_dp is None:
            gaps = self.gaps
            self.vac.step(c.dt, lambda pc: self.leak.mdot(pc, gaps))
        self.sensor.update(self.t, c.dt, self.dp)

    def hold(self, T: float, p, rec=None):
        for _ in range(int(round(T / self.cfg.dt))):
            self.step((p, (0, 0, 0), (0, 0, 0)))
            if rec:
                rec(self)

    def engage(self, press: float = 0.0006, settle: float = 0.06, lift: float = 0.02, lift_T: float = 0.15):
        """Descend until the lips press `press` m into contact, switch the vacuum on, settle, then lift gently."""
        c = self.cfg
        z_touch = -(0.001 + press)
        n = int(0.05 / c.dt)
        for k in range(n):
            s = (k + 1) / n
            self.step(((0, 0, z_touch * (3 * s * s - 2 * s ** 3)), (0, 0, 0), (0, 0, 0)))
        self.vac.on = True
        self.hold(settle, (0, 0, z_touch))
        self.seal_kpa = self.dp / 1e3
        n = int(lift_T / c.dt)
        for k in range(n):
            s = (k + 1) / n
            self.step(((0, 0, z_touch + lift * (3 * s * s - 2 * s ** 3)), (0, 0, 0), (0, 0, 0)))
        self.z_hold = z_touch + lift
        self.hold(0.05, (0, 0, self.z_hold))

    def attached(self, slip_tol: float = 0.001) -> bool:
        p, tilt = self.prod_rel()
        return self.rel0 is not None and np.linalg.norm(p - self.rel0) < slip_tol and tilt < 20

    def mark(self):
        self.rel0 = self.prod_rel()[0].copy()
