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

from .pneumatics import EJECTORS, P_ATM, LipLeak, VacuumCircuit, VacuumSensor, step_lip_fast


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
    lip_r: float = 0.00075           # m, radius of the lip contact sphere (ring) or capsule (cone)
    # lip shape. "ring": a flat cup's lip, spheres in the plane z = 0 (r_lip = lip radius). "cone": a bell cup
    # (Schmalz SPG) whose conical wall closes around the product's top edge; each segment is a capsule along the
    # wall from the mouth (radius cone_r_mouth) to the neck (cone_r_neck), at cone_beta from the axis. The seal
    # line is where the wall meets the product rim: r_lip = rim radius, and the frame origin sits at that height.
    lip: str = "ring"
    cone_beta: float = 0.0           # rad, wall angle from the cup axis
    cone_r_mouth: float = 0.0        # m
    cone_r_neck: float = 0.0         # m
    # bellows between effector and lip ring (model-suction.md §4.1). Axial stiffness: Schmalz upper bound
    # F_suction/Z; tilt k_θ = k_bz·R²/2 and lateral k_lat = 6·k_bz·R²/L² from thin-walled-tube beam theory
    # (fixed–guided beam, as in Hudoklin et al. 2022's beam model of cup bending).
    bellows: bool = False
    k_bz: float = 1240.0             # N/m (SPB1 20: ≤ 9.9 N / 8 mm)
    R_b: float = 0.0088              # m, mean bellows radius ≈ (Ds + d2)/4 for SPB1 20 (21.4, 13.8 mm)
    L_b: float = 0.012               # m, bellows length (not published; swept in V15)
    Z_stroke: float = 0.008          # m, SPB1 20 stroke
    g_det: float = 0.003             # m, contact detection band (margin = gap)
    # pneumatics
    ejector: str = "Si08-3"
    V_cup: float = 3.2e-6 + 1e-6     # m^3, SPB1 20 volume + short hose
    L_x: float = 1.0e-3
    C_d: float = 0.7
    g0: float = 0.0
    K_p: float = 0.0
    frozen_dp: float | None = None   # Pa; if set, the ODE is off and Δp is held constant
    n_gen: int = 1                   # ejector cartridges in parallel (generator sized to the cup)
    p_set_kpa: float | None = None   # vacuum regulator set point (None: the generator's own maximum)
    # product
    shape: str = "box"               # "box" (size = L, W, H) or "cyl" (size = D, D, H)
    size: tuple = (0.05, 0.05, 0.02)
    mass: float = 0.2
    com: tuple = (0.0, 0.0, 0.0)     # CoM offset from the box centre (m)
    mu: float = 0.6
    # liquid contents (pouches): a fraction of the mass on a 2-D spring–damper inside the product ([E] lumped slosh mode)
    slosh_frac: float = 0.0
    slosh_hz: float = 3.0
    slosh_zeta: float = 0.1
    # numerics
    dt: float = 2e-4
    n_sub: int = 10
    fast_ode: bool = True            # compiled pneumatics (identical equations; see tests/test_pneumatics.py)
    solref: tuple = (0.002, 1.0)
    noslip: int = 10                 # MuJoCo noslip iterations: removes soft-constraint creep inside the friction cone
    limit_solref: tuple = (0.0005, 1.0)  # stiff lip end stop at free length (default 0.02 lets a lip stretch)


def cup_parts(c: CupConfig):
    """MJCF pieces of the cup and the product, so the same cup can sit on the slide effector (CupCell) or on the
    delta platform (sim.line). Returns (cup_block, product_body, contact_pairs). The cup block's frame is the lip
    plane (cup centre site "cupc"); the product body is created at the world origin's frame given by the caller."""
    m_l = (c.cup_mass / 2 if c.bellows else c.cup_mass) / c.n_seg
    k_l = c.k_ring / c.n_seg
    d_l = 2 * c.zeta * np.sqrt(k_l * m_l)
    lips, pairs = [], []
    for i in range(c.n_seg):
        th = 2 * np.pi * i / c.n_seg
        u = np.array([np.cos(th), np.sin(th)])
        x, y = c.r_lip * u
        if c.lip == "cone":
            # wall surface through the seal circle (this body's origin); the capsule axis sits lip_r outside it
            sb, cb, tb = np.sin(c.cone_beta), np.cos(c.cone_beta), np.tan(c.cone_beta)
            ends = []
            for r_end in (c.cone_r_mouth, c.cone_r_neck):
                dr = r_end - c.r_lip
                ends.append(np.r_[(dr + c.lip_r * cb) * u, -dr / tb + c.lip_r * sb])
            ft = " ".join(f"{v:.6f}" for v in np.r_[ends[0], ends[1]])
            geom = f'<geom name="lipg{i}" type="capsule" fromto="{ft}" size="{c.lip_r}" contype="0" conaffinity="0" rgba="0.75 0.78 0.8 0.6"/>'
        else:
            geom = f'<geom name="lipg{i}" type="sphere" size="{c.lip_r}" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1"/>'
        lips.append(f'''
        <body name="lip{i}" pos="{x:.6f} {y:.6f} 0">
          <joint name="lz{i}" type="slide" axis="0 0 1" range="0 {c.delta_max}" limited="true" solreflimit="{c.limit_solref[0]} {c.limit_solref[1]}" solimplimit="0.95 0.99 0.0001" stiffness="{k_l}" springref="0" damping="{d_l:.6g}" armature="0"/>
          <inertial pos="0 0 0" mass="{m_l:.6g}" diaginertia="1e-9 1e-9 1e-9"/>
          {geom}
        </body>''')
        pairs.append(f'<pair geom1="lipg{i}" geom2="prod" condim="3" friction="{c.mu} {c.mu} 0.005 0.0001 0.0001" '
                     f'solref="{c.solref[0]} {c.solref[1]}" solimp="0.95 0.99 0.0005" margin="0" gap="{c.g_det}"/>')
    if c.bellows:
        m_b = c.cup_mass / 2
        k_t = c.k_bz * c.R_b ** 2 / 2
        k_lat = 6 * c.k_bz * c.R_b ** 2 / c.L_b ** 2
        I_b = m_b * c.L_b ** 2 / 3
        zc = lambda k, mm: 2 * c.zeta * np.sqrt(k * mm)
        cup_block = f'''<body name="cupbase" pos="0 0 {c.L_b}">
        <joint name="blx" type="slide" axis="1 0 0" stiffness="{k_lat:.6g}" damping="{zc(k_lat, m_b):.6g}"/>
        <joint name="bly" type="slide" axis="0 1 0" stiffness="{k_lat:.6g}" damping="{zc(k_lat, m_b):.6g}"/>
        <joint name="bz" type="slide" axis="0 0 1" range="0 {c.Z_stroke}" limited="true" solreflimit="{c.limit_solref[0]} {c.limit_solref[1]}" stiffness="{c.k_bz}" damping="{zc(c.k_bz, m_b):.6g}"/>
        <joint name="btx" type="hinge" axis="1 0 0" stiffness="{k_t:.6g}" damping="{zc(k_t, I_b):.6g}"/>
        <joint name="bty" type="hinge" axis="0 1 0" stiffness="{k_t:.6g}" damping="{zc(k_t, I_b):.6g}"/>
        <inertial pos="0 0 {-c.L_b / 2}" mass="{m_b:.6g}" diaginertia="{I_b:.4g} {I_b:.4g} {I_b:.4g}"/>
        <body name="cupring" pos="0 0 {-c.L_b}">
          <site name="cupc" pos="0 0 0"/>
          {''.join(lips)}
        </body>
      </body>'''
    else:
        cup_block = f'<site name="cupc" pos="0 0 0"/>{"".join(lips)}'
    sx, sy, sz = (s / 2 for s in c.size)
    m_shell = c.mass * (1 - c.slosh_frac)
    if c.shape == "cyl":
        I_xy, I_z = m_shell * (3 * sx ** 2 + c.size[2] ** 2) / 12, m_shell * sx ** 2 / 2
        inertia, geom = f"{I_xy:.4e} {I_xy:.4e} {I_z:.4e}", f'<geom name="prod" type="cylinder" size="{sx} {sz}" rgba="0.55 0.33 0.2 1"/>'
    else:
        inertia = (f"{m_shell / 12 * (c.size[1]**2 + c.size[2]**2):.4e} {m_shell / 12 * (c.size[0]**2 + c.size[2]**2):.4e} "
                   f"{m_shell / 12 * (c.size[0]**2 + c.size[1]**2):.4e}")
        geom = f'<geom name="prod" type="box" size="{sx} {sy} {sz}" rgba="0.85 0.66 0.36 1"/>'
    slosh = ""
    if c.slosh_frac > 0:
        m_s = c.mass * c.slosh_frac
        k_s = m_s * (2 * np.pi * c.slosh_hz) ** 2
        d_s = 2 * c.slosh_zeta * np.sqrt(k_s * m_s)
        slosh = f'''<body name="slosh" pos="{c.com[0]} {c.com[1]} {c.com[2]}">
        <joint name="slx" type="slide" axis="1 0 0" stiffness="{k_s:.6g}" damping="{d_s:.6g}" range="{-sx * 0.6} {sx * 0.6}" limited="true"/>
        <joint name="sly" type="slide" axis="0 1 0" stiffness="{k_s:.6g}" damping="{d_s:.6g}" range="{-sy * 0.6} {sy * 0.6}" limited="true"/>
        <inertial pos="0 0 0" mass="{m_s:.6g}" diaginertia="1e-7 1e-7 1e-7"/>
      </body>'''
    product = f'''<freejoint name="prod_free"/>
      <inertial pos="{c.com[0]} {c.com[1]} {c.com[2]}" mass="{m_shell}" diaginertia="{inertia}"/>
      {geom}{slosh}'''
    return cup_block, product, "".join(pairs)


def touch_offset(c: CupConfig) -> float:
    """Height of the cup frame (lip plane) above the product's top face at first contact, product centred: the
    sphere radius for a ring lip; 0 for a cone, whose frame sits at the seal circle on the product rim."""
    return 0.0 if c.lip == "cone" else c.lip_r


def generator(c: CupConfig):
    """The vacuum generator curve the cup sees: the ejector, behind the regulator if a set point is given."""
    ej = EJECTORS[c.ejector].parallel(c.n_gen)
    return ej.regulated(c.p_set_kpa) if c.p_set_kpa is not None else ej


class Cup:
    """Runtime physics of a cup inside any MuJoCo model built from cup_parts(): per-segment gaps from the contacts,
    vacuum force through mj_applyFT, pneumatics ODE and the vacuum sensor. `fitting_body` takes the vacuum reaction
    (the rigid fitting: the slide effector or the delta platform)."""

    def __init__(self, cfg: CupConfig, m: mujoco.MjModel, d: mujoco.MjData, fitting_body: int, seed: int = 0):
        c = self.cfg = cfg
        self.m, self.d = m, d
        self.lip_geoms = np.array([m.geom(f"lipg{i}").id for i in range(c.n_seg)])
        self.lig = {int(g): i for i, g in enumerate(self.lip_geoms)}
        self.prod_geom, self.prod_body = m.geom("prod").id, m.body("prod").id
        self.ring_body = fitting_body
        self.lip_qadr = np.array([m.jnt_qposadr[m.joint(f"lz{i}").id] for i in range(c.n_seg)])
        self.prod_qadr = m.jnt_qposadr[m.joint("prod_free").id]
        self.site = m.site("cupc").id
        self.leak = LipLeak(r_lip=c.r_lip, n_seg=c.n_seg, L_x=c.L_x, C_d=c.C_d, g0=c.g0, K_p=c.K_p)
        self.vac = VacuumCircuit(generator(c), V_cup=c.V_cup, n_sub=c.n_sub)
        self.vac.on = False
        self.sensor = VacuumSensor(rng=np.random.default_rng(seed))
        self.gaps = np.full(c.n_seg, c.g_det)
        self.ext_force = np.zeros(3)   # extra force on the product at its CoM (tests: d'Alembert loads)
        self.ext_torque = np.zeros(3)

    def read_gaps(self) -> np.ndarray:
        """Per-segment separation (m): 0 when touching or compressed, g_det when out of the detection band."""
        gaps = np.full(self.cfg.n_seg, self.cfg.g_det)
        d, lg, pg = self.d, self.lig, self.prod_geom
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in lg and g2 == pg:
                i = lg[g1]
            elif g2 in lg and g1 == pg:
                i = lg[g2]
            else:
                continue
            gaps[i] = min(gaps[i], max(0.0, con.dist))
        return gaps

    @property
    def dp(self) -> float:
        return self.cfg.frozen_dp if self.cfg.frozen_dp is not None else self.vac.vacuum

    @property
    def F_vac(self) -> float:
        return self.dp * self.cfg.A_eff

    def apply(self) -> None:
        """Before mj_step: vacuum force on the product along its top-face normal through the cup centre, reaction on
        the fitting; plus any external test load. Adds to qfrc_applied (the caller zeroes it each step)."""
        m, d = self.m, self.d
        F_vac = self.F_vac
        if F_vac > 0 and np.any(self.gaps < self.cfg.g_det):
            R = d.xmat[self.prod_body].reshape(3, 3)
            axis = R[:, 2].copy()
            if self.cfg.lip == "cone":
                # a bell seals on the product's own rim: the pressure resultant acts at the centre of its top face
                point = d.xpos[self.prod_body] + axis * self.cfg.size[2] / 2
            else:
                point = d.site_xpos[self.site].copy()
            mujoco.mj_applyFT(m, d, F_vac * axis, np.zeros(3), point, self.prod_body, d.qfrc_applied)
            mujoco.mj_applyFT(m, d, -F_vac * axis, np.zeros(3), point, self.ring_body, d.qfrc_applied)
        if np.any(self.ext_force) or np.any(self.ext_torque):
            mujoco.mj_applyFT(m, d, self.ext_force, self.ext_torque, d.xipos[self.prod_body].copy(), self.prod_body, d.qfrc_applied)

    def update(self, t: float, dt: float) -> None:
        """After mj_step: gaps from the contacts, pneumatics ODE, sensor."""
        c = self.cfg
        self.gaps = self.read_gaps()
        if c.frozen_dp is None:
            if c.fast_ode:
                step_lip_fast(self.vac, dt, self.leak, self.gaps)
            else:
                gaps = self.gaps
                self.vac.step(dt, lambda pc: self.leak.mdot(pc, gaps))
        self.sensor.update(t, dt, self.dp)

    def prod_rel(self):
        """Product position relative to the cup centre, and its tilt (deg)."""
        p = self.d.xpos[self.prod_body] - self.d.site_xpos[self.site]
        q = self.d.qpos[self.prod_qadr + 3:self.prod_qadr + 7]
        return p, np.degrees(2 * np.arccos(min(1.0, abs(q[0]))))

    def lip_compression(self) -> np.ndarray:
        return self.d.qpos[self.lip_qadr].copy()


class CupCell:
    """The cup on an x/y/z slide effector that tracks a reference exactly (the effector-only reduction)."""

    def __init__(self, cfg: CupConfig | None = None, seed: int = 0):
        self.cfg = c = cfg or CupConfig()
        sz = c.size[2] / 2
        cup_block, product, pairs = cup_parts(c)
        z0 = c.size[2] + touch_offset(c) + 0.001  # lips 1 mm above first contact at start
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
      {cup_block}
    </body>
    <body name="prod" pos="0 0 {sz}">
      {product}
    </body>
  </worldbody>
  <contact>{pairs}</contact>
  <actuator>
    <motor joint="ex" ctrlrange="-2000 2000"/><motor joint="ey" ctrlrange="-2000 2000"/><motor joint="ez" ctrlrange="-2000 2000"/>
  </actuator>
</mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.eff_body = m.body("eff").id
        # the vacuum reaction acts on the cup's ceiling (the rigid fitting on the effector); the lips and bellows
        # then carry the product's pull as compression ("lifting effect" of bellows cups)
        self.cup = Cup(c, m, self.d, self.eff_body, seed)
        for k in ("lip_geoms", "prod_geom", "prod_body", "ring_body", "lip_qadr", "prod_qadr", "site", "leak", "vac", "sensor"):
            setattr(self, k, getattr(self.cup, k))
        self.kp, self.kd, self.eff_mass = 4e5, 1.2e3, 0.5 + c.cup_mass
        self.t = 0.0
        self.rel0 = None
        mujoco.mj_forward(m, self.d)

    # gaps and external loads live on the Cup (tests set ext_force / read gaps through the cell)
    gaps = property(lambda self: self.cup.gaps)
    ext_force = property(lambda self: self.cup.ext_force, lambda self, v: setattr(self.cup, "ext_force", v))
    ext_torque = property(lambda self: self.cup.ext_torque, lambda self, v: setattr(self.cup, "ext_torque", v))

    def read_gaps(self) -> np.ndarray:
        return self.cup.read_gaps()

    @property
    def dp(self) -> float:
        return self.cup.dp

    def lip_compression(self) -> np.ndarray:
        return self.cup.lip_compression()

    def prod_rel(self):
        return self.cup.prod_rel()

    # ---------------------------------------------------------------- stepping
    def step(self, ref) -> None:
        """One MuJoCo step: effector tracks ref = (p, v, a); vacuum force from Δp; pneumatics advanced."""
        c, d, m = self.cfg, self.d, self.m
        p, v, a = ref
        q, qd = d.qpos[:3], d.qvel[:3]
        d.ctrl[:3] = self.eff_mass * (np.asarray(a) + [0, 0, 9.81]) + self.kp * (np.asarray(p) - q) + self.kd * (np.asarray(v) - qd) \
            + np.array([0, 0, self.cup.F_vac])  # feedforward for the vacuum reaction on the cup
        d.qfrc_applied[:] = 0
        self.cup.apply()
        mujoco.mj_step(m, d)
        self.t += c.dt
        self.cup.update(self.t, c.dt)

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
