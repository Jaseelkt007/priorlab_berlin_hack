"""The pick line (gate 5): the compliant suction cup on the full closed-chain delta, picking from a moving belt.

One pick = one simulated cycle, as on a real PickMaster line (docs/research/model-delta.md §3.3):
  1. approach: empty move from the previous place point to the product, tracking the belt (full speed);
  2. touch and dwell 35 ms with the vacuum on (ABB's air-activation time). The vacuum sensor's reading at the end of
     the dwell (seal_kpa) and the evacuation time are known here, BEFORE the loaded move: the speed decision point;
  3. transfer: loaded move to the place pad at the speed setting alpha (ABB AccSet: scales the acceleration and jerk
     caps; QuickMove-style limits stay on);
  4. outcome at arrival, just before release: dropped, and place_offset_mm = product centre vs the place target.

Belt tracking ("moving work object"): the planner's paths live in a plan frame; world = plan + ŷ·v_b·D(t), where the
belt clock D runs at rate 0 → 1 during the approach (quintic ramp), 1 while tracking, 1 → 0 during the transfer. The
extra acceleration v_b·1.875/T is reserved from the planner's acceleration cap.
Vision: the product position is estimated with Gaussian error (σ from ABB's tracking-repeatability table).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .cup import Cup, CupConfig, cup_parts, generator, touch_offset
from .delta import DeltaController, DeltaParams, build_model_a, plant_friction
from .delta_motion import MoveSpec, Planner

DWELL = 0.035           # s, ABB air activation at pick and at place
Z_BELT = -1.000         # m, belt top surface (robot base frame); the cycle height is ~0.99 m below the base
X_PICK, X_PLACE = -0.1525, 0.1525   # 305 mm transfer, as ABB's benchmark cycle
H_LIFT = 0.025          # m, depart and approach heights (25/305/25)


_APPROACH: dict = {}    # the last approach plan (exact reuse across the alphas of one pick instance, per process)


def belt_xml(v_belt: float) -> str:
    """Infeed belt: a static box whose surface moves at v_belt along +y (MuJoCo geom/surfacevel, official
    model/surfacevel examples). Friction drives the products; no belt degrees of freedom."""
    return (f'<geom name="belt" type="box" size="0.08 0.6 0.01" pos="{X_PICK} 0 {Z_BELT - 0.01}" '
            f'surfacevel="0 {v_belt} 0 0 0 0" friction="0.5 0.005 0.0001" rgba=".18 .19 .21 1"/>')


def tracking_sigma_mm(v_belt: float) -> float:
    """ABB IRB 360 conveyor-tracking repeatability (datasheet): 1.0 mm at 200 mm/s, 1.5 mm at 350–750 mm/s,
    5.0 mm at 800–1400 mm/s. Linear between the published points."""
    return float(np.interp(v_belt, [0.0, 0.2, 0.35, 0.75, 0.8, 1.4], [1.0, 1.0, 1.5, 1.5, 5.0, 5.0]))


@dataclass
class PickSetup:
    cup: CupConfig = field(default_factory=CupConfig)
    v_belt: float = 0.3                 # m/s
    alpha: float = 1.0                  # speed setting for the loaded move
    vision_sigma_mm: float | None = None  # None: from the ABB tracking table at v_belt
    tool_mass: float = 0.1              # kg, gripper (cup + fitting) as carried by the robot
    press: float = 0.0006               # m, commanded lip compression at touch
    seed: int = 0


def _smooth_rate(u):
    """Quintic smoothstep s(u) on [0, 1], and its integral ∫0^u s."""
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5, 2.5 * u ** 4 - 3 * u ** 5 + u ** 6


class PickCell:
    def __init__(self, s: PickSetup):
        self.s = s
        c = s.cup
        self.rng = np.random.default_rng(s.seed)
        P = self.P = DeltaParams(m_tool=s.tool_mass + c.mass, dt=c.dt)   # the controller knows the nominal payload
        self.z_top = Z_BELT + c.size[2]                       # product top
        self.z_touch = self.z_top + touch_offset(c) - s.press  # lip-plane height at touch
        self.place = np.array([X_PLACE, 0.0, Z_BELT])         # place target (product bottom centre)
        # product: on the belt, its centre will cross y = 0 at t_cross; the robot sees it through vision
        sig = (tracking_sigma_mm(s.v_belt) if s.vision_sigma_mm is None else s.vision_sigma_mm) * 1e-3
        self.vision_err = np.r_[self.rng.normal(0, sig, 2), 0.0]
        self._plan()
        y0 = -s.v_belt * self.t_touch                         # product at y = 0 at touch time
        self.prod0 = np.array([X_PICK, y0, Z_BELT + c.size[2] / 2])
        self._build()

    # ------------------------------------------------------------------ motion
    def _plan(self):
        s, P = self.s, self.P
        home = np.array([0.0, 0.0, self.z_touch + P.tcp_off])
        pl = Planner(P, home)
        vb = s.v_belt
        est = np.array([X_PICK, 0.0, 0.0]) + self.vision_err     # estimated product position at touch (y = 0)
        tcp_pick_w = np.array([est[0], est[1], self.z_touch])      # world TCP at touch
        tcp_place_w = np.array([X_PLACE, 0.0, self.z_touch])
        tcp_start_w = tcp_place_w.copy()                          # previous cycle ended at the place point
        # the belt offsets depend on the move durations: fixed-point iteration, first for the approach (which does not
        # depend on alpha, so the several alphas of one pick instance reuse it exactly), then for the transfer.
        # 1 ms in T moves the belt offset by v_b·ΔT/2 ≤ 0.2 mm, well below the 1–1.5 mm vision error.
        key = (P.m_tool, P.dt, vb, *tcp_pick_w, *tcp_start_w)
        if key in _APPROACH:
            r1, T1, pick_plan = _APPROACH[key]
        else:
            T1 = 0.13
            for _ in range(4):
                pick_plan = tcp_pick_w - [0, vb * T1 / 2, 0]
                r1, T1n, _ = pl.plan(tcp_start_w, pick_plan, H_LIFT, H_LIFT, MoveSpec(alpha=1.0, a_max=150.0 - vb * 1.875 / T1))
                done, T1 = abs(T1n - T1) < 1e-3, T1n
                if done:
                    break
            _APPROACH.clear()
            _APPROACH[key] = (r1, T1, pick_plan)
        T2 = 0.13
        for _ in range(4):
            place_plan = tcp_place_w - [0, vb * (T1 / 2 + DWELL + T2 / 2), 0]
            spec2 = MoveSpec(alpha=s.alpha, a_max=(150.0 * s.alpha - vb * 1.875 / T2) / s.alpha)
            r2, T2n, info2 = pl.plan(pick_plan, place_plan, H_LIFT, H_LIFT, spec2)
            done, T2 = abs(T2n - T2) < 1e-3, T2n
            if done:
                break
        self.move_info = info2
        self.t_touch = T1
        self.t_lift = T1 + DWELL
        self.t_arrive = T1 + DWELL + T2
        self.T_total = self.t_arrive + 0.005
        off = np.array([0, 0, P.tcp_off])

        def D(t):   # belt clock and its derivatives
            if t < T1:
                r, I = _smooth_rate(t / T1)
                return T1 * I, r, 0.0
            if t < T1 + DWELL:
                return T1 / 2 + (t - T1), 1.0, 0.0
            u = t - T1 - DWELL
            if u < T2:
                r, I = _smooth_rate(u / T2)
                return T1 / 2 + DWELL + u - T2 * I, 1.0 - r, 0.0
            return T1 / 2 + DWELL + T2 / 2, 0.0, 0.0

        def ref(t):
            """Platform-centre reference (p, v, a) in the world, belt tracking included."""
            if t < T1:
                p, v, a = r1(t)
            elif t < T1 + DWELL:
                p, v, a = r1(T1)
            else:
                p, v, a = r2(t - T1 - DWELL)
            h = 1e-5
            Dt, rate, _ = D(t)
            acc = (D(t + h)[1] - D(t - h)[1]) / (2 * h)
            y = np.array([0, 1.0, 0])
            return p + vb * Dt * y, v + vb * rate * y, a + vb * acc * y

        self.ref = ref
        self.tcp_ref = lambda t: ref(t)[0] - off

    # ------------------------------------------------------------------ model
    def _build(self):
        s, P, c = self.s, self.P, self.s.cup
        cup_block, product, pairs = cup_parts(c)
        fitting_mass = max(s.tool_mass - c.cup_mass, 1e-3)
        plat = (f'<body name="tool" pos="0 0 {-P.tcp_off}"><inertial pos="0 0 0.01" mass="{fitting_mass}" '
                f'diaginertia="1e-5 1e-5 1e-5"/><geom type="cylinder" size="0.012 0.02" pos="0 0 0.022" rgba=".1 .1 .1 1" '
                f'contype="0" conaffinity="0" mass="0"/>{cup_block}</body>')
        x, y, z = self.prod0
        world = f"""
    {belt_xml(s.v_belt)}
    <geom name="pad" type="box" size="0.06 0.06 0.01" pos="{X_PLACE} 0 {Z_BELT - 0.01}" friction="0.5 0.005 0.0001" rgba=".2 .75 .4 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" pos="0 0 {Z_BELT - 0.30}" rgba=".86 .87 .89 1"/>
    <light pos="0.6 -0.6 0.8" dir="-0.5 0.5 -1" diffuse=".7 .7 .7"/>
    <body name="prod" pos="{x} {y} {z}">{product}</body>"""
        p_start = self.ref(0.0)[0]
        self.m, _ = build_model_a(P, p_start, extra_platform_xml=plat, extra_world_xml=world, extra_contact_xml=pairs,
                                  option_xml=f'noslip_iterations="{c.noslip}"')
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1280, 720
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        # product already moving with the belt (settled; test V9 checks the belt drives it to v_b)
        self.cup = Cup(c, m, d, m.body("tool").id, seed=s.seed)
        d.qvel[m.jnt_dofadr[m.joint("prod_free").id] + 1] = s.v_belt
        mujoco.mj_forward(m, d)
        self.ctrl = DeltaController(P, m, p_start)
        self.ctrl.prepare(self.ref, self.T_total + 0.02)

    # ------------------------------------------------------------------ run
    def run(self, rec=None) -> dict:
        """Simulate the cycle. Sensor-derived features use only the vacuum sensor's sampled readings (435 Hz)."""
        m, d, c, P = self.m, self.d, self.s.cup, self.P
        cup = self.cup
        n = int(round(self.T_total / P.dt))
        thr = 0.5 * generator(c).dp_max / 1e3            # evacuation time: to half the generator's (regulated) max
        vac_on_t = rel_lift = None
        n_read = 0
        for k in range(n):
            t = k * P.dt
            if vac_on_t is None and t >= self.t_touch:
                cup.vac.on, vac_on_t = True, t
            if rel_lift is None and t >= self.t_lift:
                rel_lift = cup.prod_rel()[0].copy()
                n_read = len(cup.sensor.readings)
            d.qfrc_applied[:] = 0
            d.ctrl[:3] = self.ctrl.torque(d, t)
            plant_friction(m, d, P, self.ctrl.vadr)
            cup.apply()
            mujoco.mj_step(m, d)
            cup.update(t + P.dt, P.dt)
            if rec:
                rec(self, t)
        R = np.array(cup.sensor.readings) if cup.sensor.readings else np.zeros((0, 2))
        before = R[(R[:, 0] >= self.t_touch) & (R[:, 0] < self.t_lift)]
        seal_kpa = float(before[-3:, 1].mean()) if len(before) else 0.0        # last ~7 ms of the dwell
        hit = before[before[:, 1] >= thr]
        evac_ms = float((hit[0, 0] - self.t_touch) * 1e3) if len(hit) else np.nan  # NaN: threshold not reached
        during = R[n_read:, 1]
        # outcome at arrival, before release: held = still in contact and still where it was in the cup
        rel = cup.prod_rel()[0]
        held = bool(np.any(cup.gaps < 50e-6) and np.linalg.norm(rel - rel_lift) < 0.015)
        prod = d.xpos[cup.prod_body]
        return dict(dropped=not held,
                    place_offset_mm=float(np.linalg.norm(prod[:2] - self.place[:2]) * 1e3) if held else np.nan,
                    slip_mm=float(np.linalg.norm((rel - rel_lift)[:2]) * 1e3),
                    seal_kpa=seal_kpa, evac_ms=evac_ms,
                    vac_min_kpa=float(during.min()) if len(during) else np.nan,
                    vision_err_mm=float(np.linalg.norm(self.vision_err[:2]) * 1e3),
                    move_ms=(self.t_arrive - self.t_lift) * 1e3, a_peak=float(self.move_info["a_peak"]),
                    cycle_s=self.t_touch + 2 * DWELL + (self.t_arrive - self.t_lift))
