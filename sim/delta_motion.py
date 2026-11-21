"""Pick-cycle motion for the delta (gate 4, docs/research/model-delta.md §3.1, §5.3).

- Path: vertical up h1, transfer, vertical down h2 (ABB 25/305/25 and 90/400/90 cycles), corners blended with
  circular arcs of radius r_c (ABB's zone size is not published, so it is swept).
- Time scaling: TOPP-RA (Pham & Pham 2018), the time-optimal parameterization ABB documents for QuickMove,
  under: speed ≤ v_max; |acceleration| ≤ α·a_max (polyhedral bound, conservative); motor torque ≤ 0.9·τ_max
  from the lumped dynamics (validated within ~4 % in test V4); dynamic base force ≤ ABB's foundation loads.
- Speed: |TCP velocity| ≤ v_max and every arm |θ̇| ≤ θ̇_max (Gnad et al.'s data-sheet value for the IRB 360).
- Jerk: Biagiotti & Melchiorri's FIR method. A moving average of length T_r = 2·a_max / j_max on the path parameter
  limits jerk to ≤ j (ABB's AccSet "Ramp"). All limits are re-checked on the filtered profile, and the caps are lowered
  until they hold.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

import mujoco

from .delta import (DeltaController, DeltaParams, Feedforward, FeedforwardA, base_force_lumped, build_model_a, ik,
                    jacobians, plant_friction, to_motor, tool_xml)

ta.setup_logging("WARNING")


@dataclass
class MoveSpec:
    alpha: float = 1.0          # speed override: scales the acceleration and jerk caps (ABB AccSet)
    a_max: float = 150.0        # m/s² (IRB 360-1/1130)
    v_max: float = 10.0         # m/s
    j_max: float = 1.6e5        # m/s³: above ~1e5 jerk no longer binds the ABB cycles (calibration sweep)
    r_c: float = np.inf         # corner rounding; inf = full arch (control offset D/3). Not published; smaller only slows
    tau_margin: float = 0.9     # use ≤ 90 % of τ_max
    # ABB IRB 360-1/1130 foundation loads in operation (product spec §1.3.3): Fx ±330, Fy ±260, Fz −1500 ± 170 N.
    # The real controller keeps the robot inside them, so the planner does too (lumped momentum model, checked on
    # the full closed chain in V7).
    F_base: tuple = (330.0, 260.0, 170.0)
    # arm (gearbox output) speed limit. Prior: 1088 °/s (Gnad et al., "according to the data sheet", their IRB 360).
    # Calibrated on ABB's 90/400/90 cycle at 0.1 kg (scripts/calibrate_delta.py, docs/research/delta-validation.md).
    thd_max: float = np.deg2rad(622.0)
    n_grid: int = 160           # TOPP-RA grid points


def cycle_path(p_a: np.ndarray, p_b: np.ndarray, h1: float, h2: float, r_c: float, n: int = 400):
    """Curvature-continuous pick path: a clamped quintic B-spline over the up–across–down control polygon
    (B-spline U-turns as used for the IRB 360 by Gnad et al.). It leaves and arrives vertically, stays in the
    convex hull of the polygon, and r_c sets how far the corners are rounded. Returns arc-length samples (s, p)."""
    from scipy.interpolate import BSpline
    d = p_b[:2] - p_a[:2]
    D = np.linalg.norm(d)
    u = np.r_[d / D, 0.0] if D > 1e-9 else np.array([1.0, 0, 0])
    z_top = max(p_a[2] + h1, p_b[2] + h2)
    up1, up2 = z_top - p_a[2], z_top - p_b[2]
    c = min(r_c, D / 3)                      # r_c = inf: full arch
    top_a = np.array([p_a[0], p_a[1], z_top])
    top_b = np.array([p_b[0], p_b[1], z_top])
    ctrl = np.array([p_a, p_a + [0, 0, up1 * 0.5], top_a, top_a + c * u, top_b - c * u, top_b, p_b + [0, 0, up2 * 0.5], p_b])
    k = 5
    nc = len(ctrl)
    knots = np.r_[np.zeros(k + 1), np.linspace(0, 1, nc - k + 1)[1:-1], np.ones(k + 1)]
    spl = BSpline(knots, ctrl, k)
    uu = np.linspace(0, 1, 20001)
    pts = spl(uu)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    ss = np.linspace(0, s[-1], n)
    uq = np.interp(ss, s, uu)
    return ss, spl(uq)


def _sphere_normals(n: int = 400):
    """n near-uniform unit directions (Fibonacci sphere) for a polyhedral bound on the acceleration norm.
    TOPP-RA takes only linear constraints, and |a| ≤ A is a cone; n·a ≤ A·cos(δ_max) for all normals is an inner
    approximation (conservative). With 400 normals, cos(δ_max) ≈ 0.99, so it costs ~1 % of acceleration."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


NORMALS = _sphere_normals()


def _poly_scale(normals: np.ndarray, n_test: int = 200000) -> float:
    """cos of the worst angle between any direction and its nearest normal (dense random check)."""
    v = np.random.default_rng(0).normal(size=(n_test, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return float(np.min(np.max(v @ normals.T, axis=1)))


POLY_SCALE = _poly_scale(NORMALS)


class Planner:
    def __init__(self, P: DeltaParams, p_home_platform: np.ndarray):
        self.P = P
        # torque model: lumped ⅔/⅓ (V4: 4.1 % of peak vs the exact model)
        self.dyn = Feedforward(dataclasses.replace(P, r_elbow=2 / 3, r_plat=1 / 3), p_home_platform)

    def plan(self, tcp_a, tcp_b, h1: float, h2: float, spec: MoveSpec, max_iter: int = 15):
        """Time-optimal move under all limits. The FIR jerk filter and the path curvature couple speed and acceleration,
        so the caps are lowered in proportion to any overshoot and the plan is redone until every limit holds.
        Returns (ref, T, info); ref(t) -> platform-centre (p, v, a)."""
        # one back-off factor per limit: TOPP-RA enforces each at its grid points, the jerk filter then rounds the
        # profile, so any overshoot is removed by tightening that limit alone (and acceleration for geometric jerk)
        k_a = k_t = k_F = 1.0
        for it in range(max_iter):
            sp = dataclasses.replace(spec, a_max=spec.a_max * k_a, tau_margin=spec.tau_margin * k_t,
                                     F_base=tuple(np.array(spec.F_base) * k_F))
            ref, T, info = self._plan_once(tcp_a, tcp_b, h1, h2, sp)
            cap_tau = spec.tau_margin * self.P.tau_max
            ra = info["a_peak"] / (spec.alpha * spec.a_max)
            rj = info["j_peak"] / (spec.alpha * spec.j_max)
            rt = info["tau_peak"] / cap_tau
            rF = info["rF"] * k_F                      # rF was measured against the tightened limits
            info.update(iterations=it + 1, k_a=k_a, k_t=k_t, k_F=k_F, tau_cap=cap_tau, rF=rF)
            rv = info["thd_peak"] / spec.thd_max
            if ra <= 1.0 and rj <= 1.0 and rt <= 1.0 and rF <= 1.0 and rv <= 1.0 + 1e-3:
                return ref, T, info
            if max(ra, rj ** (2 / 3)) > 1:
                k_a *= 0.99 / max(ra, rj ** (2 / 3))
            if rt > 1:
                k_t *= 0.99 / rt
            if rF > 1:
                k_F *= 0.99 / rF
        raise RuntimeError(f"limits not met after {max_iter} iterations: {info}")

    def _plan_once(self, tcp_a, tcp_b, h1: float, h2: float, spec: MoveSpec):
        P = self.P
        ss, path_pts = cycle_path(np.asarray(tcp_a, float), np.asarray(tcp_b, float), h1, h2, spec.r_c)
        path = ta.SplineInterpolator(ss, path_pts)
        A = spec.alpha * spec.a_max * POLY_SCALE
        F_acc = NORMALS
        g_acc = np.full(len(NORMALS), A)
        acc_c = constraint.SecondOrderConstraint(lambda q, qd, qdd: qdd, lambda q: F_acc, lambda q: g_acc, dof=3,
                                                 discretization_scheme=constraint.DiscretizationType.Interpolation)
        tau_lim = spec.tau_margin * P.tau_max
        off = np.array([0, 0, P.tcp_off])

        Pl = self.dyn.P

        def inv_dyn(q, qd, qdd):
            pw, vw, aw = np.asarray(q) + off, np.asarray(qd), np.asarray(qdd)
            tau, _, _ = self.dyn.torque(pw, vw, aw)
            return np.r_[tau, base_force_lumped(pw, vw, aw, Pl)]

        I6 = np.eye(6)
        F_tau, g_tau = np.vstack([I6, -I6]), np.r_[np.full(3, tau_lim), spec.F_base, np.full(3, tau_lim), spec.F_base]
        tau_c = constraint.SecondOrderConstraint(inv_dyn, lambda q: F_tau, lambda q: g_tau, dof=3,
                                                 discretization_scheme=constraint.DiscretizationType.Interpolation)
        def sdot_max(sv):
            # ṡ ≤ v_max (arc-length path) and every arm: |θ̇_i| = |(J_q⁻¹ J_x p'(s))_i| ṡ ≤ θ̇_max
            out = np.empty(len(sv))
            for i, x in enumerate(sv):
                pm = to_motor(path(x) + off, P)
                Jx, Jq = jacobians(pm, ik(pm, P), P)
                g = np.abs(np.linalg.solve(Jq, Jx @ path(x, 1))).max()
                out[i] = min(spec.v_max, spec.thd_max / max(g, 1e-12))
            return out

        def vlim(sv):
            # TOPP-RA bounds q̇_k = p'_k(s) ṡ per axis; bounds ∝ |p'_k| make the binding one ṡ ≤ sdot_max(s)
            scalar = np.ndim(sv) == 0
            sv = np.atleast_1d(sv)
            b = sdot_max(sv)[:, None] * np.abs(path(sv, 1)) + 1e-9
            out = np.stack([-b, b], axis=-1)        # (N, dof, 2); TOPP-RA expects (dof, 2) for a scalar s
            return out[0] if scalar else out

        vel_c = constraint.JointVelocityConstraintVarying(vlim)
        grid = np.linspace(0, ss[-1], spec.n_grid)
        inst = algo.TOPPRA([vel_c, acc_c, tau_c], path, gridpoints=grid)
        sdd_g, sd_g, _ = inst.compute_parameterization(0, 0)
        if sd_g is None or np.any(np.isnan(sd_g)):
            raise RuntimeError("TOPP-RA found no feasible time scaling")
        # exact s(t): constant path acceleration between grid points (TOPP-RA's discretization)
        sdot = np.maximum(np.asarray(sd_g, float), 0.0)   # TOPP-RA returns path velocities ṡ at the grid points
        ds = np.diff(grid)
        dt_seg = 2 * ds / np.maximum(sdot[:-1] + sdot[1:], 1e-12)
        t_grid = np.r_[0.0, np.cumsum(dt_seg)]
        T0 = t_grid[-1]
        dt = 2e-5
        t = np.arange(0, T0 + dt, dt)
        k = np.clip(np.searchsorted(t_grid, t, side="right") - 1, 0, len(ds) - 1)
        tau_in = t - t_grid[k]
        acc_seg = (sdot[k + 1] - sdot[k]) / np.maximum(dt_seg[k], 1e-12)
        s_t = np.minimum(grid[k] + sdot[k] * tau_in + 0.5 * acc_seg * tau_in ** 2, grid[-1])
        # TOPP-RA's acceleration is bang-bang and can switch from +a to −a (a jump of 2a), so the moving average must
        # span T_r = 2a/j to keep jerk ≤ j. α scales a and j together, so T_r does not depend on α.
        T_r = 2 * spec.a_max / spec.j_max
        n_r = max(1, int(round(T_r / dt)))
        pad = np.r_[np.zeros(n_r), s_t, np.full(n_r, s_t[-1])]
        s_f = np.convolve(pad, np.ones(n_r) / n_r, mode="valid")[: len(s_t) + n_r]
        tf = np.arange(len(s_f)) * dt
        T = tf[-1]
        sd = np.gradient(s_f, dt)
        sdd = np.gradient(sd, dt)

        def ref(tt):
            tt = min(max(tt, 0.0), T)
            sv = np.interp(tt, tf, s_f)
            v1 = np.interp(tt, tf, sd)
            a1 = np.interp(tt, tf, sdd)
            p = path(sv)
            dp, ddp = path(sv, 1), path(sv, 2)
            return p + off, dp * v1, dp * a1 + ddp * v1 * v1

        # re-check the limits on the filtered profile
        chk = np.linspace(0, T, 600)
        acc = np.array([np.linalg.norm(ref(x)[2]) for x in chk])
        vel = np.array([np.linalg.norm(ref(x)[1]) for x in chk])
        thd_pk = max(np.abs(self.dyn.torque(*ref(x))[2]).max() for x in chk)
        jerk = np.linalg.norm(np.gradient(np.array([ref(x)[2] for x in chk]), chk, axis=0), axis=1).max()
        # motor torque incl. the gearbox friction the plant has (Gnad et al.: Coulomb + viscous)
        tau = []
        for x in chk:
            tq, _, thd = self.dyn.torque(*ref(x))
            tau.append(tq + P.fric_coulomb * np.tanh(thd / P.fric_vs) + P.fric_viscous * thd)
        tau = np.array(tau)
        Fb = np.array([base_force_lumped(*ref(x), self.dyn.P) for x in chk])
        rF = float((np.abs(Fb) / np.array(spec.F_base)).max())
        tau_static = float(np.abs(self.dyn.torque(ref(0)[0], np.zeros(3), np.zeros(3))[0]).max())
        info = dict(thd_peak=thd_pk, rF=rF, F_peak=np.abs(Fb).max(axis=0), tau_static=tau_static, T_topp=T0, T=T, a_peak=acc.max(), v_peak=vel.max(), j_peak=jerk, tau_peak=np.abs(tau).max(),
                    a_cap=spec.alpha * spec.a_max, tau_cap=tau_lim, length=ss[-1])
        return ref, T, info


# ---------------- ABB benchmark cycles (product spec §1.7.1): out, dwell, back, dwell
Z_TCP = -0.99                     # TCP height of the cycle (mid working range)
DWELL = 0.035
ABB_CYCLES = {("25/305/25", 0.1): 0.300, ("25/305/25", 1.0): 0.353, ("90/400/90", 0.1): 0.444, ("90/400/90", 1.0): 0.500}
CYCLES = {"25/305/25": (0.025, 0.305), "90/400/90": (0.090, 0.400)}


def cycle_time(P: DeltaParams, cyc: str, spec: MoveSpec):
    h, D = CYCLES[cyc]
    a = np.array([-D / 2, 0, Z_TCP])
    b = np.array([D / 2, 0, Z_TCP])
    pl = Planner(P, np.array([0, 0, Z_TCP + P.tcp_off]))
    ref, T, info = pl.plan(a, b, h, h, spec)
    return 2 * T + 2 * DWELL, (ref, T, info, a, b)


def base_loads(P: DeltaParams, cyc: str, spec: MoveSpec):
    """Run the cycle's outward move on the full closed chain; base force = −Σ m_i(a_i − g) over moving bodies."""
    t_cyc, (ref, T, info, a, b) = cycle_time(P, cyc, spec)
    p_start = ref(0)[0]
    m, _ = build_model_a(P, p_start, extra_platform_xml=tool_xml(P))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ctrl = DeltaController(P, m, p_start)
    ctrl.use_table(FeedforwardA(P, p_start, extra_platform_xml=tool_xml(P)).precompute(ref, T + 0.02, P.dt))
    Pm, e_tcp, tau_pk = [], 0.0, 0.0
    tcp = m.site("tcp").id
    M = float(m.body_mass[1:].sum())
    for k in range(int((T + 0.02) / P.dt)):
        t = k * P.dt
        d.ctrl[:3] = ctrl.torque(d, ref, t)
        plant_friction(m, d, P, ctrl.vadr)
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        # linear momentum, and angular momentum about the base origin (L_com + c × p)
        p_lin = M * d.subtree_linvel[0].copy()
        Pm.append(np.r_[p_lin, d.subtree_angmom[0] + np.cross(d.subtree_com[0], p_lin)])
        e_tcp = max(e_tcp, np.linalg.norm(d.site_xpos[tcp] - (ref(t + P.dt)[0] - [0, 0, P.tcp_off])))
        tau_pk = max(tau_pk, np.abs(ctrl.last_tau).max())
    F = np.gradient(np.array(Pm), P.dt, axis=0)       # dynamic base force/moment = dP/dt, dL/dt (weight excluded)
    return dict(Fx=float(np.abs(F[:, 0]).max()), Fy=float(np.abs(F[:, 1]).max()), Fz_dyn=float(np.abs(F[:, 2]).max()),
                Mz=float(np.abs(F[:, 5]).max()), planned_F=info["F_peak"].tolist(),
                tcp_err_mm=e_tcp * 1e3, tau_peak=tau_pk)
