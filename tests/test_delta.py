"""Gate 4 (docs/MODEL_SPEC.md §6): delta robot tests from docs/research/model-delta.md §5.6.

Run: .venv/bin/python -m pytest tests/test_delta.py -v -s
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import ruckig

from sim.delta import (PHI, DeltaController, DeltaParams, FeedforwardA, plant_friction, arm_points, build_model_a, ik, jacobians, to_motor, tool_xml,
                       udir)

P = DeltaParams()
P0 = np.array([-0.15, 0.0, -0.99 + P.tcp_off])   # platform centre; TCP 0.99 m below the base frame


def envelope_points(n_r=6, n_a=13, n_z=6):
    """ABB IRB 360-1/1130 working range: Ø1130 mm, TCP 0.865–1.115 m below the base frame."""
    for r, a, zt in itertools.product(np.linspace(0, 0.565, n_r), np.linspace(0, 2 * np.pi, n_a), np.linspace(0.865, 1.115, n_z)):
        yield np.array([r * np.cos(a), r * np.sin(a), -zt + P.tcp_off])   # platform centre (world)


# ---------------- V1 kinematics
def test_v1_ik_satisfies_strut_constraint_and_jacobian():
    rng = np.random.default_rng(0)
    worst_len, worst_jac = 0.0, 0.0
    for pw in list(envelope_points())[::7]:
        pm = to_motor(pw, P)
        th = ik(pm, P)
        assert th is not None
        for i, ph in enumerate(PHI):
            Pm, E = arm_points(th[i], ph, P)
            J = pw + P.r_p * udir(ph)
            worst_len = max(worst_len, abs(np.linalg.norm(J - E) - P.l))
        Jx, Jq = jacobians(pm, th, P)
        v = rng.normal(size=3)
        h = 1e-7
        thd_fd = (ik(pm + h * v, P) - ik(pm - h * v, P)) / (2 * h)
        thd = np.linalg.solve(Jq, Jx @ v)
        worst_jac = max(worst_jac, np.abs(thd - thd_fd).max() / max(np.abs(thd).max(), 1e-9))
    print(f"\nV1 strut length error max {worst_len * 1e9:.3f} nm; Jacobian vs finite differences rel. error {worst_jac:.2e}")
    assert worst_len < 1e-9 and worst_jac < 1e-6


def test_v1_assembled_model_matches_analytic():
    m, th0 = build_model_a(P, P0)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    viol = np.abs(d.efc_pos[:d.ne]).max()
    tcp_err = np.linalg.norm(d.site_xpos[m.site("tcp").id] - (P0 - [0, 0, P.tcp_off]))
    print(f"\nV1 assembled: max loop-closure violation {viol * 1e6:.3f} µm, TCP position error {tcp_err * 1e6:.3f} µm")
    assert viol < 1e-6 and tcp_err < 1e-6


# ---------------- V2 workspace
def test_v2_workspace_reachable_within_limits():
    lo, hi = np.deg2rad(P.q_range)
    pts, conds, th_all = 0, [], []
    for pw in envelope_points():
        pts += 1
        pm = to_motor(pw, P)
        th = ik(pm, P)
        assert th is not None, f"unreachable {pw}"
        assert np.all(th >= lo) and np.all(th <= hi), f"joint limit at {pw}: {np.rad2deg(th)}"
        Jx, Jq = jacobians(pm, th, P)
        J = np.linalg.solve(Jx, Jq)
        conds.append(np.linalg.cond(J))
        th_all.append(th)
    th_all = np.rad2deg(np.array(th_all))
    print(f"\nV2 {pts} envelope points reachable; θ range [{th_all.min():.1f}, {th_all.max():.1f}]° within "
          f"[{P.q_range[0]}, {P.q_range[1]}]°; Jacobian condition number max {max(conds):.1f}")
    assert pts == 468 and max(conds) < 50


# ---------------- V3 closed-chain integrity on a full-speed move
def x_move(p0w, dx=0.30, a=150.0, j=1e4, v=10.0):
    otg, inp, tr = ruckig.Ruckig(1), ruckig.InputParameter(1), ruckig.Trajectory(1)
    inp.current_position, inp.target_position = [0.0], [dx]
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = [v], [a], [j]
    otg.calculate(inp, tr)

    def ref(t):
        p, vv, aa = tr.at_time(min(max(t, 0.0), tr.duration))
        return p0w + np.array([p[0], 0, 0]), np.array([vv[0], 0, 0]), np.array([aa[0], 0, 0])
    return ref, tr.duration


def run_move(Pt: DeltaParams, exact_ff: bool = False):
    m, th0 = build_model_a(Pt, P0, extra_platform_xml=tool_xml(Pt))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ctrl = DeltaController(Pt, m, P0)
    ref, T = x_move(P0)
    ctrl.prepare(ref, T + 0.05, exact=FeedforwardA(Pt, P0, extra_platform_xml=tool_xml(Pt)) if exact_ff else None)
    tcp, plat = m.site("tcp").id, m.body("plat").id
    e_tcp = viol = tilt = tau_pk = fb_pk = 0.0
    for k in range(int((T + 0.05) / Pt.dt)):
        t = k * Pt.dt
        d.ctrl[:3] = ctrl.torque(d, t)
        plant_friction(m, d, Pt, ctrl.vadr)
        mujoco.mj_step(m, d)
        pw, _, _ = ref(t)   # after mj_step, site_xpos still describes the pre-integration state at time t
        e_tcp = max(e_tcp, np.linalg.norm(d.site_xpos[tcp] - (pw - [0, 0, Pt.tcp_off])))
        viol = max(viol, np.abs(d.efc_pos[:d.ne]).max())
        R = d.xmat[plat].reshape(3, 3)
        tilt = max(tilt, np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        tau_pk = max(tau_pk, np.abs(ctrl.last_tau).max())
        fb_pk = max(fb_pk, np.abs(ctrl.last_fb).max())   # PD share only (friction compensation is feedforward)
    return T, e_tcp, viol, tilt, tau_pk, fb_pk


def test_v3_closed_chain_integrity_full_speed():
    """Loop violation < 25 µm, platform tilt < 0.5 mrad (research §5.6), TCP path error below ABB's published
    path accuracy AT = 0.51 mm at 0.1 kg (ISO 9283, product spec §1.6.2)."""
    T, e_tcp, viol, tilt, tau_pk, _ = run_move(DeltaParams(m_tool=0.1))
    print(f"\nV3 300 mm in {T * 1e3:.1f} ms at 150 m/s²: TCP error {e_tcp * 1e3:.3f} mm (ABB AT 0.51), loop violation {viol * 1e6:.1f} µm, "
          f"platform tilt {tilt * 1e3:.3f} mrad, peak motor torque {tau_pk:.1f} N·m (prior inertias)")
    assert e_tcp < 0.51e-3 and viol < 25e-6 and tilt < 0.5e-3


def test_v4_feedforward_quality():
    """Feedback torque (PD share) must stay below 5 % of peak. The lumped Codourey model cannot meet this at a
    0.1 kg payload for any strut-mass split (reported), so the exact closed-chain feedforward is used and tested."""
    for r in (0.5, 2 / 3):
        T, e_tcp, viol, tilt, tau_pk, fb_pk = run_move(DeltaParams(m_tool=0.1, r_elbow=r))
        print(f"\nV4 lumped r_elbow={r:.2f}: feedback {fb_pk:.2f} N·m = {fb_pk / tau_pk * 100:.1f} % of peak; TCP {e_tcp * 1e3:.3f} mm")
    T, e_tcp, viol, tilt, tau_pk, fb_pk = run_move(DeltaParams(m_tool=0.1), exact_ff=True)
    print(f"V4 exact closed-chain feedforward: feedback {fb_pk:.2f} N·m = {fb_pk / tau_pk * 100:.1f} % of peak {tau_pk:.1f}; TCP {e_tcp * 1e3:.3f} mm")
    assert fb_pk / tau_pk < 0.05


# ---------------- V5–V7: ABB cycle times and base loads (calibrated: τ_max, θ̇_max; scripts/calibrate_delta.py)
from sim.delta_motion import ABB_CYCLES, MoveSpec, base_loads, cycle_time  # noqa: E402

import pytest  # noqa: E402


def _cycle(cyc, kg):
    t, _ = cycle_time(DeltaParams(m_tool=kg), cyc, MoveSpec())
    ref_t = ABB_CYCLES[(cyc, kg)]
    print(f"\n{cyc} @ {kg} kg: {t * 1e3:.1f} ms vs ABB {ref_t * 1e3:.0f} ms ({(t / ref_t - 1) * 100:+.1f} %)")
    return t, ref_t


@pytest.mark.parametrize("cyc,kg", [("90/400/90", 0.1), ("25/305/25", 1.0)])
def test_v5_fitted_cycles_reproduced(cyc, kg):
    """The two calibration points: arm speed limit on 90/400/90 @ 0.1 kg, torque limit on 25/305/25 @ 1 kg."""
    t, ref_t = _cycle(cyc, kg)
    assert abs(t / ref_t - 1) < 0.01


def test_v6_heldout_cycle():
    """Held out (not used in any fit): 90/400/90 @ 1 kg, ±5 % (ABB rounds ppm to 5 and calls figures approximate)."""
    t, ref_t = _cycle("90/400/90", 1.0)
    assert abs(t / ref_t - 1) < 0.05


@pytest.mark.xfail(strict=True, reason="Known gap: the short 25 mm-lift cycle at 0.1 kg is ~7 % slow. Even with every "
                   "calibratable limit non-binding, our path needs ≥ 309 ms (+3 %); the residual is ABB's unpublished "
                   "PickMaster path shape / dwell timing (docs/research/delta-validation.md).")
def test_v6_unfitted_short_cycle_light():
    t, ref_t = _cycle("25/305/25", 0.1)
    assert abs(t / ref_t - 1) < 0.05


@pytest.mark.parametrize("cyc", ["25/305/25", "90/400/90"])
def test_v7_base_loads_full_chain(cyc):
    """Full closed chain, 1 kg, exact feedforward: base loads vs ABB PS §1.3.3. The planner enforces the limits with the
    lumped momentum model; the full chain may exceed them by the lumped model's error, bounded here at 2 %."""
    bl = base_loads(DeltaParams(m_tool=1.0), cyc, MoveSpec())
    print(f"\nV7 {cyc} @ 1 kg: Fx {bl['Fx']:.0f}/330 N, Fy {bl['Fy']:.0f}/260 N, dyn Fz {bl['Fz_dyn']:.0f}/170 N, Mz {bl['Mz']:.1f}/100 N·m; "
          f"TCP tracking {bl['tcp_err_mm']:.3f} mm, peak τ {bl['tau_peak']:.1f} N·m")
    assert bl["Fx"] <= 330 * 1.02 and bl["Fy"] <= 260 * 1.02 and bl["Fz_dyn"] <= 170 * 1.02 and bl["Mz"] <= 100
    assert bl["tcp_err_mm"] < 0.51
