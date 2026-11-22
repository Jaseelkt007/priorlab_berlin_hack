"""Gate 5 (docs/MODEL_SPEC.md §6): the line. Tests from docs/research/model-delta.md §5.6 (V9, V10) plus an external
check of the drop boundary against the vendor sizing rule.

Run: .venv/bin/python -m pytest tests/test_line.py -v -s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import pytest

from sim.cup import CupConfig
from sim.line import X_PICK, Z_BELT, PickCell, PickSetup, belt_xml, tracking_sigma_mm


# ---------------- V9 conveyor
@pytest.mark.parametrize("v_b", [0.2, 0.5, 1.4])
def test_v9_belt_drives_product(v_b):
    """A product dropped on the belt reaches belt speed (< 0.5 % error) and then tracks it (< 1 mm per s). The
    measurement window is sized so the product stays on the 1.2 m belt (≤ 0.5 m of travel), and that is checked."""
    xml = f"""<mujoco><option timestep="2e-4" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="10"/>
      <worldbody>{belt_xml(v_b)}
        <body name="p" pos="{X_PICK} -0.55 {Z_BELT + 0.0105}"><freejoint/><geom type="box" size="0.025 0.025 0.01" mass="0.05"/></body>
      </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    while d.time < 0.4:
        mujoco.mj_step(m, d)
    y0, t0 = d.qpos[1], d.time
    while d.time < t0 + min(1.0, 0.5 / v_b):
        mujoco.mj_step(m, d)
    v = d.qvel[1]
    drift = abs((d.qpos[1] - y0) - v_b * (d.time - t0)) / (d.time - t0)
    on_belt = abs(d.qpos[1]) < 0.6 - 0.025 and abs(d.qpos[2] - (Z_BELT + 0.01)) < 1e-3
    print(f"\nV9 belt {v_b} m/s: product {v:.4f} m/s ({(v / v_b - 1) * 100:+.3f} %), drift {drift * 1e3:.3f} mm/s, "
          f"y {d.qpos[1]:.3f} m")
    assert on_belt and abs(v / v_b - 1) < 0.005 and drift < 1e-3


# ---------------- V10 tracked pick
@pytest.mark.parametrize("v_b", [0.2, 0.5])
def test_v10_tracked_pick_touch_velocity(v_b):
    """At touch the cup moves with the product: horizontal relative velocity < 5 mm/s (PickMaster tracks the belt)."""
    cell = PickCell(PickSetup(cup=CupConfig(mass=0.02, g0=3e-6), v_belt=v_b, alpha=0.5, vision_sigma_mm=0.0))
    rel = []

    def rec(c, t):
        if abs(t - c.t_touch) < c.P.dt / 2:
            v = np.zeros(6)
            mujoco.mj_objectVelocity(c.m, c.d, mujoco.mjtObj.mjOBJ_SITE, c.cup.site, v, 0)
            rel.append(np.linalg.norm(v[3:5] - c.d.qvel[c.m.jnt_dofadr[c.m.joint("prod_free").id]:][:2]))
    out = cell.run(rec)
    print(f"\nV10 belt {v_b} m/s: TCP–product relative velocity at touch {rel[0] * 1e3:.2f} mm/s; "
          f"held {not out['dropped']}, place offset {out['place_offset_mm']:.2f} mm (no vision error)")
    assert rel[0] < 5e-3 and not out["dropped"] and out["place_offset_mm"] < 1.0


@pytest.mark.parametrize("v_b", [0.2, 0.5])
def test_v10_placement_error_matches_tracking_table(v_b):
    """With vision error from ABB's tracking table, the placement error of a light, slow pick is that error:
    mean of the 2-D error norm ≈ σ·√(π/2) within ±35 % (20 picks)."""
    errs = [PickCell(PickSetup(cup=CupConfig(mass=0.02, g0=3e-6), v_belt=v_b, alpha=0.3, seed=s)).run()["place_offset_mm"]
            for s in range(20)]
    sig = tracking_sigma_mm(v_b)
    expect = sig * np.sqrt(np.pi / 2)
    print(f"\nV10 belt {v_b} m/s: mean placement error {np.mean(errs):.2f} mm vs {expect:.2f} mm expected (σ {sig} mm, ABB)")
    assert abs(np.mean(errs) / expect - 1) < 0.35


# ---------------- drop boundary vs the vendor sizing rule
def test_drop_boundary_matches_schmalz_sizing():
    """Schmalz horizontal load case: F_TH = m (g + a/μ) S, with S ≥ 2 recommended for dynamic picks. The simulated
    drop onset (first drop as alpha rises) must imply a safety factor S* = F / (m (g + a*/μ)) in [1.5, 2.5]."""
    cfg = CupConfig(mass=0.05, size=(0.05, 0.05, 0.02), mu=0.6, g0=3e-6)
    prev = None
    for alpha in np.round(np.arange(0.2, 1.01, 0.05), 2):
        out = PickCell(PickSetup(cup=cfg, v_belt=0.0, alpha=float(alpha), vision_sigma_mm=0.0)).run()
        if out["dropped"]:
            a_star = 0.5 * (out["a_peak"] + prev["a_peak"]) if prev else out["a_peak"]
            F = out["seal_kpa"] * 1e3 * cfg.A_eff
            S = F / (cfg.mass * (9.81 + a_star / cfg.mu))
            print(f"\nfirst drop at alpha {alpha} (a_peak {out['a_peak']:.0f} m/s²): implied Schmalz safety factor S* = {S:.2f}")
            assert 1.5 <= S <= 2.5
            return
        prev = out
    pytest.fail("no drop up to alpha 1.0")
