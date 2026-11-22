"""Gate 3 (docs/MODEL_SPEC.md §6): seal level and pre-failure warning, tests V11 and V12 of model-suction.md §6.

V11: seal at touch falls monotonically with the equivalent roughness gap g0; a smooth surface seals within
     2 kPa of the ejector maximum (Huh et al.: 74 of 85 kPa on the smoothest sandpaper).
V12: under jerk-limited sideways pick accelerations, every drop is preceded by a dip in the vacuum sensor
     reading, and loads just below the limit show dips without drops (near-misses). The lead time is reported
     (GOMP-ST: > 0.1 s from deformation to failure).
Run: .venv/bin/python -m pytest tests/test_seal.py -v -s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import ruckig

from sim.cup import CupCell, CupConfig


def seal_level(g0: float, **kw) -> float:
    cell = CupCell(CupConfig(g0=g0, **kw))
    cell.engage(lift=0.0, lift_T=0.01)
    cell.hold(0.1, (0, 0, cell.z_hold))
    return cell.dp / 1e3


# ---------------- V11
def test_v11_seal_vs_roughness():
    gs = [0.0, 2e-6, 5e-6, 10e-6, 15e-6, 20e-6, 30e-6]
    seals = [seal_level(g) for g in gs]
    print("\nV11 seal (kPa) vs g0 (µm): " + ", ".join(f"{g * 1e6:.0f}: {s:.1f}" for g, s in zip(gs, seals)))
    assert seals[0] >= 75.0 - 2.0
    assert all(b <= a + 1e-6 for a, b in zip(seals, seals[1:]))


# ---------------- V12
def lateral_profile(a_max: float, j_max: float, dist: float, dt: float):
    """Jerk-limited rest-to-rest move along x (ruckig), sampled at dt: arrays of p, v, a."""
    otg, inp, traj = ruckig.Ruckig(1), ruckig.InputParameter(1), ruckig.Trajectory(1)
    inp.current_position, inp.current_velocity, inp.current_acceleration = [0.0], [0.0], [0.0]
    inp.target_position, inp.target_velocity, inp.target_acceleration = [dist], [0.0], [0.0]
    inp.max_velocity, inp.max_acceleration, inp.max_jerk = [10.0], [a_max], [j_max]
    otg.calculate(inp, traj)
    ts = np.arange(0, traj.duration + dt, dt)
    out = np.array([traj.at_time(t) for t in ts])[:, :, 0]
    return out[:, 0], out[:, 1], out[:, 2]


def pick_move(cfg: CupConfig, a_max: float, j_max: float = 6500.0, dist: float = 0.30):
    """Seal, lift, then a sideways transfer. Returns (dropped, t_dip_onset, t_drop, min sensor kPa, seal kPa)."""
    cell = CupCell(cfg)
    cell.engage()
    seal = cell.dp
    z = cell.z_hold
    P, V, A = lateral_profile(a_max, j_max, dist, cfg.dt)
    t_dip = t_drop = None
    vmin = seal
    t0 = cell.t
    for p, v, a in zip(P, V, A):
        cell.step(((p, 0, z), (v, 0, 0), (a, 0, 0)))
        if t_dip is None and seal - cell.dp > 1000.0:          # 1 kPa below the sealed level
            t_dip = cell.t - t0
        vmin = min(vmin, cell.dp)
        if t_drop is None and np.all(cell.gaps > 50e-6):
            t_drop = cell.t - t0
            break
    if t_drop is None:
        cell.hold(0.1, (P[-1], 0, z))
        if np.all(cell.gaps > 50e-6):
            t_drop = cell.t - t0
    return t_drop is not None, t_dip, t_drop, vmin / 1e3, seal / 1e3


@pytest.mark.parametrize("bellows", [False, True])
def test_v12a_peel_drop_preceded_by_dip(bellows):
    """Tall, peel-dominated product: every drop is preceded by a vacuum dip; the lead time is reported.
    Finding: below the limit there is no dip at all (Huh et al.: vacuum flat until leak onset), so peel gives
    no graded near-miss in the vacuum signal."""
    base = dict(mass=0.2, size=(0.05, 0.05, 0.04), mu=0.5, g0=3e-6, bellows=bellows)
    leads, drops = [], 0
    for a in (4, 6, 8, 10, 12, 16, 20, 30, 60, 150):
        dropped, t_dip, t_drop, vmin, seal = pick_move(CupConfig(**base), a)
        if dropped:
            drops += 1
            assert t_dip is not None and t_dip < t_drop, f"drop at a={a} without a preceding dip"
            leads.append((t_drop - t_dip) * 1e3)
    print(f"\nV12a bellows={bellows}: {drops} drops, all preceded by a dip; lead time ms min {min(leads):.1f} "
          f"median {np.median(leads):.1f} max {max(leads):.1f}")
    assert drops >= 3


def test_v12b_shear_graded_near_miss():
    """Flat, shear-dominated product: past the friction limit the product slides in the cup but is still held,
    and the slip grows with acceleration (a graded near-miss visible as placement error and a small dip)."""
    from sim.cup import CupConfig as C
    base = dict(mass=0.05, size=(0.06, 0.045, 0.006), mu=0.5, g0=3e-6)
    F = 75e3 * C().A_eff
    a_star = 0.5 * (F - 0.05 * 9.81) / 0.05
    slips = []
    for a in (0.5 * a_star, 0.9 * a_star, 1.05 * a_star, 1.1 * a_star, 1.2 * a_star, 1.26 * a_star):
        cell = CupCell(CupConfig(**base))
        cell.engage()
        cell.mark()
        P, V, A = lateral_profile(a, 6500.0, 0.30, cell.cfg.dt)
        for p, v, ac in zip(P, V, A):
            cell.step(((p, 0, cell.z_hold), (v, 0, 0), (ac, 0, 0)))
        cell.hold(0.05, (P[-1], 0, cell.z_hold))
        assert not np.all(cell.gaps > 50e-6), f"dropped at {a:.0f} m/s²"
        rel = cell.prod_rel()[0] - cell.rel0
        slips.append(np.hypot(rel[0], rel[1]) * 1e3)
    print(f"\nV12b shear limit {a_star:.0f} m/s²; slip (mm) at 0.5/0.9/1.05/1.1/1.2/1.26 × limit: {np.round(slips, 2).tolist()}")
    assert max(slips[:2]) < 0.3                                   # below the limit: no meaningful slip
    assert all(b >= a - 0.05 for a, b in zip(slips[2:], slips[3:]))  # past the limit: slip grows
    assert slips[-1] > 1.0
