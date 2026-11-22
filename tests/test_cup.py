"""Gate 2 (docs/MODEL_SPEC.md §6): compliant cup mechanics, tests V5–V9b, V13, V14 of model-suction.md §6.

Loads are applied quasi-statically to the hanging product at its CoM (d'Alembert: an inertial load
m·a equals an external force -m·a at the CoM). f_N = F_vac − m·g is the lip normal force of a hanging product.
Run: .venv/bin/python -m pytest tests/test_cup.py -v -s
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import pytest

from sim.cup import CupCell, CupConfig

G = 9.81


def ring_wrench(cell):
    """Normal force f_N and moment (about the cup centre, in the product-face frame) carried by the lip contacts."""
    d, m = cell.d, cell.m
    R = d.xmat[cell.prod_body].reshape(3, 3)
    nz = R[:, 2]
    face = d.xpos[cell.prod_body] + nz * cell.cfg.size[2] / 2            # a point on the sealed (top) face
    site = d.site_xpos[cell.site]
    c0 = site - float(np.dot(site - face, nz)) * nz                    # cup centre projected onto the face plane
    f6 = np.zeros(6)
    fN, M = 0.0, np.zeros(3)
    lips = set(int(g) for g in cell.lip_geoms)
    for k in range(d.ncon):
        c = d.contact[k]
        if not ((int(c.geom1) in lips and c.geom2 == cell.prod_geom) or (int(c.geom2) in lips and c.geom1 == cell.prod_geom)):
            continue
        if c.efc_address < 0:
            continue
        mujoco.mj_contactForce(m, d, k, f6)
        fw = c.frame.reshape(3, 3).T @ f6[:3]
        fN += abs(float(np.dot(fw, R[:, 2])))
        M += np.cross(c.pos - c0, fw)
    return fN, R.T @ M


def ramp_to_failure(cfg: CupConfig, axis: str, top: float, T: float = 1.5, slip_tol: float = 1e-3,
                    onset_gap: float | None = None, rot_tol_deg: float = 2.0, mode: str = "detach"):
    """Ramp an external load on the product from 0 to `top` over T seconds.
    axis: 'down' (force −z), 'x' (force +x), 'yaw' (torque about z), 'roll' (torque about x).
    mode: 'detach' (all lip segments lose contact, or tilt > 20°), 'slip' (tangential slip > slip_tol), 'rot' (rotation > rot_tol_deg).
    Returns dict with the load at onset (first lip gap > onset_gap), at failure, and the dip onset."""
    cell = CupCell(cfg)
    cell.engage()
    cell.hold(0.4, (0, 0, cell.z_hold))  # let the product settle on the lip springs
    cell.mark()
    p0 = cell.d.xpos[cell.prod_body].copy()
    q0 = cell.d.qpos[cell.prod_qadr + 3:cell.prod_qadr + 7].copy()
    dp0 = cell.dp
    out = {"onset": None, "fail": None, "dip": None, "dp0": dp0}
    n = int(T / cfg.dt)
    for k in range(n):
        L = top * (k + 1) / n
        cell.ext_force[:] = 0
        cell.ext_torque[:] = 0
        if axis == "down":
            cell.ext_force[2] = -L
        elif axis == "x":
            cell.ext_force[0] = L
        elif axis == "yaw":
            cell.ext_torque[2] = L
        elif axis == "roll":
            cell.ext_torque[0] = L
        cell.step(((0, 0, cell.z_hold), (0, 0, 0), (0, 0, 0)))
        if onset_gap is not None and out["onset"] is None:
            if cell.gaps.max() > onset_gap:
                out["onset"] = L
                out["ring_at_onset"] = prev_ring
            else:
                prev_ring = ring_wrench(cell)
        if out["dip"] is None and dp0 - cell.dp > 500.0:
            out["dip"] = L
        dpos = cell.d.xpos[cell.prod_body] - p0
        q = cell.d.qpos[cell.prod_qadr + 3:cell.prod_qadr + 7]
        rot = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q, q0))))))
        detached = bool(np.all(cell.gaps > 50e-6))          # every lip segment has lost contact
        if mode == "slip":
            failed = np.linalg.norm(dpos[:2]) > slip_tol       # tangential slip (the cup has no lateral compliance)
        elif mode == "rot":
            failed = rot > rot_tol_deg
        else:
            failed = detached or rot > 20.0
        if failed:
            out["fail"] = L
            break
    out["F"] = cell.dp * cfg.A_eff if cfg.frozen_dp is not None else dp0 * cfg.A_eff
    return out


DP = 60e3
F = DP * CupConfig().A_eff  # 9.9 N


# ---------------- V5 lift to failure, pressure held: a* = F/m − g (±4 %)  [also V14 regression]
@pytest.mark.parametrize("mass", [0.2, 0.5])
def test_v5_lift(mass):
    cfg = CupConfig(frozen_dp=DP, mass=mass)
    r = ramp_to_failure(cfg, "down", top=F * 1.3)
    a_sim = r["fail"] / mass
    a_ref = F / mass - G
    print(f"\nV5 m={mass}: a* sim {a_sim:.2f} vs ref {a_ref:.2f} m/s² ({(a_sim / a_ref - 1) * 100:+.1f} %)")
    assert abs(a_sim / a_ref - 1) < 0.04


# ---------------- V6 shear, flat product, 1 mm slip criterion: a* = µ(F − m g)/m (±5 %)
@pytest.mark.parametrize("mu", [0.3, 0.6])
def test_v6_shear(mu):
    mass = 0.05
    cfg = CupConfig(frozen_dp=DP, mass=mass, mu=mu, size=(0.06, 0.045, 0.006))
    r = ramp_to_failure(cfg, "x", top=mu * F * 1.5, slip_tol=1e-3, mode="slip")
    a_sim = r["fail"] / mass
    a_ref = mu * (F - mass * G) / mass
    print(f"\nV6 mu={mu}: a* sim {a_sim:.1f} vs ref {a_ref:.1f} m/s² ({(a_sim / a_ref - 1) * 100:+.1f} %)")
    assert abs(a_sim / a_ref - 1) < 0.05


# ---------------- V7 torsion: τ_max = r·µ·f_N (Dex-Net ring) (±5 %)
def test_v7_torsion():
    mass, mu = 0.05, 0.6
    cfg = CupConfig(frozen_dp=DP, mass=mass, mu=mu, size=(0.06, 0.045, 0.006))
    fN = F - mass * G
    r = ramp_to_failure(cfg, "yaw", top=cfg.r_lip * mu * fN * 1.5, rot_tol_deg=2.0, mode="rot")
    ref = cfg.r_lip * mu * fN
    print(f"\nV7 torsion: τ* sim {r['fail'] * 1e3:.2f} vs ref {ref * 1e3:.2f} N·mm ({(r['fail'] / ref - 1) * 100:+.1f} %)")
    assert abs(r["fail"] / ref - 1) < 0.05


# ---------------- V8 peel onset: when the first lip spring unloads, the springs carry M = r·f_N/2 (Dex-Net linear ring), any stiffness (±5 %)
def spring_wrench(cell):
    """Lip-spring forces p_i = k·δz_i (the Dex-Net ring quantity): total f_N and moment about the ring centre."""
    k_l = cell.cfg.k_ring / cell.cfg.n_seg
    p = k_l * np.maximum(cell.lip_compression(), 0.0)
    th = 2 * np.pi * np.arange(cell.cfg.n_seg) / cell.cfg.n_seg
    x, y = cell.cfg.r_lip * np.cos(th), cell.cfg.r_lip * np.sin(th)
    return p.sum(), float(np.hypot((p * y).sum(), (p * x).sum()))


@pytest.mark.parametrize("k_ring", [4000.0, 12000.0])
def test_v8_peel_onset_invariant(k_ring):
    """At the step where the first lip spring reaches its free length, the spring-carried moment must equal r·f_N/2.
    The externally applied moment is reported too. It exceeds r·f_N/2 when a soft ring lets the product tilt ~10°,
    because the lips are rigid sideways and friction then carries part of the moment (open item: lateral lip compliance)."""
    mass = 0.2
    cfg = CupConfig(frozen_dp=DP, mass=mass, k_ring=k_ring)
    fN_nom = F - mass * G
    cell = CupCell(cfg)
    cell.engage()
    cell.hold(0.4, (0, 0, cell.z_hold))
    top, n = cfg.r_lip * fN_nom * 1.5, int(1.5 / cfg.dt)
    for k in range(n):
        L = top * (k + 1) / n
        cell.ext_torque[:] = [L, 0, 0]
        cell.step(((0, 0, cell.z_hold), (0, 0, 0), (0, 0, 0)))
        if cell.lip_compression().min() <= 2e-6:
            break
    fN, M = spring_wrench(cell)
    ref = cfg.r_lip * fN / 2
    tilt = np.degrees(np.arccos(cell.d.xmat[cell.prod_body].reshape(3, 3)[2, 2]))
    print(f"\nV8 k_ring={k_ring:.0f}: spring moment at onset {M * 1e3:.2f} vs r·f_N/2 {ref * 1e3:.2f} N·mm ({(M / ref - 1) * 100:+.1f} %) | "
          f"external {L * 1e3:.2f} vs {cfg.r_lip * fN_nom / 2 * 1e3:.2f} N·mm, tilt {tilt:.1f}°")
    assert abs(M / ref - 1) < 0.05


# ---------------- V9 peel ultimate, pressure frozen: M = r·f_N (rigid ring / CoP at the rim) (±10 %)
def test_v9_peel_ultimate_frozen():
    mass = 0.2
    cfg = CupConfig(frozen_dp=DP, mass=mass)
    fN = F - mass * G
    r = ramp_to_failure(cfg, "roll", top=cfg.r_lip * fN * 1.5, onset_gap=1e-6)
    ref = cfg.r_lip * fN
    print(f"\nV9 ultimate: {r['fail'] * 1e3:.2f} vs ref {ref * 1e3:.2f} N·mm ({(r['fail'] / ref - 1) * 100:+.1f} %)")
    assert abs(r["fail"] / ref - 1) < 0.10


# ---------------- V9b peel with the pressure ODE on: failure in [M_onset, M_rigid]; dip starts at onset (±5 %)
def test_v9b_peel_with_pneumatics():
    mass = 0.2
    cfg = CupConfig(frozen_dp=None, mass=mass)
    r = ramp_to_failure(cfg, "roll", top=0.17, T=1.5, onset_gap=1e-6)
    fN = r["F"] - mass * G
    m_on, m_rig = cfg.r_lip * fN / 2, cfg.r_lip * fN
    print(f"\nV9b sealed Δp {r['dp0'] / 1e3:.1f} kPa: onset {r['onset'] * 1e3:.2f}, dip {r['dip'] * 1e3 if r['dip'] else float('nan'):.2f}, "
          f"fail {r['fail'] * 1e3:.2f} N·mm; bounds [{m_on * 1e3:.2f}, {m_rig * 1e3:.2f}]")
    assert m_on * 0.95 <= r["fail"] <= m_rig * 1.05
    assert r["dip"] is not None and abs(r["dip"] / r["onset"] - 1) < 0.05


# ---------------- V13 numerical convergence: halve dt, double ODE sub-steps (< 2 %)
def test_v13_convergence():
    res = []
    for dt, n_sub in ((2e-4, 10), (1e-4, 20)):
        cfg = CupConfig(frozen_dp=None, mass=0.2, dt=dt, n_sub=n_sub)
        r = ramp_to_failure(cfg, "roll", top=0.17, T=1.5, onset_gap=1e-6)
        res.append((r["dp0"], r["fail"]))
    (s1, f1), (s2, f2) = res
    print(f"\nV13 seal {s1 / 1e3:.2f} vs {s2 / 1e3:.2f} kPa; peel failure {f1 * 1e3:.2f} vs {f2 * 1e3:.2f} N·mm")
    assert abs(s1 / s2 - 1) < 0.02 and abs(f1 / f2 - 1) < 0.02
