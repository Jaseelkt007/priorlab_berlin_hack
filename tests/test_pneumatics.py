"""Gate 1 (docs/MODEL_SPEC.md §6): pneumatics tests V1–V4 from docs/research/model-suction.md §6.

Run: .venv/bin/python -m pytest tests/test_pneumatics.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest

from sim.pneumatics import (EJECTORS, EVAC_TABLES, P_ATM, R_AIR, RHO0, T0, EjectorCurve, LipLeak,
                            VacuumCircuit, iso6358_mdot, sonic_conductance, steady_seal)


def evacuate(ej, V, targets_kpa, dt=1e-4, t_max=20.0, leak=lambda p: 0.0):
    """Integrate a sealed single-node volume and return the time (s) at which each target vacuum is reached."""
    c = VacuumCircuit(ej, V_cup=V, n_sub=2)
    t, out, i = 0.0, [], 0
    while i < len(targets_kpa) and t < t_max:
        dp = c.step(dt, leak)
        t += dt
        while i < len(targets_kpa) and dp >= targets_kpa[i] * 1e3:
            out.append(t)
            i += 1
    return out


# ---------------- V1: evacuation of a sealed volume vs Piab's published tables (±10 %, Piab's stated accuracy)
@pytest.mark.parametrize("name", ["Si08-3", "Pi12-2"])
def test_v1_evacuation_vs_piab(name):
    targets, ref = EVAC_TABLES[name]
    got = evacuate(EJECTORS[name], 1e-3, targets, dt=2e-3)
    rel = [(g - r) / r for g, r in zip(got, ref)]
    print(f"\n{name}: sim s/l {np.round(got, 2).tolist()} vs Piab {ref}; rel err {np.round(rel, 3).tolist()}")
    assert len(got) == len(ref)
    tol = 0.10 if name == "Si08-3" else 0.12  # Pi12-2 flow and evacuation tables use different feed pressures (spec §3.2)
    assert max(abs(r) for r in rel) <= tol


# ---------------- V2: time constants with a linear ejector curve (SMC: 63 % at T1, 95 % at T2 = 3·T1)
def test_v2_time_constants_linear_curve():
    q0, dpmax = 1.0, 70.0  # Nl/s, kPa
    pts = np.linspace(0, dpmax * 0.999, 30)
    ej = EjectorCurve("linear", pts, q0 * (1 - pts / dpmax), dpmax)
    V = 5e-6
    t63, t95 = evacuate(ej, V, [0.632 * dpmax, 0.950 * dpmax], dt=2e-6)
    tau = V * dpmax * 1e3 / (RHO0 * R_AIR * T0 * q0 * 1e-3)
    print(f"\nT1 {t63 * 1e3:.3f} ms (analytic tau {tau * 1e3:.3f} ms), T2/T1 {t95 / t63:.3f}")
    assert abs(t63 / tau - 1) < 0.05
    assert abs(t95 / t63 - 3.0) < 0.05 * 3.0


# ---------------- V3: steady seal with a known orifice; SMC round trip; min vs series blend
def test_v3_steady_seal_orifice_and_ode_agree():
    ej = EJECTORS["Si08-3"]
    A = 0.05e-6  # 0.05 mm^2 leak orifice
    C = sonic_conductance(A)
    leak = lambda p: float(iso6358_mdot(C, P_ATM, p))
    dp_root = steady_seal(ej, leak)
    c = VacuumCircuit(ej, V_cup=5e-6)
    for _ in range(2000):
        c.step(1e-4, leak)
    print(f"\nroot {dp_root / 1e3:.2f} kPa, ODE steady {c.vacuum / 1e3:.2f} kPa")
    assert abs(c.vacuum - dp_root) / dp_root < 0.02


def test_v3_smc_round_trip():
    """SMC adsorption test: read P1, take Q_L = Q_ej(P1), recover the conductance, recompute P1."""
    ej = EJECTORS["Si08-3"]
    for P1_kpa in (20, 40, 53, 65):
        mdot = float(ej.mdot(P1_kpa * 1e3))
        p_cup = P_ATM - P1_kpa * 1e3
        C = mdot / float(iso6358_mdot(1.0, P_ATM, p_cup))  # flow is linear in C
        P1_back = steady_seal(ej, lambda p: float(iso6358_mdot(C, P_ATM, p))) / 1e3
        assert abs(P1_back - P1_kpa) / P1_kpa < 0.02


def test_v3_min_vs_series_blend_small_gaps():
    """The min() combination of viscous and orifice flow must not matter for gaps ≤ 10 µm (< 5 % seal change)."""
    ej = EJECTORS["Si08-3"]
    for g in (2e-6, 5e-6, 10e-6):
        s_min = steady_seal(ej, lambda p: LipLeak(g0=g).mdot(p))
        s_ser = steady_seal(ej, lambda p: LipLeak(g0=g, combine="series").mdot(p))
        print(f"\ngap {g * 1e6:.0f} µm: min {s_min / 1e3:.2f} kPa, series {s_ser / 1e3:.2f} kPa")
        assert abs(s_min - s_ser) / s_min < 0.05


def test_v3_matches_research_illustration():
    """Cross-check against the research document's own calculation (§3.4 table, Si08-3 column): ±2 kPa."""
    ej = EJECTORS["Si08-3"]
    ref = {0.0: 75.0, 5e-6: 74.9, 10e-6: 74.1, 20e-6: 64.0, 40e-6: 29.9}
    for g, r in ref.items():
        s = steady_seal(ej, lambda p: LipLeak(g0=g).mdot(p)) / 1e3
        print(f"\nuniform gap {g * 1e6:.0f} µm: {s:.1f} kPa (research calc {r})")
        assert abs(s - r) <= 2.0


# ---------------- V4: static holding force F = Δp·A_eff
SPB1 = {10: 1.5, 15: 4.3, 20: 9.9, 25: 14.9, 30: 20.7, 40: 35.9}  # Schmalz suction force at -0.6 bar (N)
SPB1_D2 = {10: 5.6, 15: 9.5, 20: 13.8, 25: 16.9, 30: 20.0, 40: 26.1}  # inner diameter d2 (mm)


def test_v4_effective_area_is_inner_diameter():
    """Schmalz forces correspond to the inner diameter d2 (research found 1.00–1.06×d2)."""
    for size, F in SPB1.items():
        d_eff = np.sqrt(4 * F / (np.pi * 60e3)) * 1e3
        assert 0.99 <= d_eff / SPB1_D2[size] <= 1.07


def test_v4_force_with_vacuum_dependent_area():
    """Piab B25XP: d_eff shrinks with vacuum; tabulated A_eff(Δp) must reproduce 9.7 / 19.0 / 22.0 N within ±5 %."""
    dp = np.array([20e3, 60e3, 90e3])
    F_ref = np.array([9.7, 19.0, 22.0])
    A = F_ref / dp
    A_of = lambda x: np.interp(x, dp, A)
    for x, f in zip(dp, F_ref):
        assert abs(x * A_of(x) - f) / f < 0.05
    # and the d_eff trend the research reported: 24.8 -> 20.1 -> 17.6 mm
    d = np.sqrt(4 * A / np.pi) * 1e3
    assert np.allclose(d, [24.8, 20.1, 17.6], atol=0.15)


# ---------------- compiled fast path must reproduce the reference integrator
@pytest.mark.parametrize("two_node", [False, True])
def test_fast_path_matches_reference(two_node):
    from sim.pneumatics import step_lip_fast
    rng = np.random.default_rng(1)
    leak = LipLeak(g0=3e-6, K_p=1e-15)
    kw = dict(V_line=4e-6, C_hose=float(sonic_conductance(2e-6))) if two_node else {}
    a = VacuumCircuit(EJECTORS["Si08-3"], V_cup=4.2e-6, **kw)
    b = VacuumCircuit(EJECTORS["Si08-3"], V_cup=4.2e-6, **kw)
    worst = 0.0
    for k in range(3000):
        gaps = np.where(rng.random(16) < 0.2, rng.uniform(0, 60e-6, 16), 0.0) if k > 1500 else np.zeros(16)
        a.step(2e-4, lambda p: leak.mdot(p, gaps))
        step_lip_fast(b, 2e-4, leak, gaps)
        worst = max(worst, abs(a.p_cup - b.p_cup), abs(a.p_line - b.p_line))
    print(f"\nmax |Δp| fast vs reference: {worst:.3f} Pa (two_node={two_node})")
    assert worst < 20.0  # Pa; differences come only from the dense ejector table vs PCHIP
