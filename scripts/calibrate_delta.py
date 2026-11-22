"""Gate 4 calibration and held-out checks (docs/research/model-delta.md §5.5–5.6).

What binds (sweeps in docs/research/delta-validation.md): zone and jerk saturate (full-arch path, j ≳ 1e5 m/s³),
so they are set non-binding and reported. Two parameters are fitted, each on one ABB point:
  - arm speed limit θ̇_max on 90/400/90 at 0.1 kg (0.444 s): the long, speed-dominated cycle;
  - motor torque limit τ_max on 25/305/25 at 1.0 kg (0.353 s): the payload penalty.
Unfitted checks (±5 %): 25/305/25 at 0.1 kg (0.300 s) and the held-out 90/400/90 at 1.0 kg (0.500 s).
V7: base loads on the full closed chain during the 1 kg cycles vs ABB's operating maxima.
Cycle = move out + 35 ms + move back + 35 ms (ABB: "air activation time of 35 ms for picking and 35 ms for placing").
Usage: .venv/bin/python scripts/calibrate_delta.py
"""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sim.delta import DeltaParams
from sim.delta_motion import ABB_CYCLES as ABB, CYCLES, MoveSpec, base_loads, cycle_time

def bisect(f, lo, hi, n=14):
    """Root of a monotone f on [lo, hi]; f(lo) and f(hi) of opposite sign (RuntimeError counts as +inf)."""
    def g(x):
        try:
            return f(x)
        except RuntimeError:
            return np.inf
    s_lo = np.sign(g(lo))
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sign(g(mid)) == s_lo else (lo, mid)
    return 0.5 * (lo + hi)


def report(P, spec):
    rows = {}
    for (cyc, kg), ref_t in ABB.items():
        t, _ = cycle_time(dataclasses.replace(P, m_tool=kg), cyc, spec)
        rows[f"{cyc} @ {kg} kg"] = (t, ref_t, (t / ref_t - 1) * 100)
    return rows


if __name__ == "__main__":
    P = DeltaParams(m_tool=0.1, tau_max=1000.0)
    R_C, J_MAX = np.inf, 1.6e5
    out = {}
    print("0. saturation sweeps at 0.1 kg, prior arm speed 1088 °/s, torque not limiting:")
    for rc in (0.02, 0.06, np.inf):
        for j in (4e4, 1.6e5, 6.4e5):
            sp = MoveSpec(r_c=rc, j_max=j, thd_max=np.deg2rad(1088.0))
            t25, _ = cycle_time(P, "25/305/25", sp)
            t90, _ = cycle_time(P, "90/400/90", sp)
            print(f"   zone {rc * 1e3:5.0f} mm, j {j:7.0f}: 25/305/25 {t25 * 1e3:.1f} ms, 90/400/90 {t90 * 1e3:.1f} ms")
    spec = MoveSpec(r_c=R_C, j_max=J_MAX, thd_max=np.deg2rad(1088.0))
    thd = bisect(lambda deg: cycle_time(P, "90/400/90", dataclasses.replace(spec, thd_max=np.deg2rad(deg)))[0] - 0.444, 1088.0, 400.0)
    spec = dataclasses.replace(spec, thd_max=np.deg2rad(thd))
    out["thd_max_deg"] = thd
    print(f"1. arm speed limit fitted on 90/400/90 @ 0.1 kg: {thd:.0f} °/s (Gnad et al. prior: 1088 °/s)")
    tau_fit = bisect(lambda tq: cycle_time(dataclasses.replace(P, m_tool=1.0, tau_max=tq), "25/305/25", spec)[0] - 0.353, 400.0, 20.0)
    out["tau_max"] = tau_fit
    print(f"2. motor torque limit fitted on 25/305/25 @ 1.0 kg: {tau_fit:.1f} N·m")
    Pc = dataclasses.replace(P, tau_max=tau_fit)
    out["cycles"] = report(Pc, spec)
    print("3. all four ABB cycle points (fit: 90/400/90 @ 0.1, 25/305/25 @ 1.0; checks: the other two):")
    for k, (t, r, e) in out["cycles"].items():
        print(f"   {k:18s}: {t * 1e3:6.1f} ms vs ABB {r * 1e3:.0f} ms ({e:+.1f} %)")
    print("4. V7 base loads, full closed chain, 1 kg (ABB operating max Fx 330, Fy 260, dynamic Fz 170 N, Mz 100 N·m):")
    out["base"] = {}
    for cyc in CYCLES:
        bl = base_loads(dataclasses.replace(Pc, m_tool=1.0), cyc, spec)
        out["base"][cyc] = bl
        print(f"   {cyc}: Fx {bl['Fx']:.0f} N, Fy {bl['Fy']:.0f} N, dyn Fz {bl['Fz_dyn']:.0f} N, Mz {bl['Mz']:.1f} N·m (lumped plan: {np.round(bl['planned_F']).tolist()}) | TCP tracking {bl['tcp_err_mm']:.3f} mm, peak τ {bl['tau_peak']:.1f} N·m")
    Path("docs/research/delta-calibration.json").write_text(json.dumps(out, indent=2, default=float))
