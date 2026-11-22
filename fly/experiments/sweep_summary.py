"""Summarise the context-size sweep (results/sweep/) into one JSON for the report page.

    uv run python experiments/sweep_summary.py

Reads tabpfn_ctx<N>.json and tabpfn_ctx<N>_drift.npz for every swept size and writes
results/sweep/summary.json: per size and path, flight time, mean/max distance, ms per decision,
and the distance curve downsampled to one point per 2 ms.
"""

import json
from pathlib import Path

import numpy as np

SWEEP = Path(__file__).resolve().parents[1] / "results/sweep"
EVERY = 10  # control steps per curve point (2 ms)


def main():
    out = {}
    for meta in sorted(SWEEP.glob("tabpfn_ctx*.json"), key=lambda p: int(p.stem.split("ctx")[1].split("_")[0])):
        if meta.stem.endswith("summary"):
            continue
        size = int(meta.stem.split("ctx")[1].split("_")[0])
        info = json.loads(meta.read_text())
        drift = np.load(meta.with_name(f"{meta.stem}_drift.npz"))
        paths = {}
        for seg, (key, curve) in zip(info["segments"], sorted(drift.items(), key=lambda kv: int(kv[0].split("_")[0]))):
            paths[seg["path"]] = dict(flew_ms=seg["flew_ms"], planned_ms=seg["planned_ms"], ended=seg["ended"],
                                      mean_mm=seg["drift_mean_mm"], max_mm=seg["drift_max_mm"],
                                      ms_per_step=seg.get("ms_per_step"),
                                      curve=[round(float(v), 3) for v in curve[EVERY - 1::EVERY]])
        out[size] = dict(rows=size * 12, wall_s=info["wall_s"], paths=paths)
    (SWEEP / "summary.json").write_text(json.dumps(out))
    for size, s in out.items():
        print(size, {p: f"{v['flew_ms']:.0f}/{v['planned_ms']:.0f} ms, {v['mean_mm']:.2f} mm" for p, v in s["paths"].items()})


if __name__ == "__main__":
    main()
