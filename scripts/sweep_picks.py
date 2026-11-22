"""Speed tests: drop rate and placement error vs speed setting, per product, under random line states. The output is
also the line's commissioning log (the in-context data of the speed policies).
Usage: .venv/bin/python scripts/sweep_picks.py [picks_per_cell] [out.csv] [products] [--s0 FIRST]
Seeds are 10_000·product + 100·alpha_index + s for s in [s0, s0 + picks_per_cell): split a log between machines by s0."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.picklog import ALPHAS, run_tasks  # first: pins worker threads before numpy loads
from sim.products import CATALOGUE, LINE_PRODUCTS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=20)
    ap.add_argument("out", nargs="?", default="data/sweep.csv")
    ap.add_argument("products", nargs="?", default=",".join(LINE_PRODUCTS))
    ap.add_argument("--s0", type=int, default=0)
    a = ap.parse_args()
    only = a.products.split(",")
    assert a.s0 + a.n <= 100, "s must stay below 100 (seed layout)"
    out = Path(a.out)
    tasks = [dict(product=k, alpha=float(al), seed=10_000 * i + 100 * j + s)
             for i, k in enumerate(CATALOGUE) if k in only for j, al in enumerate(ALPHAS) for s in range(a.s0, a.s0 + a.n)]
    t = time.time()
    out.parent.mkdir(exist_ok=True)
    df = run_tasks(tasks, out, progress_every=50)
    print(f"{len(df)} picks in {time.time() - t:.0f} s -> {out}")
    g = df.groupby(["product", "alpha"])
    print("drop rate by product (rows) and speed setting alpha (columns):")
    print(g["dropped"].mean().unstack().round(2).to_string())
    print("median place offset (mm), held picks:")
    print(g["place_offset_mm"].median().unstack().round(2).to_string())
    print("seal_kpa median by product:", df.groupby("product")["seal_kpa"].median().round(1).to_dict())
