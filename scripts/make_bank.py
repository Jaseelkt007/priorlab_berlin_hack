"""Simulate the evaluation pick bank: every instance of every scenario stream (sim.scenarios) at all speed settings.
Rows stream to the CSV as they finish (crash-safe). Usage: .venv/bin/python scripts/make_bank.py [out.csv] [scenario,...]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.picklog import run_tasks, simulate_instance  # first: pins worker threads before numpy loads
from sim.scenarios import SCENARIOS, all_tasks

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/bank.csv")
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SCENARIOS)
    tasks = [t for t in all_tasks() if t["scenario"] in only]
    # long instances first would balance the pool better, but order does not matter for the result
    out.parent.mkdir(exist_ok=True)
    t0 = time.time()
    df = run_tasks(tasks, out, fn=simulate_instance, progress_every=20)
    print(f"{len(tasks)} instances, {len(df)} picks in {time.time() - t0:.0f} s -> {out}")
