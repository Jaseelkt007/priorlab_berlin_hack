"""Pick-log generation: one row per simulated pick on the full line (sim.line), in the columns a real line logs.

Columns
- known before the loaded move (TabPFN features): product, weight_g, len_mm, wid_mm, hgt_mm, prod_t, room_t, room_rh,
  cup_cycles, cup_d_mm, p_set_kpa, belt_mps, seal_kpa, evac_ms, and the action alpha (speed setting);
- outcomes: dropped, place_offset_mm; diagnostics: slip_mm, vac_min_kpa, move_ms, cycle_s, a_peak;
- hidden physics (prefixed "_"): never given to a model.
"""
from __future__ import annotations

import os

# one thread per worker process: set before numpy/numba load (spawned workers inherit the environment)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

ALPHAS = np.round(np.arange(0.2, 1.01, 0.1), 2)     # ABB AccSet: 20–100 % (values below 20 are raised to 20)
FEATURES = ["weight_g", "len_mm", "wid_mm", "hgt_mm", "prod_t", "room_t", "room_rh", "cup_cycles", "cup_d_mm",
            "p_set_kpa", "belt_mps", "seal_kpa", "evac_ms"]


def random_state(rng: np.random.Generator):
    """A plausible line state (research §3): cup anywhere in its life (incl. overdue), room climate, batch drift."""
    from .products import LineState
    return LineState(cup_cycles=int(rng.uniform(0, 900_000)), room_t=float(rng.uniform(19, 28)),
                     room_rh=float(rng.uniform(35, 75)), batch_mass_shift=float(rng.normal(0, 0.02)),
                     batch_seal_shift=float(rng.normal(0, 3.0)))


def simulate_instance(task: dict) -> list[dict]:
    """One pick instance (product item, line state, belt speed and vision error, all drawn from task["seed"]) simulated
    at each speed setting in task["alphas"] (default: all ALPHAS). The rows are exact counterfactuals of one another:
    only alpha differs. task: product, seed, optional state (LineState fields), belt_mps, alphas, and any extra keys
    (copied into the rows, e.g. a stream id). Returns one log row per alpha."""
    from .line import PickCell, PickSetup
    from .products import LineState, sample
    rows = []
    for alpha in task.get("alphas", ALPHAS):
        rng = np.random.default_rng(task["seed"])
        st = LineState(**task["state"]) if task.get("state") else random_state(rng)
        v_b = task.get("belt_mps", float(rng.uniform(0.2, 0.5)))
        cfg, logged, hidden = sample(task["product"], rng, st)
        out = PickCell(PickSetup(cup=cfg, alpha=float(alpha), v_belt=v_b, seed=task["seed"])).run()
        row = {k: v for k, v in task.items() if k not in ("product", "seed", "state", "belt_mps", "alphas")}
        row.update(logged, belt_mps=v_b, alpha=float(alpha), seed=task["seed"])
        row.update({k: out[k] for k in ("seal_kpa", "evac_ms", "dropped", "place_offset_mm", "slip_mm", "vac_min_kpa",
                                        "move_ms", "cycle_s", "a_peak")})
        row.update(hidden, _vision_err_mm=out["vision_err_mm"])
        rows.append(row)
    return rows


def simulate_pick(task: dict) -> dict:
    """task: product, alpha, seed, optional state (LineState fields) and belt_mps. Returns one log row."""
    return simulate_instance({k: v for k, v in task.items() if k != "alpha"} | {"alphas": [task["alpha"]]})[0]


def _warm_compile() -> None:
    """Compile the Numba pneumatics once in the parent so spawned workers load the on-disk cache instead of racing
    to write it."""
    from .line import PickCell, PickSetup  # noqa: F401  (imports sim.pneumatics)
    from .pneumatics import EJECTORS, LipLeak, VacuumCircuit, step_lip_fast
    circ = VacuumCircuit(EJECTORS["Si08-3"], V_cup=4e-6)
    step_lip_fast(circ, 2e-4, LipLeak(), np.zeros(16))


def run_tasks(tasks: list[dict], out_csv, workers: int | None = None, progress_every: int = 50,
              max_attempts: int = 3, fn=None) -> pd.DataFrame:
    """Simulate all tasks in parallel (fn: simulate_pick, or simulate_instance for all alphas of an instance), appending
    each finished pick to out_csv as it completes (a kill loses nothing already simulated). A worker that dies is reported by the executor (no silent hang); its unfinished tasks are
    resubmitted to a fresh pool, up to max_attempts times each."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    # worker processes: argument, else env DELTA_WORKERS (12 on the shared Nebius server), else all cores but one
    workers = workers or int(os.environ.get("DELTA_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 1)
    _warm_compile()
    t0, n_done = time.time(), 0
    pending = {i: 0 for i in range(len(tasks))}         # task index -> attempts so far
    with open(out_csv, "w", newline="") as f:
        w = None
        while pending:
            todo = [i for i, n in pending.items() if n < max_attempts]
            if not todo:
                break
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
                futs = {ex.submit(fn or simulate_pick, tasks[i]): i for i in todo}
                try:
                    for fu in as_completed(futs):
                        i = futs[fu]
                        try:
                            out = fu.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:                     # a simulation error is data, not a crash
                            pending[i] += 1
                            print(f"  task {tasks[i]} failed: {e!r}", flush=True)
                            continue
                        for row in (out if isinstance(out, list) else [out]):
                            if w is None:
                                w = csv.DictWriter(f, fieldnames=list(row))
                                w.writeheader()
                            w.writerow(row)
                        f.flush()
                        del pending[i]
                        n_done += 1
                        if n_done % progress_every == 0 or n_done == len(tasks):
                            el = time.time() - t0
                            print(f"  {n_done}/{len(tasks)} picks, {el:.0f} s, {n_done / el:.2f} picks/s", flush=True)
                except BrokenProcessPool:
                    for i in pending:
                        pending[i] += 1
                    print(f"  a worker died; restarting the pool for {len(pending)} unfinished picks", flush=True)
    if pending:
        print(f"  gave up on {len(pending)} picks after {max_attempts} attempts: {[tasks[i] for i in pending]}")
    return pd.read_csv(out_csv)
