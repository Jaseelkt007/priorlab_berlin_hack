"""torch fits, JAX answers: the live loop in one process, checked against torch every refit.

Each round refits TabPFN (torch, fp32, unquantised cache) on a new 1,000-row window of
the toy log, exports the context in memory, swaps it into the JAX runtime (no
recompile: same shapes) and compares 15-row pick probabilities with torch's.

Needs torch + tabpfn and jax in one interpreter. Until they share an env, run from the
repo root with the torch env's packages appended (see DEBT.md):
  PYTHONPATH=.:../priorlab_berlin_hack/.venv/lib/python3.12/site-packages .venv-jax/bin/python -m jax_tabpfn.refit_demo
"""
import time

from jax_tabpfn.runtime import JaxTabPFN  # import jax before torch

import numpy as np
import torch
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path.home() / "priorlab_berlin_hack/.env")
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from jax_tabpfn.export import bundle_from_classifier
from sim.delta_toy import FEATURES, SPEEDS, make_log, run_picks

rng = np.random.default_rng(0)
log = make_log(rng, 4000, prod="pouch", hum=55)
clf = TabPFNClassifier.create_default_for_version(
    ModelVersion.V3_5_FAST, n_estimators=1, device="cuda", fit_mode="fit_with_cache", random_state=0,
    inference_precision=torch.float32, kv_cache_precision="auto")
models = {}
print(f"{'round':>5s} {'window':>11s} | {'torch fit':>9s} {'export':>6s} {'JAX swap':>8s} | "
      f"{'JAX fp32 ms':>11s} {'max|dp|':>8s} | {'JAX fast ms':>11s} {'max|dp|':>8s}")
for r, start in enumerate([0, 1000, 2000, 3000]):
    window = log.iloc[start:start + 1000]
    t0 = time.perf_counter()
    clf.fit(window[FEATURES].to_numpy(), window["dropped"].to_numpy())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    b = bundle_from_classifier(clf)
    t2 = time.perf_counter()
    for fast in (False, True):
        if fast not in models:
            models[fast] = JaxTabPFN(b, fast=fast)
        else:
            models[fast].update_context(b)
    t3 = time.perf_counter()
    res = []
    for fast in (False, True):
        dps, ts = [], []
        for i in range(5):  # 5 fresh picks
            q = run_picks(log.iloc[[(start + 1000 + 37 * i) % 4000] * len(SPEEDS)].reset_index(drop=True), SPEEDS,
                          rng)[FEATURES].to_numpy()
            t = time.perf_counter()
            p = models[fast].predict_proba(q)
            ts.append(1000 * (time.perf_counter() - t))
            dps.append(np.abs(p - clf.predict_proba(q)).max())
        res.append((np.median(ts), max(dps)))
    print(f"{r:5d} {start:5d}-{start + 1000:<5d} | {1000 * (t1 - t0):7.0f}ms {1000 * (t2 - t1):4.0f}ms "
          f"{1000 * (t3 - t2) / 2:6.0f}ms | {res[0][0]:11.2f} {res[0][1]:8.1e} | {res[1][0]:11.2f} {res[1][1]:8.1e}")
