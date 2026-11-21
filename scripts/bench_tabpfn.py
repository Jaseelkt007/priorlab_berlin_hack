"""TabPFN latency + quality benchmark for the delta pick line (local GPU).

Answers: how fast is a per-pick decision (15 candidate speeds) and a speed-map refresh
(thousands of rows), for TabPFN-3.5 and 3.5-Fast, by context size and n_estimators,
with a cached context. Also reports held-out drop-prediction quality.

Usage: .venv/bin/python scripts/bench_tabpfn.py [--quick]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from sim.delta_toy import FEATURES, SPEEDS, make_log, run_picks, sample_items

QUICK = "--quick" in sys.argv
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})")

pool = make_log(rng, 4000, prod="pouch", hum=55)
test = make_log(rng, 2000, prod="pouch", hum=55)
print(f"toy log: {len(pool)} rows, drop rate {pool['dropped'].mean():.1%}")

# speed-map query: 200 (seal, weight) grid points x 15 speeds = 3000 rows
grid_items = sample_items(rng, "pouch", 200, hum=55)
grid_items["seal_kpa"] = np.repeat(np.linspace(10, 60, 20), 10)
grid_items["weight_g"] = np.tile(np.linspace(54, 66, 10), 20)
map_q = run_picks(grid_items.loc[grid_items.index.repeat(len(SPEEDS))].reset_index(drop=True),
                  np.tile(SPEEDS, len(grid_items)), rng)[FEATURES].to_numpy()
pick_q = run_picks(test.iloc[[0] * len(SPEEDS)].reset_index(drop=True), SPEEDS, rng)[FEATURES].to_numpy()
one_q = pick_q[:1]


def timed(fn, reps):
    if dev == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return 1000 * np.median(ts), 1000 * np.percentile(ts, 95)


versions = [("3.5-Fast", ModelVersion.V3_5_FAST), ("3.5", ModelVersion.V3_5)]
contexts = [300, 1000] if QUICK else [100, 300, 1000, 3000]
estims = [1, 4] if QUICK else [1, 2, 4, 8]
print(f"\n{'model':>8} {'ctx':>5} {'est':>3} | {'fit+cache ms':>12} | {'1 row ms':>9} {'15 rows ms (p95)':>17} | {'3000-row map ms':>15} | {'AUC':>5} {'Brier':>6}")
for vname, ver in versions:
    for n_ctx in contexts:
        X = pool[FEATURES].to_numpy()[:n_ctx]
        y = pool["dropped"].to_numpy()[:n_ctx]
        for ne in estims:
            clf = TabPFNClassifier.create_default_for_version(ver, n_estimators=ne, device=dev, fit_mode="fit_with_cache",
                                                              random_state=0)
            t = time.perf_counter()
            clf.fit(X, y)
            if dev == "cuda":
                torch.cuda.synchronize()
            fit_ms = 1000 * (time.perf_counter() - t)
            clf.predict_proba(pick_q)  # warm-up
            one = timed(lambda: clf.predict_proba(one_q), 5 if QUICK else 15)
            fifteen = timed(lambda: clf.predict_proba(pick_q), 5 if QUICK else 15)
            mp = timed(lambda: clf.predict_proba(map_q), 2 if QUICK else 4)
            p = clf.predict_proba(test[FEATURES].to_numpy())[:, 1]
            yt = test["dropped"].to_numpy()
            auc = roc_auc_score(yt, p) if 0 < yt.sum() < len(yt) else float("nan")
            print(f"{vname:>8} {n_ctx:>5} {ne:>3} | {fit_ms:12.0f} | {one[0]:9.1f} {fifteen[0]:8.1f} ({fifteen[1]:5.1f}) | {mp[0]:15.0f} | {auc:5.3f} {brier_score_loss(yt, p):6.4f}")
            del clf
            if dev == "cuda":
                torch.cuda.empty_cache()
if dev == "cuda":
    print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
