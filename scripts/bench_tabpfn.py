"""TabPFN latency + quality benchmark for the delta pick line (local GPU), on the simulated line's real logs.

Answers: how fast is a per-pick decision (9 candidate speed settings) and a speed-map refresh (1,800 rows), for
TabPFN-3.5 and 3.5-Fast, by context size and n_estimators, with a cached context. Context: the commissioning log;
held-out quality on the evaluation bank.

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

from policy.policies import X_COLS

QUICK = "--quick" in sys.argv
LOG = "data/commission_lap.csv,data/commission_srv.csv"
BANK = "data/bank_srv.csv,data/bank_lap.csv"
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})")

import pandas as pd

pool = pd.concat([pd.read_csv(f) for f in LOG.split(",")], ignore_index=True).sample(frac=1.0, random_state=0)
test = pd.concat([pd.read_csv(f) for f in BANK.split(",")], ignore_index=True)
test = test[test.groupby(["stream", "idx"])["alpha"].transform("nunique") == 9]
print(f"commissioning log: {len(pool)} rows, drop rate {pool['dropped'].mean():.1%}; bank {len(test)} rows")
items = test.groupby(["stream", "idx"]).ngroups
# speed map: 200 bank items x 9 speed settings = 1,800 rows; one pick: one item's 9 rows
keys = test[["stream", "idx"]].drop_duplicates().sample(n=min(200, items), random_state=0)
map_q = test.merge(keys)[X_COLS].to_numpy(float)
pick_q = test.merge(keys.iloc[:1])[X_COLS].to_numpy(float)
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
contexts = [300, 1000] if QUICK else [300, 1000, len(pool)]
estims = [1, 4] if QUICK else [1, 2, 4, 8]
print(f"\n{'model':>8} {'ctx':>5} {'est':>3} | {'fit+cache ms':>12} | {'1 row ms':>9} {'9 rows ms (p95)':>16} | {'1800-row map ms':>15} | {'AUC':>5} {'Brier':>6}")
for vname, ver in versions:
    for n_ctx in contexts:
        X = pool[X_COLS].to_numpy(float)[:n_ctx]
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
            nine = timed(lambda: clf.predict_proba(pick_q), 5 if QUICK else 15)
            mp = timed(lambda: clf.predict_proba(map_q), 2 if QUICK else 4)
            p = clf.predict_proba(test[X_COLS].to_numpy(float))[:, 1]
            yt = test["dropped"].astype(int).to_numpy()
            auc = roc_auc_score(yt, p) if 0 < yt.sum() < len(yt) else float("nan")
            print(f"{vname:>8} {n_ctx:>5} {ne:>3} | {fit_ms:12.0f} | {one[0]:9.1f} {nine[0]:8.1f} ({nine[1]:5.1f}) | {mp[0]:15.0f} | {auc:5.3f} {brier_score_loss(yt, p):6.4f}")
            del clf
            if dev == "cuda":
                torch.cuda.empty_cache()
if dev == "cuda":
    print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
