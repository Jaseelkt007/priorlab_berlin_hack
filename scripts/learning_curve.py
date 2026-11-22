"""How many logged picks does the drop model need? Prediction and decision quality vs context size.

Context: random subsets of the commissioning log (all products x all speed settings, random line states). Test: every
pick of the evaluation bank (unseen items, line states and drift; all speed settings of each item). For each size and
repeat, every model is fitted on the same rows and scored on the same test rows:
- prediction: log-loss, Brier score, ROC AUC, expected calibration error (10 equal-width bins);
- decision: per bank item, the fastest alpha with predicted P(drop) <= target, then the realised good picks per minute
  and drop rate (static model, no feedback; the closed loop is scripts/run_eval.py).
Usage: .venv/bin/python scripts/learning_curve.py LOG.csv[,LOG2.csv] BANK.csv[,BANK2.csv] OUT.csv [--sizes ...]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from policy.policies import X_COLS


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())


def decide(test, p, target):
    """Fastest alpha with p <= target per item (slowest if none); realised good picks/min and drop rate."""
    t = test.assign(p=p, ok=p <= target)
    ch = []
    for _, g in t.groupby(["stream", "idx"], sort=False):
        ok = g[g["ok"]]
        ch.append(ok.loc[ok["alpha"].idxmax()] if len(ok) else g.loc[g["alpha"].idxmin()])
    c = pd.DataFrame(ch)
    held = ~c["dropped"].astype(bool)
    return 60 * held.sum() / c["cycle_s"].sum(), 1 - held.mean()


def models(version):
    def tabpfn():
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        ver = {"3.5": ModelVersion.V3_5, "3.5-fast": ModelVersion.V3_5_FAST}[version]
        return TabPFNClassifier.create_default_for_version(ver, n_estimators=1, device="cuda", random_state=0)
    return {f"tabpfn-{version}": tabpfn,
            "gbdt": lambda: HistGradientBoostingClassifier(random_state=0),
            # evac_ms is NaN when the vacuum never reached the threshold: impute plus a missing indicator
            "logistic": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                              LogisticRegression(max_iter=2000))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("bank")
    ap.add_argument("out")
    ap.add_argument("--sizes", default="100,250,500,1000,2000,2700")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--targets", default="0.02,0.05,0.1")
    ap.add_argument("--version", default="3.5")
    a = ap.parse_args()
    log = pd.concat([pd.read_csv(f) for f in a.log.split(",")], ignore_index=True)
    test = pd.concat([pd.read_csv(f) for f in a.bank.split(",")], ignore_index=True)
    test = test[test.groupby(["stream", "idx"])["alpha"].transform("nunique") == 9].reset_index(drop=True)
    Xt, yt = test[X_COLS].to_numpy(float), test["dropped"].astype(int).to_numpy()
    targets = [float(x) for x in a.targets.split(",")]
    print(f"log {len(log)} rows ({log['dropped'].mean():.1%} drops); test {len(test)} rows, "
          f"{test.groupby(['stream', 'idx']).ngroups} items ({yt.mean():.1%} drops)", flush=True)
    rows = []
    for n in [int(x) for x in a.sizes.split(",")]:
        for r in range(a.repeats if n < len(log) else 1):
            sub = log.sample(n=min(n, len(log)), random_state=r)
            X, y = sub[X_COLS].to_numpy(float), sub["dropped"].astype(int).to_numpy()
            for name, make in models(a.version).items():
                t0 = time.time()
                m = make().fit(X, y)
                p = np.clip(m.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)
                row = dict(model=name, n=len(sub), repeat=r, log_loss=log_loss(yt, p), brier=brier_score_loss(yt, p),
                           auc=roc_auc_score(yt, p), ece=ece(yt, p), seconds=time.time() - t0)
                for tg in targets:
                    row[f"gpm@{tg:g}"], row[f"drop@{tg:g}"] = decide(test, p, tg)
                rows.append(row)
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["model", "n"]).mean(numeric_only=True).round(4).to_string())
