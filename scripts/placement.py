"""Where will the product land? Placement error of held picks predicted per speed setting, as a distribution.

Context: held picks of the commissioning log (place_offset_mm = product centre vs place target at arrival). Test: held
picks of the evaluation bank. Models: TabPFN-3.5 regressor (quantiles of its predictive distribution), gradient-boosted
quantile regression (one model per quantile), and linear quantile regression. Scores: MAE of the median, pinball loss
averaged over q 0.05/0.5/0.95, and coverage of the 90 % interval (should be 90 %), per product.
Usage: .venv/bin/python scripts/placement.py LOG.csv[,..] BANK.csv[,..] OUT.csv
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from policy.policies import X_COLS

Q = [0.05, 0.5, 0.95]


def pinball(y, q_pred):
    return np.mean([np.mean(np.maximum(q * (y - p), (q - 1) * (y - p))) for q, p in zip(Q, q_pred)])


def fit_predict(name, X, y, Xt):
    if name == "tabpfn-3.5":
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, n_estimators=1, device="cuda", random_state=0)
        m.fit(X, y)
        return list(m.predict(Xt, output_type="quantiles", quantiles=Q))
    if name == "gbdt":
        return [HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=0).fit(X, y).predict(Xt)
                for q in Q]
    pre = lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    return [make_pipeline(pre(), QuantileRegressor(quantile=q, alpha=0.0, solver="highs")).fit(X, y).predict(Xt) for q in Q]


if __name__ == "__main__":
    log = pd.concat([pd.read_csv(f) for f in sys.argv[1].split(",")], ignore_index=True)
    bank = pd.concat([pd.read_csv(f) for f in sys.argv[2].split(",")], ignore_index=True)
    log, bank = log[~log["dropped"].astype(bool)], bank[~bank["dropped"].astype(bool)]
    X, y = log[X_COLS].to_numpy(float), log["place_offset_mm"].to_numpy(float)
    Xt, yt = bank[X_COLS].to_numpy(float), bank["place_offset_mm"].to_numpy(float)
    print(f"context {len(log)} held picks, test {len(bank)} held picks; offset median {np.median(yt):.2f} mm, "
          f"90th pct {np.percentile(yt, 90):.2f} mm", flush=True)
    rows = []
    for name in ["tabpfn-3.5", "gbdt", "linear"]:
        t0 = time.time()
        qp = fit_predict(name, X, y, Xt)
        for prod in ["all"] + sorted(bank["product"].unique()):
            ix = np.ones(len(yt), bool) if prod == "all" else (bank["product"] == prod).to_numpy()
            rows.append(dict(model=name, product=prod, n=int(ix.sum()), mae_median=np.mean(np.abs(yt[ix] - qp[1][ix])),
                             pinball=pinball(yt[ix], [p[ix] for p in qp]),
                             coverage90=np.mean((yt[ix] >= qp[0][ix]) & (yt[ix] <= qp[2][ix])),
                             width90=np.mean(qp[2][ix] - qp[0][ix]), seconds=time.time() - t0))
        print(pd.DataFrame(rows[-1 - bank["product"].nunique():]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(sys.argv[3], index=False)
