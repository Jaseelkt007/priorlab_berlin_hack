"""How fast is TabPFN per radar reading, and what does speeding it up cost in accuracy? (EXPERIMENTS.md 16)

    python drone/latency.py        # needs a GPU, .env with TABPFN_TOKEN, drone/data/ tables and IMM files

The live budget: the radar gives a new reading every 100 ms, so all 4 models (forward, left, up, alarm)
must answer within that. Setup: IMM + TabPFN (experiment 15), examples from X2 + 1,000 Crazyflie rows, tested on
pilot b. For each setting: delay for one reading (1 drone, and 10 drones at once) and the error on pilot b
(every 10th reading). Writes drone/results/latency.csv.
"""

import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import tabpfn_model as T
from features import to_world

ROOT = Path(__file__).resolve().parent.parent
# (name, TabPFN options, X2 example rows)
SETTINGS = [
    ("as used so far", {}, 4000),
    ("+ example cache", {"fit_mode": "fit_with_cache"}, 4000),
    ("+ cache, 2 passes", {"fit_mode": "fit_with_cache", "n_estimators": 2}, 4000),
    ("+ cache, 1 pass", {"fit_mode": "fit_with_cache", "n_estimators": 1}, 4000),
    ("+ cache, 1 pass, 1,000 X2 rows", {"fit_mode": "fit_with_cache", "n_estimators": 1}, 1000),
]
REPEATS = 30


def main():
    import torch

    load_dotenv(ROOT / ".env")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    x2, cf2, pb = T.load("x2"), T.load("cf2"), T.load("cf2_pilotb")
    for name, t in (("x2", x2), ("cf2", cf2), ("cf2_pilotb", pb)):
        t["X"] = np.column_stack([t["X"], T.imm_columns(name, t)])
    adapt = np.isin(cf2["flight"], np.unique(cf2["flight"])[: T.ADAPT_FLIGHTS])
    test = np.where(T.complete(pb))[0][::10]
    truth = pb["Y_world"][test, :3]

    def timed(models, rows):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for kind, m in models:
            m.predict(rows, output_type="quantiles", quantiles=T.QUANTILES) if kind == "reg" else m.predict_proba(rows)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    out = []
    for name, options, n_x2 in SETTINGS:
        rng = np.random.default_rng(T.SEED)
        ctx = T.sample(rng, np.where(T.complete(x2))[0], n_x2)
        ad = T.sample(rng, np.where(T.complete(cf2) & adapt)[0], 1000)
        X = np.concatenate([x2["X"][ctx], cf2["X"][ad]])
        Y = np.concatenate([x2["Y"][ctx], cf2["Y"][ad]])
        breach = np.concatenate([x2["breach"][ctx], cf2["breach"][ad]])
        models, move = [], []
        for t in range(3):
            m = T.make_regressor(**options)
            m.fit(X, Y[:, t])
            models.append(("reg", m))
            # the median, as in every other score
            move.append(np.concatenate([m.predict(b, output_type="quantiles", quantiles=[0.5])[0]
                                        for b in T.batches(pb["X"][test], 2000)]))
        c = T.make_classifier(**options)
        c.fit(X, breach)
        models.append(("clf", c))
        error = np.median(np.linalg.norm(to_world(np.column_stack(move), pb["pos"][test], pb["psi"][test]) - truth, axis=1))
        row = {"setting": name, "examples": len(X), "error_pilot_b_m": round(float(error), 3)}
        for n in (1, 10):
            for _ in range(3):
                timed(models, pb["X"][test[:n]])  # warm up
            ms = [timed(models, pb["X"][test[i * n : (i + 1) * n]]) for i in range(REPEATS)]
            row[f"ms_{n}_drone{'s' if n > 1 else ''}"] = round(float(np.median(ms)), 1)
        out.append(row)
        print(row, flush=True)
        for _, m in models:
            T.free(m)
    import pandas as pd

    df = pd.DataFrame(out)
    df["fits_100ms"] = df["ms_1_drone"] < 100
    df.to_csv(ROOT / "drone" / "results" / "latency.csv", index=False)
    print(df.to_string(index=False))
    print(f"GPU: {torch.cuda.get_device_name()}; every 10th pilot-b reading ({len(test):,}) for the error")


if __name__ == "__main__":
    main()
