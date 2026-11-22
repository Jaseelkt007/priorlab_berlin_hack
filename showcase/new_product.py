"""S05: a product the line has never logged. The evaluation's new:bar stream (80 bar picks, random line states and belt
speeds; every pick simulated at all 9 speeds in results/logs/bank_*.csv), with the bar REMOVED from the context.

Policies, each seeing only what a real line knows before the loaded move:
  fixed      : FixedRecipe(cost=10) from the commissioning log; an unknown product runs at the slowest setting (20 %)
  tabpfn_jax : TabPFN-3.5-Fast answered in JAX (as in showcase/line_shift.py): cost=10, safe-step (never more than one
               setting above the fastest it has seen this product hold), refit every 2 picks, an exactly fixed-size
               rolling context (the commissioning log without the bar, every new pick in, the oldest held row out)
Output: results/new_bar/decisions_jax.csv (same columns as the shift decisions).
Usage (server): .venv/bin/python showcase/new_product.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax_tabpfn.runtime  # noqa: F401,E402  (jax before torch)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STREAM = "new:bar"
OUT = ROOT / "results" / "new_bar"


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from policy.policies import X_COLS, FixedRecipe, replay
    from showcase.line_shift import jax_policy_class
    bank = pd.concat([pd.read_csv(ROOT / "results/logs/bank_lap.csv"), pd.read_csv(ROOT / "results/logs/bank_srv.csv")],
                     ignore_index=True)
    bank = bank[bank["stream"] == STREAM]
    assert bank["idx"].nunique() == 80 and (bank.groupby("idx")["alpha"].nunique() == 9).all(), "incomplete bank"
    log = pd.concat([pd.read_csv(ROOT / "results/logs/commission_lap.csv"),
                     pd.read_csv(ROOT / "results/logs/commission_srv.csv")], ignore_index=True)
    ctx = log[log["product"] != "bar"]                 # the line has never logged this product
    out = []
    for name, pol in (("fixed", FixedRecipe(cost=10)),
                      ("tabpfn_jax", jax_policy_class()(cost=10, safe_step=True, refit_every=2))):
        risk, nctx, lat = {}, {}, {}
        if name.startswith("tabpfn"):
            orig = pol.choose

            def choose(cand, pol=pol, orig=orig):
                i = int(cand["idx"].iloc[0])
                t0 = time.perf_counter()
                p = pol._proba(cand[X_COLS].to_numpy(float)) if pol.const is None else np.full(len(cand), pol.const)
                lat[i] = (time.perf_counter() - t0) * 1e3
                risk[i], nctx[i] = [float(x) for x in p], len(pol.log)
                return orig(cand)
            pol.choose = choose
        runs = replay(bank, pol, ctx)
        runs["policy_key"] = name
        runs["risk"] = runs["idx"].map(lambda i: json.dumps(risk.get(int(i), [])))
        runs["n_ctx"] = runs["idx"].map(lambda i: nctx.get(int(i), len(ctx)))
        runs["tabpfn_ms"] = runs["idx"].map(lambda i: lat.get(int(i), np.nan))
        out.append(runs)
        good = (~runs["dropped"].astype(bool)).sum()
        ms = runs["tabpfn_ms"].dropna()
        print(f"{name:10s}: drops {int(runs['dropped'].sum())}/{len(runs)}, good picks/min "
              f"{60 * good / runs['cycle_s'].sum():.1f}, speeds {runs['alpha'].round(1).tolist()}"
              + (f", query ms median {ms.median():.2f} max {ms.max():.2f}" if len(ms) else ""), flush=True)
    dec = pd.concat(out, ignore_index=True)
    dec["sku"] = "bar 45 g"
    dec["film"] = "standard"
    keep = ["policy_key", "policy", "idx", "sku", "film", "cup_cycles", "weight_g", "belt_mps", "alpha", "seed",
            "seal_kpa", "evac_ms", "dropped", "place_offset_mm", "cycle_s", "a_peak", "risk", "n_ctx", "tabpfn_ms",
            "prod_t", "room_rh"]
    OUT.mkdir(parents=True, exist_ok=True)
    dec[keep].to_csv(OUT / "decisions_jax.csv", index=False)
    print("wrote", OUT / "decisions_jax.csv")


if __name__ == "__main__":
    main()
