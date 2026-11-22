"""Experiment C: copy the flight policy offline (behaviour cloning in context, no training).

    uv run python experiments/exp_c_clone.py --dry-run
    uv run python experiments/exp_c_clone.py

Data: pass1_clean at full rate (every 0.2 ms step, so all wingbeat phases are covered).
Table (DuckPFN's pooled layout): one row per (step, action): the policy's 104 inputs plus an
action-ID column (categorical) -> that action, standardised per action. One TabPFN call per
context size covers all 12 actions. Context: N random steps from training flight pairs.
Test: 1,000 random steps from held-out pairs (same split as A and B).
Comparison on the same N steps: gradient-boosted trees (one per action) and an MLP.
"""

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flydata import ROOT, load, split_pairs, tabpfn_client_ready

OUT = ROOT / "results/exp_c"
SIZES = (100, 500, 2000, 8000)
N_TEST = 1000


def pooled(obs, y_std):
    """(n, 104) obs and (n, 12) standardised actions -> (n*12, 105) rows and (n*12,) targets."""
    n, a = y_std.shape
    X = np.column_stack([np.repeat(obs, a, axis=0), np.tile(np.arange(a), n)]).astype(np.float32)
    return X, y_std.reshape(-1)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2, axis=0) / np.sum((y - y.mean(0)) ** 2, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=float, default=2e6)
    args = parser.parse_args()
    tabpfn_client = tabpfn_client_ready()

    d, meta = load("pass1_clean", wingbeat=False, keys=["ids", "obs", "action_mean"])
    names = meta["columns"]["action_mean"]
    test_mask, _ = split_pairs(d["ids"])
    rng = np.random.default_rng(0)
    train_idx = rng.permutation(np.flatnonzero(~test_mask))
    test_idx = rng.choice(np.flatnonzero(test_mask), N_TEST, replace=False)
    Xte_obs, Yte = d["obs"][test_idx], d["action_mean"][test_idx]

    total = 0.0
    for n in SIZES:
        Xp, _ = pooled(d["obs"][train_idx[:n]], d["action_mean"][train_idx[:n]])
        Xtp, _ = pooled(Xte_obs, Yte)
        c = float(tabpfn_client.estimate_cost(Xp, Xtp).estimated_cost)
        total += c
        print(f"  {n:5d} steps -> {len(Xp):6d} context rows, {len(Xtp)} test rows, ~{c:,.0f} tokens")
    print(f"estimated total: {total:,.0f} tokens for {len(SIZES)} calls")
    if args.dry_run:
        return
    if total > args.max_tokens:
        raise SystemExit(f"estimate {total:,.0f} exceeds --max-tokens {args.max_tokens:,.0f}")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    for n in SIZES:
        obs, act = d["obs"][train_idx[:n]], d["action_mean"][train_idx[:n]]
        mu, sd = act.mean(0), np.maximum(act.std(0), 1e-6)
        Xp, yp = pooled(obs, (act - mu) / sd)
        Xtp, _ = pooled(Xte_obs, (Yte - mu) / sd)
        path = cache / f"n{n}.npy"
        if path.exists():
            pred = np.load(path)
        else:
            t0 = time.time()
            model = tabpfn_client.TabPFNRegressor(random_state=0, categorical_features_indices=[Xp.shape[1] - 1])
            pred = np.asarray(model.fit(Xp, yp).predict(Xtp))
            np.save(path, pred)
            print(f"  n={n}: TabPFN fit+predict {time.time() - t0:.1f} s", flush=True)
        tab = pred.reshape(N_TEST, -1) * sd + mu
        gbt = np.column_stack([HistGradientBoostingRegressor(random_state=0).fit(obs, act[:, j]).predict(Xte_obs)
                               for j in range(act.shape[1])])
        mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=2000, early_stopping=n >= 500,
                                                           random_state=0)).fit(obs, (act - mu) / sd)
        mlp = mlp.predict(Xte_obs) * sd + mu
        results[n] = {m: {"r2_mean": float(np.mean(r2(Yte, p))), "r2_per_action": dict(zip(names, r2(Yte, p).round(3).tolist()))}
                      for m, p in (("tabpfn", tab), ("gbt", gbt), ("mlp", mlp))}

    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nmean R² over 12 actions on held-out flights ({N_TEST} random steps)")
    print(f"{'steps':>6s} {'TabPFN':>8s} {'GBT':>8s} {'MLP':>8s}")
    for n, r in results.items():
        print(f"{n:6d} {r['tabpfn']['r2_mean']:8.3f} {r['gbt']['r2_mean']:8.3f} {r['mlp']['r2_mean']:8.3f}")
    big = results[max(SIZES)]
    print(f"\nper action at {max(SIZES)} steps (TabPFN / GBT / MLP):")
    for a in names:
        print(f"  {a:16s} {big['tabpfn']['r2_per_action'][a]:6.3f} {big['gbt']['r2_per_action'][a]:6.3f} {big['mlp']['r2_per_action'][a]:6.3f}")
    print("\nusage:", tabpfn_client.get_api_usage())


if __name__ == "__main__":
    main()
