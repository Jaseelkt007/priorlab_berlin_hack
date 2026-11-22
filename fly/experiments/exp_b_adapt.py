"""Experiment B: adapt to a changed fly by changing the context rows only (no training).

    uv run python experiments/exp_b_adapt.py --dry-run
    uv run python experiments/exp_b_adapt.py

Task: experiment A's forward model at 3 wingbeats (~14 ms), tested on the *changed* fly's
held-out flights (pass 4: left wing at 70% strength; pass 5: body mass x1.2).
Context conditions:
  clean        normal-fly training flights only (what a model knows before the damage)
  clean+N      the same, plus all wingbeats of N changed-fly training flights (N = 4, 16)
  clean+all    plus every changed-fly training flight
  changed      changed-fly training flights only
Comparison: gradient-boosted trees *retrained* on exactly the same rows for every condition.
Same flight-pair split as experiment A. API results are cached in results/exp_b/cache.
"""

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from flydata import FORWARD_TARGETS, ROOT, build_forward, load, metrics, split_pairs, tabpfn_client_ready

OUT = ROOT / "results/exp_b"
H = 3
QUANTILES = [0.1, 0.5, 0.9]
CHANGED = {"weak_wing": "pass4_weak_wing", "heavy": "pass5_heavy"}
N_FLIGHTS = (4, 16)


def table(name):
    d, meta = load(name)
    X, _, per_h = build_forward(d, meta, (H,))
    p = per_h[H]
    test, _ = split_pairs(d["ids"])
    return X[p["rows"]], p["Y"], d["ids"][p["rows"]], test[p["rows"]], p["const_vel"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tokens", type=float, default=2e6)
    args = parser.parse_args()
    tabpfn_client = tabpfn_client_ready()
    rng = np.random.default_rng(0)

    Xc, Yc, _, test_c, _ = table("pass1_clean")
    clean_ctx = ~test_c

    jobs = []  # (changed, condition, X_context, Y_context, X_test, Y_test, const_vel_test)
    for changed, name in CHANGED.items():
        Xd, Yd, idsd, test_d, cv = table(name)
        train_flights = np.unique(idsd[~test_d, 0])
        conds = {"clean": (Xc[clean_ctx], Yc[clean_ctx])}
        for n in N_FLIGHTS:
            pick = np.isin(idsd[:, 0], rng.choice(train_flights, n, replace=False)) & ~test_d
            conds[f"clean+{n}"] = (np.vstack([Xc[clean_ctx], Xd[pick]]), np.vstack([Yc[clean_ctx], Yd[pick]]))
        conds["clean+all"] = (np.vstack([Xc[clean_ctx], Xd[~test_d]]), np.vstack([Yc[clean_ctx], Yd[~test_d]]))
        conds["changed"] = (Xd[~test_d], Yd[~test_d])
        for cond, (Xx, Yy) in conds.items():
            jobs.append((changed, cond, Xx, Yy, Xd[test_d], Yd[test_d], cv[test_d]))

    total = 0.0
    for changed, cond, Xx, _, Xt, _, _ in jobs:
        c = float(tabpfn_client.estimate_cost(Xx, Xt).estimated_cost)
        total += c * len(FORWARD_TARGETS)
        print(f"  {changed:9s} {cond:10s}: {len(Xx):6d} context rows, {len(Xt)} test rows, ~{c:,.0f} tokens/target")
    print(f"estimated total: {total:,.0f} tokens for {len(jobs) * len(FORWARD_TARGETS)} calls")
    if args.dry_run:
        return
    if total > args.max_tokens:
        raise SystemExit(f"estimate {total:,.0f} exceeds --max-tokens {args.max_tokens:,.0f}")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    for changed, cond, Xx, Yy, Xt, Yt, cv in jobs:
        for k, target in enumerate(FORWARD_TARGETS):
            key = f"{changed}_{cond}_{target}"
            path = cache / f"{key}.npy"
            if path.exists():
                q = np.load(path)
            else:
                t0 = time.time()
                model = tabpfn_client.TabPFNRegressor(random_state=0).fit(Xx, Yy[:, k])
                q = np.stack(model.predict(Xt, output_type="quantiles", quantiles=QUANTILES))
                np.save(path, q)
                print(f"  {key}: {time.time() - t0:.1f} s", flush=True)
            gbt = HistGradientBoostingRegressor(random_state=0).fit(Xx, Yy[:, k]).predict(Xt)
            results[key] = {"changed": changed, "condition": cond, "target": target, "n_context": len(Xx),
                            "tabpfn": metrics(Yt[:, k], q[1], q[0], q[2]), "gbt": metrics(Yt[:, k], gbt),
                            "const_vel": metrics(Yt[:, k], cv[:, k])}

    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    for changed in CHANGED:
        print(f"\n{changed}: MAE on held-out changed-fly flights (TabPFN / retrained GBT), 80% coverage TabPFN")
        print(f"{'context':11s} " + " ".join(f"{t:>24s}" for t in FORWARD_TARGETS))
        for cond in ["clean"] + [f"clean+{n}" for n in N_FLIGHTS] + ["clean+all", "changed"]:
            cells = []
            for t in FORWARD_TARGETS:
                r = results[f"{changed}_{cond}_{t}"]
                cells.append(f"{r['tabpfn']['mae']:.3f}/{r['gbt']['mae']:.3f} ({r['tabpfn']['coverage_80']:.2f})")
            print(f"{cond:11s} " + " ".join(f"{c:>24s}" for c in cells))
    print("\nusage:", tabpfn_client.get_api_usage())


if __name__ == "__main__":
    main()
