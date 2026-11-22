"""Experiment A: where will the fly be in 1, 3 and 10 wingbeats? (TabPFN vs baselines)

    uv run python experiments/exp_a_forward.py              # prices first, then runs
    uv run python experiments/exp_a_forward.py --dry-run    # price only, no API calls

Data: pass1_clean wingbeat file. Table and targets: flydata.build_forward (heading-frame
displacement and Δyaw; body sensors + action + wing phase + previous beat; no reference
inputs). Split by flight pair. Every API result is cached in results/exp_a/cache.
"""

import argparse
import json
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from flydata import FORWARD_TARGETS, ROOT, build_forward, load, metrics, split_pairs, tabpfn_client_ready

OUT = ROOT / "results/exp_a"
HORIZONS = (1, 3, 10)
QUANTILES = [0.1, 0.5, 0.9]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="price the API calls, do not run them")
    parser.add_argument("--max-tokens", type=float, default=5e6, help="refuse to run above this estimate")
    args = parser.parse_args()
    tabpfn_client = tabpfn_client_ready()

    d, meta = load("pass1_clean")
    X, names, per_h = build_forward(d, meta, HORIZONS)
    is_test_row, test_pairs = split_pairs(d["ids"])
    print(f"{X.shape[1]} features; held-out pairs ({len(test_pairs)}): {test_pairs}")

    plan, total = [], 0.0
    for h, p in per_h.items():
        tr, te = ~is_test_row[p["rows"]], is_test_row[p["rows"]]
        est = tabpfn_client.estimate_cost(X[p["rows"]][tr], X[p["rows"]][te])
        plan.append((h, int(tr.sum()), int(te.sum()), float(est.estimated_cost)))
        total += est.estimated_cost * len(FORWARD_TARGETS)
    for h, ntr, nte, c in plan:
        print(f"  horizon {h:>2} beats: {ntr} context rows, {nte} test rows, ~{c:,.0f} tokens per target")
    print(f"estimated total: {total:,.0f} tokens for {len(plan) * len(FORWARD_TARGETS)} calls")
    if args.dry_run:
        return
    if total > args.max_tokens:
        raise SystemExit(f"estimate {total:,.0f} exceeds --max-tokens {args.max_tokens:,.0f}")

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    results, preds = {}, {}
    for h, p in per_h.items():
        Xh, Y = X[p["rows"]], p["Y"]
        tr, te = ~is_test_row[p["rows"]], is_test_row[p["rows"]]
        for k, target in enumerate(FORWARD_TARGETS):
            key = f"h{h}_{target}"
            path = cache / f"{key}.npy"
            if path.exists():
                q = np.load(path)
            else:
                t0 = time.time()
                model = tabpfn_client.TabPFNRegressor(random_state=0)
                model.fit(Xh[tr], Y[tr, k])
                q = np.stack(model.predict(Xh[te], output_type="quantiles", quantiles=QUANTILES))
                np.save(path, q)
                print(f"  {key}: TabPFN fit+predict {time.time() - t0:.1f} s", flush=True)

            y = Y[te, k]
            gbt = lambda **kw: HistGradientBoostingRegressor(random_state=0, **kw).fit(Xh[tr], Y[tr, k]).predict(Xh[te])  # noqa: E731
            hgb, hgb_lo, hgb_hi = gbt(), gbt(loss="quantile", quantile=0.1), gbt(loss="quantile", quantile=0.9)
            results[key] = {
                "tabpfn": metrics(y, q[1], q[0], q[2]),
                "gbt": metrics(y, hgb, hgb_lo, hgb_hi),
                "const_vel": metrics(y, p["const_vel"][te, k]),
                "last_beat": metrics(y, p["last_beat"][te, k]),
                "n_context": int(tr.sum()), "n_test": int(te.sum()),
            }
            preds[key] = {"y": y, "tabpfn_q": q, "gbt": hgb, "test_ids": d["ids"][p["rows"]][te]}

    np.savez_compressed(OUT / "predictions.npz", **{f"{k}_{n}": v for k, dct in preds.items() for n, v in dct.items()})
    (OUT / "results.json").write_text(json.dumps({"test_pairs": test_pairs, "features": names,
                                                  "results": results}, indent=1))
    print(f"\n{'target':22s} {'TabPFN MAE':>10s} {'GBT MAE':>8s} {'ConstVel':>8s} {'LastBeat':>8s}"
          f" {'TabPFN R2':>9s} {'cov80 TabPFN/GBT':>17s}")
    for key, r in results.items():
        print(f"{key:22s} {r['tabpfn']['mae']:10.3f} {r['gbt']['mae']:8.3f} {r['const_vel']['mae']:8.3f}"
              f" {r['last_beat']['mae']:8.3f} {r['tabpfn']['r2']:9.3f}"
              f" {r['tabpfn']['coverage_80']:8.2f}/{r['gbt']['coverage_80']:.2f}")
    print("\nusage:", tabpfn_client.get_api_usage())


if __name__ == "__main__":
    main()
