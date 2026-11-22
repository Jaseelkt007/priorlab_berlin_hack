"""TabPFN predicts where the intruder will be 1, 1.5 and 2 s ahead, and whether it breaches within 2 s.

    python drone/tabpfn_model.py              # needs TABPFN_TOKEN in .env (repo root)
    python drone/tabpfn_model.py --context 2000 --every 10   # smaller and faster

No training: TabPFN is shown example rows (its "context") and predicts the test rows from them.

Two versions, both scored on the same Crazyflie test rows as the IMM:
  A "X2 only":           context = rows from big-drone (X2) flights only. The small drone is new to it.
  B "X2 + a few cf2":    the same, plus rows from ADAPT_FLIGHTS Crazyflie flights that are not in the
                         test set. This is "adapting to a new drone without retraining".
The first ADAPT_FLIGHTS Crazyflie flights are kept out of the test for both versions and the IMM.

Positions are learned as movement in the drone's view (forward, left, up), one TabPFN regressor per
number (TabPFN has no multi-output), with the 5% / 50% / 95% quantiles; scored in world x, y, z.
The alarm is a TabPFN classifier on `breach`.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from features import ADAPT_FLIGHTS, DATA, FEATURES, HORIZONS, rotate, to_world

QUANTILES = [0.05, 0.5, 0.95]
FAST = {"fit_mode": "fit_with_cache", "n_estimators": 1}  # the real-time setting (latency.py, experiment 16)
SEED = 0


def load(drone):
    return dict(np.load(DATA / f"table_{drone}.npz"))


def imm_columns(name, table):
    """The IMM's own answer as 6 extra inputs (experiment 15, "IMM + TabPFN"): its predicted move in 1 s in the
    drone's frame (forward, left, up) and how sure it is the drone goes straight / accelerates / turns.
    Needs drone/data/imm_<name>.npz from `imm.py --drone <name>` (X2 tuning). NaN where the IMM has no answer yet;
    TabPFN handles missing values."""
    imm = np.load(DATA / f"imm_{name}.npz")
    hi = int(np.argmin(np.abs(imm["horizons"] - 1.0)))
    d = imm["mean"][:, hi].astype(float) - table["pos"]
    return np.column_stack([rotate(d[:, :2], table["psi"]), d[:, 2], imm["mu"].astype(float)])


def complete(table):
    """Rows where every target is known (the flight lasts long enough)."""
    return ~np.isnan(table["Y"]).any(1)


def sample(rng, idx, n):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def make_regressor(**options):
    """options: TabPFN settings, e.g. fit_mode="fit_with_cache", n_estimators=1 (latency.py measures them)."""
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(device="cuda" if _cuda() else "cpu", random_state=SEED, **options)


def make_classifier(**options):
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(device="cuda" if _cuda() else "cpu", random_state=SEED, **options)


def _cuda():
    import torch

    return torch.cuda.is_available()


def test_rows(cf2, every=5):
    """Crazyflie test rows (flights after the first ADAPT_FLIGHTS, every Nth reading).
    Returns (position rows: future known, alarm rows: all)."""
    flights = np.unique(cf2["flight"])
    is_test = ~np.isin(cf2["flight"], flights[:ADAPT_FLIGHTS])
    return np.where(is_test & complete(cf2))[0][::every], np.where(is_test)[0][::every]


def free(model):
    """Release a fitted model's GPU memory: the 4 GB card cannot hold several TabPFN models at once."""
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batches(X, batch):
    return [X[i : i + batch] for i in range(0, len(X), batch)]


def run_version(name, ctx_X, ctx_Y, ctx_breach_X, ctx_breach, test_X, alarm_X, batch, options=None):
    """Fit one regressor per target and one classifier; return drone-frame quantiles (n, 9, 3) and P(breach)."""
    n_targets = ctx_Y.shape[1]
    q = np.full((len(test_X), n_targets, len(QUANTILES)), np.nan)
    for t in range(n_targets):
        t0 = time.time()
        model = make_regressor(**(options or {}))
        model.fit(ctx_X, ctx_Y[:, t])
        parts = [model.predict(b, output_type="quantiles", quantiles=QUANTILES) for b in batches(test_X, batch)]
        q[:, t] = np.concatenate([np.column_stack(p) for p in parts])
        free(model)
        print(f"  {name}: target {t + 1}/{n_targets} done in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    clf = make_classifier(**(options or {}))
    clf.fit(ctx_breach_X, ctx_breach)
    p_breach = np.concatenate([clf.predict_proba(b)[:, 1] for b in batches(alarm_X, batch)])
    free(clf)
    print(f"  {name}: alarm done in {time.time() - t0:.0f} s", flush=True)
    return q, p_breach


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=4000, help="X2 rows shown to TabPFN")
    parser.add_argument("--adapt-rows", type=int, default=1000, help="Crazyflie rows added in version B")
    parser.add_argument("--every", type=int, default=5, help="test on every Nth reading (5 = one per 0.5 s)")
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--versions", default="A,B", help="which versions to run: A, B or A,B")
    parser.add_argument("--out", type=Path, default=DATA / "tabpfn_cf2.npz")
    parser.add_argument("--fast", action="store_true", help="the real-time setting: example cache, 1 pass (use with --context 1000)")
    parser.add_argument("--imm-features", action="store_true", help="add the IMM's answer as 6 extra inputs (IMM + TabPFN)")
    parser.add_argument("--test", default="cf2", help="test table: cf2 (flights 50-299) or cf2_pilotb (all flights, a new pilot)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not os.environ.get("TABPFN_TOKEN"):
        raise SystemExit("TABPFN_TOKEN missing: put it in .env at the repo root (see README).")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")

    rng = np.random.default_rng(SEED)
    x2, cf2 = load("x2"), load("cf2")
    cf2_flights = np.unique(cf2["flight"])
    adapt_flights = cf2_flights[:ADAPT_FLIGHTS]

    # Positions: only rows whose future is known. Alarm: all rows, because a flight ends when the drone
    # breaches, so the rows just before a breach have no "2 s later" position but are the alarm cases.
    is_adapt = np.isin(cf2["flight"], adapt_flights)
    ctx = sample(rng, np.where(complete(x2))[0], args.context)
    ctx_alarm = sample(rng, np.arange(len(x2["X"])), args.context)
    adapt = sample(rng, np.where(complete(cf2) & is_adapt)[0], args.adapt_rows)
    adapt_alarm = sample(rng, np.where(is_adapt)[0], args.adapt_rows)
    tt = cf2 if args.test == "cf2" else load(args.test)  # the examples always come from cf2 flights 0-49
    if args.imm_features:
        for name, table in {"x2": x2, "cf2": cf2, args.test: tt}.items():
            if table["X"].shape[1] == len(FEATURES):  # tt may be cf2 itself: add the columns once
                table["X"] = np.column_stack([table["X"], imm_columns(name, table)])
    if args.test == "cf2":
        test, alarm_test = test_rows(cf2, args.every)
    else:
        test, alarm_test = np.where(complete(tt))[0][:: args.every], np.arange(len(tt["X"]))[:: args.every]
    print(f"context: {len(ctx)} X2 rows (+{len(adapt)} Crazyflie rows in version B); test: {len(test)} position rows, "
          f"{len(alarm_test)} alarm rows ({tt['breach'][alarm_test].sum()} breaches), test set {args.test}")

    def both(a, b):
        return np.concatenate([a, b])

    versions = {
        "A_x2_only": (x2["X"][ctx], x2["Y"][ctx], x2["X"][ctx_alarm], x2["breach"][ctx_alarm]),
        "B_x2_plus_cf2": (both(x2["X"][ctx], cf2["X"][adapt]), both(x2["Y"][ctx], cf2["Y"][adapt]),
                          both(x2["X"][ctx_alarm], cf2["X"][adapt_alarm]),
                          both(x2["breach"][ctx_alarm], cf2["breach"][adapt_alarm])),
    }
    out = {"test_rows": test, "alarm_rows": alarm_test, "horizons": np.array(HORIZONS),
           "quantiles": np.array(QUANTILES)}
    wanted = {v.strip() for v in args.versions.split(",")}
    for name, (cX, cY, aX, aB) in versions.items():
        if name[0] not in wanted:
            continue
        t0 = time.time()
        q, p = run_version(name, cX, cY, aX, aB, tt["X"][test], tt["X"][alarm_test], args.batch, FAST if args.fast else None)
        world_median = np.stack([to_world(q[:, 3 * j : 3 * j + 3, 1], tt["pos"][test], tt["psi"][test])
                                 for j in range(len(HORIZONS))], axis=1)  # (n, H, 3)
        out[f"{name}_quantiles"] = q
        out[f"{name}_world"] = world_median
        out[f"{name}_p_breach"] = p
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    path = args.out
    out["adapt_rows"] = np.array(len(adapt))
    out["test_set"] = np.array(args.test)
    out["imm_features"] = np.array(args.imm_features)
    out["fast"] = np.array(args.fast)
    np.savez_compressed(path, **out)
    print(f"-> {path}; score with: python drone/compare.py")


if __name__ == "__main__":
    main()
