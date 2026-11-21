"""Turn recorded radar tracks into model tables: one row per radar reading.

    python drone/features.py                 # data/intruder_x2.npz  -> data/table_x2.npz
    python drone/features.py --drone cf2     # data/intruder_cf2.npz -> data/table_cf2.npz
    python drone/features.py --preview 2000  # first 2000 rows of data/table_x2.npz -> data/table_x2_preview.csv

Inputs use only the radar measurements a real system would have (never the true state, never
the flight clock). Targets use the true future position. Spec: drone/README.md, "Inputs and outputs".

Per reading at time t:
1. position p and velocity v at t: straight-line fit to the last 1.5 s of radar positions;
   acceleration a: quadratic fit to the last 2 s. (Tested on 200 flights: a quadratic over only
   1 s gave 3x the velocity error and a fake +6 m/s2 forward acceleration from noise.)
2. heading psi = direction of v; horizontal vectors are rotated into the drone frame (forward, left);
3. FEATURES below; TARGETS = true position after 1, 3, 5 s minus p, in the same frame;
4. breach_5s = the flight breaches the zone within 5 s.
Rows whose fit fails are dropped; history features (1 s / 2 s earlier) and targets past the end of
the flight are NaN. Crashed flights are dropped.
"""

import argparse
from pathlib import Path

import numpy as np

RADAR_HZ = 10
VEL_WINDOW = 15  # readings = 1.5 s, straight-line fit for position and velocity
ACC_WINDOW = 20  # readings = 2 s, quadratic fit for acceleration
MISSING_WINDOW = 10  # n_missing counts gaps in the last 1 s
# A track needs this many valid readings before we trust it; with only 3-4 readings at the start of a
# flight the fits gave nonsense (e.g. -69 m/s2). A real tracker also waits before confirming a track.
MIN_VALID_VEL = 8  # of 15
MIN_VALID_ACC = 12  # of 20
HORIZONS = (1, 3, 5)  # s

FEATURES = [
    "range", "height", "zone_fwd", "zone_left",
    "speed", "climb", "acc_fwd", "acc_left", "acc_up", "turn_rate",
    "speed_1s", "turn_rate_1s", "climb_1s", "turn_rate_2s",
    "n_missing", "fit_error",
]
TARGETS = [f"d{h}_{axis}" for h in HORIZONS for axis in ("fwd", "left", "up")]

DATA = Path(__file__).resolve().parent.parent / "data"


def rotate(xy, psi):
    """Rotate horizontal vectors (n, 2) by -psi: world frame -> drone heading frame (forward, left)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]], axis=1)


def fit_window(meas, window, degree, min_valid=None):
    """Least-squares polynomial fit over each sliding window of one flight's radar track (n, 3).

    NaN readings are skipped. Returns coefficients (n, degree + 1, 3) for the polynomial in time
    relative to the newest reading (so coefficient 0 is the value now, 1 the slope), the RMS
    residual, and the number of valid readings. Windows with fewer than `min_valid` valid readings
    (default degree + 2) get NaN.
    """
    n = len(meas)
    tau = (np.arange(window) - (window - 1)) / RADAR_HZ  # e.g. -1.4 .. 0 s, newest last
    basis = np.stack([tau**d for d in range(degree + 1)], axis=1)
    padded = np.concatenate([np.full((window - 1, 3), np.nan), meas])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)  # (n, W, 3)
    valid = ~np.isnan(windows[:, :, 0])
    w = valid.astype(float)
    normal = np.einsum("nk,ki,kj->nij", w, basis, basis)
    rhs = np.einsum("nk,ki,nkd->nid", w, basis, np.nan_to_num(windows))
    ok = valid.sum(1) >= (min_valid or degree + 2)
    coef = np.full((n, degree + 1, 3), np.nan)
    coef[ok] = np.linalg.solve(normal[ok], rhs[ok])
    sq = np.where(valid[:, :, None], (np.einsum("ki,nid->nkd", basis, coef) - windows) ** 2, 0.0)
    rms = np.full(n, np.nan)
    rms[ok] = np.sqrt(sq[ok].sum((1, 2)) / (3 * valid[ok].sum(1)))
    return coef, rms, valid.sum(1)


def flight_rows(meas, true, breach_at):
    """Features, targets and labels for one flight. `breach_at` = index of the breach reading or None."""
    n = len(meas)
    line, fit_error, _ = fit_window(meas, VEL_WINDOW, 1, MIN_VALID_VEL)
    curve, _, _ = fit_window(meas, ACC_WINDOW, 2, MIN_VALID_ACC)
    _, _, n_valid = fit_window(meas, MISSING_WINDOW, 0)
    pos, vel, acc = line[:, 0], line[:, 1], 2 * curve[:, 2]
    n_missing = MISSING_WINDOW - n_valid
    psi = np.arctan2(vel[:, 1], vel[:, 0])
    speed = np.hypot(vel[:, 0], vel[:, 1])
    climb = vel[:, 2]
    acc_h = rotate(acc[:, :2], psi)
    turn_rate = (vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / np.maximum(speed, 0.3) ** 2
    zone = rotate(-pos[:, :2], psi)

    def lag(x, seconds):
        k = seconds * RADAR_HZ
        return np.concatenate([np.full(k, np.nan), x[:-k]]) if k < n else np.full(n, np.nan)

    X = np.column_stack([
        np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], zone[:, 0], zone[:, 1],
        speed, climb, acc_h[:, 0], acc_h[:, 1], acc[:, 2], turn_rate,
        lag(speed, 1), lag(turn_rate, 1), lag(climb, 1), lag(turn_rate, 2),
        n_missing, fit_error,
    ])

    Y = np.full((n, len(TARGETS)), np.nan)
    for j, h in enumerate(HORIZONS):
        k = h * RADAR_HZ
        if k >= n:
            continue
        delta = true[k:] - pos[:-k]
        Y[:-k, 3 * j : 3 * j + 2] = rotate(delta[:, :2], psi[:-k])
        Y[:-k, 3 * j + 2] = delta[:, 2]

    steps_to_breach = (breach_at - np.arange(n)) if breach_at is not None else np.full(n, np.inf)
    breach_5s = (steps_to_breach <= 5 * RADAR_HZ).astype(int)
    keep = ~np.isnan(pos[:, 0]) & ~np.isnan(acc[:, 0])
    return X[keep], Y[keep], breach_5s[keep], pos[keep], psi[keep], keep


def build(drone):
    raw = dict(np.load(DATA / f"intruder_{drone}.npz"))  # dict: read each compressed array once
    outcomes = list(raw["outcomes"])
    outcome_of = dict(zip(raw["flight_seed"], raw["flight_outcome"]))
    intent_of = dict(zip(raw["flight_seed"], raw["flight_intent"]))
    parts = {k: [] for k in ["X", "Y", "breach_5s", "pos", "psi", "true", "flight", "t", "behaviour", "intent"]}
    dropped = 0
    seeds, starts = np.unique(raw["flight"], return_index=True)  # each flight's readings are contiguous
    ends = np.append(starts[1:], len(raw["flight"]))[np.argsort(starts)]
    starts = np.sort(starts)
    seeds = raw["flight"][starts]
    for seed, a, b in zip(seeds, starts, ends):
        outcome = outcomes[outcome_of[seed]]
        if outcome == "crash":
            dropped += 1
            continue
        m = slice(a, b)
        meas, true = raw["meas"][m], raw["true"][m]
        breach_at = len(meas) - 1 if outcome == "breach" else None
        X, Y, b, pos, psi, keep = flight_rows(meas, true, breach_at)
        for k, v in [("X", X), ("Y", Y), ("breach_5s", b), ("pos", pos), ("psi", psi), ("true", true[keep]),
                     ("flight", np.full(len(X), seed)), ("t", raw["t"][m][keep]),
                     ("behaviour", raw["behaviour"][m][keep]), ("intent", np.full(len(X), intent_of[seed]))]:
            parts[k].append(v)
    table = {k: np.concatenate(v) for k, v in parts.items()}
    out = DATA / f"table_{drone}.npz"
    np.savez_compressed(out, **table, features=np.array(FEATURES), targets=np.array(TARGETS),
                        behaviours=raw["behaviours"], intents=raw["intents"], drone=np.array(drone))

    X, Y = table["X"], table["Y"]
    print(f"{drone}: {len(X):,} rows from {len(np.unique(table['flight']))} flights ({dropped} crashed flights dropped) -> {out}")
    print(f"  rows with all targets: {(~np.isnan(Y).any(1)).sum():,}; alarm positives (breach within 5 s): {table['breach_5s'].mean():.1%}")
    err = np.linalg.norm(table["pos"] - table["true"], axis=1)
    print(f"  radar fit vs true position: median error {np.median(err):.2f} m")
    for name, col in zip(FEATURES, X.T):
        print(f"  {name:13s} NaN {np.isnan(col).mean():5.1%}  median {np.nanmedian(col):8.2f}  [5%, 95%] = [{np.nanpercentile(col, 5):8.2f}, {np.nanpercentile(col, 95):8.2f}]")


def preview(drone, rows):
    """Write the first `rows` rows of a built table as a CSV with named columns, to look at by eye."""
    import pandas as pd

    z = np.load(DATA / f"table_{drone}.npz")
    head = slice(0, rows)
    df = pd.DataFrame({
        "flight": z["flight"][head], "t": z["t"][head],
        "intent": z["intents"][z["intent"][head]], "behaviour": z["behaviours"][z["behaviour"][head]],
    })
    df = pd.concat([
        df,
        pd.DataFrame(z["X"][head], columns=[f"X_{n}" for n in z["features"]]),
        pd.DataFrame(z["Y"][head], columns=[f"Y_{n}" for n in z["targets"]]),
        pd.DataFrame({"Y_breach_5s": z["breach_5s"][head]}),
    ], axis=1)
    out = DATA / f"table_{drone}_preview.csv"
    df.to_csv(out, index=False, float_format="%.3f")
    print(f"{len(df)} rows, {len(df.columns)} columns -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone", default="x2")
    parser.add_argument("--preview", type=int, metavar="ROWS", help="only export a CSV preview of an existing table")
    args = parser.parse_args()
    if args.preview:
        preview(args.drone, args.preview)
    else:
        build(args.drone)


if __name__ == "__main__":
    main()
