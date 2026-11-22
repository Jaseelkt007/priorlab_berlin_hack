"""IMM tracker: the real-world baseline TabPFN has to beat.

    python drone/imm.py --tune           # pick the noise settings on 200 X2 (training) flights
    python drone/imm.py --drone cf2      # run on the Crazyflie test flights -> data/imm_cf2.npz

IMM (Interacting Multiple Model) is what radar trackers use for manoeuvring targets. It runs three
filters at once and blends them by how well each explains the latest radar readings:
  CV  constant velocity      (Kalman filter)
  CA  constant acceleration  (Kalman filter)
  CT  coordinated turn       (extended Kalman filter: the turn rate makes it nonlinear)
All three share one 10-number state: position (3), velocity (3), acceleration (3), turn rate.
CV and CT keep acceleration at zero; CV and CA keep the turn rate at zero.

It reads the radar track reading by reading (10 Hz), skips missing readings (predict only), and
after every reading predicts where the drone will be 0.5 ... 5 s ahead, as a mean and a covariance.
It is given the true radar noise model (0.3 m + 1% of range), which is generous to it.
"""

import argparse
import itertools
from dataclasses import asdict, dataclass

import numpy as np

from features import ADAPT_FLIGHTS, DATA, HORIZONS as TABLE_HORIZONS
from intruder import RADAR_EVERY, RADAR_NOISE

DT = RADAR_EVERY
HORIZONS = np.arange(1, 9) * 0.25  # 0.25 .. 2 s: covers the table's 1 s and the 2 s alarm window
N = 10  # state size
CV, CA, CT = 0, 1, 2
# Which state components each model actually tracks. When the models are blended, a model that does not
# track a component must not pull it toward its forced zero: it contributes the receiving model's own
# value instead. (Without this, the CA model lost its acceleration within a second: tested on synthetic tracks.)
MODELED = {CV: np.r_[0:6], CA: np.r_[0:9], CT: np.r_[0:6, 9]}


@dataclass
class Settings:
    q_cv: float = 4.0  # white-noise acceleration intensity of CV, (m/s^2)^2/s
    q_ca: float = 10.0  # white-noise jerk intensity of CA, (m/s^3)^2/s
    q_ct: float = 1.0  # white-noise acceleration intensity of CT
    q_turn: float = 0.5  # turn-rate random walk of CT, (rad/s)^2/s
    p_stay: float = 0.9  # chance per reading that the drone keeps its current mode


def transition(p_stay):
    off = (1 - p_stay) / 2
    return np.full((3, 3), off) + np.eye(3) * (p_stay - off)


def move(model, s, t):
    """Propagate states s (..., 10) by t seconds under one model (exact for constant inputs)."""
    p, v, a, w = s[..., 0:3], s[..., 3:6], s[..., 6:9], s[..., 9]
    out = np.zeros_like(s)
    if model == CV:
        out[..., 0:3] = p + v * t
        out[..., 3:6] = v
    elif model == CA:
        out[..., 0:3] = p + v * t + 0.5 * a * t**2
        out[..., 3:6] = v + a * t
        out[..., 6:9] = a
    else:  # CT: horizontal velocity turns at rate w, vertical is constant velocity
        wt = w * t
        small = np.abs(w) < 1e-4
        ws = np.where(small, 1.0, w)
        sin_w = np.where(small, t, np.sin(wt) / ws)
        cos_w = np.where(small, 0.5 * w * t**2, (1 - np.cos(wt)) / ws)
        vx, vy = v[..., 0], v[..., 1]
        out[..., 0] = p[..., 0] + sin_w * vx - cos_w * vy
        out[..., 1] = p[..., 1] + cos_w * vx + sin_w * vy
        out[..., 2] = p[..., 2] + v[..., 2] * t
        out[..., 3] = np.cos(wt) * vx - np.sin(wt) * vy
        out[..., 4] = np.sin(wt) * vx + np.cos(wt) * vy
        out[..., 5] = v[..., 2]
        out[..., 9] = w
    return out


def jacobian(model, s, t):
    """d move / d state, (B, 10, 10). Linear models: exact; CT: central differences."""
    if model != CT:
        eye = np.broadcast_to(np.eye(N), s.shape[:-1] + (N, N))
        return np.stack([move(model, eye[..., k], t) - move(model, np.zeros_like(s), t) for k in range(N)], axis=-1)
    eps = 1e-5
    cols = []
    for k in range(N):
        d = np.zeros(N)
        d[k] = eps
        cols.append((move(CT, s + d, t) - move(CT, s - d, t)) / (2 * eps))
    return np.stack(cols, axis=-1)


def noise(model, t, st):
    """Process noise added over t seconds (continuous white-noise models, so any t composes exactly)."""
    q = np.eye(N) * 1e-6  # the components a model forces to zero
    if model in (CV, CT):
        k = st.q_cv if model == CV else st.q_ct
        block = k * np.array([[t**3 / 3, t**2 / 2], [t**2 / 2, t]])
        for i in range(3):
            q[np.ix_([i, i + 3], [i, i + 3])] = block
        if model == CT:
            q[9, 9] = st.q_turn * t
    else:
        block = st.q_ca * np.array([[t**5 / 20, t**4 / 8, t**3 / 6], [t**4 / 8, t**3 / 3, t**2 / 2], [t**3 / 6, t**2 / 2, t]])
        for i in range(3):
            q[np.ix_([i, i + 3, i + 6], [i, i + 3, i + 6])] = block
    return q


def predict(model, m, P, t, st):
    F = jacobian(model, m, t)
    return move(model, m, t), F @ P @ np.swapaxes(F, -1, -2) + noise(model, t, st)


P0_INIT = np.diag([0, 0, 0, 100, 100, 100, 25, 25, 25, 0.25]).astype(float)


class Tracker:
    """The IMM for B tracks at once, one radar reading at a time (live use: B = 1)."""

    def __init__(self, B, st, horizons=HORIZONS):
        self.st, self.horizons, self.PI = st, horizons, transition(st.p_stay)
        self.m = np.zeros((B, 3, N))
        self.P = np.tile(np.eye(N), (B, 3, 1, 1))
        self.mu = np.full((B, 3), 1 / 3)
        self.started = np.zeros(B, bool)

    def step(self, z, alive=None):
        """Take one reading per track: z (B, 3), NaN if missing; alive (B,) marks tracks still running.
        Returns the blended prediction at each horizon: mean (B, H, 3), covariance (B, H, 3, 3),
        model probabilities (B, 3); NaN for tracks without a first reading yet."""
        B, H = len(z), len(self.horizons)
        alive = np.ones(B, bool) if alive is None else alive
        out_mean = np.full((B, H, 3), np.nan)
        out_cov = np.full((B, H, 3, 3), np.nan)
        out_mu = np.full((B, 3), np.nan)
        seen = alive & ~np.isnan(z[:, 0])
        sigma = RADAR_NOISE + 0.01 * np.linalg.norm(np.nan_to_num(z), axis=1)

        new = seen & ~self.started  # first reading of a flight: start every model there
        if new.any():
            self.m[new] = 0
            self.m[new, :, 0:3] = z[new][:, None]
            self.P[new] = P0_INIT
            self.P[new, :, 0:3, 0:3] = (sigma[new] ** 2)[:, None, None, None] * np.eye(3)
            self.mu[new] = 1 / 3
            self.started |= new
        step = alive & self.started & ~new
        if step.any():
            # 1. mix: each model starts from a blend of all models, weighted by switching chances
            c = self.mu[step] @ self.PI  # (b, j)
            w = self.mu[step][:, :, None] * self.PI[None] / c[:, None, :]  # (b, i, j)
            ms, Ps = self.m[step], self.P[step]
            m0 = np.zeros_like(ms)
            P0 = np.zeros_like(Ps)
            for j in range(3):
                src_m, src_P = ms.copy(), Ps.copy()
                for i in range(3):
                    gap = np.setdiff1d(MODELED[j], MODELED[i])  # components j tracks but i does not
                    if i == j or not len(gap):
                        continue
                    src_m[:, i][:, gap] = ms[:, j][:, gap]
                    src_P[:, i][:, gap, :] = 0
                    src_P[:, i][:, :, gap] = 0
                    src_P[:, i][:, gap[:, None], gap] = Ps[:, j][:, gap[:, None], gap]
                m0[:, j] = np.einsum("bi,bin->bn", w[:, :, j], src_m)
                d = src_m - m0[:, j][:, None]
                P0[:, j] = np.einsum("bi,binm->bnm", w[:, :, j], src_P + d[..., :, None] * d[..., None, :])
            # 2. predict one radar interval
            for j in range(3):
                m0[:, j], P0[:, j] = predict(j, m0[:, j], P0[:, j], DT, self.st)
            # 3. update with the reading, if there is one
            zs, ok = z[step], seen[step]
            loglik = np.zeros((len(zs), 3))
            if ok.any():
                R = (sigma[step][ok] ** 2)[:, None, None, None] * np.eye(3)
                Pm, mm = P0[ok], m0[ok]
                S = Pm[:, :, 0:3, 0:3] + R
                Sinv = np.linalg.inv(S)
                K = Pm[:, :, :, 0:3] @ Sinv
                innov = zs[ok][:, None, :] - mm[:, :, 0:3]
                mm = mm + np.einsum("bjnk,bjk->bjn", K, innov)
                Pm = Pm - K @ S @ np.swapaxes(K, -1, -2)
                m0[ok], P0[ok] = mm, 0.5 * (Pm + np.swapaxes(Pm, -1, -2))
                loglik[ok] = -0.5 * (np.einsum("bjk,bjkl,bjl->bj", innov, Sinv, innov) + np.log(np.linalg.det(S)))
            # 4. model probabilities
            lw = np.log(np.maximum(c, 1e-300)) + loglik
            lw -= lw.max(1, keepdims=True)
            mu_new = np.exp(lw)
            self.m[step], self.P[step], self.mu[step] = m0, P0, mu_new / mu_new.sum(1, keepdims=True)

        live = alive & self.started
        if live.any():
            ml, Pl, mul = self.m[live], self.P[live], self.mu[live]
            comp_m = np.zeros((live.sum(), 3, H, 3))
            comp_P = np.zeros((live.sum(), 3, H, 3, 3))
            for j in range(3):
                for hi, h in enumerate(self.horizons):
                    mj, Pj = predict(j, ml[:, j], Pl[:, j], h, self.st)
                    comp_m[:, j, hi], comp_P[:, j, hi] = mj[:, 0:3], Pj[:, 0:3, 0:3]
            mean_h = np.einsum("bj,bjhn->bhn", mul, comp_m)
            dm = comp_m - mean_h[:, None]
            cov_h = np.einsum("bj,bjhnm->bhnm", mul, comp_P + dm[..., :, None] * dm[..., None, :])
            out_mean[live], out_cov[live], out_mu[live] = mean_h, cov_h, mul
        return out_mean, out_cov, out_mu


def run(meas, lengths, st, horizons=HORIZONS):
    """Run the IMM over a batch of padded radar tracks.

    meas (B, T, 3) with NaN for missing readings; lengths (B,). Returns the blended prediction after
    every reading: mean (B, T, H, 3), covariance (B, T, H, 3, 3), and model probabilities (B, T, 3).
    Before a flight's first reading the outputs are NaN.
    """
    B, T, _ = meas.shape
    H = len(horizons)
    tracker = Tracker(B, st, horizons)
    out_mean = np.full((B, T, H, 3), np.nan, np.float32)
    out_cov = np.full((B, T, H, 3, 3), np.nan, np.float32)
    out_mu = np.full((B, T, 3), np.nan, np.float32)
    for k in range(T):
        out_mean[:, k], out_cov[:, k], out_mu[:, k] = tracker.step(meas[:, k], k < lengths)
    return out_mean, out_cov, out_mu


def load_tracks(drone, flights=None):
    raw = dict(np.load(DATA / f"intruder_{drone}.npz"))
    seeds, starts = np.unique(raw["flight"], return_index=True)
    order = np.argsort(starts)
    seeds, starts = seeds[order], starts[order]
    ends = np.append(starts[1:], len(raw["flight"]))
    keep = [i for i, s in enumerate(seeds) if raw["outcomes"][raw["flight_outcome"][raw["flight_seed"] == s][0]] != "crash"]
    if flights is not None:
        keep = keep[:flights]
    lengths = np.array([ends[i] - starts[i] for i in keep])
    meas = np.full((len(keep), lengths.max(), 3), np.nan)
    for row, i in enumerate(keep):
        meas[row, : lengths[row]] = raw["meas"][starts[i] : ends[i]]
    return seeds[keep], meas, lengths


def align(seeds, out, table):
    """Pick the IMM output (B, T, ...) for every table row, matched by flight and reading index."""
    row_of = {s: r for r, s in enumerate(seeds)}
    rows = np.array([row_of.get(f, -1) for f in table["flight"]])
    idx = np.rint(table["t"] / DT).astype(int)
    ok = rows >= 0
    picked = np.full((len(rows),) + out.shape[2:], np.nan, np.float32)
    picked[ok] = out[rows[ok], idx[ok]]
    return picked


def score(mean, table, label=""):
    """Median 3D error (m) at the table's horizons against the true future position."""
    errs = []
    for j, h in enumerate(TABLE_HORIZONS):
        hi = int(np.argmin(np.abs(HORIZONS - h)))
        truth = table["Y_world"][:, 3 * j : 3 * j + 3]
        ok = ~np.isnan(truth).any(1) & ~np.isnan(mean[:, hi]).any(1)
        errs.append(np.median(np.linalg.norm(mean[ok, hi] - truth[ok], axis=1)))
    if label:
        print(f"{label}: median error " + ", ".join(f"{h:g} s {e:.2f} m" for h, e in zip(TABLE_HORIZONS, errs)))
    return errs


def tune(drone, n_flights):
    """Coordinate search on the first n_flights flights of this drone: lowest median error 1 s ahead.
    x2: training flights. cf2: flights 0-49, the same ones TabPFN B may see (never tested on)."""
    table = dict(np.load(DATA / f"table_{drone}.npz"))
    seeds, meas, lengths = load_tracks(drone)
    first = np.unique(table["flight"])[:n_flights]  # flight seeds in order: flight 0, 1, 2, ...
    keep = np.isin(seeds, first)
    seeds, meas, lengths = seeds[keep], meas[keep], lengths[keep]
    print(f"tuning on {len(seeds)} {drone} flights", flush=True)
    sel = np.isin(table["flight"], seeds)
    table = {k: v[sel] if v.ndim and len(v) == len(sel) else v for k, v in table.items()}
    grid = {"q_cv": [0.1, 0.3, 1, 3, 10], "q_ca": [0.0001, 0.0003, 0.001, 0.003, 0.03, 0.3, 3], "q_ct": [0.001, 0.003, 0.01, 0.05, 0.3, 1, 4],
            "q_turn": [0.00003, 0.0001, 0.0003, 0.003, 0.03, 0.3], "p_stay": [0.9, 0.95, 0.98, 0.995, 0.999, 0.9997, 0.9999]}
    best = None
    # coordinate search: vary one setting at a time around the current best, two passes
    current = SETTINGS.get(drone, TUNED)  # start from this drone's last tuning
    for _ in range(2):
        for key, values in grid.items():
            for v in values:
                st = Settings(**{**asdict(current), key: v})
                mean, _, _ = run(meas, lengths, st)
                e = score(align(seeds, mean, table), table)
                if best is None or e[0] < best[0][0]:
                    best = (e, st)
                    print(f"  better: {asdict(st)} -> " + " / ".join(f"{x:.2f}" for x in e) + " m", flush=True)
            current = best[1]
    print("best settings:", asdict(best[1]))
    return best[1]


# Best settings from tuning on 150 X2 flights (--tune; EXPERIMENTS.md, experiments 7-9). Tuning for 1 s ahead
# picked the same values as for 1.5 s. X2 median error 1 s ahead: 2.16 m.
TUNED = Settings(q_cv=1.0, q_ca=0.001, q_ct=0.3, q_turn=0.003, p_stay=0.995)
# Fairness check: tuned on Crazyflie flights 0-49 (the flights TabPFN B may see). Filled in after --tune --drone cf2.
TUNED_CF2 = Settings(q_cv=0.3, q_ca=0.0001, q_ct=0.001, q_turn=0.0001, p_stay=0.9997)  # 2 searches, widened grid
SETTINGS = {"x2": TUNED, "cf2": TUNED_CF2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="grid search on the first --tune-flights flights of --drone")
    parser.add_argument("--tune-flights", type=int, default=None, help="default: 150 for x2, 50 (the context flights) for cf2")
    parser.add_argument("--drone", default="cf2")
    parser.add_argument("--settings", choices=list(SETTINGS), default="x2",
                        help="which tuning to run with: x2 (default) or cf2 (tuned on Crazyflie flights 0-49)")
    args = parser.parse_args()
    if args.tune:
        tune(args.drone, args.tune_flights or (150 if args.drone == "x2" else ADAPT_FLIGHTS))
        return
    table = dict(np.load(DATA / f"table_{args.drone}.npz"))
    seeds, meas, lengths = load_tracks(args.drone)
    settings = SETTINGS[args.settings]
    mean, cov, mu = run(meas, lengths, settings)
    m, c = align(seeds, mean, table), align(seeds, cov, table)
    score(m, table, f"IMM ({args.settings} tuning) on {args.drone}")
    out = DATA / (f"imm_{args.drone}.npz" if args.settings == "x2" else f"imm_{args.drone}_{args.settings}tuned.npz")
    np.savez_compressed(out, mean=m, cov=c, horizons=HORIZONS, settings=np.array(str(asdict(settings))))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
