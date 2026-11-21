"""Shared loading, table building, splitting and scoring for the TabPFN fly experiments.

Data layout: docs/data/collection-plan.md. All experiments split by flight pair (an original
flight and its mirror stay together) with the same seed, so held-out flights are identical
across experiments and passes.
"""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLIGHTS = ROOT / "data/flights"
STEPS_PER_BEAT = 23
DT_BEAT = STEPS_PER_BEAT * 2e-4  # ~4.6 ms
FORWARD_TARGETS = ["forward_mm", "lateral_mm", "up_mm", "dyaw_deg"]


def load(name, wingbeat=True, keys=None):
    """Load data/flights/<name>[_wingbeat].npz (e.g. name='pass1_clean') and meta.json."""
    path = FLIGHTS / f"{name}{'_wingbeat' if wingbeat else ''}.npz"
    npz = np.load(path)
    data = {k: npz[k] for k in (keys or npz.files)}
    return data, json.loads((FLIGHTS / "meta.json").read_text())


def tabpfn_client_ready():
    """Import tabpfn_client with the key from .env (never printed)."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    return tabpfn_client


def yaw_of(quat):
    w, x, y, z = quat.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def to_heading(dxy, yaw):
    """Rotate world-frame xy displacement into the frame of heading `yaw`."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1]])


def world_yaw_rate(quat, ang_vel_body):
    """z component of the body angular velocity expressed in the world frame."""
    w, x, y, z = quat.T
    r20, r21, r22 = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    return r20 * ang_vel_body[:, 0] + r21 * ang_vel_body[:, 1] + r22 * ang_vel_body[:, 2]


def build_forward(d, meta, horizons):
    """Forward-model table on a wingbeat file: features at beat k, motion from beat k to k+h.

    Features: the 62 body-sensor observations (reference inputs excluded), the applied action,
    the pattern-generator wing targets (wingbeat phase), beat frequency and the previous
    beat's motion. Targets are in the current heading frame: forward/lateral/up (mm), Δyaw (deg).
    Returns X, feature names, and per horizon: rows, Y, and the two physics baselines.
    """
    ids, st = d["ids"], d["state"]
    cols = meta["columns"]
    obs_keep = [i for i, c in enumerate(cols["obs"]) if not c.startswith("ref_")]
    sc = {c: i for i, c in enumerate(cols["state"])}
    com = st[:, [sc["com_x"], sc["com_y"], sc["com_z"]]]
    quat = st[:, [sc["quat_w"], sc["quat_x"], sc["quat_y"], sc["quat_z"]]]
    yaw = yaw_of(quat)
    wpg = st[:, [sc[c] for c in cols["state"] if c.startswith("wpg_target_")]]

    flight, beat = ids[:, 0], ids[:, 4] // STEPS_PER_BEAT
    prev = np.full((len(ids), 4), np.nan, np.float32)
    has_prev = np.r_[False, (flight[1:] == flight[:-1]) & (beat[1:] == beat[:-1] + 1)]
    i = np.flatnonzero(has_prev)
    prev[i, :2] = to_heading(com[i, :2] - com[i - 1, :2], yaw[i]) * 10
    prev[i, 2] = (com[i, 2] - com[i - 1, 2]) * 10
    prev[i, 3] = np.degrees(wrap(yaw[i] - yaw[i - 1]))

    X = np.column_stack([d["obs"][:, obs_keep], d["action_applied"], wpg, st[:, sc["beat_freq"]], prev])
    names = ([cols["obs"][j] for j in obs_keep] + [f"action_{a}" for a in cols["action_applied"]]
             + [c for c in cols["state"] if c.startswith("wpg_target_")] + ["beat_freq"]
             + ["prev_forward_mm", "prev_lateral_mm", "prev_up_mm", "prev_dyaw_deg"])

    per_h = {}
    for h in horizons:
        j = np.arange(len(ids)) + h
        ok = np.zeros(len(ids), bool)
        valid = j < len(ids)
        ok[valid] = (flight[j[valid]] == flight[valid]) & (beat[j[valid]] == beat[valid] + h)
        ok &= has_prev
        r = np.flatnonzero(ok)
        f = r + h
        dxy = to_heading(com[f, :2] - com[r, :2], yaw[r]) * 10
        Y = np.column_stack([dxy, (com[f, 2] - com[r, 2]) * 10, np.degrees(wrap(yaw[f] - yaw[r]))])
        v = to_heading(st[r][:, [sc["lin_vel_x"], sc["lin_vel_y"]]], yaw[r]) * 10 * h * DT_BEAT
        ang = st[r][:, [sc["ang_vel_x"], sc["ang_vel_y"], sc["ang_vel_z"]]]
        const_vel = np.column_stack([v, st[r, sc["lin_vel_z"]] * 10 * h * DT_BEAT,
                                     np.degrees(world_yaw_rate(quat[r], ang) * h * DT_BEAT)])
        per_h[h] = dict(rows=r, Y=Y.astype(np.float32), const_vel=const_vel.astype(np.float32),
                        last_beat=prev[r] * h)
    return X.astype(np.float32), names, per_h


def split_pairs(ids, test_frac=0.2, seed=0):
    """Boolean test mask over rows, holding out ~20% of flight pairs per flight type."""
    rng = np.random.default_rng(seed)
    test = []
    for t in (0, 1):  # evasion, saccade
        pairs = np.unique(ids[ids[:, 2] == t, 1])
        test += list(rng.choice(pairs, size=max(1, round(test_frac * len(pairs))), replace=False))
    return np.isin(ids[:, 1], test), sorted(int(p) for p in test)


def metrics(y, pred, lo=None, hi=None):
    out = {"mae": float(np.mean(np.abs(pred - y))),
           "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))}
    if lo is not None:
        out["coverage_80"] = float(np.mean((y >= lo) & (y <= hi)))
        out["width_80"] = float(np.mean(hi - lo))
    return out
