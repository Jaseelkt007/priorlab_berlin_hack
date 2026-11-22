"""Toy pick-log generator for the delta suction line (Python port of docs/explainer/delta-pick-line.html).

Illustrative physics: holding force = seal x cup area x efficiency; failure when shear, lift
or peel load exceeds it. Used for TabPFN latency/accuracy benchmarks until the MuJoCo delta
cell produces real logs. Not a validated model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G, AMAX, SETPOINT, ETA = 9.81, 150.0, 65.0, 0.85
SPEEDS = np.round(np.arange(0.30, 1.0001, 0.05), 2)

PRODUCTS = {
    "biscuit": dict(sku="BIS-0412", w=(12, 1.2), dims=(60, 45, 8), cup=15, seal=(36, 5), porous=1.0, mu=0.60, h=4, kappa=1.0),
    "pouch": dict(sku="COF-2210", w=(60, 3), dims=(110, 70, 22), cup=25, seal=(44, 8), porous=0.3, mu=0.55, h=9, kappa=0.6),
    "cheese": dict(sku="CHE-0125", w=(125, 4), dims=(80, 60, 35), cup=30, seal=(56, 4), porous=0.0, mu=0.45, h=17, kappa=1.0),
}

# columns TabPFN sees (known before the move) + the action
FEATURES = ["weight_g", "len_mm", "wid_mm", "hgt_mm", "pick_off_mm", "seal_kpa", "evac_ms",
            "humidity_pct", "cup_cycles", "cup_d_mm", "speed"]


def hold_force(seal_kpa, cup_mm):
    return seal_kpa * 1000 * np.pi * (cup_mm / 2000) ** 2 * ETA


def sample_items(rng: np.random.Generator, prod: str, n: int, hum: float = 40.0, wear: float = 0.0, cup_cycles0: int = 0) -> pd.DataFrame:
    p = PRODUCTS[prod]
    hum_f = 1 - p["porous"] * np.clip((hum - 45) / 100, 0, 1) * 1.6
    off = np.abs(rng.normal(0, 2.5, n))
    seal = (p["seal"][0] + rng.normal(0, p["seal"][1], n)) * hum_f * (1 - 0.45 * wear) * (1 - 0.02 * off)
    seal = np.clip(seal, 3, SETPOINT)
    dims = np.array(p["dims"]) * (1 + rng.normal(0, 0.02, (n, 3)))
    return pd.DataFrame({
        "product": prod,
        "weight_g": p["w"][0] + rng.normal(0, p["w"][1], n),
        "len_mm": dims[:, 0], "wid_mm": dims[:, 1], "hgt_mm": dims[:, 2],
        "pick_off_mm": off, "seal_kpa": seal, "evac_ms": 14 + 520 / seal + np.abs(rng.normal(0, 3, n)),
        "humidity_pct": np.full(n, hum), "cup_cycles": cup_cycles0 + np.arange(n), "cup_d_mm": np.full(n, p["cup"]),
        # hidden physics, never given to the model
        "_mu": p["mu"] * np.exp(rng.normal(0, 0.06, n)), "_h": p["h"] * (1 + rng.normal(0, 0.08, n)),
        "_kappa": np.full(n, p["kappa"]), "_noise": np.exp(rng.normal(0, 0.04, n)),
    })


def utilization(df: pd.DataFrame, speed) -> np.ndarray:
    m = df["weight_g"].to_numpy() / 1000
    a = np.asarray(speed) * AMAX
    F = hold_force(df["seal_kpa"].to_numpy(), df["cup_d_mm"].to_numpy())
    r = df["cup_d_mm"].to_numpy() / 2000
    shear = m * a / (df["_mu"].to_numpy() * F)
    lift = m * (G + 0.5 * a) / F
    peel = m * a * (df["_h"].to_numpy() / 1000) / (F * r * df["_kappa"].to_numpy())
    return np.maximum.reduce([shear, lift, peel]) * df["_noise"].to_numpy()


def run_picks(df: pd.DataFrame, speed, rng: np.random.Generator) -> pd.DataFrame:
    """Attach the action and the measured outputs to a batch of items."""
    out = df.copy()
    out["speed"] = np.broadcast_to(np.asarray(speed, dtype=float), len(df)).copy()
    U = utilization(df, out["speed"].to_numpy())
    dropped = U > 1
    dip = np.where(dropped, 1.0, 0.5 * np.maximum(0, U - 0.55) / 0.45)
    out["dropped"] = dropped.astype(int)
    out["vac_min_kpa"] = np.where(dropped, np.maximum(0.5, 2.5 + rng.normal(0, 1, len(df))),
                                  np.maximum(0.5, df["seal_kpa"] * (1 - dip) + rng.normal(0, 0.5, len(df))))
    out["_U"] = U
    return out


def make_log(rng: np.random.Generator, n: int, prod: str = "pouch", hum: float = 50.0, wear: float = 0.0,
             speed_range=(0.3, 1.0)) -> pd.DataFrame:
    """A calibration-run style log: varied speeds so the boundary is visible."""
    items = sample_items(rng, prod, n, hum, wear)
    speeds = rng.uniform(*speed_range, n)
    return run_picks(items, speeds, rng)
