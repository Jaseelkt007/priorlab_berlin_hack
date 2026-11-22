"""One shift on the bar line, fast-forwarded: real-world events, a fixed speed recipe vs TabPFN choosing every pick.

Product family: flow-wrapped bars on one cell with one tool (Ø25 flat cup, regulated 50 kPa), belt 0.30 m/s,
room 22 °C / 50 % RH. 120 picks; every pick is simulated at all 9 speed settings (exact counterfactuals, full
physics: closed-chain delta, compliant cup, pneumatics, belt), so both policies run on identical items.

Events (state per pick, sim.products.LineState):
  picks   0–29   normal bars (45 g)
  picks  30–119  heavy SKU mixed in: every 3rd bar is a 60 g protein bar (mass +33 %; the commissioning log has
                 bars of 42–50 g only, so this is new to every policy). The checkweigher reports the weight.
  picks  60–89   new film roll: creased film, seal class −12 kPa (the vacuum sensor sees it at touch)
  picks  90–109  overdue cup: 750k → 1.04M cycles (vendor interval 500k–750k), leak grows
  picks 110–119  cup changed: a fresh cup
The day is fast-forwarded: cup cycles advance 150k → 330k over picks 0–89.

Policies (policy/policies.py), each sees only what a real line knows before the loaded move:
  fixed  : FixedRecipe(cost=10), one speed per product from the commissioning log (today's practice)
  tabpfn : TabPFNPolicy(cost=10, safe_step=True, refit_every=2): the log is the context; each executed pick's outcome
           is appended, and the speed map is refreshed every 2 picks (about once a second on a live line)
Context: the full commissioning log (results/logs/commission_lap.csv + commission_srv.csv, 2,700 picks).

Stages:
  bank    simulate all 120 x 9 picks          -> results/shift/bank.csv
  decide      replay both policies on the bank    -> results/shift/decisions.csv (+ risk vectors, events)
  decide_jax  the same with TabPFN-3.5-Fast answered in JAX (rolling 2,700-row context), plus the fixed 50 % recipe
              ("fastest speed with <= 5 % commissioning drops")  -> results/shift/decisions_jax.csv
Usage: .venv/bin/python showcase/line_shift.py bank|decide|decide_jax [--workers 10]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

OUT = ROOT / "results" / "shift"
N = 120
STREAM = "shift:bar"
BELT = 0.30
EVENTS = [  # (first pick, last pick, label) for the timeline
    (30, 119, "heavy SKU mixed in (60 g, every 3rd bar)"),
    (60, 89, "new film roll (creased film)"),
    (90, 109, "overdue cup (750k–1.04M cycles)"),
    (110, 119, "cup changed"),
]


def shift_tasks() -> list[dict]:
    tasks = []
    for i in range(N):
        heavy = i >= 30 and i % 3 == 0
        film = 60 <= i < 90
        if i < 90:
            cyc = 150_000 + 2_000 * i
        elif i < 110:
            cyc = 750_000 + 15_000 * (i - 90)
        else:
            cyc = 500 * (i - 110)
        tasks.append(dict(stream=STREAM, scenario="shift", idx=i, product="bar", seed=7_000_000 + i, belt_mps=BELT,
                          sku="protein bar 60 g" if heavy else "bar 45 g", film="new roll" if film else "standard",
                          state=dict(cup_cycles=cyc, room_t=22.0, room_rh=50.0,
                                     batch_mass_shift=0.33 if heavy else 0.0, batch_seal_shift=-12.0 if film else 0.0)))
    return tasks


def stage_bank(workers: int):
    from sim.picklog import run_tasks, simulate_instance
    OUT.mkdir(parents=True, exist_ok=True)
    df = run_tasks(shift_tasks(), OUT / "bank.csv", workers=workers, progress_every=10, fn=simulate_instance)
    t = df.pivot_table(index="idx", columns="alpha", values="dropped", aggfunc="first").astype(int)
    print(f"bank: {len(df)} picks ({df['idx'].nunique()} instances x {df['alpha'].nunique()} speeds)")
    print("drops by speed (normal / heavy / film / worn / new cup):")
    groups = {"normal": [i for i in range(30)], "heavy": [i for i in range(30, 120) if i % 3 == 0],
              "film": [i for i in range(60, 90) if i % 3], "worn": [i for i in range(90, 110) if i % 3],
              "new cup": [i for i in range(110, 120) if i % 3]}
    for k, ids in groups.items():
        print(f"  {k:8s}", t.loc[ids].mean().round(2).to_dict())


def jax_policy_class():
    """TabPFN-3.5-Fast with the JAX query path (jax_tabpfn, PR #1/#2): the context is fitted by PyTorch in fp32 and
    exported (export.bundle_from_classifier); every pick's 9-speed query runs in JAX fp32, which the verification
    measured identical in decisions to PyTorch fp32 (0 of 3,825 differ). The context is an exactly fixed-size
    rolling window (the commissioning log's 2,700 rows: each new pick in, the oldest held pick out, every drop kept),
    so JAX compiles once. jax is imported before torch (shared GPU)."""
    from jax_tabpfn.export import bundle_from_classifier
    from jax_tabpfn.runtime import JaxTabPFN
    import torch
    from policy.policies import TabPFNPolicy
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    class JaxTabPFNPolicy(TabPFNPolicy):
        def __init__(self, **kw):
            super().__init__(version="3.5-fast", **kw)
            self.clf = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3_5_FAST, n_estimators=1, device="cuda", fit_mode="fit_with_cache", random_state=0,
                inference_precision=torch.float32, kv_cache_precision="auto")
            self.jm = None
            self.name = self.name.replace("tabpfn-3.5-fast(", "tabpfn-3.5-fast-jax(")

        def start(self, log):
            self.n_fixed = len(log)            # the context stays exactly this size: JAX compiles once
            super().start(log)

        def _refit(self):
            """Exactly fixed-size rolling window: after the new picks are appended, the same number of the oldest
            held rows leave (drops are never evicted)."""
            excess = len(self.log) - self.n_fixed
            if excess > 0:
                held = np.where(~self.log["dropped"].astype(bool).to_numpy())[0][:excess]
                self.log = self.log.drop(self.log.index[held]).reset_index(drop=True)
            super()._refit()

        def _fit(self, X, y):
            self.clf.fit(X, y)
            b = bundle_from_classifier(self.clf)
            if self.jm is None:
                self.jm = JaxTabPFN(b, fast=False)
                self.jm.warmup(9)                  # compile (once, then the persistent cache) before the first pick
            else:
                self.jm.update_context(b)

        def _proba(self, X):
            return self.jm.predict_proba(X)[:, 1]
    return JaxTabPFNPolicy


def stage_decide(jax: bool = False):
    if jax:
        import jax_tabpfn.runtime  # noqa: F401  (jax before torch)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from policy.policies import X_COLS, FixedRecipe, TabPFNPolicy, replay
    bank = pd.read_csv(OUT / "bank.csv")
    assert (bank.groupby("idx")["alpha"].nunique() == 9).all() and bank["idx"].nunique() == N, "incomplete bank"
    log = pd.concat([pd.read_csv(ROOT / "results/logs/commission_lap.csv"),
                     pd.read_csv(ROOT / "results/logs/commission_srv.csv")], ignore_index=True)
    if jax:
        pols = (("fixed", FixedRecipe(cost=10)), ("fixed50", FixedRecipe(target=0.05)),
                ("tabpfn_jax", jax_policy_class()(cost=10, safe_step=True, refit_every=2)))
        out_csv = OUT / "decisions_jax.csv"
    else:
        pols = (("fixed", FixedRecipe(cost=10)), ("tabpfn", TabPFNPolicy(cost=10, safe_step=True, refit_every=2)))
        out_csv = OUT / "decisions.csv"
    out = []
    for name, pol in pols:
        risk: dict[int, list] = {}
        nctx: dict[int, int] = {}
        lat: dict[int, float] = {}
        if name.startswith("tabpfn"):
            orig = pol.choose

            def choose(cand, pol=pol, orig=orig):
                i = int(cand["idx"].iloc[0])
                t0 = time.perf_counter()     # measured per-pick query time of this run
                p = pol._proba(cand[X_COLS].to_numpy(float)) if pol.const is None else np.full(len(cand), pol.const)
                lat[i] = (time.perf_counter() - t0) * 1e3
                risk[i], nctx[i] = [float(x) for x in p], len(pol.log)
                return orig(cand)
            pol.choose = choose
        runs = replay(bank, pol, log)
        runs["policy_key"] = name
        runs["risk"] = runs["idx"].map(lambda i: json.dumps(risk.get(int(i), [])))
        runs["n_ctx"] = runs["idx"].map(lambda i: nctx.get(int(i), len(log)))
        runs["tabpfn_ms"] = runs["idx"].map(lambda i: lat.get(int(i), np.nan))
        out.append(runs)
        good = (~runs["dropped"].astype(bool)).sum()
        ms = runs["tabpfn_ms"].dropna()
        print(f"{name:10s}: drops {int(runs['dropped'].sum())}/{len(runs)}, good picks/min "
              f"{60 * good / runs['cycle_s'].sum():.1f}, mean speed {runs['alpha'].mean():.2f}"
              + (f", query ms median {ms.median():.2f} p95 {ms.quantile(0.95):.2f} max {ms.max():.2f}" if len(ms) else ""))
        for a, b, lab in EVENTS:
            seg = runs[(runs["idx"] >= a) & (runs["idx"] <= b)]
            print(f"    {lab:42s} speed {seg['alpha'].mean():.2f}  drops {int(seg['dropped'].sum())}")
    dec = pd.concat(out, ignore_index=True)
    keep = ["policy_key", "policy", "idx", "sku", "film", "cup_cycles", "weight_g", "belt_mps", "alpha", "seed",
            "seal_kpa", "evac_ms", "dropped", "place_offset_mm", "cycle_s", "a_peak", "risk", "n_ctx", "tabpfn_ms",
            "prod_t", "room_rh"]
    dec[keep].to_csv(out_csv, index=False)
    (OUT / "events.json").write_text(json.dumps([dict(first=a, last=b, label=l) for a, b, l in EVENTS], indent=1))
    print("wrote", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["bank", "decide", "decide_jax"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    stage_bank(a.workers) if a.stage == "bank" else stage_decide(jax=a.stage == "decide_jax")
