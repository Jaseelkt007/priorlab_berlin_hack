"""Replay the speed policies on the pick bank and report throughput vs drop rate per scenario.

Context (the line's log before each stream): the commissioning speed tests (scripts/sweep_picks.py: all products x all
speed settings under random line states); for the "new" scenario the stream's product is removed from it.
Suites: main (oracle, fixed recipe with and without a one-step margin, logistic, GBDT, TabPFN), frozen (TabPFN with a
growing vs a frozen log), explore (TabPFN without and with bounded exploration), tabpfn (TabPFN only, e.g. with
--version 3.5-fast), cost (every policy maximizes expected value per second, a drop costing --costs picks).
Usage: .venv/bin/python scripts/run_eval.py BANK.csv[,..] LOG.csv[,..] OUT_DIR [--suite main] [--targets 0.02,0.05,0.1]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import pandas as pd

from policy.policies import FixedRecipe, GBDTPolicy, LogisticPolicy, Oracle, TabPFNPolicy, replay, summarize

ap = argparse.ArgumentParser()
ap.add_argument("bank", help="bank CSV(s), comma-separated")
ap.add_argument("log", help="commissioning log CSV(s), comma-separated")
ap.add_argument("out")
ap.add_argument("--suite", default="main", choices=["main", "frozen", "explore", "tabpfn", "cost"])
ap.add_argument("--costs", default="1,3,10,30", help="cost suite: a drop costs this many picks' worth")
ap.add_argument("--targets", default="0.02,0.05,0.1")
ap.add_argument("--refit", type=int, default=5)
ap.add_argument("--version", default="3.5")
ap.add_argument("--streams", default="")
a = ap.parse_args()

bank = pd.concat([pd.read_csv(f) for f in a.bank.split(",")], ignore_index=True)
log = pd.concat([pd.read_csv(f) for f in a.log.split(",")], ignore_index=True)
targets = [float(x) for x in a.targets.split(",")]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
streams = a.streams.split(",") if a.streams else sorted(bank["stream"].unique())
tag = a.suite if a.version == "3.5" else f"{a.suite}_{a.version}"


def suite(tg):
    if a.suite == "cost":      # tg is the drop cost here: every policy maximizes the same expected value per second
        return [FixedRecipe(cost=tg), LogisticPolicy(cost=tg, refit_every=a.refit),
                GBDTPolicy(cost=tg, refit_every=a.refit),
                TabPFNPolicy(cost=tg, version=a.version, refit_every=a.refit)]
    tab = lambda **kw: TabPFNPolicy(tg, version=a.version, refit_every=a.refit, **kw)
    if a.suite == "main":
        return [FixedRecipe(tg), FixedRecipe(tg, margin=1), LogisticPolicy(tg, refit_every=a.refit),
                GBDTPolicy(tg, refit_every=a.refit), tab()]
    if a.suite == "frozen":
        return [tab(), tab(frozen=True)]
    if a.suite == "tabpfn":
        return [tab()]
    return [tab(), tab(eps=0.1, seed=1)]


runs = []
for st in streams:
    b = bank[bank["stream"] == st]
    b = b[b.groupby("idx")["alpha"].transform("nunique") == 9]          # unfinished instances are skipped
    prod, scen = b["product"].iloc[0], b["scenario"].iloc[0]
    ctx = log[log["product"] != prod] if scen == "new" else log
    levels = [float(x) for x in a.costs.split(",")] if a.suite == "cost" else targets
    pols = ([Oracle()] if a.suite in ("main", "cost") else []) + [p for tg in levels for p in suite(tg)]
    for pol in pols:
        t0 = time.time()
        r = replay(b, pol, ctx)
        r["stream"], r["scenario"] = st, scen
        r["target"], r["cost"] = getattr(pol, "target", None), getattr(pol, "cost", None)
        runs.append(r)
        s = summarize(r).iloc[0]
        print(f"{st:24s} {pol.name:40s} good/min {s.good_per_min:6.1f}  drop {s.drop_rate:6.1%}  mean alpha "
              f"{s.mean_alpha:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(runs).to_csv(out / f"runs_{tag}.csv", index=False)

R = pd.concat(runs)
R.to_csv(out / f"runs_{tag}.csv", index=False)
tab = pd.concat([summarize(g).assign(scenario=sc) for sc, g in R.groupby("scenario")] + [summarize(R).assign(scenario="all")])
tab.to_csv(out / f"summary_{tag}.csv", index=False)
print(tab.round(3).to_string(index=False))
