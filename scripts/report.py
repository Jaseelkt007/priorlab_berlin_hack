"""Markdown tables from the evaluation runs (scripts/run_eval.py) for docs/research/evaluation.md and the README.

Net value per minute at drop cost c: 60 · (held − c · dropped) / Σ cycle time. Uplift intervals are paired bootstrap
intervals: the same resampled picks of a stream for both policies (they ran on the same items).
Usage: .venv/bin/python scripts/report.py [data/eval]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EV = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval")
FAM = {"oracle": "oracle", "fixed": "fixed recipe", "logistic": "logistic", "gbdt": "boosted trees", "tabpfn": "TabPFN"}


def fam(name):
    return next(v for k, v in FAM.items() if name.startswith(k))


def net(g, c):
    held = (~g["dropped"].astype(bool)).to_numpy(float)
    return 60 * (held - c * (1 - held)).sum() / g["cycle_s"].sum()


def paired_uplift(a, b, c, n_boot=2000, seed=0):
    """Net value per minute of a minus b (same items, aligned by stream and idx), with a 95 % bootstrap interval that
    resamples items within each stream."""
    rng = np.random.default_rng(seed)
    a, b = a.sort_values(["stream", "idx"]).reset_index(drop=True), b.sort_values(["stream", "idx"]).reset_index(drop=True)
    assert (a[["stream", "idx"]].to_numpy() == b[["stream", "idx"]].to_numpy()).all()
    ha, hb = (~a["dropped"].astype(bool)).to_numpy(float), (~b["dropped"].astype(bool)).to_numpy(float)
    va, vb = ha - c * (1 - ha), hb - c * (1 - hb)
    ta, tb = a["cycle_s"].to_numpy(float), b["cycle_s"].to_numpy(float)
    groups = [np.flatnonzero(a["stream"].to_numpy() == s) for s in a["stream"].unique()]
    est = 60 * (va.sum() / ta.sum() - vb.sum() / tb.sum())
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([rng.choice(g, len(g)) for g in groups])
        bs.append(60 * (va[ix].sum() / ta[ix].sum() - vb[ix].sum() / tb[ix].sum()))
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cost_tables(R):
    R = R.assign(fam=R["policy"].map(fam))
    out = []
    scen = ["wear", "humid", "new", "batch", "all"]
    for c in sorted(R["cost"].dropna().unique()):
        out.append(f"\n**Drop cost {c:g} picks** (good picks/min · drop rate · net value/min)\n")
        out.append("| scenario | " + " | ".join(FAM.values()) + " | TabPFN − fixed, net/min (95 % CI) |")
        out.append("|---|" + "---|" * (len(FAM) + 1))
        for s in scen:
            g = R if s == "all" else R[R["scenario"] == s]
            cells = []
            for f in FAM.values():
                x = g[(g["fam"] == f) & ((g["cost"] == c) | (f == "oracle"))]
                held = (~x["dropped"].astype(bool)).mean()
                cells.append(f"{60 * (~x['dropped'].astype(bool)).sum() / x['cycle_s'].sum():.1f} · {1 - held:.1%} · "
                             f"{net(x, c):.1f}")
            t = g[(g["fam"] == "TabPFN") & (g["cost"] == c)]
            f0 = g[(g["fam"] == "fixed recipe") & (g["cost"] == c)]
            e, lo, hi = paired_uplift(t, f0, c)
            out.append(f"| {s} | " + " | ".join(cells) + f" | {e:+.1f} ({lo:+.1f}, {hi:+.1f}) |")
    return "\n".join(out)


def simple(R, label):
    out = [f"\n**{label}** (good picks/min · drop rate, per scenario)\n", "| policy | wear | humid | new | batch | all |",
           "|---|---|---|---|---|---|"]
    for p, g in R.groupby("policy", sort=False):
        cells = []
        for s in ["wear", "humid", "new", "batch", "all"]:
            x = g if s == "all" else g[g["scenario"] == s]
            held = (~x["dropped"].astype(bool))
            cells.append(f"{60 * held.sum() / x['cycle_s'].sum():.1f} · {1 - held.mean():.1%}")
        out.append(f"| {p} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    for name, fn in [("runs_cost.csv", lambda R: cost_tables(R)), ("runs_main.csv", lambda R: simple(R, "Risk-target rule")),
                     ("runs_frozen.csv", lambda R: simple(R, "Growing vs frozen log (TabPFN)")),
                     ("runs_explore.csv", lambda R: simple(R, "Bounded exploration (TabPFN)")),
                     ("runs_tabpfn_3.5-fast.csv", lambda R: simple(R, "TabPFN-3.5-Fast, risk-target rule"))]:
        f = EV / name
        if f.exists():
            print(fn(pd.read_csv(f)))
