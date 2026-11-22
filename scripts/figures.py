"""Result figures (PNG) for the README and slides: learning curve, net value vs drop cost, drop-rate heatmap.
Usage: .venv/bin/python scripts/figures.py [--eval data/eval] [--lc data/eval/learning_curve.csv] [--log a.csv,b.csv]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# reference palette (dataviz skill, light mode): categorical slots in fixed order; text and surface tokens
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"tabpfn": ("#2a78d6", "TabPFN (log in context)"), "fixed": ("#eb6834", "Fixed recipe (today)"),
          "fixed+1": ("#1baf7a", "Fixed recipe, 1 step margin"), "logistic": ("#eda100", "Logistic regression"),
          "gbdt": ("#e87ba4", "Gradient-boosted trees")}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False})


def learning(lc: pd.DataFrame, out: Path, target: str = "0.05"):
    cols = [("log_loss", "log-loss (lower is better)"), ("ece", "calibration error (lower is better)"),
            (f"gpm@{target}", f"good picks/min at a {float(target):.0%} risk target"),
            (f"drop@{target}", f"realised drop rate at a {float(target):.0%} target")]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.0 * len(cols), 3.6))
    names = {m: ("tabpfn" if m.startswith("tabpfn") else m) for m in lc["model"].unique()}
    for ax, (c, lab) in zip(axes, cols):
        for m, fam in names.items():
            g = lc[lc["model"] == m].groupby("n")[c].agg(["mean", "min", "max"]).reset_index()
            col, nm = SERIES[fam]
            ax.plot(g["n"], g["mean"], color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1.2, label=nm)
            ax.fill_between(g["n"], g["min"], g["max"], color=col, alpha=0.15, lw=0)
        ax.set_xscale("log")
        ns = sorted(lc["n"].unique())
        ax.set_xticks(ns, [f"{n / 1000:g}k" if n >= 1000 else str(n) for n in ns], fontsize=9)
        ax.minorticks_off()
        if c.startswith("drop@"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("logged picks in context")
        ax.set_title(lab, fontsize=10.5, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def heatmap(log: pd.DataFrame, out: Path):
    order = ["blister", "praline", "digestive", "bar", "pouch85", "cheese_wedge"]
    t = log.groupby(["product", "alpha"])["dropped"].mean().unstack().reindex(order)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.imshow(t.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#2a78d6", "#0d3b75"]),
              vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(t.shape[1]), [f"{a:.0%}" for a in t.columns])
    ax.set_yticks(range(t.shape[0]), [p.replace("_", " ") for p in t.index])
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color="white" if v >= 0.45 else INK)
    ax.grid(False)
    ax.set_xlabel("speed setting (ABB AccSet: share of 150 m/s² and the jerk cap)")
    ax.set_title(f"Simulated drop rate, {len(log):,} picks under random line states", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _net(g, c):
    h = (~g["dropped"].astype(bool)).to_numpy(float)
    return 60 * (h - c * (1 - h)).sum() / g["cycle_s"].sum()


def value_vs_cost(panels, out: Path):
    """Net value per minute (held picks minus cost x drops, per minute) against the cost of a drop, all streams pooled.
    panels: [(title, [(label, colour, dashed, runs DataFrame with a cost column), ...]), ...]."""
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.2), sharey=True)
    for ax, (title, series) in zip(np.atleast_1d(axes), panels):
        for lab, col, dash, R in series:
            cs = sorted(R["cost"].dropna().unique())
            v = [_net(R[R["cost"] == c], c) for c in cs]
            ax.plot(cs, v, color=col, lw=2.2, ls="--" if dash else "-", marker="o", ms=5, mec=SURF, mew=1.2, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
        ax.minorticks_off()
        ax.set_ylim(0, 170)
        ax.set_xlabel("cost of one drop (in picks)")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.text(0.99, 0.02, "lines leaving the bottom go below zero", transform=ax.transAxes, ha="right",
                fontsize=8, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel("net value per minute\n(good picks − cost × drops)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--lc", default="data/eval/learning_curve.csv")
    ap.add_argument("--log", default="data/commission_lap.csv,data/commission_srv.csv")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if Path(a.lc).exists():
        learning(pd.read_csv(a.lc), out / "learning_curve.png")
    small, full = Path("data/eval_small"), Path(a.eval)
    if (small / "runs_cost_safe.csv").exists() and (full / "runs_cost_safe.csv").exists():
        def fam_runs(R, prefix, extra=None):
            x = R[R["policy"].str.startswith(prefix)]
            return x if extra is None else x[x["policy"].str.contains(extra, regex=False)]
        S, F = pd.read_csv(small / "runs_cost_safe.csv"), pd.read_csv(full / "runs_cost_safe.csv")
        M = pd.read_csv(small / "runs_margin.csv") if (small / "runs_margin.csv").exists() else None
        sm = [("TabPFN (log in context)", SERIES["tabpfn"][0], False, fam_runs(S, "tabpfn")),
              ("Recipe tuned on the log", SERIES["fixed"][0], False, fam_runs(S, "fixed"))]
        if M is not None:
            sm += [("Recipe, 1 step slower", SERIES["fixed+1"][0], True, M[M["policy"].str.contains("margin 1")]),
                   ("Recipe, 2 steps slower", "#6250d6", True, M[M["policy"].str.contains("margin 2")])]
        sm += [("Logistic regression", SERIES["logistic"][0], False, fam_runs(S, "logistic")),
               ("Gradient-boosted trees", SERIES["gbdt"][0], False, fam_runs(S, "gbdt"))]
        fl = [("TabPFN (log in context)", SERIES["tabpfn"][0], False, fam_runs(F, "tabpfn")),
              ("Recipe tuned on the log", SERIES["fixed"][0], False, fam_runs(F, "fixed")),
              ("Logistic regression", SERIES["logistic"][0], False, fam_runs(F, "logistic")),
              ("Gradient-boosted trees", SERIES["gbdt"][0], False, fam_runs(F, "gbdt"))]
        value_vs_cost([("Small commissioning log: 10 test picks per speed and product (540)", sm),
                       ("Large commissioning log: 50 per speed and product (2,700)", fl)], out / "value_vs_cost.png")
    logs = [f for f in a.log.split(",") if Path(f).exists()]
    if logs:
        heatmap(pd.concat([pd.read_csv(f) for f in logs]), out / "drop_heatmap.png")
    print("figures in", out)
