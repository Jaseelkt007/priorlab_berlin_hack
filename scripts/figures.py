"""Result figures (PNG) for the README and slides, from the evaluation outputs.
Usage: .venv/bin/python scripts/figures.py [--eval data/eval] [--lc data/eval/learning_curve.csv] [--log a.csv,b.csv]
"""
import argparse
import re
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


def family(name: str) -> str:
    if name.startswith("tabpfn"):
        return "tabpfn"
    if name.startswith("fixed"):
        return "fixed+1" if "margin" in name else "fixed"
    return name.split("(")[0]


def frontier(summary: pd.DataFrame, out: Path, rule: str = "Each line: risk targets 2, 5, 10 %."):
    sc = [s for s in ["wear", "humid", "new", "batch", "all"] if s in set(summary["scenario"])]
    titles = {"wear": "Cup wears out (300k → 1.05M cycles)", "humid": "Humid afternoon (RH 40 → 80 %)",
              "new": "New product, never logged", "batch": "New film batch mid-run", "all": "All streams"}
    fig, axes = plt.subplots(1, len(sc), figsize=(4.2 * len(sc), 3.9), sharey=False)
    for ax, s in zip(np.atleast_1d(axes), sc):
        d = summary[summary["scenario"] == s].copy()
        d["fam"] = d["policy"].map(family)
        d["level"] = d["policy"].str.extract(r"(?:target|cost)=([0-9.]+)").astype(float)
        for fam, (col, lab) in SERIES.items():
            g = d[d["fam"] == fam].sort_values(["drop_rate", "good_per_min"])
            if g.empty:
                continue
            ax.errorbar(100 * g["drop_rate"], g["good_per_min"],
                        xerr=[100 * (g["drop_rate"] - g["dr_lo"]), 100 * (g["dr_hi"] - g["drop_rate"])],
                        yerr=[g["good_per_min"] - g["gpm_lo"], g["gpm_hi"] - g["good_per_min"]],
                        color=col, lw=2, marker="o", ms=6, elinewidth=0.8, capsize=0, label=lab,
                        mec=SURF, mew=1.5, zorder=3 if fam == "tabpfn" else 2)
        o = d[d["fam"] == "oracle"]
        if len(o):
            ax.scatter(100 * o["drop_rate"], o["good_per_min"], marker="*", s=140, color=INK, zorder=4,
                       label="Oracle (knows the outcome)")
        ax.set_title(titles[s], fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("drop rate (%)")
    np.atleast_1d(axes)[0].set_ylabel("good picks per minute")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Good picks per minute vs drop rate. {rule} Whiskers: 95 % bootstrap CI.", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    fig.savefig(out, dpi=160)
    plt.close(fig)


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/eval")
    ap.add_argument("--lc", default="data/eval/learning_curve.csv")
    ap.add_argument("--log", default="data/commission_lap.csv,data/commission_srv.csv")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Path(a.eval) / "summary_main.csv"
    if s.exists():
        frontier(pd.read_csv(s), out / "frontier_targets.png")
    s = Path(a.eval) / "summary_cost.csv"
    if s.exists():
        frontier(pd.read_csv(s), out / "frontier.png",
                 "Each line: one policy as the cost of a drop goes from 1 to 100 picks.")
    if Path(a.lc).exists():
        learning(pd.read_csv(a.lc), out / "learning_curve.png")
    logs = [f for f in a.log.split(",") if Path(f).exists()]
    if logs:
        heatmap(pd.concat([pd.read_csv(f) for f in logs]), out / "drop_heatmap.png")
    print("figures in", out)
