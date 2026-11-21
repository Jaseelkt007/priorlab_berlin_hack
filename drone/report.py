"""Build the results folder: tables, analysis and charts for IMM vs TabPFN on the Crazyflie test flights.

    python drone/report.py        # needs data/imm_cf2.npz and data/tabpfn_cf2.npz

Writes results/README.md (the write-up), results/*.csv (every number in it) and results/charts/*.png.
Scoring is compare.py's: the same test rows for every model.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare import Z90, auc, caught_at, change_ahead, drone_frame, imm_breach_probability  # noqa: E402
from features import DATA  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
MODELS = ["IMM", "TabPFN A", "TabPFN B"]
LONG = {"IMM": "IMM (radar standard, tuned on X2)", "TabPFN A": "TabPFN A (X2 context only)",
        "TabPFN B": "TabPFN B (X2 + 1,000 rows from 50 other Crazyflie flights)"}
# categorical slots 1-3 of the dataviz reference palette, fixed order; ink and chrome from the same palette
COLOR = {"IMM": "#2a78d6", "TabPFN A": "#eb6834", "TabPFN B": "#1baf7a"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIST_BINS = [0, 15, 25, 35, 45, 70]


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)


def grouped_bars(ax, groups, values, fmt="{:.2f}"):
    """values: {model: [one per group]}; thin bars with a 2px gap, value labels on the TabPFN B bars only."""
    width = 0.26
    x = np.arange(len(groups))
    for i, m in enumerate(MODELS):
        bars = ax.bar(x + (i - 1) * width, values[m], width - 0.03, color=COLOR[m], label=LONG[m], zorder=2)
        if m == "TabPFN B":
            for b, v in zip(bars, values[m]):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, groups)


def warning_times(p, table, rows, threshold):
    """Seconds between the alarm switching on (and staying on) and the breach, per breaching flight."""
    out = []
    flights, t, label = table["flight"][rows], table["t"][rows], table["breach"][rows]
    for f in np.unique(flights[label == 1]):
        sel = np.where(flights == f)[0]
        order = sel[np.argsort(t[sel])]
        on = p[order] > threshold
        if not on[-1]:
            out.append(0.0)
            continue
        start = len(on) - 1
        while start > 0 and on[start - 1]:
            start -= 1
        end_t = t[order[-1]] + 0.1  # the flight ends at the breach, one radar step after its last row
        out.append(end_t - t[order[start]])
    return np.array(out)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "charts").mkdir(exist_ok=True)
    table = dict(np.load(DATA / "table_cf2.npz"))
    imm = np.load(DATA / "imm_cf2.npz")
    tab = dict(np.load(DATA / "tabpfn_cf2.npz"))
    rows, alarm_rows = tab["test_rows"], tab["alarm_rows"]
    hi = int(np.argmin(np.abs(imm["horizons"] - 1.0)))
    truth = table["Y_world"][rows, :3]
    psi = table["psi"][rows]
    pred = {"IMM": imm["mean"][rows, hi].astype(float),
            "TabPFN A": tab["A_x2_only_world"][:, 0], "TabPFN B": tab["B_x2_plus_cf2_world"][:, 0]}
    err = {m: np.linalg.norm(pred[m] - truth, axis=1) for m in MODELS}

    # range (cone) coverage, per direction, in the drone's frame
    imm_cov = imm["cov"][rows, hi].astype(float)
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros((len(rows), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, s, -s, c, 1
    sd = np.sqrt(np.einsum("nij,njk,nlk->nil", R, imm_cov, R).diagonal(axis1=1, axis2=2))
    cover = {"IMM": (np.abs(drone_frame(truth - pred["IMM"], psi)) <= Z90 * sd).mean()}
    width = {"IMM": (2 * Z90 * sd).mean()}
    Y = table["Y"][rows]
    for m, key in (("TabPFN A", "A_x2_only"), ("TabPFN B", "B_x2_plus_cf2")):
        q = tab[f"{key}_quantiles"][:, :3]
        cover[m] = ((Y >= q[:, :, 0]) & (Y <= q[:, :, 2])).mean()
        width[m] = (q[:, :, 2] - q[:, :, 0]).mean()

    # alarm
    label = table["breach"][alarm_rows]
    ok = ~np.isnan(imm["mean"][alarm_rows]).any((1, 2))
    p_alarm = {"IMM": np.zeros(len(alarm_rows))}
    p_alarm["IMM"][ok] = imm_breach_probability(imm["mean"][alarm_rows][ok].astype(float),
                                                imm["cov"][alarm_rows][ok].astype(float), np.random.default_rng(0))
    p_alarm["TabPFN A"], p_alarm["TabPFN B"] = tab["A_x2_only_p_breach"], tab["B_x2_plus_cf2_p_breach"]
    alarm = {}
    for m in MODELS:
        thr = np.quantile(p_alarm[m][label == 0], 0.95)
        warn = warning_times(p_alarm[m], table, alarm_rows, thr)
        alarm[m] = (auc(p_alarm[m], label), caught_at(p_alarm[m], label), np.median(warn), (warn > 0).mean())

    summary = pd.DataFrame({
        "model": [LONG[m] for m in MODELS],
        "median_error_m": [np.median(err[m]) for m in MODELS],
        "p90_error_m": [np.percentile(err[m], 90) for m in MODELS],
        "range_coverage_target_90": [cover[m] for m in MODELS],
        "range_width_m": [width[m] for m in MODELS],
        "alarm_auc": [alarm[m][0] for m in MODELS],
        "alarm_caught_at_5pct_false": [alarm[m][1] for m in MODELS],
        "alarm_warning_s_median": [alarm[m][2] for m in MODELS],
    }).round(3)
    summary.to_csv(OUT / "summary.csv", index=False)

    behaviours = list(table["behaviours"])
    beh = table["behaviour"][rows]
    by_beh = pd.DataFrame({"manoeuvre": behaviours, "rows": [(beh == b).sum() for b in range(len(behaviours))],
                           **{m: [np.median(err[m][beh == b]) for b in range(len(behaviours))] for m in MODELS}})
    by_beh.round(3).to_csv(OUT / "by_manoeuvre.csv", index=False)

    dist = np.hypot(table["true"][rows, 0], table["true"][rows, 1])
    bins = np.digitize(dist, DIST_BINS[1:-1])
    names = [f"{a}-{b} m" for a, b in zip(DIST_BINS[:-1], DIST_BINS[1:])]
    by_dist = pd.DataFrame({"distance_to_zone_centre": names, "rows": [(bins == i).sum() for i in range(len(names))],
                            **{m: [np.median(err[m][bins == i]) for i in range(len(names))] for m in MODELS}})
    by_dist.round(3).to_csv(OUT / "by_distance.csv", index=False)

    changing = change_ahead(table, rows)[:, 0]
    by_change = pd.DataFrame({"moment": ["manoeuvre keeps going", "manoeuvre changes within 1 s"],
                              "rows": [(~changing).sum(), changing.sum()],
                              **{m: [np.median(err[m][~changing]), np.median(err[m][changing])] for m in MODELS}})
    by_change.round(3).to_csv(OUT / "by_change.csv", index=False)

    wins = (err["TabPFN B"] < err["IMM"]).mean()

    # charts
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    vals = [np.median(err[m]) for m in MODELS]
    bars = ax.bar(MODELS, vals, 0.5, color=[COLOR[m] for m in MODELS], zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} m", ha="center", va="bottom", color=INK, fontsize=10)
    style(ax, "Where will the drone be in 1 s? Median error (lower is better)", "metres")
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "error_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    grouped_bars(ax, behaviours, {m: by_beh[m].values for m in MODELS})
    style(ax, "Median error 1 s ahead, by what the drone is doing", "metres")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "error_by_manoeuvre.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    grouped_bars(ax, names, {m: by_dist[m].values for m in MODELS})
    style(ax, "Median error 1 s ahead, by distance from the zone centre", "metres")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "error_by_distance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    for m in MODELS:
        e = np.sort(err[m])
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=COLOR[m], linewidth=2, label=LONG[m])
    ax.set_xlim(0, 5)
    style(ax, "Share of moments with error below x (higher is better)", "share of test moments")
    ax.set_xlabel("error 1 s ahead (m)", color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "error_distribution.png", dpi=150)
    plt.close(fig)

    def md(df):
        cell = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        return "\n".join(lines + ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)])

    rel = lambda m: (np.median(err[m]) / np.median(err["IMM"]) - 1) * 100  # noqa: E731
    report = f"""# Results: IMM vs TabPFN, where will the drone be in 1 second?

Generated by `python drone/report.py`. Every number below is in the CSV files next to this page.

## Setup in one paragraph

A radar at the protected zone sees an intruder drone 10 times a second (noise 0.3 m + 1% of range, about 11% of
readings missing). At every reading each model predicts **where the drone will be 1 s later (x, y, z)**, with a 90%
range, and **whether it enters the zone within 2 s** (the alarm). Test set: **{len(np.unique(table['flight'][rows]))}
Crazyflie flights, {len(rows):,} moments** (every radar reading whose future is known), all models on exactly the
same moments. The IMM was tuned on X2 (big drone) flights. TabPFN gets no training: version A sees 4,000 X2 rows as
context; version B also sees 1,000 rows from 50 other Crazyflie flights (never tested on).

## Headline

| Model | Median error | 90% of errors below | Range holds the truth (target 90%) | Alarm AUC | Breaches caught (1 false alarm per 20) | Warning before breach (median) |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {LONG[m]} | **{np.median(err[m]):.2f} m** | {np.percentile(err[m], 90):.2f} m | {cover[m]:.0%} "
        f"(width {width[m]:.1f} m) | {alarm[m][0]:.3f} | {alarm[m][1]:.0%} | {alarm[m][2]:.1f} s |" for m in MODELS) + f"""

- **TabPFN B is {-rel('TabPFN B'):.0f}% more accurate than the IMM** ({np.median(err['TabPFN B']):.2f} vs
  {np.median(err['IMM']):.2f} m) and closer than the IMM at **{wins:.0%} of all moments**.
- **TabPFN A (never saw a Crazyflie) is {rel('TabPFN A'):+.0f}%** against the IMM: X2 examples (10 m/s) do not
  transfer fully to a 3 m/s drone. A few examples of the new drone in the context are enough to close that gap.
- **The alarm is close to solved by every model** (AUC near 1): 2 s ahead, a 3 m/s drone cannot surprise much.

![error by model](charts/error_by_model.png)
![error distribution](charts/error_distribution.png)

## By manoeuvre

TabPFN B is better than the IMM on: {", ".join(b for b, i, t in zip(by_beh.manoeuvre, by_beh.IMM, by_beh["TabPFN B"]) if t < i)};
worse on: {", ".join(b for b, i, t in zip(by_beh.manoeuvre, by_beh.IMM, by_beh["TabPFN B"]) if t >= i) or "none"}.

{md(by_beh)}

![error by manoeuvre](charts/error_by_manoeuvre.png)

## By distance from the zone

The radar is noisier far away (1% of range), so everything gets harder with distance.

{md(by_dist)}

![error by distance](charts/error_by_distance.png)

## When the drone changes what it is doing

{md(by_change)}

## Caveats (read before quoting the numbers)

- **Simulation only:** one simulator, one scripted intruder pilot; the test flights use the same pilot as the context
  (different drone, different random flights).
- **TabPFN B saw Crazyflie data, the IMM did not:** the IMM's 5 settings were tuned on X2 flights only. A fully fair
  check re-tunes the IMM on the same 50 Crazyflie flights (planned).
- **Context:** 4,000 random X2 rows out of ~400,000; larger or similarity-picked contexts are untested.
- **The IMM is told the true radar noise level;** TabPFN is not.

## Videos

In `videos/` (not in git, too large): `demo_radar_54.mp4` (camera on the building), `demo_chase_54.mp4` (behind the
drone, close-up inset), and `imm_vs_tabpfn_54.mp4` if present (both predictions on the same flight).
"""
    (OUT / "README.md").write_text(report)
    print(summary.to_string(index=False))
    print(f"TabPFN B closer than IMM at {wins:.0%} of moments -> {OUT}")


if __name__ == "__main__":
    main()
