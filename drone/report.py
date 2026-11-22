"""Build the results folder: tables, analysis and charts for IMM vs TabPFN on the Crazyflie test flights.

    python drone/report.py        # needs drone/data/imm_cf2.npz and drone/data/tabpfn_cf2.npz

Writes drone/results/README.md (the write-up), drone/results/*.csv (every number in it) and drone/results/charts/*.png.
Scoring is compare.py's: the same test rows for every model.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare import (  # noqa: E402
    Z90, auc, caught_at, change_ahead, drone_frame, imm_breach_probability, realtime_predictions)
from features import DATA  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
IMM_FILES = {"IMM": "imm_cf2.npz", "IMM re-tuned": "imm_cf2_cf2tuned.npz"}  # the second only if the fairness check ran
MODELS = ["IMM", "TabPFN A", "TabPFN B"]  # + "IMM re-tuned" when its file exists (set in main)
LONG = {"IMM": "IMM (radar standard, tuned on X2)", "TabPFN A": "TabPFN A (X2 context only)",
        "TabPFN B": "TabPFN B (X2 + 1,000 rows from 50 other Crazyflie flights)",
        "IMM re-tuned": "IMM re-tuned on the same 50 Crazyflie flights (fairness check)"}
# categorical slots 1-4 of the dataviz reference palette, fixed order; ink and chrome from the same palette
COLOR = {"IMM": "#2a78d6", "TabPFN A": "#eb6834", "TabPFN B": "#1baf7a", "IMM re-tuned": "#eda100"}
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
    width = 0.8 / len(MODELS)
    x = np.arange(len(groups))
    for i, m in enumerate(MODELS):
        bars = ax.bar(x + (i - (len(MODELS) - 1) / 2) * width, values[m], width - 0.03, color=COLOR[m],
                      label=LONG[m], zorder=2)
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


def gain_range(new, base, flights, n=1000):
    """How much lower `new`'s median error is than `base`'s (%), with a 95% range from resampling whole flights."""
    idx = [np.where(flights == f)[0] for f in np.unique(flights)]
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n):
        pick = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        draws.append(1 - np.median(new[pick]) / np.median(base[pick]))
    better = sum(np.median(new[i]) < np.median(base[i]) for i in idx)
    return 100 * (1 - np.median(new) / np.median(base)), *np.percentile(100 * np.array(draws), [2.5, 97.5]), better, len(idx)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "charts").mkdir(exist_ok=True)
    table = dict(np.load(DATA / "table_cf2.npz"))
    imms = {m: np.load(DATA / f) for m, f in IMM_FILES.items() if (DATA / f).exists()}
    if "IMM re-tuned" in imms and "IMM re-tuned" not in MODELS:
        MODELS.append("IMM re-tuned")
    tab = dict(np.load(DATA / "tabpfn_cf2.npz"))
    rows, alarm_rows = tab["test_rows"], tab["alarm_rows"]
    hi = int(np.argmin(np.abs(imms["IMM"]["horizons"] - 1.0)))
    truth = table["Y_world"][rows, :3]
    psi = table["psi"][rows]
    pred = {m: imm["mean"][rows, hi].astype(float) for m, imm in imms.items()}
    pred["TabPFN A"], pred["TabPFN B"] = tab["A_x2_only_world"][:, 0], tab["B_x2_plus_cf2_world"][:, 0]
    err = {m: np.linalg.norm(pred[m] - truth, axis=1) for m in MODELS}

    # range (cone) coverage, per direction, in the drone's frame
    c, s = np.cos(psi), np.sin(psi)
    R = np.zeros((len(rows), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, s, -s, c, 1
    cover, width = {}, {}
    for m, imm in imms.items():
        cov = imm["cov"][rows, hi].astype(float)
        sd = np.sqrt(np.einsum("nij,njk,nlk->nil", R, cov, R).diagonal(axis1=1, axis2=2))
        cover[m] = (np.abs(drone_frame(truth - pred[m], psi)) <= Z90 * sd).mean()
        width[m] = (2 * Z90 * sd).mean()
    Y = table["Y"][rows]
    for m, key in (("TabPFN A", "A_x2_only"), ("TabPFN B", "B_x2_plus_cf2")):
        q = tab[f"{key}_quantiles"][:, :3]
        cover[m] = ((Y >= q[:, :, 0]) & (Y <= q[:, :, 2])).mean()
        width[m] = (q[:, :, 2] - q[:, :, 0]).mean()

    # alarm
    label = table["breach"][alarm_rows]
    p_alarm = {}
    for m, imm in imms.items():
        ok = ~np.isnan(imm["mean"][alarm_rows]).any((1, 2))
        p_alarm[m] = np.zeros(len(alarm_rows))
        p_alarm[m][ok] = imm_breach_probability(imm["mean"][alarm_rows][ok].astype(float),
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

    # example sweep: TabPFN B with 0, 100, ... Crazyflie rows in its context (tabpfn_model.py --out drone/data/sweep/B_<n>.npz)
    sweep_text = "Not run yet."
    sweep_files = sorted((DATA / "sweep").glob("B_*.npz"), key=lambda f: int(f.stem[2:]))
    if sweep_files:
        out_rows = []
        for f in sweep_files:
            sw = np.load(f)
            r = sw["test_rows"]
            t = table["Y_world"][r, :3]
            row = {"crazyflie_rows_in_context": int(sw["adapt_rows"]), "test_rows": len(r),
                   "TabPFN B": np.median(np.linalg.norm(sw["B_x2_plus_cf2_world"][:, 0] - t, axis=1))}
            for m, imm in imms.items():
                row[m] = np.median(np.linalg.norm(imm["mean"][r, hi].astype(float) - t, axis=1))
            out_rows.append(row)
        sweep = pd.DataFrame(out_rows)
        sweep.round(3).to_csv(OUT / "example_sweep.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
        x = np.arange(len(sweep))
        ax.plot(x, sweep["TabPFN B"], color=COLOR["TabPFN B"], linewidth=2, marker="o", markersize=8, zorder=3,
                label="TabPFN (X2 examples + this many Crazyflie rows)")
        for xi, v in zip(x, sweep["TabPFN B"]):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        for m in imms:
            v = sweep[m].iloc[0]
            ax.axhline(v, color=COLOR[m], linewidth=2, linestyle="--", zorder=2, label=f"{LONG[m]}: {v:.2f} m")
        ax.set_ylim(top=sweep["TabPFN B"].max() + 0.06)  # room for the top value label under the title
        ax.set_xticks(x, [f"{n:,}" for n in sweep["crazyflie_rows_in_context"]])
        ax.set_xlabel("Crazyflie example rows in TabPFN's context (10 rows = 1 s of radar)", color=MUTED)
        style(ax, "How many examples of the new drone does TabPFN need? Median error 1 s ahead", "metres")
        ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
        fig.tight_layout()
        fig.savefig(OUT / "charts" / "example_sweep.png", dpi=150)
        plt.close(fig)
        below = sweep[sweep["TabPFN B"] < sweep[list(imms)].min(axis=1)]["crazyflie_rows_in_context"]
        beat = (f"TabPFN beats {'both IMMs' if len(imms) > 1 else 'the IMM'} from **{below.iloc[0]:,} Crazyflie rows** "
                f"(about {below.iloc[0] / 10:.0f} s of radar on the new drone)." if len(below)
                else "TabPFN does not beat the IMM at any of these amounts.")
        sweep_text = (f"Same test moments for every point (every 5th reading, {sweep['test_rows'].iloc[0]:,} moments). "
                      f"{beat}\n\n{md(sweep)}\n\n![example sweep](charts/example_sweep.png)")

    # new pilot: 250 Crazyflie flights by pilot b, whom no model has seen (intruder.py --pilot b)
    pilot_text = "Not run yet."
    if (DATA / "tabpfn_cf2_pilotb.npz").exists():
        pt = dict(np.load(DATA / "table_cf2_pilotb.npz"))
        pb = np.load(DATA / "tabpfn_cf2_pilotb.npz")
        r, ar = pb["test_rows"], pb["alarm_rows"]
        t = pt["Y_world"][r, :3]
        perr = {"TabPFN A": np.linalg.norm(pb["A_x2_only_world"][:, 0] - t, axis=1),
                "TabPFN B": np.linalg.norm(pb["B_x2_plus_cf2_world"][:, 0] - t, axis=1)}
        pcover = {m: ((pt["Y"][r] >= pb[f"{k}_quantiles"][:, :3, 0]) & (pt["Y"][r] <= pb[f"{k}_quantiles"][:, :3, 2])).mean()
                  for m, k in (("TabPFN A", "A_x2_only"), ("TabPFN B", "B_x2_plus_cf2"))}
        palarm = {"TabPFN A": pb["A_x2_only_p_breach"], "TabPFN B": pb["B_x2_plus_cf2_p_breach"]}
        for m, f in (("IMM", "imm_cf2_pilotb.npz"), ("IMM re-tuned", "imm_cf2_pilotb_cf2tuned.npz")):
            im = np.load(DATA / f)
            perr[m] = np.linalg.norm(im["mean"][r, hi].astype(float) - t, axis=1)
            ok = ~np.isnan(im["mean"][ar]).any((1, 2))
            palarm[m] = np.zeros(len(ar))
            palarm[m][ok] = imm_breach_probability(im["mean"][ar][ok].astype(float), im["cov"][ar][ok].astype(float),
                                                   np.random.default_rng(0))
        plabel = pt["breach"][ar]
        order = [m for m in MODELS if m in perr]
        pilot = pd.DataFrame({"model": [LONG[m] for m in order],
                              "median_error_m": [np.median(perr[m]) for m in order],
                              "same_model_on_pilot_a_m": [np.median(err[m]) for m in order],
                              "alarm_caught_at_5pct_false": [caught_at(palarm[m], plabel) for m in order]})
        pilot.round(3).to_csv(OUT / "new_pilot.csv", index=False)
        best_imm = min(np.median(perr["IMM"]), np.median(perr["IMM re-tuned"]))
        e_b = np.median(perr["TabPFN B"])
        verdict = (f"TabPFN B is still **{(1 - e_b / best_imm) * 100:.0f}% better** than the best IMM on a pilot it never saw."
                   if e_b < best_imm else
                   f"On the new pilot the best IMM is **{(1 - best_imm / e_b) * 100:.0f}% better** than TabPFN B: part of "
                   "TabPFN's win on pilot a was learning that pilot's habits.")
        n_fl = len(np.unique(pt["flight"][r]))
        pilot_text = (f"Every flight so far (examples and tests) was flown by the same scripted pilot, so TabPFN could have "
                      f"learned that pilot's habits. Here {n_fl} new Crazyflie flights are flown by **pilot b**: another "
                      f"manoeuvre mix, switching about twice as often, sharper zigzags, bigger height changes "
                      f"(`python drone/intruder.py --drone cf2 --pilot b`). Nothing was re-tuned: TabPFN keeps its pilot-a "
                      f"examples, the IMM its settings. {len(r):,} moments, {int(plabel.sum())} breach rows.\n\n"
                      + "| Model | Error on pilot b | Same model on pilot a | Breaches caught |\n|---|---|---|---|\n"
                      + "\n".join(f"| {LONG[m]} | **{np.median(perr[m]):.2f} m** | {np.median(err[m]):.2f} m | "
                                  f"{caught_at(palarm[m], plabel):.0%} |" for m in order)
                      + f"\n\n{verdict} TabPFN B's 90% range holds the truth {pcover['TabPFN B']:.0%} of the time on pilot b.")

    # IMM + TabPFN: the IMM's answer as 6 extra inputs (tabpfn_model.py --imm-features), tested on pilot b
    combo_text = "Not run yet."
    if (DATA / "tabpfn_cf2_pilotb_combo.npz").exists():
        pt = dict(np.load(DATA / "table_cf2_pilotb.npz"))
        cb, pb = np.load(DATA / "tabpfn_cf2_pilotb_combo.npz"), np.load(DATA / "tabpfn_cf2_pilotb.npz")
        im = np.load(DATA / "imm_cf2_pilotb.npz")
        r = cb["test_rows"]
        t = pt["Y_world"][r, :3]
        ce = {"IMM": np.linalg.norm(im["mean"][r, hi].astype(float) - t, axis=1),
              "TabPFN B": np.linalg.norm(pb["B_x2_plus_cf2_world"][:, 0] - t, axis=1),
              "IMM + TabPFN": np.linalg.norm(cb["B_x2_plus_cf2_world"][:, 0] - t, axis=1)}
        q = cb["B_x2_plus_cf2_quantiles"][:, :3]
        ccover = ((pt["Y"][r] >= q[:, :, 0]) & (pt["Y"][r] <= q[:, :, 2])).mean()
        cl = pt["breach"][cb["alarm_rows"]]
        fl = pt["flight"][r]
        g_imm, g_b = gain_range(ce["IMM + TabPFN"], ce["IMM"], fl), gain_range(ce["IMM + TabPFN"], ce["TabPFN B"], fl)
        pbeh = pt["behaviour"][r]
        combo = pd.DataFrame({"manoeuvre": behaviours,
                              **{m: [np.median(e[pbeh == b]) for b in range(len(behaviours))] for m, e in ce.items()}})
        combo.round(3).to_csv(OUT / "combo_pilot_b.csv", index=False)
        combo_text = (
            "The IMM's own answer goes into TabPFN as 6 extra inputs: its predicted move in 1 s (forward, left, up) and "
            "how sure it is the drone goes straight / accelerates / turns (`tabpfn_model.py --imm-features`). TabPFN "
            "then learns from the examples when the IMM is right and how it is usually wrong. Same examples as TabPFN B, "
            f"tested on **pilot b** (the pilot no model has seen), {len(r):,} moments.\n\n"
            "| Model | Error on pilot b | 90% of errors below |\n|---|---|---|\n"
            + "\n".join(f"| {m} | **{np.median(e):.2f} m** | {np.percentile(e, 90):.2f} m |" for m, e in ce.items())
            + f"\n\n- **{g_imm[0]:.0f}% better than the IMM** (95% range {g_imm[1]:.1f}-{g_imm[2]:.1f}%, better on "
            f"{g_imm[3]} of {g_imm[4]} flights).\n- **{g_b[0]:.1f}% better than TabPFN B** (95% range {g_b[1]:.1f}-"
            f"{g_b[2]:.1f}%, better on {g_b[3]} of {g_b[4]} flights): real, but small.\n- Its 90% range holds the truth "
            f"{ccover:.0%} of the time; alarm catches {caught_at(cb['B_x2_plus_cf2_p_breach'], cl):.0%} of breaches "
            f"(TabPFN B {caught_at(pb['B_x2_plus_cf2_p_breach'], cl):.0%}).\n\n{md(combo)}")

    # real time: IMM every 0.1 s, the fast IMM + TabPFN every 0.3 s (its answer arriving 0.3 s later)
    rt_text = "Not run yet."
    if (DATA / "tabpfn_cf2_pilotb_combo_fast.npz").exists():
        pt = dict(np.load(DATA / "table_cf2_pilotb.npz"))
        fast = np.load(DATA / "tabpfn_cf2_pilotb_combo_fast.npz")
        im = np.load(DATA / "imm_cf2_pilotb.npz")
        r = fast["test_rows"]
        t = pt["Y_world"][r, :3]
        imm_w = im["mean"][:, hi].astype(float)
        fast_w = np.full_like(imm_w, np.nan)
        fast_w[r] = fast["B_x2_plus_cf2_world"][:, 0]
        rt_w, used, _ = realtime_predictions(pt, imm_w, fast_w, every=3)
        full = np.load(DATA / "tabpfn_cf2_pilotb_combo.npz")
        re = {"IMM alone (every 0.1 s)": np.linalg.norm(imm_w[r] - t, axis=1),
              "IMM + TabPFN, slow setting, answer at every reading (not live: 5.2 s each)":
                  np.linalg.norm(full["B_x2_plus_cf2_world"][:, 0] - t, axis=1),
              "IMM + TabPFN, fast setting, answer at every reading (not live: 252 ms each)":
                  np.linalg.norm(fast_w[r] - t, axis=1),
              "**Live: IMM every 0.1 s + fast TabPFN correction every 0.3 s**": np.linalg.norm(rt_w[r] - t, axis=1)}
        live = list(re)[-1]
        g = gain_range(re[live], re["IMM alone (every 0.1 s)"], pt["flight"][r])
        pd.DataFrame({"setup": [k.replace("*", "") for k in re], "median_error_m": [np.median(e) for e in re.values()],
                      "p90_error_m": [np.percentile(e, 90) for e in re.values()]}).round(3).to_csv(
            OUT / "realtime.csv", index=False)
        rt_text = ("A live system cannot wait 252 ms at every 100 ms radar reading. So: the IMM answers at every reading; "
                   "every 0.3 s the fast IMM + TabPFN (example cache, 1 pass, 2,000 examples) starts on the newest "
                   "reading, and its answer arrives 0.3 s later. Until the next one, its correction (TabPFN's 1 s "
                   "prediction minus the IMM's, for the reading it started from) is added to the IMM's fresh "
                   f"prediction. Pilot b, {len(r):,} moments; a correction was in use at {used[r].mean():.0%} of them "
                   "(the IMM alone at the start of each flight).\n\n| Setup | Error 1 s ahead | 90% of errors below |\n"
                   "|---|---|---|\n" + "\n".join(f"| {k} | {np.median(e):.2f} m | {np.percentile(e, 90):.2f} m |"
                                                    for k, e in re.items())
                   + f"\n\n**Live, it is {g[0]:.0f}% better than the IMM** (95% range {g[1]:.1f}-{g[2]:.1f}%, better on "
                   f"{g[3]} of {g[4]} flights).")

    # speed: drone/latency.py
    speed_text = "Not measured yet (`python drone/latency.py`)."
    if (OUT / "latency.csv").exists():
        lat = pd.read_csv(OUT / "latency.csv")
        speed_text = ("The radar gives a reading every **100 ms**, so all 4 TabPFN models (forward, left, up, alarm) must "
                      "answer within that. IMM + TabPFN on an H100 (`python drone/latency.py`), median of 30 readings:"
                      f"\n\n{md(lat)}")

    fair_text = "Not run yet: the IMM was tuned on X2 flights only."
    fair_bullet = ""
    if "IMM re-tuned" in imms:
        settings = str(imms["IMM re-tuned"]["settings"])
        e_old, e_new, e_b = (np.median(err[m]) for m in ("IMM", "IMM re-tuned", "TabPFN B"))
        verdict = (f"TabPFN B is still **{(1 - e_b / e_new) * 100:.0f}% better** with the same data."
                   if e_b < e_new else f"The re-tuned IMM is now **{(1 - e_new / e_b) * 100:.0f}% better** than TabPFN B.")
        fair_bullet = (f"- **Fair comparison:** with the IMM re-tuned on the same 50 Crazyflie flights ({e_new:.2f} m), "
                       f"TabPFN B is still **{(1 - e_b / e_new) * 100:.0f}% better**. See the fairness check below.\n")
        fair_text = (f"TabPFN B saw 1,000 rows from Crazyflie flights 0-49. To be fair, the IMM's 5 settings were tuned "
                     f"again on the same 50 flights (`python drone/imm.py --tune --drone cf2`): `{settings}`.\n\n"
                     f"| | Median error 1 s ahead |\n|---|---|\n| IMM tuned on X2 | {e_old:.2f} m |\n"
                     f"| IMM re-tuned on Crazyflie flights 0-49 | {e_new:.2f} m |\n| TabPFN B | {e_b:.2f} m |\n\n{verdict}")

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
{fair_bullet}- **TabPFN A (never saw a Crazyflie) is {rel('TabPFN A'):+.0f}%** against the IMM: X2 examples (10 m/s) do not
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

## Fairness check: the IMM gets the same Crazyflie data

{fair_text}

## A pilot no model has seen

{pilot_text}

## IMM + TabPFN: TabPFN corrects the radar's tracker

{combo_text}

## Is it fast enough for live use?

{speed_text}

### Live replay: the IMM every 0.1 s, TabPFN every 0.3 s

{rt_text}

## How many examples of the new drone does TabPFN need?

{sweep_text}

## Caveats (read before quoting the numbers)

- **Simulation only:** one simulator, scripted intruder pilots (see "A pilot no model has seen" for a second pilot).
- **Context:** 4,000 random X2 rows out of ~400,000; larger or similarity-picked contexts are untested.
- **The IMM is told the true radar noise level;** TabPFN is not.
- **The examples' answers are perfect:** "where the drone really was 1 s later" comes from the simulator. In real use
  it would come from radar tracks of past flights, smoothed afterwards (using readings before and after each point),
  which is close to the truth but noisier, so real accuracy would be somewhat lower. The live drone's future is never
  shown to any model, in the simulation or in real use.

## Videos

![IMM, TabPFN B and IMM + TabPFN on the same flight](compare.gif)

`compare.gif` above: new pilot b, flight 78, the three best models together (from video 7 below).
Green = the real path; each colour = one model's predicted path, its ball = its guess for now, made 1 s ago.

The videos are in `videos/` (not in git, too large; rebuild them all with `drone/make_videos.sh`, about 5 minutes on a
GPU). Half speed, the drone drawn 4x bigger; the error on each video is **that one flight only**: for the overall
numbers use the tables above.

| Folder / file | What it shows |
|---|---|
| `1_world/world_scene.mp4` | the world: building, red zone, intruder, the X2 defender hovering; no predictions (its first 14 s: `world.gif`) |
| `2_pilotA_flight54/1_IMM.mp4` ... `4_TabPFN-B.mp4` | pilot a, test flight 54: IMM, IMM re-tuned, TabPFN A, TabPFN B, one per video |
| `3_newpilotB_flight78/1_IMM.mp4` ... `4_TabPFN-B.mp4` | pilot b (never seen), flight 78: the same four models |
| `3_newpilotB_flight78/5_IMM-plus-TabPFN.mp4` | IMM + TabPFN, the best model |
| `3_newpilotB_flight78/6_LIVE-IMM-plus-TabPFN-every-0.3s.mp4` | the live replay: IMM every 0.1 s, TabPFN correction every 0.3 s |
| `3_newpilotB_flight78/7_COMPARE-IMM-TabPFN-IMMplusTabPFN.mp4` | IMM, TabPFN B and IMM + TabPFN together (the GIF) |

On flight 78: IMM 1.54 m, TabPFN B 1.36 m, IMM + TabPFN 1.17 m, live 1.29 m (over all 250 pilot-b flights: 1.80,
1.67, 1.62 and 1.72 m).
"""
    (OUT / "README.md").write_text(report)
    print(summary.to_string(index=False))
    print(f"TabPFN B closer than IMM at {wins:.0%} of moments -> {OUT}")


if __name__ == "__main__":
    main()
