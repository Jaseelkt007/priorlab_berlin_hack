"""Score the IMM and TabPFN on the same Crazyflie test rows.

    python drone/compare.py

Uses data/imm_cf2.npz (python drone/imm.py --drone cf2) and, if present, data/tabpfn_cf2.npz
(python drone/tabpfn_model.py). The test rows are the ones tabpfn_model.py chose; without that file,
the same rule is applied here (tabpfn_model.test_rows). The alarm is scored on all test rows, including
the last seconds before a breach, whose 2 s-later position does not exist.

Scores:
- error: median distance between predicted and real position, at 1 / 1.5 / 2 s;
- error at a change: the same, only for rows where the drone switches manoeuvre before that time;
- cone: how often the real position is inside the 5%-95% range, per direction (should be ~90%);
- alarm: AUC (1 = perfect, 0.5 = guessing), and how many breaches are caught at 1 false alarm per 20 safe rows.
"""

import numpy as np

from features import DATA, HORIZONS, RADAR_HZ
from tabpfn_model import test_rows

Z90 = 1.645  # 5%-95% of a normal distribution is mean ± 1.645 sigma


def drone_frame(v, psi):
    """World vectors (n, 3) -> drone view (forward, left, up)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=1)


def change_ahead(table, rows):
    """For each row and horizon: does the true manoeuvre change between now and then?"""
    key = {(f, k): i for i, (f, k) in enumerate(zip(table["flight"], np.rint(table["t"] * RADAR_HZ).astype(int)))}
    out = np.zeros((len(rows), len(HORIZONS)), bool)
    for n, i in enumerate(rows):
        f, k0 = table["flight"][i], int(round(table["t"][i] * RADAR_HZ))
        for j, h in enumerate(HORIZONS):
            ahead = [key.get((f, k)) for k in range(k0 + 1, k0 + round(h * RADAR_HZ) + 1)]
            out[n, j] = any(a is not None and table["behaviour"][a] != table["behaviour"][i] for a in ahead)
    return out


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def caught_at(score, label, false_rate=0.05):
    threshold = np.quantile(score[label == 0], 1 - false_rate)
    return (score[label == 1] > threshold).mean()


def imm_breach_probability(mean, cov, rng, samples=200):
    """P(inside the zone at some predicted time up to 2 s), by sampling each horizon's Gaussian."""
    from sim import ZONE_HEIGHT, ZONE_RADIUS

    n, H = mean.shape[:2]
    inside_any = np.zeros((n, samples), bool)
    for h in range(H):
        L = np.linalg.cholesky(cov[:, h].astype(float) + 1e-6 * np.eye(3))
        pts = mean[:, h, None, :] + np.einsum("nij,nsj->nsi", L, rng.normal(size=(n, samples, 3)))
        inside_any |= (np.hypot(pts[..., 0], pts[..., 1]) < ZONE_RADIUS) & (pts[..., 2] < ZONE_HEIGHT)
    return inside_any.mean(1)


def realtime_predictions(table, imm_world, tab_world, every=3):
    """Replay a live system: the IMM answers at every radar reading (0.1 s); TabPFN starts a new answer every
    `every` readings and, taking up to that long (252 ms < 0.3 s, experiment 16), its answer arrives `every`
    readings later. Until the next one arrives, its correction (TabPFN's 1 s prediction minus the IMM's, for the
    reading it started from) is added to the IMM's fresh prediction. Before the first answer: the IMM alone.

    table: dict with flight, t; imm_world, tab_world: (n, 3) 1 s predictions per table row (NaN = none).
    Returns (predictions (n, 3), whether a TabPFN correction was used (n,), the row each correction came from (n,))."""
    k = np.rint(table["t"] / 0.1).astype(np.int64)
    flight = table["flight"].astype(np.int64)
    key = flight * 100_000 + k
    order = np.argsort(key)
    start = (k // every) * every - every  # the last answer that has arrived started at this reading
    want = flight * 100_000 + start
    pos = np.clip(np.searchsorted(key[order], want), 0, len(key) - 1)
    src = order[pos]
    found = (key[src] == want) & (start >= 0)
    corr = np.where(found[:, None], tab_world[src] - imm_world[src], np.nan)
    used = ~np.isnan(corr).any(1)
    return imm_world + np.where(used[:, None], corr, 0.0), used, src


def main():
    table = dict(np.load(DATA / "table_cf2.npz"))
    imm = np.load(DATA / "imm_cf2.npz")
    try:
        tab = dict(np.load(DATA / "tabpfn_cf2.npz"))
        rows, alarm_rows = tab["test_rows"], tab["alarm_rows"]
    except FileNotFoundError:
        tab = None
        rows, alarm_rows = test_rows(table)
        print("(no TabPFN results yet: scoring the IMM only)")
    print(f"test rows: {len(rows)} from {len(np.unique(table['flight'][rows]))} Crazyflie flights\n")

    imm_h = list(np.round(imm["horizons"], 2))
    hi = [imm_h.index(round(h, 2)) for h in HORIZONS]
    truth = table["Y_world"][rows].reshape(len(rows), len(HORIZONS), 3)
    pos, psi = table["pos"][rows], table["psi"][rows]
    changing = change_ahead(table, rows)

    models = {"IMM": imm["mean"][rows][:, hi].astype(float)}
    if tab is not None:
        models["TabPFN A (X2 only)"] = tab["A_x2_only_world"]
        models["TabPFN B (X2 + a few cf2)"] = tab["B_x2_plus_cf2_world"]

    def line(label, values):
        print(f"{label:28s}" + "".join(f"{v:>17}" for v in values))

    print("Median error (m)")
    line("", [f"{h:g} s" for h in HORIZONS] + [f"{h:g} s at change" for h in HORIZONS])
    for name, pred in models.items():
        err = np.linalg.norm(pred - truth, axis=2)
        line(name, [f"{np.median(err[:, j]):.2f}" for j in range(len(HORIZONS))] +
             [f"{np.median(err[changing[:, j], j]):.2f}" for j in range(len(HORIZONS))])
    print(f"{'':28s}(rows with a manoeuvre change ahead: " +
          ", ".join(f"{changing[:, j].mean():.0%}" for j in range(len(HORIZONS))) + ")")

    print("\nCone: real position inside the 5%-95% range, per direction (target 90%)")
    line("", [f"{h:g} s" for h in HORIZONS])
    imm_cov = imm["cov"][rows][:, hi].astype(float)
    cover = []
    for j in range(len(HORIZONS)):
        c, s = np.cos(psi), np.sin(psi)
        R = np.zeros((len(rows), 3, 3))
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, s, -s, c, 1
        sd = np.sqrt(np.einsum("nij,njk,nlk->nil", R, imm_cov[:, j], R).diagonal(axis1=1, axis2=2))
        off = drone_frame(truth[:, j] - models["IMM"][:, j], psi)
        cover.append(f"{(np.abs(off) <= Z90 * sd).mean():.0%}")
    line("IMM", cover)
    if tab is not None:
        Y = table["Y"][rows]
        for name, key in (("TabPFN A (X2 only)", "A_x2_only"), ("TabPFN B (X2 + a few cf2)", "B_x2_plus_cf2")):
            q = tab[f"{key}_quantiles"]
            inside = (Y >= q[:, :, 0]) & (Y <= q[:, :, 2])
            line(name, [f"{inside[:, 3 * j : 3 * j + 3].mean():.0%}" for j in range(len(HORIZONS))])

    label = table["breach"][alarm_rows]
    print(f"\nAlarm: breach within 2 s ({label.sum()} of {len(label)} test rows)")
    line("", ["AUC", "caught @5% FA"])
    ok = ~np.isnan(imm["mean"][alarm_rows]).any((1, 2))  # the IMM's first reading of a track has no prediction
    p_imm = np.zeros(len(alarm_rows))
    p_imm[ok] = imm_breach_probability(imm["mean"][alarm_rows][ok].astype(float), imm["cov"][alarm_rows][ok].astype(float),
                                       np.random.default_rng(0))
    line("IMM", [f"{auc(p_imm, label):.3f}", f"{caught_at(p_imm, label):.0%}"])
    if tab is not None:
        for name, key in (("TabPFN A (X2 only)", "A_x2_only"), ("TabPFN B (X2 + a few cf2)", "B_x2_plus_cf2")):
            p = tab[f"{key}_p_breach"]
            line(name, [f"{auc(p, label):.3f}", f"{caught_at(p, label):.0%}"])


if __name__ == "__main__":
    main()
