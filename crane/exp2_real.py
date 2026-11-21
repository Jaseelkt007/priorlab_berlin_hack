"""Experiment 2 on REAL lab-crane data (same dataset and rows as exp1_real.py).

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 crane/.venv/bin/python crane/exp2_real.py

Step 1 - adaptation curve on the unseen beam payload (run 12), no retraining:
  base context = 3,000 rows sampled from runs 1-8, 10, 11 (exp1 split B)
  + the FIRST N beam rows in time (N = 0, 10, 20, 50, 100, 200 -> 0-20 s of watching the beam swing, t = 2-22 s)
  test = beam rows with t >= 30 s (never in any context).
  Compared on exactly the same rows: trees retrained on base + N; pendulum with L_eff refitted on the N beam rows.
  Note: run 12 has a single trolley move in the first 1.6 s, so this is mostly predicting the free swing of an unknown payload.

Step 2 - physics + TabPFN (residual model), splits A, B, C of exp1:
  TabPFN predicts y - pendulum_fitted(x); prediction = pendulum + TabPFN median, range = pendulum + TabPFN quantiles.

Results -> crane/results/exp2_real.json
"""
import json
import os
import random
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from tabpfn import TabPFNRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_data as rd
from exp1_real import SPLITS, fit_pendulum, OUT

CTX = 3000
NS = (0, 10, 20, 50, 100, 200)
H = rd.HORIZONS
KEY = {h: f'th_p{int(h * 10):02d}' for h in H}


def feats(rows): return np.array([[r['f'][k] for k in rd.FEATURES] for r in rows])
def targ(rows, h): return np.array([r['y'][KEY[h]] for r in rows])
def cm(err, rows): return float(np.mean(np.abs(err) * np.array([r['f']['L'] for r in rows]) * 100))


def phys(runs, rows, p, h):
    out = []
    for r in rows:
        run = runs[r['run']]; i = int(round(r['t'] / rd.DT))
        out.append(rd.pendulum_predict(run, i, h, run['L'][i] + p['delta'], p['zeta'], p['sign']))
    return np.array(out)


def pfn(Xc, yc, Xt):
    m = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache')
    t0 = time.time(); m.fit(Xc, yc); tf = time.time() - t0
    t0 = time.time(); lo, med, hi = (np.asarray(v) for v in m.predict(Xt, output_type='quantiles', quantiles=[0.05, 0.5, 0.95])); tp = time.time() - t0
    return lo, med, hi, tf, 1000 * tp / len(Xt)


def main():
    runs = {i: rd.load_run(i) for i in rd.RUNS}
    allrows = {i: rd.rows(runs[i]) for i in runs}
    res = {'dataset': 'doi:10.17632/x69hyyv3zw.2 (CC BY 4.0)', 'ctx_rows': CTX}
    random.seed(0)

    # ---------- step 1: adaptation curve on the beam ----------
    base_ids = SPLITS['B_payload_beam'][0]
    base = random.sample([r for i in base_ids for r in allrows[i]], CTX)
    beam = allrows[12]
    pool = [r for r in beam if r['t'] < 30]; test = [r for r in beam if r['t'] >= 30]
    p_base = fit_pendulum(runs, [r for i in base_ids for r in allrows[i]], False)
    curve = []
    for n in NS:
        add = pool[:n]; ctx = base + add
        row = {'N': n, 'seconds_watched': n / 10}
        p = fit_pendulum(runs, add, False) if n >= 20 else p_base
        row['pendulum_params'] = p
        for h in H:
            y = targ(test, h); Xc, Xt = feats(ctx), feats(test)
            lo, med, hi, tf, ms = pfn(Xc, targ(ctx, h), Xt)
            tr = HistGradientBoostingRegressor(random_state=0).fit(Xc, targ(ctx, h)).predict(Xt)
            ph = phys(runs, test, p, h)
            row[f'+{h}s'] = dict(tabpfn_cm=round(cm(med - y, test), 3), coverage90=round(float(np.mean((y >= lo) & (y <= hi))), 3),
                                 trees_cm=round(cm(tr - y, test), 3), pendulum_cm=round(cm(ph - y, test), 3), fit_s=round(tf, 1), ms_per_row=round(ms, 2))
        curve.append(row)
        print('adapt', json.dumps(row))
    res['step1_adaptation_beam'] = {'n_test': len(test), 'test_t_from_s': 30, 'curve': curve}

    # ---------- step 2: physics + TabPFN residual ----------
    res['step2_residual'] = {}
    for name, (ctx_ids, test_ids) in SPLITS.items():
        full = [r for i in ctx_ids for r in allrows[i]]; test = [r for i in test_ids for r in allrows[i]]
        p = fit_pendulum(runs, full, False)
        ctx = full if len(full) <= CTX else random.sample(full, CTX)
        S = {'pendulum_params': p}
        for h in H:
            y = targ(test, h); pc, pt = phys(runs, ctx, p, h), phys(runs, test, p, h)
            lo, med, hi, tf, ms = pfn(feats(ctx), targ(ctx, h) - pc, feats(test))
            S[f'+{h}s'] = dict(pendulum_cm=round(cm(pt - y, test), 3), hybrid_cm=round(cm(pt + med - y, test), 3),
                               coverage90=round(float(np.mean((y >= pt + lo) & (y <= pt + hi))), 3), fit_s=round(tf, 1), ms_per_row=round(ms, 2))
        res['step2_residual'][name] = S
        print('residual', name, json.dumps(S))
    with open(os.path.join(OUT, 'exp2_real.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', os.path.join(OUT, 'exp2_real.json'))


if __name__ == '__main__':
    main()
