"""Experiment 1 on REAL lab-crane data: predict the swing angle 0.5 s and 1.0 s ahead on held-out runs.

    python3 -I crane/exp1_real.py                  # physics baselines only (standard library)
    crane/.venv/bin/python crane/exp1_real.py  # + gradient-boosted trees + TabPFN, when installed
    ... --ctx 3000                                     # cap TabPFN context rows (GPU memory); default 3000

Splits (whole runs held out, never random rows: lesson 4):
  A  held-out cable length : context = runs 1-4, 6-8 (L 0.1-0.8 m, default payload), test = run 5 (L 0.5 m)
  B  held-out payload      : context = runs 1-8, 10, 11 (default, ball, cylinder), test = run 12 (500 g beam)
  C  held-out hoisting     : context = fixed-L runs 1-8, 10-12, test = runs 9 and 13 (hoisting during travel)
Predictors, all on the same context rows:
  textbook  pendulum with the logged cable length, no damping (only the sign convention is fitted)
  fitted    pendulum with L_eff = L + delta and damping zeta, both fitted on the context runs (strong baseline)
  trees     HistGradientBoostingRegressor, retrained per split
  tabpfn    TabPFNRegressor (in-context, no gradient steps), quantiles 0.05 / 0.5 / 0.95
Results -> crane/results/exp1_real.json (TabPFN predictions cached there too).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_data as rd

SPLITS = {
    'A_cable_length': ([1, 2, 3, 4, 6, 7, 8], [5]),
    'B_payload_beam': ([1, 2, 3, 4, 5, 6, 7, 8, 10, 11], [12]),
    'C_hoisting': ([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], [9, 13]),
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def rmse(e): return math.sqrt(sum(v * v for v in e) / len(e))


def fit_pendulum(runs, rows, textbook):
    """Grid-search sign (and delta, zeta unless textbook) on every 20th context row, horizon 1.0 s."""
    sub = rows[::20]
    best = None
    for sign in (1, -1):
        for delta in ([0.0] if textbook else [d / 100 for d in range(0, 42, 2)]):
            for zeta in ([0.0] if textbook else [0.0, 0.01, 0.02, 0.05, 0.1]):
                e = []
                for r in sub:
                    run = runs[r['run']]; i = int(round(r['t'] / rd.DT))
                    e.append(rd.pendulum_predict(run, i, 1.0, run['L'][i] + delta, zeta, sign) - r['y']['th_p10'])
                s = rmse(e)
                if best is None or s < best[0]:
                    best = (s, sign, delta, zeta)
    return dict(sign=best[1], delta=best[2], zeta=best[3])


def eval_pendulum(runs, rows, p):
    err = {h: [] for h in rd.HORIZONS}
    for r in rows:
        run = runs[r['run']]; i = int(round(r['t'] / rd.DT))
        for h in rd.HORIZONS:
            y = r['y'][f'th_p{int(h * 10):02d}']
            err[h].append(rd.pendulum_predict(run, i, h, run['L'][i] + p['delta'], p['zeta'], p['sign']) - y)
    return err


def summarize(err, rows, extra=None):
    out = {}
    for h, e in err.items():
        cm = [abs(v) * r['f']['L'] * 100 for v, r in zip(e, rows)]   # arc length at the payload, approx
        out[f'+{h}s'] = dict(rmse_rad=round(rmse(e), 4), mae_cm=round(sum(cm) / len(cm), 2))
        if extra and h in extra:
            out[f'+{h}s'].update(extra[h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ctx', type=int, default=3000)
    ap.add_argument('--no-ml', action='store_true')
    a = ap.parse_args()
    runs = {i: rd.load_run(i) for i in rd.RUNS}
    allrows = {i: rd.rows(runs[i]) for i in runs}
    try:
        if a.no_ml: raise ImportError
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor
        have_sk = True
    except ImportError:
        have_sk = False
    try:
        if a.no_ml: raise ImportError
        from tabpfn import TabPFNRegressor
        have_pfn = True
    except ImportError:
        have_pfn = False
    print(f'trees: {have_sk}  tabpfn: {have_pfn}')
    os.makedirs(OUT, exist_ok=True)
    res = {'dataset': 'doi:10.17632/x69hyyv3zw.2 (CC BY 4.0)', 'features': rd.FEATURES, 'splits': {}}
    for name, (ctx_ids, test_ids) in SPLITS.items():
        ctx = [r for i in ctx_ids for r in allrows[i]]; test = [r for i in test_ids for r in allrows[i]]
        S = res['splits'][name] = {'context_runs': ctx_ids, 'test_runs': test_ids, 'n_ctx': len(ctx), 'n_test': len(test)}
        t0 = time.time()
        for kind in ('textbook', 'fitted'):
            p = fit_pendulum(runs, ctx, kind == 'textbook')
            S[kind] = dict(params=p, **summarize(eval_pendulum(runs, test, p), test))
        print(f'{name}: physics done in {time.time() - t0:.0f} s')
        if have_sk:
            X = np.array([[r['f'][k] for k in rd.FEATURES] for r in ctx]); Xt = np.array([[r['f'][k] for k in rd.FEATURES] for r in test])
            err = {}
            for h in rd.HORIZONS:
                key = f'th_p{int(h * 10):02d}'
                m = HistGradientBoostingRegressor(random_state=0).fit(X, [r['y'][key] for r in ctx])
                err[h] = list(m.predict(Xt) - np.array([r['y'][key] for r in test]))
            S['trees'] = summarize(err, test)
        if have_pfn:
            random.seed(0)
            sel = ctx if len(ctx) <= a.ctx else random.sample(ctx, a.ctx)
            X = np.array([[r['f'][k] for k in rd.FEATURES] for r in sel]); Xt = np.array([[r['f'][k] for k in rd.FEATURES] for r in test])
            err, extra = {}, {}
            for h in rd.HORIZONS:
                key = f'th_p{int(h * 10):02d}'; y = np.array([r['y'][key] for r in test])
                m = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache')
                t1 = time.time(); m.fit(X, np.array([r['y'][key] for r in sel])); tf = time.time() - t1
                t1 = time.time(); q = m.predict(Xt, output_type='quantiles', quantiles=[0.05, 0.5, 0.95]); tp = time.time() - t1
                lo, med, hi = (np.asarray(v) for v in q)
                err[h] = list(med - y)
                extra[h] = dict(coverage90=round(float(np.mean((y >= lo) & (y <= hi))), 3), width90_rad=round(float(np.mean(hi - lo)), 4),
                                fit_s=round(tf, 2), predict_ms_per_row=round(1000 * tp / len(test), 3), n_ctx_used=len(sel))
            S['tabpfn'] = summarize(err, test, extra)
        print(json.dumps({k: v for k, v in S.items() if k not in ('context_runs', 'test_runs')}, indent=1))
    with open(os.path.join(OUT, 'exp1_real.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('wrote', os.path.join(OUT, 'exp1_real.json'))


if __name__ == '__main__':
    main()
