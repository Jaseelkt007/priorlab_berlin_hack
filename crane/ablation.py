"""Ablation: can the 12 forecast inputs (and 5 decision inputs) be cut to the few that physics says matter?

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl crane/.venv/bin/python crane/ablation.py

A pendulum's next second depends on three things: how the load moves now, how long the pendulum is, and what push
comes next. The simplified sets keep one or two inputs per thing. "Planned push" = average planned trolley
acceleration over the next 1 s, derived from columns we already log: 2 * (planned move in 1 s - speed * 1 s).
Forecast target: load offset (MuJoCo) / swing angle (lab crane) 1 s ahead. Same context rows, same test rows for every set.
Decision target: settle wait and residual swing per move (5-fold cross-validation on the 120 commissioning moves).
Output: crane/results/ablation.json
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

SETS_M = {
    'all 12 (current)': ['off', 'off_m02', 'off_m05', 'doff', 'amp2s', 'x', 'dx', 'ddx', 'L1', 'payload_t', 'plan_dx05', 'plan_dx10'],
    '6: motion + length + push + 0.5 s history': ['off', 'doff', 'off_m05', 'L1', 'ddx', 'plan_acc10'],
    '5: motion + length + push': ['off', 'doff', 'L1', 'ddx', 'plan_acc10'],
    '4: motion + push (no length)': ['off', 'doff', 'ddx', 'plan_acc10'],
    '10: all minus position & speed': ['off', 'off_m02', 'off_m05', 'doff', 'amp2s', 'ddx', 'L1', 'payload_t', 'plan_dx05', 'plan_dx10'],
}
SETS_R = {
    'all 12 (current)': ['th', 'th_m02', 'th_m05', 'dth', 'amp2s', 'x', 'dx', 'ddx', 'L', 'dL', 'plan_dx05', 'plan_dx10'],
    '6: motion + length + push + 0.5 s history': ['th', 'dth', 'th_m05', 'L', 'ddx', 'plan_acc10'],
    '5: motion + length + push': ['th', 'dth', 'L', 'ddx', 'plan_acc10'],
}
SETS_D = {'5 (current): rope, payload, period, speed, tuning error': [0, 1, 2, 3, 4],
          '3: period, speed, tuning error': [2, 3, 4], '2: period, tuning error': [2, 4]}


def add_derived(f):
    f = dict(f); f['plan_acc10'] = 2 * (f['plan_dx10'] - f['dx'] * 1.0); return f


def fit_eval(TabPFNRegressor, ctx, test, cols, scale):
    X = np.array([[r['f'][c] for c in cols] for r in ctx]); y = np.array([r['y'] for r in ctx])
    Xt = np.array([[r['f'][c] for c in cols] for r in test]); yt = np.array([r['y'] for r in test]); s = np.asarray(scale)
    lo, md, hi = (np.asarray(v) for v in TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(X, y).predict(Xt, output_type='quantiles', quantiles=[0.05, 0.5, 0.95]))
    return dict(err_cm=round(float(np.mean(np.abs(md - yt) * s)) * 100, 2), coverage90=round(float(np.mean((yt >= lo) & (yt <= hi))), 2))


def main():
    from tabpfn import TabPFNRegressor
    random.seed(0); res = {}
    # ---- MuJoCo forecast: context = live-style fleet; tests = playground grid and new back-and-forth jobs
    import playground_precompute as pp
    from mujoco_demo import rows_from_log
    from live_crane import build_fleet
    fleet = [dict(f=add_derived(r['f']), y=r['y']) for r in json.load(open(os.path.join(OUT, 'live_fleet_rows.json')))]
    grid = []
    for L1 in pp.ROPES:
        for L2 in pp.SLINGS:
            for pt in pp.PAYLOADS:
                for sp in pp.SPEEDS:
                    c = pp.cond_of(L1, L2, pt); plan = pp.Delayed(sp)
                    grid += [dict(f=add_derived(r['f']), y=r['y']) for r in rows_from_log(pp.simulate(c, plan)[:, :4], plan, c, 0)]
    grid = random.sample(grid, 2000)
    print('simulating 10 new back-and-forth jobs ...', flush=True)
    jobs = [dict(f=add_derived(r['f']), y=r['y']) for r in build_fleet(n_jobs=10, seed=99)]; jobs = random.sample(jobs, min(2000, len(jobs)))
    res['mujoco'] = {}
    for name, cols in SETS_M.items():
        res['mujoco'][name] = dict(inputs=cols, playground=fit_eval(TabPFNRegressor, fleet, grid, cols, 1.0), new_jobs=fit_eval(TabPFNRegressor, fleet, jobs, cols, 1.0))
        print('mujoco', name, res['mujoco'][name]['playground'], res['mujoco'][name]['new_jobs'], flush=True)
    # ---- real lab crane forecast: splits B (beam) and C (hoisting), target +1 s
    import real_data as rd
    from exp1_real import SPLITS
    runs = {i: rd.load_run(i) for i in rd.RUNS}; allrows = {i: rd.rows(runs[i]) for i in runs}
    res['real'] = {}
    for name, cols in SETS_R.items():
        res['real'][name] = dict(inputs=cols)
        for sp in ('B_payload_beam', 'C_hoisting'):
            ci, ti = SPLITS[sp]; random.seed(0)
            ctx = [dict(f=add_derived(r['f']), y=r['y']['th_p10']) for r in random.sample([r for i in ci for r in allrows[i]], 3000)]
            test = [dict(f=add_derived(r['f']), y=r['y']['th_p10']) for i in ti for r in allrows[i]]
            res['real'][name][sp] = fit_eval(TabPFNRegressor, ctx, test, cols, [r['f']['L'] for r in test])
        print('real', name, res['real'][name]['B_payload_beam'], res['real'][name]['C_hoisting'], flush=True)
    # ---- decision model: 5-fold CV on the commissioning moves
    D = json.load(open(os.path.join(OUT, 'pickplace_decision_fleet.json'))); idx = list(range(len(D))); random.seed(1); random.shuffle(idx)
    res['decision'] = {}
    for name, cols in SETS_D.items():
        e_s, e_a = [], []
        for k in range(5):
            te = set(idx[k::5]); tr = [D[i] for i in idx if i not in te]; ts = [D[i] for i in te]
            X = np.array([[r['x'][c] for c in cols] for r in tr]); Xt = np.array([[r['x'][c] for c in cols] for r in ts])
            for key, acc in (('settle', e_s), ('amp', e_a)):
                p = np.asarray(TabPFNRegressor(n_estimators=1).fit(X, np.array([r[key] for r in tr])).predict(Xt))
                acc += list(np.abs(p - np.array([r[key] for r in ts])))
        res['decision'][name] = dict(settle_err_s=round(float(np.mean(e_s)), 2), swing_err_cm=round(float(np.mean(e_a)) * 100, 1))
        print('decision', name, res['decision'][name], flush=True)
    json.dump(res, open(os.path.join(OUT, 'ablation.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
