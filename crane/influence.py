"""How each input influences the output: measured, for the explainer page's "inputs and outputs" section.

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl crane/.venv/bin/python crane/influence.py

1. Slider effects (MuJoCo playground, 81 runs from results/playground.json): for each level of rope, sling, payload
   and speed, averaged over the other three sliders: peak load swing, TabPFN error 1 s ahead, physics error.
2. Permutation importance of TabPFN's inputs: shuffle one input column of the test rows, re-predict, and report how
   much the mean error grows (cm). Bigger growth = TabPFN relies more on that input.
   a) real lab crane: exp1 split B (context runs 1-8, 10, 11; test = beam run 12), target swing angle +1.0 s
   b) MuJoCo: context = the 3,000 fleet rows; test = 2,000 rows sampled from the 81 playground runs, target offset +1 s
Output: crane/results/influence.json
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_data as rd
from exp1_real import SPLITS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def perm_importance(model, Xt, y, scale, names, rng, reps=2):
    base = np.mean(np.abs(model.predict(Xt) - y)) * scale
    out = {}
    for j, n in enumerate(names):
        inc = []
        for _ in range(reps):
            Xp = Xt.copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            inc.append(np.mean(np.abs(model.predict(Xp) - y)) * scale - base)
        out[n] = round(float(np.mean(inc)), 3)
    return round(float(base), 3), out


def main():
    from tabpfn import TabPFNRegressor
    rng = np.random.default_rng(0); random.seed(0)
    res = {}

    # 1. slider effects
    pg = json.load(open(os.path.join(OUT, 'playground.json')))
    names = ['rope', 'sling', 'payload_t', 'speed']
    eff = {}
    for k, n in enumerate(names):
        eff[n] = []
        for v in pg['grid'][n]:
            rs = [r for key, r in pg['runs'].items() if abs(float(key.split('|')[k]) - v) < 1e-9]
            eff[n].append(dict(value=v, peak_swing_m=round(float(np.mean([max(abs(x) for x in r['off']) for r in rs])) / 100, 2),
                               tabpfn_cm=round(float(np.mean([r['err_cm'] for r in rs])), 2), physics_cm=round(float(np.mean([r['err_phys_cm'] for r in rs])), 2),
                               coverage90=round(float(np.mean([r['cover'] for r in rs])), 2), runs=len(rs)))
    res['slider_effects'] = eff
    print('slider effects done')

    # 2a. real lab crane
    runs = {i: rd.load_run(i) for i in rd.RUNS}; allrows = {i: rd.rows(runs[i]) for i in runs}
    ctx_ids, test_ids = SPLITS['B_payload_beam']
    ctx = random.sample([r for i in ctx_ids for r in allrows[i]], 3000); test = [r for i in test_ids for r in allrows[i]]
    X = lambda rows: np.array([[r['f'][k] for k in rd.FEATURES] for r in rows])
    Lt = np.mean([r['f']['L'] for r in test])
    m = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(X(ctx), np.array([r['y']['th_p10'] for r in ctx]))
    base, imp = perm_importance(m, X(test), np.array([r['y']['th_p10'] for r in test]), Lt * 100, rd.FEATURES, rng)
    res['real_importance'] = dict(target='swing angle +1.0 s, beam run held out', base_error_cm=base, increase_cm=imp)
    print('real', base, imp)

    # 2b. MuJoCo
    import playground_precompute as pp
    from mujoco_demo import FEAT, rows_from_log
    fleet = json.load(open(os.path.join(OUT, 'fleet_rows.json')))
    grid = []
    for L1 in pp.ROPES:
        for L2 in pp.SLINGS:
            for pt in pp.PAYLOADS:
                for sp in pp.SPEEDS:
                    c = pp.cond_of(L1, L2, pt); plan = pp.Delayed(sp)
                    grid += rows_from_log(pp.simulate(c, plan)[:, :4], plan, c, 0)
    test = random.sample(grid, 2000)
    Xm = lambda rows: np.array([[r['f'][k] for k in FEAT] for r in rows])
    m = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(Xm(fleet), np.array([r['y'] for r in fleet]))
    base, imp = perm_importance(m, Xm(test), np.array([r['y'] for r in test]), 100, FEAT, rng)
    res['mujoco_importance'] = dict(target='load offset +1 s, 81 playground runs', base_error_cm=base, increase_cm=imp)
    print('mujoco', base, imp)
    json.dump(res, open(os.path.join(OUT, 'influence.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
