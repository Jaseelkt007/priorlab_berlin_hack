"""Precompute the explainer page's interactive playground (option A) and the fleet context for render_clip.py (option B).

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl crane/.venv/bin/python crane/playground_precompute.py

Grid (3 x 3 x 3 x 3 = 81 MuJoCo runs): rope L1 {3, 4.5, 6} m, sling L2 {0.5, 1.5, 2.5} m, payload {0.5, 1.5, 3} t,
speed {50, 75, 100} % (speed and acceleration scaled together; no anti-sway). Each run: 2 s hold, one 10 m move, 10 s
after arrival. Logged at 10 Hz: trolley x, hook and load offset from the trolley.
TabPFN context: 3,000 rows sampled from 40 random fleet jobs spanning the same ranges but never a grid point
(rope 3-6 m, sling 0.5-2.5 m, 0.5-3 t, 50-100 %). TabPFN predicts the load offset 1 s ahead at every 0.1 s of every
grid run (quantiles 0.05 / 0.5 / 0.95), plus the single-pendulum physics prediction with a 0.5 m entered sling.
Outputs: crane/results/playground.json (compact, embedded in explainer.html), crane/results/fleet_rows.json
"""
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_crane as mc
import mujoco
from mujoco_demo import FEAT, H, X, physics_pred, rows_from_log

OUT = os.path.join(mc.HERE, 'results')
ROPES, SLINGS, PAYLOADS, SPEEDS = [3.0, 4.5, 6.0], [0.5, 1.5, 2.5], [0.5, 1.5, 3.0], [0.5, 0.75, 1.0]
PRE, POST = 2.0, 10.0
BEAM = 2.0


class Delayed:
    """A one-move Plan that starts after PRE seconds of standing still, and ends POST s after arrival."""
    def __init__(self, frac):
        self.p = mc.Plan([frac], pause=POST); self.T = self.p.T + PRE; self.x0 = self.p.x0
    def state(self, t): return self.p.state(t - PRE) if t >= PRE else (self.x0, 0.0, 0.0)
    def ref(self, t): return self.state(t)[0]


def cond_of(L1, L2, t_payload): return dict(L1=L1, L2=L2, m_hook=250.0, m_load=t_payload * 1000, beam=BEAM)


def simulate(cond, plan):
    """Same control law as mujoco_crane.run, also logging the hook offset. Columns: t, x, xref, load_off, hook_off."""
    m, d = mc.make(cond); tid, hid = m.body('trolley').id, m.body('hook').id
    M = float(m.body_subtreemass[tid]); kp, kv = M * 25.0, M * 10.0
    log, k, n = [], 0, int(round(0.1 / mc.DT))
    while d.time < plan.T:
        xr, vr, ar = plan.state(d.time)
        d.ctrl[0] = M * ar + kp * (xr - d.qpos[0]) + kv * (vr - d.qvel[0])
        mujoco.mj_step(m, d); k += 1
        if k % n == 0:
            log.append((d.time, d.xpos[tid][0], xr, d.site_xpos[0][0] - d.xpos[tid][0], d.xpos[hid][0] - d.xpos[tid][0]))
    return np.array(log)


def main():
    from tabpfn import TabPFNRegressor
    rng = random.Random(3); t0 = time.time(); fleet = []
    for j in range(40):
        c = cond_of(rng.uniform(3, 6), rng.uniform(0.5, 2.5), rng.uniform(0.5, 3.0)); plan = Delayed(rng.uniform(0.5, 1.0))
        fleet += rows_from_log(simulate(c, plan)[:, :4], plan, c, j)
    print(f'fleet: {len(fleet)} rows in {time.time() - t0:.0f} s')
    ctx = rng.sample(fleet, 3000)
    json.dump([dict(f=r['f'], y=r['y']) for r in ctx], open(os.path.join(OUT, 'fleet_rows.json'), 'w'))
    model = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(X(ctx), np.array([r['y'] for r in ctx]))
    runs, allrows = [], []
    for L1 in ROPES:
        for L2 in SLINGS:
            for pt in PAYLOADS:
                for sp in SPEEDS:
                    c = cond_of(L1, L2, pt); plan = Delayed(sp); log = simulate(c, plan)
                    rows = rows_from_log(log[:, :4], plan, c, len(runs))
                    runs.append(dict(c=c, sp=sp, log=log, rows=rows, plan=plan)); allrows += rows
    print(f'grid: {len(runs)} runs, {len(allrows)} rows to predict ({time.time() - t0:.0f} s)')
    tp = time.time(); lo, md, hi = (np.asarray(v) for v in model.predict(X(allrows), output_type='quantiles', quantiles=[0.05, 0.5, 0.95]))
    print(f'TabPFN predict: {time.time() - tp:.1f} s for {len(allrows)} rows')
    k = 0; out = {'grid': dict(rope=ROPES, sling=SLINGS, payload_t=PAYLOADS, speed=SPEEDS), 'dt': 0.1, 'horizon_s': H, 'clear_m': mc.CLEAR, 'beam_m': BEAM,
                  'bays': list(mc.BAYS), 'runs': {}}
    errs = {'tabpfn': [], 'physics': [], 'cover': []}
    for r in runs:
        n = len(r['rows']); log = r['log']
        p_lo, p_md, p_hi = lo[k:k + n], md[k:k + n], hi[k:k + n]; k += n
        phys = np.array([physics_pred(row, r['plan'], r['c']) for row in r['rows']])
        y = np.array([row['y'] for row in r['rows']])
        errs['tabpfn'] += list(np.abs(p_md - y)); errs['physics'] += list(np.abs(phys - y)); errs['cover'] += list((y >= p_lo) & (y <= p_hi))
        i0 = r['rows'][0]['i']                                        # first log index that has a prediction
        cm = lambda a: [int(round(v * 100)) for v in a]
        key = f"{r['c']['L1']}|{r['c']['L2']}|{r['c']['m_load'] / 1000}|{r['sp']}"
        out['runs'][key] = dict(x=cm(log[:, 1]), off=cm(log[:, 3]), hook=cm(log[:, 4]), i0=int(i0),
                                lo=cm(p_lo), md=cm(p_md), hi=cm(p_hi), ph=cm(phys),
                                err_cm=round(float(np.mean(np.abs(p_md - y))) * 100, 1), err_phys_cm=round(float(np.mean(np.abs(phys - y))) * 100, 1),
                                cover=round(float(np.mean((y >= p_lo) & (y <= p_hi))), 2))
    out['summary'] = dict(tabpfn_cm=round(100 * float(np.mean(errs['tabpfn'])), 1), physics_cm=round(100 * float(np.mean(errs['physics'])), 1),
                          coverage90=round(float(np.mean(errs['cover'])), 2), fleet_jobs=40, ctx_rows=3000)
    json.dump(out, open(os.path.join(OUT, 'playground.json'), 'w'), separators=(',', ':'))
    print('summary', out['summary'], f"size {os.path.getsize(os.path.join(OUT, 'playground.json')) // 1024} KB")


if __name__ == '__main__':
    main()
