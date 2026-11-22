"""Set-down advisor (no change to the crane's control): when may the final lowering start?

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl crane/.venv/bin/python crane/setdown_advisor.py

Data: the saved closed-loop moves (results/closed_loop_run.json: 3 strategies x 12 loads, load state at 10 Hz from
arrival until the crane's own rule confirmed "settled"). Final lowering takes LOWER = 1 s; touchdown is OK if the
load is within +-TOL of the bay centre at touchdown.
  today   : start lowering once |offset| < TOL has held for 1 s (the end of each saved trace)
  physics : start at the first moment the single-pendulum model (entered 0.5 m sling) predicts |offset(t+1 s)| < TOL
  tabpfn  : start at the first moment TabPFN's whole 90% range for offset(t+1 s) lies inside +-TOL
TabPFN context: results/live_fleet_rows.json (same features as live_crane.py). Output: results/setdown_advisor.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_crane as mc
import mujoco
from live_crane import Mover, features, physics_pred, FEAT

OUT = os.path.join(mc.HERE, 'results')
WHO = ('physics', 'tabpfn', 'physics_hold', 'tabpfn_hold', 'tabpfn_1s')
TOL, LOWER, VMAX_TD, HOLD_N = 0.05, 1.0, 0.05, 5   # tolerance m, lowering s, max sideways speed at touchdown m/s, gate window rows


def main():
    from tabpfn import TabPFNRegressor
    run = json.load(open(os.path.join(OUT, 'closed_loop_run.json')))
    fleet = json.load(open(os.path.join(OUT, 'live_fleet_rows.json')))
    model = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(np.array([[r['f'][k] for k in FEAT] for r in fleet]), np.array([r['y'] for r in fleet]))
    res = {}
    for lane, moves in run['lanes'].items():
        out = []
        for j, mv in zip(run['jobs'], moves):
            c = j['c']; m, d = mc.make(c); tid = m.body('trolley').id
            trace = []
            for q in mv['frames']:                                             # 10 Hz, from the start of the move
                d.qpos[:] = q; mujoco.mj_forward(m, d); trace.append((d.xpos[tid][0], d.site_xpos[0][0] - d.xpos[tid][0]))
            k0 = int(np.ceil(mv['move_s'] / 0.1))                              # first sample after arrival
            mover = Mover(trace[-1][0])                                        # parked at the bay: plan = stand still
            hist = [(i * 0.1, x, o) for i, (x, o) in enumerate(trace)]
            ks = [k for k in range(max(k0, 21), len(trace) - 10)]              # need 2 s history and the 1-s outcome
            if not ks:
                out.append(dict(today_s=0.0, **{w: None for w in WHO})); continue
            F = [features(hist[k - 39:k + 1], mover, k * 0.1, c, 1) for k in ks]
            lo, md, hi = (np.asarray(v) for v in model.predict(np.array([[f[x] for x in FEAT] for f in F]), output_type='quantiles', quantiles=[0.05, 0.5, 0.95]))
            ph = [physics_pred(f, mover, k * 0.1, c['L1']) for f, k in zip(F, ks)]
            today = (len(trace) - 1) * 0.1                                     # the crane's own rule fired here
            def first(ok, hold=1):
                for n_, (k, g) in enumerate(zip(ks, ok)):
                    if n_ + 1 >= hold and all(ok[n_ - hold + 1:n_ + 1]):
                        i = k + 10; v = (trace[min(i + 1, len(trace) - 1)][1] - trace[i - 1][1]) / 0.2
                        pos_ok, spd_ok = abs(trace[i][1]) < TOL, abs(v) < VMAX_TD
                        return dict(start_s=round(k * 0.1, 1), saved_s=round(today - k * 0.1, 1), touchdown_offset_cm=round(abs(trace[i][1]) * 100, 1),
                                    touchdown_speed_cm_s=round(abs(v) * 100, 1), ok=bool(pos_ok and spd_ok), pos_ok=bool(pos_ok))
                return None
            gp = [abs(p) < TOL for p in ph]; gt = [bool((l > -TOL) and (h < TOL)) for l, h in zip(lo, hi)]
            out.append(dict(today_s=round(today, 1), physics=first(gp), tabpfn=first(gt), physics_hold=first(gp, HOLD_N), tabpfn_hold=first(gt, HOLD_N), tabpfn_1s=first(gt, 10)))
        res[lane] = out
        for who in WHO:
            got = [o[who] for o in out if o[who]]
            print(f"{lane:8s} {who:12s}: fired {len(got):2d}/{len(out)}  saved {sum(g['saved_s'] for g in got):5.1f} s  inside +-5 cm {sum(g['pos_ok'] for g in got):2d}  safe (also < 5 cm/s) {sum(g['ok'] for g in got):2d}  median speed {np.median([g['touchdown_speed_cm_s'] for g in got]) if got else 0:.0f} cm/s")
    summ = {lane: {who: dict(fired=sum(1 for o in out if o[who]), moves=len(out), saved_s=round(sum(o[who]['saved_s'] for o in out if o[who]), 1),
                             ok=sum(o[who]['ok'] for o in out if o[who]), pos_ok=sum(o[who]['pos_ok'] for o in out if o[who]), bad=sum(not o[who]['ok'] for o in out if o[who]),
                             median_speed_cm_s=float(np.median([o[who]['touchdown_speed_cm_s'] for o in out if o[who]])) if any(o[who] for o in out) else None) for who in WHO}
            for lane, out in res.items()}
    json.dump(dict(tol_m=TOL, lower_s=LOWER, max_touchdown_speed_m_s=VMAX_TD, hold_rows=HOLD_N, summary=summ, moves=res), open(os.path.join(OUT, 'setdown_advisor.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
