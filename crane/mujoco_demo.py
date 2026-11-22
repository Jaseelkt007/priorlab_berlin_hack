"""MuJoCo demo: TabPFN predicts where a crane load will be 1 s ahead, on a load it has never seen,
adapting by adding that load's own rows to its table while the crane works. Renders an MP4.

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_demo.py
    ... --no-video        # numbers only

1. Fleet logs (context): 30 simulated jobs on SHORT slings (0.5-1.5 m), heavy beams (1-3 t), rope 3-7 m,
   random move speeds. 10 Hz rows, 3,000 sampled.
2. Held-out job: light 400 kg, 2.4 m beam on LONG 2.6 m slings (the page's phase 2), 4 moves.
3. Online: every 5 s, refit TabPFN on fleet rows + this job's rows whose 1 s outcome is already known (t_row <= t - 1 s).
   Predict the load offset (load x - trolley x) 1 s ahead for the next 5 s of rows.
4. Physics baseline: single pendulum with the operator-entered 0.5 m sling, driven by the planned trolley motion.
5. Video: teal ghost = TabPFN's predicted load position in 1 s (band = 90% range), orange ghost = physics.
Outputs: crane/results/mujoco_demo.json, crane/results/mujoco_demo.mp4
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_crane as mc
import mujoco

OUT = os.path.join(mc.HERE, 'results')
FEAT = ['off', 'off_m02', 'off_m05', 'doff', 'amp2s', 'x', 'dx', 'ddx', 'L1', 'payload_t', 'plan_dx05', 'plan_dx10']
H = 1.0                  # prediction horizon, s
REFIT = 5.0              # refit period during the job, s
ENTERED_SLING = 0.5      # what the operator typed into the anti-sway, m
DEMO = dict(L1=4.0, L2=2.6, m_hook=250.0, m_load=400.0, beam=2.4)
DEMO_FRACS = [1.0, 0.9, 1.0, 0.8]


def rows_from_log(log, plan, cond, job):
    """10 Hz log (t, x, xref, off) -> feature rows + target offset at +H."""
    t, x, off = log[:, 0], log[:, 1], log[:, 3]
    k = int(round(H * 10)); out = []
    for i in range(20, len(t) - k):
        f = dict(off=off[i], off_m02=off[i - 2], off_m05=off[i - 5], doff=(off[i] - off[i - 1]) / 0.1,
                 amp2s=float(np.abs(off[i - 20:i + 1]).max()), x=x[i], dx=(x[i] - x[i - 1]) / 0.1,
                 ddx=(x[i] - 2 * x[i - 1] + x[i - 2]) / 0.01, L1=cond['L1'], payload_t=cond['m_load'] / 1000,
                 plan_dx05=plan.ref(t[i] + 0.5) - plan.ref(t[i]), plan_dx10=plan.ref(t[i] + 1.0) - plan.ref(t[i]))
        out.append(dict(job=job, i=i, t=float(t[i]), f=f, y=float(off[i + k]), xt=float(x[i])))
    return out


def physics_pred(r, plan, cond):
    """Single pendulum (rope + hook + entered sling + half load height) along the planned trolley motion."""
    L = cond['L1'] + 0.3 + ENTERED_SLING + 0.18
    th = math.asin(max(-0.99, min(0.99, r['f']['off'] / L))); w = r['f']['doff'] / L / max(0.2, math.cos(th))
    dt = 0.005; t = r['t']
    for _ in range(int(H / dt)):
        a = plan.state(t)[2]
        w += (-(mc.G / L) * math.sin(th) - (a / L) * math.cos(th)) * dt; th += w * dt; t += dt
    return L * math.sin(th)


def X(rows): return np.array([[r['f'][k] for k in FEAT] for r in rows])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--no-video', action='store_true'); ap.add_argument('--render-only', action='store_true'); ap.add_argument('--jobs', type=int, default=30)
    a = ap.parse_args()
    cache = os.path.join(OUT, 'mujoco_demo_rows.json')
    if a.render_only:
        c = json.load(open(cache)); render(c['job'], mc.Plan(DEMO_FRACS), c['res']); return
    from tabpfn import TabPFNRegressor
    rng = random.Random(1); t0 = time.time()
    fleet = []
    for j in range(a.jobs):
        cond = dict(L1=rng.uniform(3, 7), L2=rng.uniform(0.5, 1.5), m_hook=250.0, m_load=rng.uniform(1000, 3000), beam=rng.uniform(1.0, 2.0))
        plan = mc.Plan([rng.uniform(0.5, 1.0) for _ in range(4)])
        fleet += rows_from_log(mc.run(cond, plan), plan, cond, j)
    print(f'fleet: {a.jobs} jobs, {len(fleet)} rows, {time.time() - t0:.0f} s')
    base = rng.sample(fleet, 3000)

    plan = mc.Plan(DEMO_FRACS); log = mc.run(DEMO, plan)
    job = rows_from_log(log, plan, DEMO, 'demo')
    print(f'demo job: {plan.T:.0f} s, {len(job)} rows, max |offset| {np.abs(log[:, 3]).max():.2f} m')

    preds = {}; fits = []
    T = job[0]['t']
    while T <= job[-1]['t']:
        known = [r for r in job if r['t'] <= T - H]                      # outcomes already observed
        ctx = base + known
        m = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache')
        tf = time.time(); m.fit(X(ctx), np.array([r['y'] for r in ctx])); tf = time.time() - tf
        batch = [r for r in job if T <= r['t'] < T + REFIT]
        if batch:
            tp = time.time(); lo, med, hi = (np.asarray(v) for v in m.predict(X(batch), output_type='quantiles', quantiles=[0.05, 0.5, 0.95])); tp = time.time() - tp
            for r, l, md, h in zip(batch, lo, med, hi): preds[r['i']] = (float(l), float(md), float(h), len(known))
            fits.append(dict(t=round(T, 1), own_rows=len(known), fit_s=round(tf, 2), ms_per_row=round(1000 * tp / len(batch), 2)))
            print(f'  t={T:5.1f}s own rows {len(known):4d}  fit {tf:.1f}s')
        T += REFIT

    err_p, err_f, cov, seg = [], [], [], []
    for r in job:
        l, md, h, n = preds[r['i']]; ph = physics_pred(r, plan, DEMO)
        err_p.append(abs(md - r['y'])); err_f.append(abs(ph - r['y'])); cov.append(l <= r['y'] <= h); seg.append(r['t'])
        r['pfn'] = (l, md, h, n); r['phys'] = ph
    seg = np.array(seg); err_p = np.array(err_p); err_f = np.array(err_f); cov = np.array(cov)
    first = seg < plan.seg[1][0]; rest = ~first
    res = dict(demo=DEMO, fracs=DEMO_FRACS, entered_sling=ENTERED_SLING, horizon_s=H, fleet_jobs=a.jobs, fleet_rows=len(fleet), ctx_base=len(base), fits=fits,
               first_move=dict(tabpfn_cm=round(100 * err_p[first].mean(), 1), physics_cm=round(100 * err_f[first].mean(), 1), coverage90=round(float(cov[first].mean()), 2)),
               later_moves=dict(tabpfn_cm=round(100 * err_p[rest].mean(), 1), physics_cm=round(100 * err_f[rest].mean(), 1), coverage90=round(float(cov[rest].mean()), 2)))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'mujoco_demo.json'), 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('first_move', 'later_moves')}))
    json.dump(dict(job=job, res=res), open(cache, 'w'))
    if not a.no_video:
        render(job, plan, res)


def render(job, plan, res):
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    m, d = mc.make(DEMO); r = mujoco.Renderer(m, 720, 1280)
    byt = {round(row['t'], 1): row for row in job}
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22); small = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = small = ImageFont.load_default()
    path = os.path.join(OUT, 'mujoco_demo.mp4'); w = iio.get_writer(path, fps=30, quality=8, macro_block_size=8)
    z = mc.RAIL_Z - 0.15 - DEMO['L1'] - 0.3 - DEMO['L2'] - 0.18
    state = {'row': None}

    def ghost(scn, x, half, rgba, size_y=0.26, h=0.19):
        if scn.ngeom >= scn.maxgeom: return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([half, size_y, h]), np.array([x, 0.0, z]), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    def strip(scn, x, half, rgba, zz=0.03, hy=0.9):
        if scn.ngeom >= scn.maxgeom: return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([half, hy, 0.02]), np.array([x, 0.0, zz]), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    byi = {row['i']: row for row in job}

    def on_frame(m_, d_, t):
        row = byt.get(round(math.floor(t * 10) / 10, 1)) or state['row']
        state['row'] = row
        r.update_scene(d_, camera='side'); scn = r.scene
        have = row is not None and 'pfn' in row
        if have:
            l, md, h, n = row['pfn']; xf = plan.ref(row['t'] + H)
            strip(scn, xf + (l + h) / 2, DEMO['beam'] / 2 + (h - l) / 2, (0.04, 0.48, 0.49, 0.22), 0.02, 1.1)   # 90% band on the floor
            strip(scn, xf + md, DEMO['beam'] / 2, (0.04, 0.48, 0.49, 0.85), 0.05, 0.55)                       # TabPFN
            strip(scn, xf + row['phys'], DEMO['beam'] / 2, (0.77, 0.35, 0.09, 0.85), 0.08, 0.3)              # physics
            ghost(scn, xf + md, DEMO['beam'] / 2, (0.04, 0.48, 0.49, 0.28))
        img = Image.fromarray(r.render()).convert('RGBA'); ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        dr.rounded_rectangle((14, 12, 700, 236), 12, fill=(255, 255, 255, 225))
        dr.text((28, 22), 'Where will the load be in 1 s?', fill=(15, 26, 34), font=font)
        dr.text((28, 54), f'new load: 400 kg beam on 2.6 m slings · anti-sway set for 0.5 m · t = {t:4.1f} s', fill=(88, 102, 114), font=small)
        if have:
            act = row['y']
            dr.text((28, 88), f'TabPFN        {md:+.2f} m   (90%: {l:+.2f} to {h:+.2f})   error {abs(md - act) * 100:4.1f} cm', fill=(10, 122, 126), font=small)
            dr.text((28, 114), f'physics       {row["phys"]:+.2f} m                                   error {abs(row["phys"] - act) * 100:4.1f} cm', fill=(196, 90, 23), font=small)
            dr.text((28, 140), f'actual        {act:+.2f} m   (load offset from trolley, revealed 1 s later)', fill=(15, 26, 34), font=small)
            dr.text((28, 172), f'rows from this load in TabPFN\'s table: {n}  ·  no retraining, rows added only', fill=(88, 102, 114), font=small)
            dr.text((28, 198), 'floor markers: teal = TabPFN (light band = 90% range), orange = physics', fill=(88, 102, 114), font=small)
        fm, lm = res['first_move'], res['later_moves']
        dr.rounded_rectangle((14, 664, 1266, 706), 10, fill=(255, 255, 255, 225))
        dr.text((28, 674), f"mean error 1 s ahead (simulation) · first move: TabPFN {fm['tabpfn_cm']} cm, physics {fm['physics_cm']} cm · later moves: TabPFN {lm['tabpfn_cm']} cm, physics {lm['physics_cm']} cm", fill=(88, 102, 114), font=small)
        w.append_data(np.asarray(Image.alpha_composite(img, ov).convert('RGB')))

    mc.run(DEMO, plan, on_frame=on_frame, fps=30)
    w.close(); print('wrote', path)


if __name__ == '__main__':
    main()
