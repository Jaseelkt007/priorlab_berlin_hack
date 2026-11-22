"""Option B: render a MuJoCo clip for any rope / sling / payload / speed, with TabPFN's prediction 1 s ahead.

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl \
      crane/.venv/bin/python crane/render_clip.py --rope 4.5 --sling 2.5 --payload 0.5 --speed 1.0
    ... --no-tabpfn          # physics only, no TabPFN key needed
    ... --out my_clip.mp4    # default: crane/results/clip_<params>.mp4

One move of 10 m (2 s hold before, 10 s after arrival), no anti-sway, speed and acceleration scaled by --speed.
TabPFN context: the 3,000 fleet rows saved by playground_precompute.py (run that first).
Teal ghost and floor band = TabPFN's predicted load position in 1 s (90% range); orange floor marker = physics model
(single pendulum with the operator-entered 0.5 m sling).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_crane as mc
import mujoco
from mujoco_demo import FEAT, H, physics_pred, rows_from_log
from playground_precompute import Delayed, cond_of, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rope', type=float, default=4.5); ap.add_argument('--sling', type=float, default=2.5)
    ap.add_argument('--payload', type=float, default=0.5, help='tonnes'); ap.add_argument('--speed', type=float, default=1.0, help='0.3-1.0')
    ap.add_argument('--beam', type=float, default=2.0); ap.add_argument('--no-tabpfn', action='store_true'); ap.add_argument('--out')
    a = ap.parse_args()
    c = cond_of(a.rope, a.sling, a.payload); c['beam'] = a.beam; plan = Delayed(a.speed)
    m, d = mc.make(c); tid = m.body('trolley').id
    M = float(m.body_subtreemass[tid]); kp, kv = M * 25.0, M * 10.0
    log, qs, k = [], [], 0
    while d.time < plan.T:
        xr, vr, ar = plan.state(d.time)
        d.ctrl[0] = M * ar + kp * (xr - d.qpos[0]) + kv * (vr - d.qvel[0])
        mujoco.mj_step(m, d); k += 1
        if k % 50 == 0: log.append((d.time, d.xpos[tid][0], xr, d.site_xpos[0][0] - d.xpos[tid][0]))
        if k % 17 == 0: qs.append((d.time, d.qpos.copy()))                       # ~30 fps
    log = np.array(log); rows = rows_from_log(log, plan, c, 'clip')
    for r in rows: r['phys'] = physics_pred(r, plan, c)
    if not a.no_tabpfn:
        from tabpfn import TabPFNRegressor
        ctx = json.load(open(os.path.join(OUT, 'fleet_rows.json')))
        Xc = np.array([[r['f'][f] for f in FEAT] for r in ctx]); Xq = np.array([[r['f'][f] for f in FEAT] for r in rows])
        lo, md, hi = (np.asarray(v) for v in TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(Xc, np.array([r['y'] for r in ctx]))
                      .predict(Xq, output_type='quantiles', quantiles=[0.05, 0.5, 0.95]))
        for r, l, mm, h in zip(rows, lo, md, hi): r['pfn'] = (float(l), float(mm), float(h))
        e = np.mean([abs(r['pfn'][1] - r['y']) for r in rows]) * 100
    ep = np.mean([abs(r['phys'] - r['y']) for r in rows]) * 100
    print(f"rows {len(rows)}  physics error {ep:.1f} cm" + ('' if a.no_tabpfn else f"  TabPFN error {e:.1f} cm"))

    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    try: font = ImageFont.truetype('DejaVuSans.ttf', 22); small = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError: font = small = ImageFont.load_default()
    path = a.out or os.path.join(OUT, f'clip_rope{a.rope}_sling{a.sling}_load{a.payload}t_speed{int(a.speed * 100)}.mp4')
    r_ = mujoco.Renderer(m, 720, 1280); w = iio.get_writer(path, fps=30, quality=8, macro_block_size=8)
    byt = {round(r['t'], 1): r for r in rows}; z = mc.RAIL_Z - 0.15 - c['L1'] - 0.3 - c['L2'] - 0.18

    def box(scn, x, hx, hy, hz, zz, rgba):
        if scn.ngeom >= scn.maxgeom: return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([hx, hy, hz]), np.array([x, 0.0, zz]), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

    for t, q in qs:
        d.qpos[:] = q; mujoco.mj_forward(m, d); r_.update_scene(d, camera='side'); scn = r_.scene
        row = byt.get(round(math.floor(t * 10) / 10, 1)); xf = plan.ref(t + H)
        if row:
            box(scn, xf + row['phys'], c['beam'] / 2, 0.3, 0.02, 0.08, (0.77, 0.35, 0.09, 0.85))
            if 'pfn' in row:
                l, md_, h = row['pfn']
                box(scn, xf + (l + h) / 2, c['beam'] / 2 + (h - l) / 2, 1.1, 0.02, 0.02, (0.04, 0.48, 0.49, 0.22))
                box(scn, xf + md_, c['beam'] / 2, 0.55, 0.02, 0.05, (0.04, 0.48, 0.49, 0.85))
                box(scn, xf + md_, c['beam'] / 2, 0.26, 0.19, z, (0.04, 0.48, 0.49, 0.28))
        img = Image.fromarray(r_.render()).convert('RGBA'); ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
        dr.rounded_rectangle((14, 12, 760, 196), 12, fill=(255, 255, 255, 225))
        dr.text((28, 22), 'Where will the load be in 1 s?', fill=(15, 26, 34), font=font)
        dr.text((28, 54), f'rope {a.rope} m · sling {a.sling} m · payload {a.payload} t · speed {int(a.speed * 100)}% · t = {t:4.1f} s', fill=(88, 102, 114), font=small)
        if row:
            if 'pfn' in row:
                dr.text((28, 88), f"TabPFN   {row['pfn'][1]:+.2f} m  (90%: {row['pfn'][0]:+.2f} to {row['pfn'][2]:+.2f})  error {abs(row['pfn'][1] - row['y']) * 100:4.1f} cm", fill=(10, 122, 126), font=small)
            dr.text((28, 114), f"physics  {row['phys']:+.2f} m  error {abs(row['phys'] - row['y']) * 100:4.1f} cm", fill=(196, 90, 23), font=small)
            dr.text((28, 140), f"actual   {row['y']:+.2f} m  (load offset from trolley, revealed 1 s later) · simulation", fill=(15, 26, 34), font=small)
        w.append_data(np.asarray(Image.alpha_composite(img, ov).convert('RGB')))
    w.close(); print('wrote', path)


if __name__ == '__main__':
    main()
