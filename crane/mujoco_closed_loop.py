"""Closed loop in MuJoCo: the crane uses TabPFN to DECIDE each move, versus industry anti-sway.

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_closed_loop.py
    ... --render-only     # re-render the video from the saved run

Every move is one job: a freshly hooked load travels 10 m from bay A to bay B and must settle within
+-5 cm for 1 s before set-down. If its residual swing exceeds the 0.45 m rack clearance, that is a collision.

Strategies, on the same 12 new loads (long slings 1.8-3.0 m, light beams; never seen in the fleet logs):
  industry : full speed, ZV anti-sway tuned for the operator-entered 0.5 m sling
  slow     : 40% speed and acceleration, same anti-sway
  tabpfn   : before each move, ONE batched TabPFN call scores 28 candidates (7 speeds x 4 shaper tunings):
             predicted residual swing (median, 99th percentile) and settle time. It takes the fastest expected
             cycle whose 99th-percentile swing stays inside the clearance. After the move its real outcome
             is added to the table (no retraining).
Context at the start: 150 simulated commissioning moves on short slings (0.5-1.0 m), heavy loads, random speeds/tunings.
Row: rope L1, payload (load cell), lift-off sway period (sensor, 3% noise), speed fraction, shaper tuning error.
Outputs: crane/results/closed_loop.json, closed_loop_run.json (replay), closed_loop.mp4
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
D = mc.BAYS[1] - mc.BAYS[0]
VMAX, AMAX = 1.2, 0.4
TOL, HOLD, CAP = 0.05, 1.0, 30.0
SPEEDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TUNES = [0.5, 1.5, 2.5, 3.5]          # sling length the shaper assumes, m
ENTERED = 0.5
HIT_PENALTY = 20.0
HOOK_DROP, LOAD_HALF = 0.3, 0.18      # hook block and half load height (geometry in mujoco_crane)


def shaped_state(t, frac, tune_L):
    """Trapezoid move (speed and accel scaled by frac), ZV-shaped for a pendulum of length tune_L.
    ZV (Singer & Seering): two half-size copies of the command, half a period apart."""
    v, a = frac * VMAX, frac * AMAX; ta = v / a; T = D / v + ta
    def one(s):
        if s <= 0: return 0.0, 0.0, 0.0
        if s < ta: return 0.5 * a * s * s, a * s, a
        if s < T - ta: return 0.5 * a * ta * ta + v * (s - ta), v, 0.0
        if s < T: r = T - s; return D - 0.5 * a * r * r, a * r, -a
        return D, 0.0, 0.0
    dz = math.pi * math.sqrt(tune_L / mc.G)
    p1, p2 = one(t), one(t - dz)
    return tuple(0.5 * (u + w) for u, w in zip(p1, p2)), T + dz


def period_true(c):
    w1, _ = mc.modes(c['L1'] + HOOK_DROP, c['L2'] + LOAD_HALF, c['m_load'] / c['m_hook'])
    return 2 * math.pi / w1


def simulate_move(c, frac, tune, record=False):
    """One job. Returns outcome dict; with record=True also qpos snapshots at 10 Hz for rendering."""
    m, d = mc.make(c); tid = m.body('trolley').id
    M = float(m.body_subtreemass[tid]); kp, kv = M * 25.0, M * 10.0
    tune_L = c['L1'] + HOOK_DROP + tune + LOAD_HALF
    _, Tm = shaped_state(0.0, frac, tune_L)
    frames, amp, inTol, k, nfr = [], 0.0, 0.0, 0, int(round(0.1 / mc.DT))
    off = lambda: d.site_xpos[0][0] - d.xpos[tid][0]
    while True:
        t = d.time
        if t < Tm: xr, vr, ar = shaped_state(t, frac, tune_L)[0]
        else: xr, vr, ar = D, 0.0, 0.0
        d.ctrl[0] = M * ar + kp * (mc.BAYS[0] + xr - d.qpos[0]) + kv * (vr - d.qvel[0])
        mujoco.mj_step(m, d); k += 1
        if d.time >= Tm:
            o = abs(off()); amp = max(amp, o)
            inTol = inTol + mc.DT if o < TOL else 0.0
            if inTol >= HOLD or d.time - Tm >= CAP: break
        if record and k % nfr == 0: frames.append(d.qpos.copy().tolist())
    settle = d.time - Tm
    amp = float(amp)
    return dict(move_s=Tm, settle_s=settle, amp=amp, hit=bool(amp > mc.CLEAR), cycle_s=Tm + settle + (HIT_PENALTY if amp > mc.CLEAR else 0.0), frames=frames)


def feats(c, meas, frac, tune):
    Ts = 2 * math.pi * math.sqrt((c['L1'] + HOOK_DROP + tune + LOAD_HALF) / mc.G)
    return [c['L1'], c['m_load'] / 1000, meas, frac, Ts / meas - 1]


def decide(ctx, c, meas):
    from tabpfn import TabPFNRegressor
    X = np.array([r['x'] for r in ctx]); cands = [(f, t) for f in SPEEDS for t in TUNES]
    Xc = np.array([feats(c, meas, f, t) for f, t in cands])
    t0 = time.time()
    ma = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(X, np.array([r['amp'] for r in ctx]))
    q50, q99 = (np.asarray(v) for v in ma.predict(Xc, output_type='quantiles', quantiles=[0.5, 0.99]))
    ms = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(X, np.array([r['settle'] for r in ctx]))
    s50 = np.asarray(ms.predict(Xc, output_type='median'))
    wall = time.time() - t0
    rows = []
    for (f, t), a50, a99, s in zip(cands, q50, q99, s50):
        mv = shaped_state(0.0, f, c['L1'] + HOOK_DROP + t + LOAD_HALF)[1]
        rows.append(dict(frac=f, tune=t, amp50=float(a50), amp99=float(a99), settle=float(max(0.0, s)), move_s=mv, total=mv + float(max(0.0, s)), safe=bool(a99 < mc.CLEAR)))
    ok = [r for r in rows if r['safe']]
    best = min(ok, key=lambda r: r['total']) if ok else min((r for r in rows if r['frac'] == SPEEDS[0]), key=lambda r: r['amp99'])
    return best, rows, wall


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--render-only', action='store_true'); ap.add_argument('--moves', type=int, default=12)
    a = ap.parse_args()
    runf = os.path.join(OUT, 'closed_loop_run.json')
    if a.render_only:
        render(json.load(open(runf))); return
    rng = random.Random(7); t0 = time.time()
    fleet = []
    for _ in range(150):
        c = dict(L1=rng.uniform(3, 6), L2=rng.uniform(0.5, 1.0), m_hook=250.0, m_load=rng.uniform(1000, 3000), beam=rng.uniform(1.0, 2.0))
        meas = period_true(c) * (1 + 0.03 * rng.gauss(0, 1)); f, t = rng.choice(SPEEDS), rng.choice(TUNES)
        o = simulate_move(c, f, t); fleet.append(dict(x=feats(c, meas, f, t), amp=o['amp'], settle=o['settle_s']))
    print(f'fleet: {len(fleet)} commissioning moves in {time.time() - t0:.0f} s; hits {sum(r["amp"] > mc.CLEAR for r in fleet)}')
    jobs = []
    for _ in range(a.moves):
        c = dict(L1=rng.uniform(3, 6), L2=rng.uniform(1.8, 3.0), m_hook=250.0, m_load=rng.uniform(300, 800), beam=rng.uniform(1.5, 2.5))
        jobs.append(dict(c=c, meas=period_true(c) * (1 + 0.03 * rng.gauss(0, 1))))
    run = {'jobs': jobs, 'lanes': {}}
    for name, pick in (('industry', lambda j: (1.0, ENTERED)), ('slow', lambda j: (0.4, ENTERED))):
        run['lanes'][name] = [dict(frac=f, tune=t, **simulate_move(j['c'], f, t, record=True)) for j in jobs for f, t in [pick(j)]]
        L = run['lanes'][name]; print(f'{name:8s} total {sum(r["cycle_s"] for r in L):6.0f} s  hits {sum(r["hit"] for r in L)}  mean settle {np.mean([r["settle_s"] for r in L]):.1f} s')
    ctx = list(fleet); lane = []
    for k, j in enumerate(jobs):
        best, rows, wall = decide(ctx, j['c'], j['meas'])
        o = simulate_move(j['c'], best['frac'], best['tune'], record=True)
        lane.append(dict(frac=best['frac'], tune=best['tune'], pred_amp50=best['amp50'], pred_amp99=best['amp99'], pred_settle=best['settle'], cands=rows, decide_s=wall, own_rows=k, **o))
        ctx.append(dict(x=feats(j['c'], j['meas'], best['frac'], best['tune']), amp=o['amp'], settle=o['settle_s']))
        print(f'  move {k + 1:2d}: sling {j["c"]["L2"]:.1f} m  chose {int(best["frac"] * 100)}% / tune {best["tune"]} m  pred swing {best["amp50"]:.2f} (99%: {best["amp99"]:.2f})  actual {o["amp"]:.2f}  settle {o["settle_s"]:.1f} s  {"HIT" if o["hit"] else ""}  ({wall:.0f} s)')
    run['lanes']['tabpfn'] = lane
    summ = {}
    for name, L in run['lanes'].items():
        summ[name] = dict(total_s=round(sum(r['cycle_s'] for r in L), 1), hits=int(sum(r['hit'] for r in L)), mean_settle_s=round(float(np.mean([r['settle_s'] for r in L])), 1),
                          mean_move_s=round(float(np.mean([r['move_s'] for r in L])), 1), moves_per_hour=round(3600 * len(L) / sum(r['cycle_s'] for r in L), 1))
    print(json.dumps(summ, indent=1))
    res = dict(summary=summ, moves=a.moves, fleet_moves=len(fleet), clearance_m=mc.CLEAR, entered_sling_m=ENTERED,
               tabpfn_choices=[dict(move=k + 1, sling_m=round(j['c']['L2'], 2), frac=r['frac'], tune=r['tune'], pred_amp50=round(r['pred_amp50'], 3), pred_amp99=round(r['pred_amp99'], 3),
                                    actual_amp=round(r['amp'], 3), settle_s=round(r['settle_s'], 1), hit=bool(r['hit'])) for k, (j, r) in enumerate(zip(jobs, lane))])
    json.dump(res, open(os.path.join(OUT, 'closed_loop.json'), 'w'), indent=1)
    json.dump(run, open(runf, 'w'))
    render(run)


def render(run, speedup=3):
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    try: font = ImageFont.truetype('DejaVuSans.ttf', 20); small = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError: font = small = ImageFont.load_default()
    lanes = [('industry', 'Industry anti-sway (set for 0.5 m sling)', (196, 90, 23)), ('tabpfn', 'TabPFN decides speed + tuning', (10, 122, 126))]
    W, Hh = 960, 540
    seqs = {}
    for key, title, col in lanes:            # frames: (image array) per 0.1 s of sim time, then hold
        imgs = []; clock = 0.0; hits = 0
        for k, (j, r) in enumerate(zip(run['jobs'], run['lanes'][key])):
            m, d = mc.make(j['c']); ren = mujoco.Renderer(m, Hh, W)
            rack_ids = [g for g in range(m.ngeom) if m.geom_size[g][0] == 0.15 and abs(m.geom_size[g][1] - 1.0) < 1e-6]
            t_move = r['move_s']
            for i, q in enumerate(r['frames']):
                if i % speedup: continue
                t = (i + 1) * 0.1
                d.qpos[:] = q; mujoco.mj_forward(m, d)
                hit_now = r['hit'] and t > t_move and abs(d.site_xpos[0][0] - d.xpos[1][0]) > mc.CLEAR * 0.98
                for g in rack_ids: m.geom_rgba[g] = (0.85, 0.2, 0.15, 1) if hit_now else (0.6, 0.65, 0.7, 1)
                ren.update_scene(d, camera='side')
                img = Image.fromarray(ren.render()).convert('RGBA'); ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
                dr.rounded_rectangle((10, 8, W - 10, 128), 10, fill=(255, 255, 255, 225))
                dr.text((22, 14), title, fill=col, font=font)
                phase = 'moving' if t < t_move else 'waiting for the swing to settle'
                dr.text((22, 44), f"move {k + 1}/12 · sling {j['c']['L2']:.1f} m (hidden) · {int(r['frac'] * 100)}% speed, shaper for {r['tune']} m sling", fill=(60, 70, 80), font=small)
                if key == 'tabpfn':
                    dr.text((22, 66), f"TabPFN predicted swing {r['pred_amp50']:.2f} m (99%: {r['pred_amp99']:.2f}) · rows from new loads: {r['own_rows']}", fill=col, font=small)
                else:
                    dr.text((22, 66), 'fixed choice for every load', fill=(60, 70, 80), font=small)
                dr.text((22, 90), f"{phase} · this move {t:4.1f} s · total {clock + t:5.0f} s · collisions {hits + (1 if hit_now or (r['hit'] and t > t_move + r['settle_s'] - 0.2) else 0)}", fill=(15, 26, 34), font=small)
                imgs.append(np.asarray(Image.alpha_composite(img, ov).convert('RGB')))
            clock += r['cycle_s']; hits += int(r['hit'])
            ren.close()
        seqs[key] = (imgs, clock, hits)
    n = max(len(s[0]) for s in seqs.values())
    path = os.path.join(OUT, 'closed_loop.mp4'); w = iio.get_writer(path, fps=30, quality=8, macro_block_size=8)
    for i in range(n):
        parts = []
        for key, title, col in lanes:
            imgs, total, hits = seqs[key]
            if i < len(imgs): parts.append(imgs[i])
            else:
                im = Image.fromarray(imgs[-1]).convert('RGBA'); ov = Image.new('RGBA', im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
                dr.rounded_rectangle((W // 2 - 230, Hh // 2 - 50, W // 2 + 230, Hh // 2 + 50), 12, fill=(255, 255, 255, 240))
                dr.text((W // 2 - 210, Hh // 2 - 36), f'DONE: 12 moves in {total:.0f} s', fill=col, font=font)
                dr.text((W // 2 - 210, Hh // 2 + 2), f'collisions: {hits}', fill=(15, 26, 34), font=font)
                parts.append(np.asarray(Image.alpha_composite(im, ov).convert('RGB')))
        w.append_data(np.concatenate(parts, axis=1))
    w.close(); print('wrote', path, f'{n / 30:.0f} s at {speedup}x')


if __name__ == '__main__':
    main()
