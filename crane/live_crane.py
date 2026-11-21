"""LIVE MuJoCo crane with TabPFN predicting where the load will be 1 s ahead, in real time.

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=glfw crane/.venv/bin/python crane/live_crane.py
    ... --rope 4.5 --sling 2.5 --payload 0.5 --speed 1.0      # starting values
    ... --headless 60                                          # no window: run 60 s and print errors (for testing)

A 3D window opens (on WSL via WSLg). The crane shuttles between the two bays on its own; between moves it waits 4 s.
Change the crane while it runs; a change is applied before the next move (the load is re-hooked):
    Up / Down          speed and acceleration  +/- 10 %   (30-100 %)
    Page Up / Down     sling length            +/- 0.5 m  (0.5-3.0 m, hidden from TabPFN)
    Home / End         rope length             +/- 0.5 m  (3.0-7.0 m)
    Insert / Delete    payload                 +/- 0.5 t  (0.5-3.0 t)
or type in this terminal:  rope 5 | sling 2.5 | payload 0.5 | speed 0.8 | quit
Drawn in the window: teal ghost beam = TabPFN's predicted load position 1 s from now, light teal floor band = its 90%
range, orange floor marker = physics model (single pendulum with the operator-entered 0.5 m sling).
TabPFN context: 3,000 rows from 40 simulated fleet jobs logged exactly like the live crane (4 back-and-forth moves
 with pauses, random rope/sling/payload/speed), cached in results/live_fleet_rows.json (built on first run, ~1 min).
Predictions run in a background thread (one row per 0.1 s), so the simulation stays real-time.
"""
import argparse
import collections
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_crane as mc
import mujoco
import mujoco.viewer
from mujoco_demo import FEAT, H, ENTERED_SLING

OUT = os.path.join(mc.HERE, 'results')
D = mc.BAYS[1] - mc.BAYS[0]
VMAX, AMAX, PAUSE = 1.2, 0.4, 6.0
LIMITS = dict(rope=(3.0, 7.0), sling=(0.5, 3.0), payload=(0.5, 3.0), speed=(0.3, 1.0))
KEYS = {265: ('speed', +0.1), 264: ('speed', -0.1), 266: ('sling', +0.5), 267: ('sling', -0.5),
        268: ('rope', +0.5), 269: ('rope', -0.5), 260: ('payload', +0.5), 261: ('payload', -0.5)}
COPY = ['body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass', 'body_inertia', 'body_subtreemass',
        'geom_pos', 'geom_quat', 'geom_size', 'geom_rbound', 'geom_aabb', 'dof_damping', 'dof_invweight0', 'body_invweight0']


class Mover:
    """Back-and-forth moves. state(t) -> commanded (x, v, a); the controller knows its own plan."""
    def __init__(self, x0):
        self.x0 = x0; self.t0 = None; self.frac = 1.0; self.dir = 1
    def start(self, t, frac):
        self.t0, self.frac = t, frac; self.dir = 1 if self.x0 < sum(mc.BAYS) / 2 else -1
        v, a = frac * VMAX, frac * AMAX; self.v, self.a, self.ta = v, a, v / a; self.T = D / v + v / a
    def state(self, t):
        if self.t0 is None or t < self.t0: return self.x0, 0.0, 0.0
        s = t - self.t0; v, a, ta, T, d = self.v, self.a, self.ta, self.T, self.dir
        if s < ta: return self.x0 + d * 0.5 * a * s * s, d * a * s, d * a
        if s < T - ta: return self.x0 + d * (0.5 * a * ta * ta + v * (s - ta)), d * v, 0.0
        if s < T: r = T - s; return self.x0 + d * (D - 0.5 * a * r * r), d * a * r, -d * a
        return self.x0 + d * D, 0.0, 0.0
    def ref(self, t): return self.state(t)[0]
    def done(self, t): return self.t0 is not None and t >= self.t0 + self.T


def physics_pred(f, mover, t, L1):
    L = L1 + 0.3 + ENTERED_SLING + 0.18
    th = math.asin(max(-0.99, min(0.99, f['off'] / L))); w = f['doff'] / L / max(0.2, math.cos(th)); dt = 0.005
    for k in range(int(H / dt)):
        a = mover.state(t + k * dt)[2]; w += (-(mc.G / L) * math.sin(th) - (a / L) * math.cos(th)) * dt; th += w * dt
    return L * math.sin(th)


def features(hist, mover, t, c, sign=1):
    """sign = -1 mirrors a right-to-left move into the left-to-right frame the fleet rows were logged in
    (x -> 12 - x, every displacement, speed and swing negated). The physics is mirror-symmetric, so this is exact."""
    t_, x, off = (np.array(v) for v in zip(*hist))
    if sign < 0: x, off = sum(mc.BAYS) - x, -off
    return dict(off=off[-1], off_m02=off[-3], off_m05=off[-6], doff=(off[-1] - off[-2]) / 0.1, amp2s=float(np.abs(off[-21:]).max()),
                x=x[-1], dx=(x[-1] - x[-2]) / 0.1, ddx=(x[-1] - 2 * x[-2] + x[-3]) / 0.01, L1=c['L1'], payload_t=c['m_load'] / 1000,
                plan_dx05=sign * (mover.ref(t + 0.5) - mover.ref(t)), plan_dx10=sign * (mover.ref(t + 1.0) - mover.ref(t)))


def build_fleet(n_jobs=40, moves=4, seed=5):
    """Fleet table logged exactly like the live crane: back-and-forth moves with PAUSE s at each bay (so moves can
    start with leftover swing), random rope 3-7 m, sling 0.5-3 m, payload 0.5-3 t, speed 30-100 %. Rows are in the
    mirrored (left-to-right) frame, target = sign * load offset 1 s later."""
    import random
    rng = random.Random(seed); rows = []
    for j in range(n_jobs):
        c = dict(L1=rng.uniform(3, 7), L2=rng.uniform(0.5, 3.0), m_hook=250.0, m_load=rng.uniform(500, 3000), beam=2.0)
        m, d = mc.make(c); tid = m.body('trolley').id; M = float(m.body_subtreemass[tid])
        mover = Mover(mc.BAYS[0]); hist = collections.deque(maxlen=40); pending_rows = []; done_moves = 0; wait_until = 1.0; last = -1.0
        while done_moves < moves or d.time < wait_until:
            t = d.time
            if mover.t0 is not None and mover.done(t):
                end = mover.t0 + mover.T
                if wait_until < end + PAUSE: wait_until = end + PAUSE; done_moves += 1
                if t >= wait_until and done_moves < moves: mover.x0 = mover.ref(t); mover.t0 = None
            if mover.t0 is None and t >= wait_until: mover.start(t, rng.uniform(0.3, 1.0))
            xr, vr, ar = mover.state(t)
            d.ctrl[0] = M * ar + M * 25 * (xr - d.qpos[0]) + M * 10 * (vr - d.qvel[0]); mujoco.mj_step(m, d)
            if d.time - last >= 0.1 - 1e-9:
                last = d.time; off = d.site_xpos[0][0] - d.xpos[tid][0]; hist.append((d.time, d.xpos[tid][0], off))
                for r in pending_rows:
                    if r['y'] is None and d.time >= r['t'] + H - 1e-6: r['y'] = r['sign'] * off
                if len(hist) >= 21:
                    pending_rows.append(dict(t=d.time, sign=mover.dir, f=features(list(hist), mover, d.time, c, mover.dir), y=None))
        rows += [dict(f=r['f'], y=r['y']) for r in pending_rows if r['y'] is not None]
        print(f'  fleet job {j + 1}/{n_jobs}: {len(rows)} rows', end='\r', flush=True)
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rope', type=float, default=4.5); ap.add_argument('--sling', type=float, default=2.5)
    ap.add_argument('--payload', type=float, default=0.5); ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--headless', type=float, default=0, help='run this many sim seconds without a window')
    ap.add_argument('--rebuild-fleet', action='store_true', help='re-simulate results/live_fleet_rows.json')
    a = ap.parse_args()
    P = dict(rope=a.rope, sling=a.sling, payload=a.payload, speed=a.speed); pending = dict(P); lock = threading.Lock()
    cond = lambda p: dict(L1=p['rope'], L2=p['sling'], m_hook=250.0, m_load=p['payload'] * 1000, beam=2.0)

    from tabpfn import TabPFNRegressor
    fpath = os.path.join(OUT, 'live_fleet_rows.json')
    if a.rebuild_fleet or not os.path.exists(fpath):
        print('simulating the live fleet table (once, ~1 min) ...', flush=True)
        allrows = build_fleet(); import random as _r; _r.seed(0)
        json.dump(_r.sample(allrows, min(3000, len(allrows))), open(fpath, 'w'))
    fleet = json.load(open(fpath))
    print('fitting TabPFN on', len(fleet), 'fleet rows ...', flush=True)
    model = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(np.array([[r['f'][k] for k in FEAT] for r in fleet]), np.array([r['y'] for r in fleet]))
    model.predict(np.zeros((1, len(FEAT))))                                   # warm-up
    req, res = queue.Queue(maxsize=1), {'pred': None, 'ms': 0.0}

    def worker():
        while True:
            item = req.get()
            if item is None: return
            t, row, sign = item; t0 = time.time()
            lo, md, hi = (float(np.asarray(v)[0]) for v in model.predict(np.array([[row[k] for k in FEAT]]), output_type='quantiles', quantiles=[0.05, 0.5, 0.95]))
            if sign < 0: lo, md, hi = -hi, -md, -lo                                # mirror the answer back
            res['pred'] = (t, lo, md, hi); res['ms'] = 1000 * (time.time() - t0)
    threading.Thread(target=worker, daemon=True).start()

    c = cond(P); m, d = mc.make(c); tid = m.body('trolley').id
    mover = Mover(mc.BAYS[0]); hist = collections.deque(maxlen=40); preds = {}; phys_at = {}; errs = collections.deque(maxlen=100)
    state = dict(wait_until=1.0, phys=None, last_row_t=-1.0, quit=False)

    def apply(p):
        nonlocal c
        c = cond(p); m2 = mujoco.MjModel.from_xml_string(mc.build_xml(**c))
        for name in COPY: getattr(m, name)[:] = getattr(m2, name)
        d.qpos[1:] = 0; d.qvel[1:] = 0; hist.clear(); preds.clear(); phys_at.clear(); errs.clear(); res['pred'] = None; mujoco.mj_forward(m, d)

    def on_key(k):
        if k in KEYS:
            n, step = KEYS[k]; lo_, hi_ = LIMITS[n]
            with lock: pending[n] = round(min(hi_, max(lo_, pending[n] + step)), 2)
            print('next move:', pending, flush=True)

    def stdin_reader():
        for line in sys.stdin:
            w = line.split()
            if not w: continue
            if w[0] == 'quit': state['quit'] = True; return
            if len(w) == 2 and w[0] in LIMITS:
                try: v = float(w[1])
                except ValueError: continue
                lo_, hi_ = LIMITS[w[0]]
                with lock: pending[w[0]] = min(hi_, max(lo_, v))
                print('next move:', pending, flush=True)
    if not a.headless: threading.Thread(target=stdin_reader, daemon=True).start()

    def ghost(scn, typ, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom: return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], typ, np.array(size, float), np.array(pos, float), np.eye(3).flatten(), np.array(rgba, dtype=np.float32)); scn.ngeom += 1

    def step_once():
        t = d.time
        if mover.t0 is not None and mover.done(t):                         # arrived: hold PAUSE s at the bay
            end = mover.t0 + mover.T
            if state['wait_until'] < end + PAUSE: state['wait_until'] = end + PAUSE
            if t >= state['wait_until']: mover.x0 = mover.ref(t); mover.t0 = None
        if mover.t0 is None and t >= state['wait_until']:                  # next move (apply pending changes first)
            with lock: new = dict(pending)
            if new != P: P.update(new); apply(P)
            mover.start(t, P['speed'])
        M = float(m.body_subtreemass[tid]); xr, vr, ar = mover.state(t)
        d.ctrl[0] = M * ar + M * 25 * (xr - d.qpos[0]) + M * 10 * (vr - d.qvel[0])
        mujoco.mj_step(m, d)
        if d.time - state['last_row_t'] >= 0.1 - 1e-9:                         # 10 Hz rows
            state['last_row_t'] = d.time; off = d.site_xpos[0][0] - d.xpos[tid][0]
            hist.append((d.time, d.xpos[tid][0], off))
            tt = round(d.time, 1)
            if tt - 1.0 in [round(k, 1) for k in preds]:                       # outcome of a prediction made 1 s ago
                p = preds.pop(round(tt - 1.0, 1)); errs.append((abs(p[0] - off), abs(p[1] - off) if p[1] is not None else None))
            if len(hist) >= 21:
                f = features(list(hist), mover, d.time, c)                        # world frame, for physics
                state['phys'] = (d.time, physics_pred(f, mover, d.time, c['L1'])); phys_at[round(d.time, 1)] = state['phys'][1]
                sign = mover.dir                     # direction of the current or last move (+1 before the first)
                try: req.put_nowait((d.time, features(list(hist), mover, d.time, c, sign), sign))
                except queue.Full: pass
                r = res['pred']
                if r: preds[round(r[0], 1)] = (r[2], phys_at.get(round(r[0], 1)))

    def overlay(viewer):
        scn = viewer.user_scn; scn.ngeom = 0; r = res['pred']; t = d.time
        zl = mc.RAIL_Z - 0.15 - c['L1'] - 0.3 - c['L2'] - 0.18
        if r and t - r[0] < 0.5:
            xf = mover.ref(r[0] + H); _, lo, md, hi = r
            ghost(scn, mujoco.mjtGeom.mjGEOM_BOX, [1.0 + (hi - lo) / 2, 1.1, 0.02], [xf + (lo + hi) / 2, 0, 0.02], (0.04, 0.48, 0.49, 0.25))
            ghost(scn, mujoco.mjtGeom.mjGEOM_BOX, [1.0, 0.55, 0.02], [xf + md, 0, 0.05], (0.04, 0.48, 0.49, 0.9))
            ghost(scn, mujoco.mjtGeom.mjGEOM_BOX, [1.0, 0.26, 0.19], [xf + md, 0, zl], (0.04, 0.48, 0.49, 0.3))
        if state['phys']:
            ghost(scn, mujoco.mjtGeom.mjGEOM_BOX, [1.0, 0.3, 0.02], [mover.ref(state['phys'][0] + H) + state['phys'][1], 0, 0.08], (0.77, 0.35, 0.09, 0.9))
        e = list(errs); ep = np.mean([x[0] for x in e]) * 100 if e else float('nan'); ef = [x[1] for x in e if x[1] is not None]; ef = np.mean(ef) * 100 if ef else float('nan')
        with lock: nxt = dict(pending)
        left = (f"Where will the load be in 1 s?\nrope {P['rope']} m  sling {P['sling']} m (hidden)  payload {P['payload']} t  speed {int(P['speed'] * 100)}%\n"
                f"TabPFN (teal): {('%+.2f m  90%%: %+.2f..%+.2f' % (r[2], r[1], r[3])) if r else '-'}\nphysics (orange): {('%+.2f m' % state['phys'][1]) if state['phys'] else '-'}\n"
                f"mean error, last 10 s: TabPFN {ep:.1f} cm  physics {ef:.1f} cm\nTabPFN {res['ms']:.0f} ms per prediction   sim speed {state.get('rt', 0):.2f}x real time")
        keys = (f"next move: rope {nxt['rope']}  sling {nxt['sling']}  payload {nxt['payload']}  speed {int(nxt['speed'] * 100)}%\n"
                "Up/Down speed  PgUp/PgDn sling  Home/End rope  Ins/Del payload")
        return [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, None),
                (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, keys, None)]

    if a.headless:
        t0 = time.time()
        while d.time < a.headless:
            step_once()
            if d.time > time.time() - t0: time.sleep(max(0.0, d.time - (time.time() - t0)))   # real time, like the window
            if abs(d.time % 5.0) < mc.DT / 2 and errs:
                e = list(errs); ef = [x[1] for x in e if x[1] is not None]
                print(f"t={d.time:5.1f}  {P}  last 10 s: TabPFN {np.mean([x[0] for x in e]) * 100:.1f} cm  physics {(np.mean(ef) * 100 if ef else float('nan')):.1f} cm  pred {res['ms']:.0f} ms", flush=True)
            if d.time > 40 and P['sling'] == a.sling:                          # exercise a live parameter change
                with lock: pending['sling'] = 1.0; pending['speed'] = 0.7
        print(f'headless: {a.headless:.0f} s sim in {time.time() - t0:.0f} s wall'); req.put(None); time.sleep(0.5)
        return


    with mujoco.viewer.launch_passive(m, d, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED; viewer.cam.fixedcamid = m.camera('side').id
        print('window open: close it, or type quit, to stop', flush=True)
        wall0, sim0 = time.time(), d.time; frames = collections.deque(maxlen=60)
        while viewer.is_running() and not state['quit']:
            f0 = time.time(); target = f0 - wall0
            with viewer.lock():                                               # one lock per frame, at most 0.1 s of physics
                n = 0
                while d.time - sim0 < target and n < 50: step_once(); n += 1
                texts = overlay(viewer)
            viewer.set_texts(texts)                                           # takes the viewer lock itself
            lag = target - (d.time - sim0)
            if lag > 0.25: wall0 += lag                                       # too slow: drop time instead of freezing
            viewer.sync(); frames.append((time.time(), d.time))
            if len(frames) > 10: state['rt'] = (frames[-1][1] - frames[0][1]) / max(1e-6, frames[-1][0] - frames[0][0])
            if os.environ.get('LIVE_DEBUG') and int(time.time() - wall0) % 2 == 0 and time.time() - state.get('dbg', 0) > 1.5:
                state['dbg'] = time.time(); print(f"[debug] wall {time.time() - wall0:5.1f}  sim {d.time:5.1f}  steps/frame {n}  rt {state.get('rt', 0):.2f}  quit {state['quit']}", flush=True)
            time.sleep(0.002)
    req.put(None)
    e = list(errs); ef = [x[1] for x in e if x[1] is not None]
    print(f"sim speed {state.get('rt', 0):.2f}x real time", flush=True)
    if e: print(f"last 10 s: TabPFN {np.mean([x[0] for x in e]) * 100:.1f} cm, physics {(np.mean(ef) * 100 if ef else float('nan')):.1f} cm", flush=True)
    os._exit(0)                    # GL / CUDA threads can keep the interpreter alive after the window closes


if __name__ == '__main__':
    main()
