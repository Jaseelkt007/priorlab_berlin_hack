"""LIVE pick-and-place crane in MuJoCo with TabPFN predicting the load 1 s ahead.

    set -a; . ./.env; set +a; TABPFN_NO_BROWSER=1 MUJOCO_GL=glfw crane/.venv/bin/python crane/live_pick_place.py
    ... --rope 4.5 --sling 2.0 --payload 0.5 --speed 1.0       # starting values
    ... --headless 120                                          # no window, for testing

The beam is a separate body resting in a bay. Each cycle: LOWER the hook on the hoist -> GRAB (weld on) -> RAISE to
travel height -> TRAVEL to the other bay -> SETTLE (|swing| < 5 cm for 1 s, the crane's own rule) -> LOWER into the bay
-> RELEASE (weld off) -> RAISE empty -> pick the same beam again and carry it back.
TabPFN (teal ghost + 90% floor band) and the physics model (orange) predict the load 1 s ahead during TRAVEL and SETTLE.
TabPFN chooses the anti-sway setting: key 3, 'tune auto' or --auto-tune. While the beam is lifted, one batched
TabPFN call per target predicts residual swing and settle time for every setting (off, 0.5 ... 3.0 m) from 120
simulated commissioning moves (results/pickplace_decision_fleet.json, built once) plus this session's own moves, and
takes the fastest expected cycle whose 99% swing clears the racks. Each move's outcome is added as a row.
Start / stop: Enter (or type start / stop). Modes: 1 = AUTO (cycles until stopped; a stop finishes the current cycle and
sets the beam down), 2 = MANUAL (one pick-and-place cycle per Enter). Flags: --manual, --stopped (start idle).
Anti-sway: ZV shaper, by default tuned for the operator-entered 0.5 m sling (--tune; 0 = off). Left/Right arrow (or
'tune 2.0' / 'tune off') changes the sling it assumes: match the real sling and the swing dies out quickly.
Keys / terminal commands as in live_crane.py (Up/Down speed, PgUp/PgDn sling, Home/End rope, Ins/Del payload;
'rope 5', 'sling 2', 'payload 0.5', 'speed 0.8', 'quit'); changes apply at the next pick.
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
from live_crane import Mover, features, physics_pred, FEAT, KEYS, LIMITS, build_fleet, OUT
from mujoco_demo import H

RAIL_Z, PIVOT_Z = mc.RAIL_Z, mc.RAIL_Z - 0.15
HOOK_DROP, LOAD_H, BEAM, M_HOOK = 0.3, 0.36, 2.0, 250.0
HOIST_V, TOL, HOLD, CAP = 0.6, 0.05, 1.0, 30.0
COPY = ['body_pos', 'body_quat', 'body_ipos', 'body_iquat', 'body_mass', 'body_inertia', 'body_subtreemass', 'geom_pos', 'geom_quat',
        'geom_size', 'geom_rbound', 'geom_aabb', 'dof_damping', 'dof_invweight0', 'body_invweight0', 'site_pos', 'jnt_range', 'actuator_ctrlrange']


TUNES = [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]          # anti-sway: sling length the ZV shaper assumes (None = off)


class ShapedMover(Mover):
    """Mover with a ZV anti-sway shaper (two half-size copies of the move, half a pendulum period apart)."""
    dz = 0.0
    def start(self, t, frac, tune_len=None):
        super().start(t, frac); self.dz = math.pi * math.sqrt(tune_len / mc.G) if tune_len else 0.0
    def state(self, t):
        if not self.dz or self.t0 is None: return super().state(t)
        a, b = super().state(t), super().state(t - self.dz)
        return tuple(0.5 * (u + w) for u, w in zip(a, b))
    def done(self, t): return self.t0 is not None and t >= self.t0 + self.T + self.dz


def build_xml(L1, L2, payload_t, zeta=0.03):
    m_load = payload_t * 1000; Lmax = PIVOT_Z - HOOK_DROP - LOAD_H + 0.2
    d1 = 2 * zeta * math.sqrt(mc.G / L1) * (M_HOOK + m_load) * L1 ** 2; d2 = 2 * zeta * math.sqrt(mc.G / L2) * m_load * L2 ** 2
    racks = ''.join(f'<geom type="box" pos="{bx + s * (mc.CLEAR + BEAM / 2 + 0.15):.3f} 0 0.5" size="0.15 1.0 0.5" rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>' for bx in mc.BAYS for s in (-1, 1))
    bays = ''.join(f'<geom type="box" pos="{bx} 0 0.005" size="{BEAM / 2 + 0.05:.3f} 0.5 0.005" rgba="0.95 0.8 0.2 1" contype="0" conaffinity="0"/>' for bx in mc.BAYS)
    return f'''
<mujoco model="pick_place_crane">
  <option timestep="{mc.DT}" gravity="0 0 -{mc.G}" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.35 0.35 0.35"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.94 0.95" rgb2="0.86 0.88 0.9" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="16 6"/>
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.98 0.99" rgb2="0.82 0.86 0.9" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="6 -8 16" dir="0 0.4 -1" directional="true" diffuse="0.7 0.7 0.7"/>
    <geom type="plane" pos="6 0 0" size="16 7 0.1" material="floor" friction="1 0.01 0.001"/>
    <geom type="box" pos="6 0 {RAIL_Z + 0.35}" size="7.5 0.18 0.2" rgba="0.18 0.43 0.62 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-1.6 0 {(RAIL_Z + 0.5) / 2}" size="0.25 0.25 {(RAIL_Z + 0.5) / 2}" rgba="0.35 0.4 0.45 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="13.6 0 {(RAIL_Z + 0.5) / 2}" size="0.25 0.25 {(RAIL_Z + 0.5) / 2}" rgba="0.35 0.4 0.45 1" contype="0" conaffinity="0"/>
    {racks}{bays}
    <body name="trolley" pos="0 0 {RAIL_Z}">
      <joint name="tx" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.45 0.35 0.15" mass="2000" rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0"/>
      <site name="drum" pos="0 0 -0.15" size="0.03"/>
      <body name="rope" pos="0 0 -0.15">
        <joint name="q1" type="hinge" axis="0 1 0" damping="{d1:.3f}"/>
        <geom type="sphere" size="0.03" mass="0.5" rgba="0 0 0 0" contype="0" conaffinity="0"/>
        <body name="hook" pos="0 0 0">
          <joint name="hoist" type="slide" axis="0 0 -1" range="1.5 {Lmax:.2f}" damping="2000"/>
          <geom type="box" size="0.18 0.12 0.22" pos="0 0 -0.1" mass="{M_HOOK}" rgba="0.95 0.7 0.1 1" contype="0" conaffinity="0"/>
          <site name="hook_top" pos="0 0 0" size="0.03"/>
          <body name="sling" pos="0 0 -{HOOK_DROP}">
            <joint name="q2" type="hinge" axis="0 1 0" damping="{d2:.3f}"/>
            <site name="sling_top" pos="0 0 0" size="0.02"/>
            <geom type="sphere" size="0.03" mass="0.2" rgba="0 0 0 0" contype="0" conaffinity="0"/>
            <body name="grip" pos="0 0 -{L2}">
              <geom type="box" size="{BEAM * 0.42:.3f} 0.06 0.03" mass="20" rgba="0.3 0.3 0.32 1" contype="0" conaffinity="0"/>
              <site name="grip_l" pos="{-BEAM * 0.4:.3f} 0 0" size="0.02"/><site name="grip_r" pos="{BEAM * 0.4:.3f} 0 0" size="0.02"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="load" pos="{mc.BAYS[0]} 0 {LOAD_H / 2 + 0.01}">
      <freejoint/>
      <geom type="box" size="{BEAM / 2} 0.25 {LOAD_H / 2}" mass="{m_load}" rgba="0.04 0.48 0.49 1" friction="1 0.01 0.001"/>
      <site name="load_c" pos="0 0 0" size="0.05"/>
    </body>
    <camera name="side" pos="6 -23 6.8" xyaxes="1 0 0 0 0.08 1" fovy="42"/>
  </worldbody>
  <tendon>
    <spatial name="rope_t" width="0.025" rgba="0.25 0.25 0.25 1"><site site="drum"/><site site="hook_top"/></spatial>
    <spatial name="sling_l" width="0.012" rgba="0.3 0.3 0.3 1"><site site="sling_top"/><site site="grip_l"/></spatial>
    <spatial name="sling_r" width="0.012" rgba="0.3 0.3 0.3 1"><site site="sling_top"/><site site="grip_r"/></spatial>
  </tendon>
  <equality><weld name="grab" body1="grip" body2="load" active="false" solref="0.02 1"/></equality>
  <actuator>
    <motor name="drive" joint="tx" ctrllimited="false"/>
    <position name="hoist_m" joint="hoist" kp="400000" kv="60000" ctrllimited="false"/>
  </actuator>
</mujoco>'''


# ---------- TabPFN chooses the anti-sway setting (one decision per move) ----------
CLEAR = mc.CLEAR


def sway_period(L1, L2, payload_t, rng=None):
    """Lift-off sway period from a rope-angle sensor: true first double-pendulum mode, 3% noise."""
    w1, _ = mc.modes(L1 + HOOK_DROP, L2 + LOAD_H / 2, payload_t * 1000 / M_HOOK)
    return 2 * math.pi / w1 * (1 + 0.03 * (rng.gauss(0, 1) if rng else 0.0))


def tune_len(L1, tune): return (L1 + HOOK_DROP + tune + LOAD_H / 2) if tune else None


def dfeat(L1, payload_t, meas, speed, tune):
    """Decision row: rope, payload, measured sway period, speed, shaper period / measured period - 1 (NaN = off)."""
    te = (2 * math.pi * math.sqrt(tune_len(L1, tune) / mc.G) / meas - 1) if tune else float('nan')
    return [L1, payload_t, meas, speed, te]


def simulate_move_attached(L1, L2, payload_t, speed, tune):
    """One loaded move A->B with the pick-and-place model (beam welded under the grip from the start), then settle."""
    m = mujoco.MjModel.from_xml_string(build_xml(L1, L2, payload_t)); d = mujoco.MjData(m)
    tid, lid, eq = m.body('trolley').id, m.body('load').id, m.equality('grab').id
    jtx, jh, hid = m.joint('tx').qposadr[0], m.joint('hoist').qposadr[0], m.actuator('hoist_m').id
    d.qpos[jtx] = mc.BAYS[0]; d.qpos[jh] = L1; d.ctrl[hid] = L1; mujoco.mj_forward(m, d)
    gz = d.xpos[m.body('grip').id][2]; ja = m.joint(m.body_jntadr[lid]).qposadr[0]
    d.qpos[ja:ja + 3] = [mc.BAYS[0], 0, gz - LOAD_H / 2]; d.qpos[ja + 3:ja + 7] = [1, 0, 0, 0]
    m.eq_data[eq][:] = 0; m.eq_data[eq][5] = -LOAD_H / 2; m.eq_data[eq][6] = 1; m.eq_data[eq][10] = 1; d.eq_active[eq] = 1; mujoco.mj_forward(m, d)
    for _ in range(250): mujoco.mj_step(m, d)                                  # let the weld settle
    mv = ShapedMover(mc.BAYS[0]); mv.start(d.time, speed, tune_len(L1, tune)); M = float(m.body_subtreemass[tid]) + payload_t * 1000
    sid = m.site('load_c').id; amp = 0.0; inTol = 0.0; t_arr = None
    while True:
        xr, vr, ar = mv.state(d.time)
        d.ctrl[m.actuator('drive').id] = M * ar + M * 25 * (xr - d.qpos[jtx]) + M * 10 * (vr - d.qvel[m.joint('tx').dofadr[0]])
        mujoco.mj_step(m, d)
        if mv.done(d.time):
            t_arr = t_arr or d.time; o = abs(d.site_xpos[sid][0] - d.xpos[tid][0]); amp = max(amp, o)
            inTol = inTol + mc.DT if o < TOL else 0.0
            if inTol >= HOLD or d.time - t_arr >= CAP: break
    return dict(amp=amp, settle=d.time - t_arr, move=t_arr - mv.t0)


def build_decision_fleet(n=120, seed=11):
    import random
    rng = random.Random(seed); rows = []
    for k in range(n):
        L1, L2, pt, sp = rng.uniform(3, 7), rng.uniform(0.5, 3.0), rng.uniform(0.5, 3.0), rng.uniform(0.5, 1.0); tu = rng.choice(TUNES)
        o = simulate_move_attached(L1, L2, pt, sp, tu)
        rows.append(dict(x=dfeat(L1, pt, sway_period(L1, L2, pt, rng), sp, tu), **o))
        print(f'  commissioning move {k + 1}/{n}', end='\r', flush=True)
    print(); return rows


def decide_tune(ctx, L1, payload_t, meas, speed):
    """One batched TabPFN call per target over all anti-sway settings: fastest expected cycle with 99% swing < clearance."""
    from tabpfn import TabPFNRegressor
    X = np.array([r['x'] for r in ctx]); C = np.array([dfeat(L1, payload_t, meas, speed, tu) for tu in TUNES])
    a50, a99 = (np.asarray(v) for v in TabPFNRegressor(n_estimators=1).fit(X, np.array([r['amp'] for r in ctx])).predict(C, output_type='quantiles', quantiles=[0.5, 0.99]))
    s50 = np.asarray(TabPFNRegressor(n_estimators=1).fit(X, np.array([r['settle'] for r in ctx])).predict(C, output_type='median'))
    cands = [dict(tune=tu, amp50=float(x), amp99=float(y), settle=float(max(0, z)), extra=(math.pi * math.sqrt(tune_len(L1, tu) / mc.G) if tu else 0.0)) for tu, x, y, z in zip(TUNES, a50, a99, s50)]
    for c_ in cands: c_['score'] = c_['settle'] + c_['extra']                  # shaper delay + predicted settle wait
    ok = [c_ for c_ in cands if c_['amp99'] < CLEAR] or cands
    return min(ok, key=lambda c_: c_['score']), cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rope', type=float, default=4.5); ap.add_argument('--sling', type=float, default=2.0)
    ap.add_argument('--payload', type=float, default=0.5); ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--headless', type=float, default=0)
    ap.add_argument('--tune', type=float, default=0.5, help='anti-sway assumed sling (m); 0 = off')
    ap.add_argument('--auto-tune', action='store_true', help='TabPFN chooses the anti-sway setting (key 3 toggles)')
    ap.add_argument('--manual', action='store_true', help='manual mode: one cycle per Enter / "start"')
    ap.add_argument('--stopped', action='store_true', help='start idle; press Enter or type start')
    a = ap.parse_args()
    P = dict(rope=a.rope, sling=a.sling, payload=a.payload, speed=a.speed); pending = dict(P); lock = threading.Lock()
    T = {'i': TUNES.index(a.tune) if a.tune in TUNES else 1, 'auto': a.auto_tune}

    from tabpfn import TabPFNRegressor
    fpath = os.path.join(OUT, 'live_fleet_rows.json')
    if not os.path.exists(fpath):
        import random as _r; rows = build_fleet(); _r.seed(0); json.dump(_r.sample(rows, min(3000, len(rows))), open(fpath, 'w'))
    fleet = json.load(open(fpath))
    dpath = os.path.join(OUT, 'pickplace_decision_fleet.json')
    if not os.path.exists(dpath):
        print('simulating commissioning moves for the anti-sway decision (once, ~2-3 min) ...', flush=True); json.dump(build_decision_fleet(), open(dpath, 'w'))
    dctx = json.load(open(dpath)); own = []; import random as _rr; srng = _rr.Random(5)
    print('fitting TabPFN on', len(fleet), 'fleet rows ...', flush=True)
    model = TabPFNRegressor(n_estimators=1, fit_mode='fit_with_cache').fit(np.array([[r['f'][k] for k in FEAT] for r in fleet]), np.array([r['y'] for r in fleet]))
    model.predict(np.zeros((1, len(FEAT))))
    req, res = queue.Queue(maxsize=1), {'pred': None, 'ms': 0.0}

    def worker():
        while True:
            item = req.get()
            if item is None: return
            t, row, sign = item; t0 = time.time()
            lo, md, hi = (float(np.asarray(v)[0]) for v in model.predict(np.array([[row[k] for k in FEAT]]), output_type='quantiles', quantiles=[0.05, 0.5, 0.95]))
            if sign < 0: lo, md, hi = -hi, -md, -lo
            res['pred'] = (t, lo, md, hi); res['ms'] = 1000 * (time.time() - t0)
    threading.Thread(target=worker, daemon=True).start()

    m = mujoco.MjModel.from_xml_string(build_xml(P['rope'], P['sling'], P['payload'])); d = mujoco.MjData(m)
    tid, gid, lid = m.body('trolley').id, m.body('grip').id, m.body('load').id
    jtx, jh = m.joint('tx').qposadr[0], m.joint('hoist').qposadr[0]; eq = m.equality('grab').id; hid = m.actuator('hoist_m').id
    d.qpos[jtx] = mc.BAYS[0]; d.qpos[jh] = P['rope']; d.ctrl[hid] = P['rope']; mujoco.mj_forward(m, d)
    S = dict(mode='manual' if a.manual else 'auto', run=not a.stopped and not a.manual, phase='IDLE' if (a.stopped or a.manual) else 'LOWER', t0=0.0, hoist_from=P['rope'], hoist_to=None, attached=False, inTol=0.0, cycle0=0.0, cycles=[], last_row=-1.0, quit=False, phys=None)
    mover = ShapedMover(mc.BAYS[0]); hist = collections.deque(maxlen=40); preds, phys_at, errs = {}, {}, collections.deque(maxlen=100)
    cond = lambda: dict(L1=P['rope'], L2=P['sling'], m_hook=M_HOOK, m_load=P['payload'] * 1000, beam=BEAM)
    pick_len = lambda: PIVOT_Z - HOOK_DROP - P['sling'] - LOAD_H - 0.01 - 0.15   # rope length that puts the grip on the beam top

    def rebuild():
        m2 = mujoco.MjModel.from_xml_string(build_xml(P['rope'], P['sling'], P['payload']))
        for name in COPY: getattr(m, name)[:] = getattr(m2, name)
        d.qpos[m.joint('q1').qposadr[0]] = 0; d.qpos[m.joint('q2').qposadr[0]] = 0; d.qvel[:] = 0; mujoco.mj_forward(m, d)

    def set_phase(p): S['phase'] = p; S['t0'] = d.time
    def offset(): return d.site_xpos[m.site('load_c').id][0] - d.xpos[tid][0]

    def step_once():
        t = d.time; ph = S['phase']; el = t - S['t0']
        if ph == 'LOWER':                                                     # hook down onto the beam
            tgt = pick_len(); h = min(tgt, S['hoist_from'] + HOIST_V * el); d.ctrl[hid] = h
            if h >= tgt and abs(d.qpos[jh] - tgt) < 0.03 and el > 0.5:
                if not S['attached']:                                         # weld the beam to the grip where it is
                    gp, gq = d.xpos[gid], d.xquat[gid]; lp, lq = d.xpos[lid], d.xquat[lid]
                    nq = np.zeros(4); mujoco.mju_negQuat(nq, gq); rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, lp - gp, nq); rq = np.zeros(4); mujoco.mju_mulQuat(rq, nq, lq)
                    m.eq_data[eq][:3] = 0; m.eq_data[eq][3:6] = rp; m.eq_data[eq][6:10] = rq; m.eq_data[eq][10] = 1; d.eq_active[eq] = 1; S['attached'] = True
                    set_phase('GRAB')
                else:
                    d.eq_active[eq] = 0; S['attached'] = False; S['place_cm'] = round((d.xpos[lid][0] - mover.ref(t)) * 100); set_phase('RELEASE')
        elif ph == 'GRAB' and el > 0.6:
            S['hoist_from'] = d.qpos[jh]; S['decision'] = None; S['deciding'] = T['auto']; S['meas'] = None
            if T['auto']:                                                     # decide while the beam is lifted
                S['meas'] = sway_period(P['rope'], P['sling'], P['payload'], srng)
                def _dec(args=(list(dctx) + list(own), P['rope'], P['payload'], S['meas'], P['speed'])):
                    t0_ = time.time(); best, cands = decide_tune(*args); S['decision'] = dict(best=best, cands=cands, s=time.time() - t0_)
                threading.Thread(target=_dec, daemon=True).start()
            set_phase('RAISE')
        elif ph == 'RAISE':
            h = max(P['rope'], S['hoist_from'] - HOIST_V * el); d.ctrl[hid] = h
            if h <= P['rope'] + 1e-6 and abs(d.qpos[jh] - P['rope']) < 0.03:
                if S['attached'] and S.get('deciding') and S.get('decision') is None: pass   # still deciding: hold at height
                elif S['attached']:
                    if S.get('deciding'): T['i'] = TUNES.index(S['decision']['best']['tune'])
                    tl = TUNES[T['i']]; hist.clear(); mover.x0 = mover.ref(t); mover.t0 = None
                    mover.start(t, P['speed'], (P['rope'] + HOOK_DROP + tl + LOAD_H / 2) if tl else None); set_phase('TRAVEL')
                else: set_phase('NEXT')
        elif ph == 'TRAVEL' and mover.done(t): S['inTol'] = 0.0; set_phase('SETTLE')
        elif ph == 'SETTLE':
            S['inTol'] = S['inTol'] + mc.DT if abs(offset()) < TOL else 0.0
            S['amp'] = max(S.get('amp', 0.0), abs(offset()))
            if S['inTol'] >= HOLD or el >= CAP:
                S['settle_s'] = el; S['hoist_from'] = d.qpos[jh]
                if S.get('meas'):                                             # the outcome becomes one more row
                    own.append(dict(x=dfeat(P['rope'], P['payload'], S['meas'], P['speed'], TUNES[T['i']]), amp=S['amp'], settle=el, move=0.0)); S['meas'] = None
                S['amp'] = 0.0; set_phase('LOWER')
        elif ph == 'RELEASE' and el > 0.6: S['hoist_from'] = d.qpos[jh]; set_phase('RAISE')
        elif ph == 'NEXT':                                                    # cycle done
            S['cycles'].append(dict(cycle_s=round(t - S['cycle0'], 1), settle_s=round(S.get('settle_s', 0), 1), placed_cm=S.get('place_cm'), antisway=TUNES[T['i']], chosen_by='TabPFN' if S.get('deciding') else 'operator', mode=S['mode'], **P))
            mover.x0 = mover.ref(t); mover.t0 = None
            if S['mode'] == 'manual': S['run'] = False                        # manual: one cycle per start
            set_phase('IDLE')
        elif ph == 'IDLE' and S['run']:                                      # start (auto: keeps going; manual: one cycle)
            with lock: new = dict(pending)
            if new != P: P.update(new); rebuild()
            S['cycle0'] = t; S['hoist_from'] = d.qpos[jh]; set_phase('LOWER')
        M = float(m.body_subtreemass[tid]) + (P['payload'] * 1000 if S['attached'] else 0.0)
        xr, vr, ar = mover.state(t)                                          # holds at the end bay after the move
        d.ctrl[m.actuator('drive').id] = M * ar + M * 25 * (xr - d.qpos[jtx]) + M * 10 * (vr - d.qvel[m.joint('tx').dofadr[0]])
        mujoco.mj_step(m, d)
        if S['phase'] in ('TRAVEL', 'SETTLE') and d.time - S['last_row'] >= 0.1 - 1e-9:
            S['last_row'] = d.time; off = offset(); hist.append((d.time, d.qpos[jtx], off)); tt = round(d.time, 1)
            if round(tt - 1.0, 1) in preds:
                p = preds.pop(round(tt - 1.0, 1)); errs.append((abs(p[0] - off), abs(p[1] - off) if p[1] is not None else None))
            if len(hist) >= 21:
                c = cond(); f = features(list(hist), mover, d.time, c)
                S['phys'] = (d.time, physics_pred(f, mover, d.time, c['L1'])); phys_at[round(d.time, 1)] = S['phys'][1]
                try: req.put_nowait((d.time, features(list(hist), mover, d.time, c, mover.dir), mover.dir))
                except queue.Full: pass
                r = res['pred']
                if r: preds[round(r[0], 1)] = (r[2], phys_at.get(round(r[0], 1)))
        elif S['phase'] not in ('TRAVEL', 'SETTLE'): S['phys'] = None

    def control(cmd):
        if cmd == 'start': S['run'] = True
        elif cmd == 'stop': S['run'] = False
        elif cmd == 'toggle': S['run'] = not S['run']
        elif cmd in ('auto', 'manual'): S['mode'] = cmd; S['run'] = False if cmd == 'manual' else S['run']
        print(f"mode {S['mode']}, {'running' if S['run'] else ('stopping after this cycle' if S['phase'] != 'IDLE' else 'stopped')}", flush=True)

    def on_key(k):
        if k in (257, 335): control('toggle'); return                       # Enter / keypad Enter: start / stop
        if k == 49: control('auto'); return                                  # '1'
        if k == 50: control('manual'); return                                # '2'
        if k == 51: T['auto'] = not T['auto']; print('anti-sway chosen by', 'TabPFN' if T['auto'] else 'you', flush=True); return   # '3'
        if k in (262, 263):                                                   # Right / Left: anti-sway assumed sling
            T['auto'] = False; T['i'] = (T['i'] + (1 if k == 262 else -1)) % len(TUNES); print('anti-sway for next move:', TUNES[T['i']], flush=True); return
        if k in KEYS:
            n, step = KEYS[k]; lo_, hi_ = LIMITS[n]
            with lock: pending[n] = round(min(hi_, max(lo_, pending[n] + step)), 2)
            print('next pick:', pending, flush=True)

    def stdin_reader():
        for line in sys.stdin:
            w = line.split()
            if not w: continue
            if w[0] == 'quit': S['quit'] = True; return
            if w[0] in ('start', 'stop', 'auto', 'manual'): control(w[0]); continue
            if len(w) == 2 and w[0] == 'tune' and w[1] == 'auto': T['auto'] = True; print('anti-sway chosen by TabPFN', flush=True); continue
            if len(w) == 2 and w[0] == 'tune': T['auto'] = False
            if len(w) == 2 and w[0] == 'tune':
                v = None if w[1] in ('off', '0') else float(w[1]); T['i'] = TUNES.index(v) if v in TUNES else T['i']; print('anti-sway for next move:', TUNES[T['i']], flush=True); continue
            if len(w) == 2 and w[0] in LIMITS:
                try: v = float(w[1])
                except ValueError: continue
                with lock: pending[w[0]] = min(LIMITS[w[0]][1], max(LIMITS[w[0]][0], v))
                print('next pick:', pending, flush=True)

    def ghost(scn, size, pos, rgba):
        if scn.ngeom >= scn.maxgeom: return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array(size, float), np.array(pos, float), np.eye(3).flatten(), np.array(rgba, dtype=np.float32)); scn.ngeom += 1

    def overlay(viewer):
        scn = viewer.user_scn; scn.ngeom = 0; r = res['pred']; show = S['phase'] in ('TRAVEL', 'SETTLE')
        if show and r and d.time - r[0] < 0.5:
            xf = mover.ref(r[0] + H); _, lo, md, hi = r; zl = d.site_xpos[m.site('load_c').id][2]
            ghost(scn, [BEAM / 2 + (hi - lo) / 2, 1.1, 0.02], [xf + (lo + hi) / 2, 0, 0.02], (0.04, 0.48, 0.49, 0.25))
            ghost(scn, [BEAM / 2, 0.55, 0.02], [xf + md, 0, 0.05], (0.04, 0.48, 0.49, 0.9))
            ghost(scn, [BEAM / 2, 0.26, LOAD_H / 2], [xf + md, 0, zl], (0.04, 0.48, 0.49, 0.3))
        if show and S['phys']:
            ghost(scn, [BEAM / 2, 0.3, 0.02], [mover.ref(S['phys'][0] + H) + S['phys'][1], 0, 0.08], (0.77, 0.35, 0.09, 0.9))
        e = list(errs); ep = np.mean([x[0] for x in e]) * 100 if e else float('nan'); ef = [x[1] for x in e if x[1] is not None]; ef = np.mean(ef) * 100 if ef else float('nan')
        label = dict(IDLE='stopped: press Enter (or type start)' if S['mode'] == 'auto' else 'manual: press Enter for one cycle', LOWER='lowering the hook' if not S['attached'] else 'lowering into the bay', GRAB='grabbing the beam', RAISE='lifting' if S['attached'] else 'raising the empty hook',
                     TRAVEL='travelling', SETTLE=f"waiting for the swing to settle ({d.time - S['t0']:.1f} s)", RELEASE='releasing', NEXT='next pick')[S['phase']]
        last = S['cycles'][-1] if S['cycles'] else None
        with lock: nxt = dict(pending)
        runst = 'running' if S['run'] else ('stopping after this cycle' if S['phase'] != 'IDLE' else 'stopped')
        left = (f"Pick and place · {S['mode'].upper()} · {runst} · {label}\nrope {P['rope']} m  sling {P['sling']} m (hidden)  payload {P['payload']} t  speed {int(P['speed'] * 100)}%\n"
                f"TabPFN (teal) in 1 s: {('%+.2f m  90%%: %+.2f..%+.2f' % (r[2], r[1], r[3])) if (r and show) else '-'}\nphysics (orange): {('%+.2f m' % S['phys'][1]) if S['phys'] else '-'}\n"
                f"mean error last 10 s: TabPFN {ep:.1f} cm  physics {ef:.1f} cm\n"
                f"anti-sway ({'TabPFN chooses' if T['auto'] else 'you choose'}): " + (f"ZV tuned for a {TUNES[T['i']]} m sling" if TUNES[T['i']] else 'off')
                + ((f"  · TabPFN predicted settle {S['decision']['best']['settle']:.1f} s, swing {S['decision']['best']['amp50']:.2f} m (99%: {S['decision']['best']['amp99']:.2f}) · own rows {len(own)}") if (T['auto'] and S.get('decision')) else ('  · deciding...' if T['auto'] and S['phase'] == 'RAISE' and S['attached'] else '')) + "\n"
                f"cycles {len(S['cycles'])}" + (f"  last: {last['cycle_s']} s, settle {last['settle_s']} s, placed {last['placed_cm']:+d} cm from bay centre" if last else ''))
        keys = (f"next pick: rope {nxt['rope']}  sling {nxt['sling']}  payload {nxt['payload']}  speed {int(nxt['speed'] * 100)}%\n"
                "Enter start/stop  1 auto  2 manual  3 TabPFN tunes anti-sway  |  Up/Down speed  PgUp/PgDn sling  Home/End rope  Ins/Del payload  Left/Right anti-sway")
        return [(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, None), (mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, keys, None)]

    if a.headless:
        t0 = time.time(); last_print = -5
        while d.time < a.headless:
            step_once()
            if d.time > time.time() - t0: time.sleep(max(0.0, d.time - (time.time() - t0)))
            if a.manual and S['phase'] == 'IDLE' and not S['run'] and len(S['cycles']) < 2 and d.time > 2: control('start')   # test: press start
            if d.time - last_print >= 5:
                last_print = d.time; e = list(errs); ef = [x[1] for x in e if x[1] is not None]
                print(f"t={d.time:5.1f} {S['mode']:6s} run={S['run']!s:5s} {S['phase']:8s} attached={S['attached']} trolley x={d.qpos[jtx]:5.2f} load x={d.xpos[lid][0]:5.2f} z={d.xpos[lid][2]:4.2f}  "
                      f"TabPFN {np.mean([x[0] for x in e]) * 100 if e else float('nan'):.1f} cm  physics {np.mean(ef) * 100 if ef else float('nan'):.1f} cm  cycles {len(S['cycles'])}", flush=True)
        print('cycles:', S['cycles'], flush=True); req.put(None); time.sleep(0.3); os._exit(0)

    threading.Thread(target=stdin_reader, daemon=True).start()
    with mujoco.viewer.launch_passive(m, d, key_callback=on_key, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED; viewer.cam.fixedcamid = m.camera('side').id
        print('window open: close it, or type quit, to stop', flush=True)
        wall0, sim0 = time.time(), d.time
        while viewer.is_running() and not S['quit']:
            target = time.time() - wall0
            with viewer.lock():
                n = 0
                while d.time - sim0 < target and n < 50: step_once(); n += 1
                texts = overlay(viewer)
            viewer.set_texts(texts)
            lag = target - (d.time - sim0)
            if lag > 0.25: wall0 += lag
            viewer.sync(); time.sleep(0.002)
    req.put(None); print('cycles:', S['cycles'], flush=True); os._exit(0)


if __name__ == '__main__':
    main()
