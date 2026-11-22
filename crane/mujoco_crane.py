"""MuJoCo overhead crane: rail, trolley (slide joint), rope -> hook -> sling -> beam load (double pendulum
with a distributed-mass payload), racks either side of two set-down bays.

    MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_crane.py --frame   # render one test frame
    MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_crane.py --check   # physics sanity check

Used by crane/mujoco_demo.py (data collection, TabPFN, video).
Scale: industrial overhead crane (rope 3-7 m, slings 0.5-3 m, hook 250 kg, beam 0.3-3 t).
"""
import argparse
import math
import os

import numpy as np

import sys
if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    os.environ.setdefault('MUJOCO_GL', 'egl')       # headless Linux only; Windows / macOS / desktop Linux use the default (glfw)
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
RAIL_Z = 12.0
BAYS = (1.0, 11.0)          # set-down bay centres, m
CLEAR = 0.45                # rack clearance either side of the load, m
DT = 0.002                  # physics step, s


def build_xml(L1=4.0, L2=2.6, m_hook=250.0, m_load=400.0, beam=2.4, zeta=0.03):
    """Rope length L1, sling length L2 (hook to beam), masses in kg, beam length in m."""
    d1 = 2 * zeta * math.sqrt(G / L1) * (m_hook + m_load) * L1 ** 2     # hinge damping ~ zeta on each link
    d2 = 2 * zeta * math.sqrt(G / L2) * m_load * L2 ** 2
    racks = ''.join(
        f'<geom type="box" pos="{bx + s * (CLEAR + beam / 2 + 0.15):.3f} 0 1.0" size="0.15 1.0 1.0" rgba="0.6 0.65 0.7 1" contype="0" conaffinity="0"/>'
        for bx in BAYS for s in (-1, 1))
    bays = ''.join(f'<geom type="box" pos="{bx} 0 0.01" size="{beam / 2 + 0.05:.3f} 0.5 0.01" rgba="0.95 0.8 0.2 1" contype="0" conaffinity="0"/>' for bx in BAYS)
    return f'''
<mujoco model="overhead_crane">
  <option timestep="{DT}" gravity="0 0 -{G}" integrator="RK4"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.94 0.95" rgb2="0.86 0.88 0.9" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="16 6"/>
    <texture type="skybox" builtin="gradient" rgb1="0.97 0.98 0.99" rgb2="0.82 0.86 0.9" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="6 -8 16" dir="0 0.4 -1" directional="true" diffuse="0.7 0.7 0.7"/>
    <geom type="plane" pos="6 0 0" size="16 7 0.1" material="floor" contype="0" conaffinity="0"/>
    <geom type="box" pos="6 0 {RAIL_Z + 0.35}" size="7.5 0.18 0.2" rgba="0.18 0.43 0.62 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-1.6 0 {(RAIL_Z + 0.5) / 2}" size="0.25 0.25 {(RAIL_Z + 0.5) / 2}" rgba="0.35 0.4 0.45 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="13.6 0 {(RAIL_Z + 0.5) / 2}" size="0.25 0.25 {(RAIL_Z + 0.5) / 2}" rgba="0.35 0.4 0.45 1" contype="0" conaffinity="0"/>
    {racks}{bays}
    <body name="trolley" pos="0 0 {RAIL_Z}">
      <joint name="tx" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.45 0.35 0.15" mass="2000" rgba="0.12 0.12 0.14 1" contype="0" conaffinity="0"/>
      <body name="rope" pos="0 0 -0.15">
        <joint name="q1" type="hinge" axis="0 1 0" damping="{d1:.3f}"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{L1}" size="0.025" mass="0.5" rgba="0.25 0.25 0.25 1" contype="0" conaffinity="0"/>
        <body name="hook" pos="0 0 -{L1}">
          <geom type="box" size="0.18 0.12 0.22" pos="0 0 -0.1" mass="{m_hook}" rgba="0.95 0.7 0.1 1" contype="0" conaffinity="0"/>
          <body name="sling" pos="0 0 -0.3">
            <joint name="q2" type="hinge" axis="0 1 0" damping="{d2:.3f}"/>
            <geom type="capsule" fromto="0 0 0 {-beam * 0.4:.3f} 0 -{L2}" size="0.012" mass="0.05" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
            <geom type="capsule" fromto="0 0 0 {beam * 0.4:.3f} 0 -{L2}" size="0.012" mass="0.05" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
            <body name="load" pos="0 0 -{L2}">
              <geom type="box" size="{beam / 2:.3f} 0.25 0.18" pos="0 0 -0.18" mass="{m_load}" rgba="0.04 0.48 0.49 1" contype="0" conaffinity="0"/>
              <site name="load_c" pos="0 0 -0.18" size="0.05"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    <camera name="side" pos="6 -23 6.8" xyaxes="1 0 0 0 0.08 1" fovy="42"/>
  </worldbody>
  <actuator><motor joint="tx" gear="1" ctrllimited="false"/></actuator>
</mujoco>'''


class Plan:
    """Back-and-forth moves between the bays. Each move: trapezoid at speed v = frac*vmax, accel frac*amax,
    then a pause. ref(t) is the commanded trolley position, known to the controller in advance."""
    def __init__(self, fracs, pause=6.0, vmax=1.2, amax=0.4, x0=BAYS[0]):
        self.seg = []; t = 0.0; x = x0
        for i, f in enumerate(fracs):
            v, a = f * vmax, f * amax; D = BAYS[1] - BAYS[0]; ta = v / a; T = D / v + ta
            self.seg.append((t, T, x, 1 if x == BAYS[0] else -1, v, a, ta))
            t += T + pause; x = BAYS[1] if x == BAYS[0] else BAYS[0]
        self.T = t; self.x0 = x0

    def state(self, t):
        """(position, velocity, acceleration) of the commanded trolley motion at time t."""
        x = self.x0
        for (t0, T, xs, d, v, a, ta) in self.seg:
            if t < t0: return x, 0.0, 0.0
            s = t - t0
            if s >= T: x = xs + d * (BAYS[1] - BAYS[0]); continue
            if s < ta: return xs + d * 0.5 * a * s * s, d * a * s, d * a
            if s < T - ta: return xs + d * (0.5 * a * ta * ta + v * (s - ta)), d * v, 0.0
            r = T - s; return xs + d * ((BAYS[1] - BAYS[0]) - 0.5 * a * r * r), d * a * r, -d * a
        return x, 0.0, 0.0

    def ref(self, t): return self.state(t)[0]


def make(cond):
    m = mujoco.MjModel.from_xml_string(build_xml(**cond))
    d = mujoco.MjData(m)
    d.qpos[0] = BAYS[0]; mujoco.mj_forward(m, d)
    return m, d


def run(cond, plan, hz=10, on_frame=None, fps=30):
    """Simulate the whole plan. Returns a log at `hz`: t, trolley x, ref x, load offset (load - trolley, m).
    on_frame(m, d, t) is called at `fps` for video."""
    m, d = make(cond)
    lid, tid = m.body('load').id, m.body('trolley').id
    M = float(m.body_subtreemass[tid])              # trolley + rope + hook + load
    kp, kv = M * 25.0, M * 10.0                     # ~5 rad/s tracking loop, critically damped
    log = []; nlog = int(round(1 / (hz * DT))); nfr = int(round(1 / (fps * DT))); k = 0
    while d.time < plan.T:
        xr, vr, ar = plan.state(d.time)
        d.ctrl[0] = M * ar + kp * (xr - d.qpos[0]) + kv * (vr - d.qvel[0])   # feedforward + PD
        mujoco.mj_step(m, d); k += 1
        if k % nlog == 0:
            log.append((d.time, d.xpos[tid][0], plan.ref(d.time), d.site_xpos[0][0] - d.xpos[tid][0]))
        if on_frame and k % nfr == 0:
            on_frame(m, d, d.time)
    return np.array(log)


def modes(L1, L2, R):  # linearised point-mass double pendulum, rad/s (same formula as the page)
    s = (1 + R) * (1 / L1 + 1 / L2); q = math.sqrt(max(0, s * s - 4 * (1 + R) / (L1 * L2)))
    return math.sqrt(G / 2 * (s - q)), math.sqrt(G / 2 * (s + q))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--frame', action='store_true'); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    cond = dict(L1=4.0, L2=2.6, m_hook=250.0, m_load=400.0, beam=2.4)
    if a.check:
        # free swing period vs. the double-pendulum formula (beam inertia makes it differ a little)
        m, d = make(cond); d.qpos[1] = 0.05; mujoco.mj_forward(m, d)
        xs, ts = [], []
        while d.time < 30:
            Mt = float(m.body_subtreemass[1]); d.ctrl[0] = Mt * 25 * (BAYS[0] - d.qpos[0]) - Mt * 10 * d.qvel[0]; mujoco.mj_step(m, d); xs.append(d.site_xpos[0][0] - d.xpos[1][0]); ts.append(d.time)
        xs = np.array(xs); up = np.where((xs[:-1] < 0) & (xs[1:] >= 0))[0]
        Tm = np.mean(np.diff(np.array(ts)[up])) if len(up) > 2 else float('nan')
        w1, w2 = modes(cond['L1'] + 0.3, cond['L2'] + 0.18, cond['m_load'] / cond['m_hook'])
        print(f'free swing: measured period {Tm:.2f} s, point-mass formula mode 1 {2 * math.pi / w1:.2f} s, mode 2 {2 * math.pi / w2:.2f} s')
        log = run(cond, Plan([1.0, 0.7]))
        print(f'plan {log[-1, 0]:.0f} s, max |offset| {np.abs(log[:, 3]).max():.2f} m, trolley tracking err max {np.abs(log[:, 1] - log[:, 2]).max() * 100:.1f} cm')
    if a.frame:
        import imageio.v3 as iio
        m, d = make(cond); d.qpos[1] = 0.15; mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 720, 1280); r.update_scene(d, camera='side')
        out = os.path.join(HERE, 'results', 'mujoco_frame.png'); os.makedirs(os.path.dirname(out), exist_ok=True)
        iio.imwrite(out, r.render()); print('wrote', out)
