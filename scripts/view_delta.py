"""Watch the IRB 360-class delta run ABB's benchmark pick cycle on the full closed-chain model.

The motion is the calibrated planner (TOPP-RA + ABB limits), tracked by the exact feedforward + PD controller,
with gearbox friction in the plant. The planned TCP path is drawn as a dotted line.

  .venv/bin/python scripts/view_delta.py                       # interactive window, 10x slow motion
  .venv/bin/python scripts/view_delta.py --slow 1              # real speed (one cycle ≈ 0.32 s)
  .venv/bin/python scripts/view_delta.py --cycle 90/400/90 --kg 1.0
  .venv/bin/python scripts/view_delta.py --record out.mp4      # offscreen: real speed, then 10x slow motion
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import mujoco.viewer
import numpy as np

from sim.delta import DeltaController, DeltaParams, FeedforwardA, build_model_a, plant_friction, tool_xml
from sim.delta_motion import CYCLES, DWELL, Z_TCP, MoveSpec, Planner

SCENE = """
    <light pos="0.6 -0.6 0.8" dir="-0.5 0.5 -1" diffuse=".7 .7 .7" castshadow="true"/>
    <light pos="-0.8 0.4 0.6" dir="0.6 -0.3 -1" diffuse=".35 .35 .38" castshadow="false"/>
    <geom type="plane" size="1.2 1.2 0.01" pos="0 0 {z_floor}" rgba=".86 .87 .89 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.6 0.15 0.01" pos="0 0 {z_belt}" rgba=".18 .19 .21 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.03 0.03 0.004" pos="{xa} 0 {z_pad}" rgba=".2 .55 .95 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.03 0.03 0.004" pos="{xb} 0 {z_pad}" rgba=".2 .75 .4 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="0.012 0.25" pos="0.32 0.32 -0.25" rgba=".55 .57 .6 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="0.012 0.25" pos="-0.32 0.32 -0.25" rgba=".55 .57 .6 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="0.012 0.25" pos="0.32 -0.32 -0.25" rgba=".55 .57 .6 1" contype="0" conaffinity="0"/>
    <geom type="cylinder" size="0.012 0.25" pos="-0.32 -0.32 -0.25" rgba=".55 .57 .6 1" contype="0" conaffinity="0"/>
    <geom type="box" size="0.34 0.34 0.012" pos="0 0 0.012" rgba=".55 .57 .6 1" contype="0" conaffinity="0"/>
"""
CUP = """<geom type="cylinder" size="0.012 0.02" pos="0 0 {z}" rgba=".1 .1 .1 1" contype="0" conaffinity="0" mass="0"/>"""


def build(cyc: str, kg: float):
    P = DeltaParams(m_tool=kg)
    h, D = CYCLES[cyc]
    a, b = np.array([-D / 2, 0, Z_TCP]), np.array([D / 2, 0, Z_TCP])
    spec = MoveSpec()
    pl = Planner(P, np.array([0, 0, Z_TCP + P.tcp_off]))
    out, T, info = pl.plan(a, b, h, h, spec)
    back, T2, _ = pl.plan(b, a, h, h, spec)
    Tc = T + DWELL + T2 + DWELL

    def ref(t):
        t = t % Tc
        if t < T:
            return out(t)
        if t < T + DWELL:
            return out(T)
        if t < T + DWELL + T2:
            return back(t - T - DWELL)
        return back(T2)

    p0 = ref(0)[0]
    z_pad = Z_TCP - 0.006
    world = SCENE.format(z_floor=Z_TCP - 0.30, z_belt=Z_TCP - 0.016, z_pad=z_pad, xa=a[0], xb=b[0])
    plat = tool_xml(P) + CUP.format(z=-P.tcp_off + 0.02)
    m, _ = build_model_a(P, p0, extra_platform_xml=plat, extra_world_xml=world)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    m.vis.map.shadowclip = 2.0
    n_cyc = 6
    print(f"{cyc} @ {kg} kg: cycle {Tc * 1e3:.0f} ms ({60 / Tc:.0f} picks/min); peak a {info['a_peak']:.0f} m/s², "
          f"v {info['v_peak']:.1f} m/s; precomputing exact feedforward for {n_cyc} cycles ...")
    exact = FeedforwardA(P, p0, extra_platform_xml=tool_xml(P))
    path = np.array([ref(t)[0] - [0, 0, P.tcp_off] for t in np.linspace(0, T, 40)])
    return P, m, ref, exact, Tc, n_cyc, path


def add_path(scn, path):
    for p in path:
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.003, 0, 0]), p,
                            np.eye(3).ravel(), np.array([0.95, 0.3, 0.2, 0.8], dtype=np.float32))
        scn.ngeom += 1


def camera(cam):
    cam.lookat[:] = [0, 0, Z_TCP + 0.35]
    cam.distance, cam.azimuth, cam.elevation = 2.1, 125, -14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default="25/305/25", choices=list(CYCLES))
    ap.add_argument("--kg", type=float, default=0.1)
    ap.add_argument("--slow", type=float, default=10.0, help="slow-motion factor for the window")
    ap.add_argument("--record", default=None, help="write an MP4 instead of opening a window")
    args = ap.parse_args()
    P, m, ref, exact, Tc, n_cyc, path = build(args.cycle, args.kg)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ctrl = DeltaController(P, m, ref(0)[0])
    ctrl.prepare(ref, n_cyc * Tc + 0.01, exact=exact)
    N = int(n_cyc * Tc / P.dt)

    def step(k):
        t = (k % N) * P.dt
        if k % N == 0 and k > 0:   # restart the loop from the planned state
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
        d.ctrl[:3] = ctrl.torque(d, t)
        plant_friction(m, d, P, ctrl.vadr)
        mujoco.mj_step(m, d)

    if args.record:
        fps = 60
        r = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        camera(cam)
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
                               "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "18", args.record],
                              stdin=subprocess.PIPE)
        k = 0
        for slow in (1.0, 1.0, 10.0):          # two passes at real speed, then 10x slow motion
            per_frame = 1.0 / fps / slow
            n_frames = int(N * P.dt / per_frame)
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
            k, t_next = 0, 0.0
            for _ in range(n_frames):
                while k * P.dt < t_next and k < N:
                    step(k)
                    k += 1
                r.update_scene(d, cam)
                add_path(r.scene, path)
                ff.stdin.write(r.render().tobytes())
                t_next += per_frame
        ff.stdin.close()
        ff.wait()
        print(f"wrote {args.record}")
        return


    with mujoco.viewer.launch_passive(m, d) as v:
        camera(v.cam)
        k, t0 = 0, time.perf_counter()
        while v.is_running():
            target = (time.perf_counter() - t0) / args.slow
            while k * P.dt < target:
                step(k)
                k += 1
            with v.lock():
                v.user_scn.ngeom = 0
                add_path(v.user_scn, path)
            v.sync()
            time.sleep(1 / 120)


if __name__ == "__main__":
    main()
