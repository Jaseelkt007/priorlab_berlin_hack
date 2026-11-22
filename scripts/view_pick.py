"""Watch real picks on the line: delta + compliant suction cup + moving belt + a catalogue product.

Each pick is simulated on the full model (sim.line.PickCell), its states are recorded, then replayed in slow motion.
After arrival the vacuum is switched off and the product is released onto the green pad.

  .venv/bin/python scripts/view_pick.py --product bar --alphas 0.4,1.0         # window, 15x slow motion
  .venv/bin/python scripts/view_pick.py --product praline --alphas 0.3,1.0 --record media/praline.mp4
Products: blister, praline, digestive, bar, cheese_wedge, pouch85, vial10r
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

from sim.delta import plant_friction
from sim.line import PickCell, PickSetup
from sim.products import LineState, sample

FRAME_DT = 1e-3       # recorded state every 1 ms of simulated time


def simulate(product: str, alpha: float, seed: int, v_belt: float):
    cfg, logged, hidden = sample(product, np.random.default_rng(seed), LineState())
    cell = PickCell(PickSetup(cup=cfg, alpha=alpha, v_belt=v_belt, seed=seed))
    frames = []
    every = int(round(FRAME_DT / cell.P.dt))

    def rec(c, t):
        if int(round(t / c.P.dt)) % every == 0:
            frames.append(c.d.qpos.copy())
    out = cell.run(rec)
    # release: hold 40 ms at the place point, vacuum off, let the product settle on the pad
    m, d, P, cup = cell.m, cell.d, cell.P, cell.cup
    for k in range(int(0.25 / P.dt)):
        t = cell.T_total + k * P.dt
        if k == int(0.04 / P.dt):
            cup.vac.on = False
        d.qfrc_applied[:] = 0
        d.ctrl[:3] = cell.ctrl.torque(d, cell.T_total)
        plant_friction(m, d, P, cell.ctrl.vadr)
        cup.apply()
        mujoco.mj_step(m, d)
        cup.update(t, P.dt)
        if k % every == 0:
            frames.append(d.qpos.copy())
    verdict = "DROPPED" if out["dropped"] else f"placed, error {out['place_offset_mm']:.1f} mm"
    print(f"{product} {logged['weight_g']:.1f} g, speed setting {alpha:.0%}: seal {out['seal_kpa']:.1f} kPa at touch, "
          f"loaded move {out['move_ms']:.0f} ms (peak {out['a_peak']:.0f} m/s²) -> {verdict}")
    return cell, frames, out


def camera(cam):
    cam.lookat[:] = [0.0, 0.0, -0.95]
    cam.distance, cam.azimuth, cam.elevation = 0.95, 150, -18


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="bar")
    ap.add_argument("--alphas", default="0.4,1.0", help="speed settings to show one after another")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--belt", type=float, default=0.3)
    ap.add_argument("--slow", type=float, default=15.0)
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    runs = [simulate(args.product, float(a), args.seed, args.belt) for a in args.alphas.split(",")]

    if args.record:
        fps = 30
        step = max(1, int(round(1000 / fps / args.slow)))                     # recorded frames per video frame
        ff = None
        for cell, frames, _ in runs:
            m, d = cell.m, mujoco.MjData(cell.m)
            r = mujoco.Renderer(m, 720, 1280)
            cam = mujoco.MjvCamera()
            camera(cam)
            if ff is None:
                ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                       "-s", "1280x720", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec",
                                       "libx264", "-crf", "18", args.record], stdin=subprocess.PIPE)
            for q in frames[::step] + [frames[-1]] * fps:
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam)
                ff.stdin.write(r.render().tobytes())
        ff.stdin.close()
        ff.wait()
        print(f"wrote {args.record} ({args.slow:g}x slow motion)")
        return

    i = 0
    while True:
        cell, frames, _ = runs[i % len(runs)]
        m, d = cell.m, mujoco.MjData(cell.m)
        with mujoco.viewer.launch_passive(m, d) as v:
            camera(v.cam)
            t0 = time.perf_counter()
            while v.is_running():
                k = int((time.perf_counter() - t0) / args.slow / FRAME_DT)
                if k >= len(frames) + 300:
                    break                                   # next speed setting
                d.qpos[:] = frames[min(k, len(frames) - 1)]
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(1 / 60)
            if not v.is_running() and k < len(frames) + 300:
                return
        i += 1


if __name__ == "__main__":
    main()
