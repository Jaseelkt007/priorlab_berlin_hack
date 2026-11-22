"""Launch-film footage for the fruit-fly project (fly/): clean 4K60 shots flown live with the fly project's own
environment and controllers (imported unchanged from its collect/ folder), plus per-frame sidecars.

  F01 teacher   the trained RL policy (flybody's flight policy) flies the 4 s tour; rendered t = 0.40-1.30 s:
                the 180°/s curve, the sudden 80° turn at 0.80 s, then straight
  F02 scurve    TabPFN-3.5-Fast flies the 2 s S-curve, all 12 actions predicted from a 32,768-step context
                (393,216 rows: the README's setting for "0.10 mm off the path"); rendered t = 0.60-1.40 s around
                the left-to-right switch at 1.00 s; flown to the end so the whole-path error is measured
  F03 turnfail  TabPFN (same context) on "smooth, then sudden turn": it holds the smooth curve, then the 80° turn
                at 0.60 s throws it off; rendered t = 0.40-0.85 s, flown past 5 mm so the ghost visibly leaves it

Two stages, so TabPFN flies once and the cameras render in parallel from the saved states:
  fly     fly the shot; save qpos/qvel/act per rendered frame (1 ms of flight) -> <F##>_<shot>.states.npz as soon as
          the render window is flown, then <F##>_<shot>.summary.json (error over the whole flight, beats)
  render  replay the saved states (exact: the ghost is part of qpos, the path sites are fixed per episode) through one
          camera at 3840x2160, encode 60 fps CFR (1000 sim fps = 16.7x slow motion), write <clip>.frames.csv
  manifest  collect the clips' json into manifest.json
Look: dark floor (the arena's checker darkened), black sky, the task's green crosshair hidden; the translucent ghost
(the reference fly) and the cyan path dots are kept. No text.

Server (fly env, MUJOCO_GL=egl), e.g.:
  ~/priorlab_berlin_hack/.venv-fly/bin/python showcase/launch_fly.py fly --shot scurve --out ~/delta_line/footage_fly
  ~/priorlab_berlin_hack/.venv-fly/bin/python showcase/launch_fly.py render --shot scurve --cam hero --out ...
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.dont_write_bytecode = True          # never write into the fly project's folders
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")       # shared server

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

FLY = Path(os.environ.get("FLY_ROOT", Path.home() / "priorlab_berlin_hack"))   # the fly project (code + data)
sys.path.insert(0, str(FLY / "collect"))

import numpy as np  # noqa: E402

DT = 2e-4                 # control step (s), synthetic_paths.DT
EVERY = 5                 # control steps per frame: 1 ms of flight
SIM_FPS = 1000
W, H = 3840, 2160
CONTEXT_STEPS = 32768     # x 12 actions = 393,216 rows
SHOTS = {
    "teacher": dict(num="F01", path="tour", controller="policy", t0=0.40, t1=1.30, until=1.30,
                    cams=["track1", "hero"], beats={"sudden 80° turn (peak)": 0.80}),
    "scurve": dict(num="F02", path="S-curve", controller="tabpfn", t0=0.60, t1=1.40, until=2.00,
                   cams=["track1", "hero", "side"], beats={"left-to-right switch": 1.00}),
    "turnfail": dict(num="F03", path="smooth, then sudden turn", controller="tabpfn", t0=0.40, t1=0.85, until=0.85,
                     cams=["track1", "hero"], beats={"sudden 80° turn (peak)": 0.60}),
}
CAMERAS = {"track1": 1, "side": 5, "hero": 7}   # flybody walker cameras (chase 3/4, side profile, close-up)


def build(shot):
    """The fly project's env on this shot's path, styled for the film. Returns env, physics."""
    from preview_flight import find
    from showreel import _profile
    from synthetic_paths import make_env, path_from_yaw_rate
    prof = _profile(SHOTS[shot]["path"])
    env, loader = make_env(str(find(FLY / "data", "wing_pattern_fmech.npy")), len(prof) * DT + 0.01)
    qpos, qvel, _ = path_from_yaw_rate(prof)
    pad = 10                              # the task trims future_steps + 1 at the end (as showreel.py)
    loader.set_next_trajectory(np.vstack([qpos, np.repeat(qpos[-1:], pad, 0)]),
                               np.vstack([qvel, np.repeat(qvel[-1:], pad, 0)]))
    ts = env.reset()
    m = env.physics.model
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.quality.offsamples = 8
    m.mat_rgba[m.name2id("groundplane", "material")] = [0.10, 0.11, 0.13, 1.0]   # dark floor
    for i in range(m.nsite):
        if (m.id2name(i, "site") or "").startswith("crosshair"):
            m.site_rgba[i, 3] = 0.0                                                  # hide the HUD crosshair
    return env, ts


def stage_fly(shot, out):
    sh = SHOTS[shot]
    if sh["controller"] == "policy":       # TensorFlow on the CPU: the GPU is shared
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    env, ts = build(shot)
    spec = env.action_spec()
    from preview_flight import Policy, find
    if sh["controller"] == "policy":
        ctrl = Policy(find(FLY / "data", "flight", want_dir=True), spec)
        name = "trained RL policy (flybody flight policy, the teacher)"
    else:
        from closed_loop import TabPFNController
        ctrl = TabPFNController(CONTEXT_STEPS, "cuda", bounds=(spec.minimum, spec.maximum))
        name = f"TabPFN-3.5-Fast, {CONTEXT_STEPS} context steps x 12 actions = {CONTEXT_STEPS * 12:,} rows"
    ph = env.physics
    keep = {k: [] for k in ("t", "time", "qpos", "qvel", "act", "drift")}
    drift_all, n, drift, saved, t_wall = [], 0, 0.0, False, time.time()
    base = out / f"{sh['num']}_{shot}"
    while ts.step_type != 2 and n * DT < sh["until"] - 1e-9:
        t = n * DT
        if n % EVERY == 0 and sh["t0"] - 1e-9 <= t <= sh["t1"] + 1e-9:
            for k, v in (("t", t), ("time", ph.data.time), ("qpos", ph.data.qpos.copy()),
                         ("qvel", ph.data.qvel.copy()), ("act", ph.data.act.copy()), ("drift", drift)):
                keep[k].append(v)
            if len(keep["t"]) == 1:        # a live reference frame to check the replay against
                np.save(str(base) + ".check.npy", ph.render(camera_id=CAMERAS["track1"], width=960, height=540))
        if not saved and t > sh["t1"] + 1e-9:
            np.savez(str(base) + ".states.tmp.npz", **{k: np.asarray(v) for k, v in keep.items()})
            os.replace(str(base) + ".states.tmp.npz", str(base) + ".states.npz")
            saved = True
        a = ctrl(ts.observation)
        ts = env.step(np.asarray(a, np.float32))
        drift = 10 * float(np.linalg.norm(ts.observation["walker/ref_displacement"][0]))   # cm -> mm, as showreel
        drift_all.append(drift)
        n += 1
        if n % 500 == 0:
            print(f"{shot}: t = {n * DT:.3f} s, distance {drift:.2f} mm, {(time.time() - t_wall) / n * 1e3:.0f} ms/step",
                  flush=True)
    if not saved:
        np.savez(str(base) + ".states.tmp.npz", **{k: np.asarray(v) for k, v in keep.items()})
        os.replace(str(base) + ".states.tmp.npz", str(base) + ".states.npz")
    d = np.asarray(drift_all)
    cross = {f"{x} mm": round(float((np.argmax(d > x) + 1) * DT), 4) for x in (1, 2, 5) if (d > x).any()}
    win = np.asarray(keep["drift"])
    summary = dict(shot=shot, path=sh["path"], controller=name, flown_s=round(n * DT, 4),
                   ended="path end" if env.task._reached_traj_end else ("fell" if ts.step_type == 2 else "stopped"),
                   distance_to_path_mm=dict(mean_whole_flight=round(float(d.mean()), 3),
                                            max_whole_flight=round(float(d.max()), 3),
                                            mean_rendered=round(float(win.mean()), 3),
                                            max_rendered=round(float(win.max()), 3)),
                   first_time_off_by=cross, wall_s=round(time.time() - t_wall))
    (Path(str(base) + ".summary.json")).write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def ffmpeg(path):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60",
         "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-profile:v", "high",
         "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-g", "60", "-keyint_min", "60",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv",
         "-movflags", "+faststart", "-an", str(path)], stdin=subprocess.PIPE)


def project(matrix, p):
    """World point -> pixel (x right, y down from the top-left) through dm_control's camera matrix.
    Points in front of a dm_control camera have a negative third coordinate."""
    x = matrix @ np.r_[np.asarray(p, float), 1.0]
    if x[2] >= -1e-9:
        return np.nan, np.nan
    return float(x[0] / x[2]), float(x[1] / x[2])


def stage_render(shot, cam, out):
    from dm_control.mujoco.engine import Camera
    sh = SHOTS[shot]
    base = out / f"{sh['num']}_{shot}"
    st = np.load(str(base) + ".states.npz")
    env, _ = build(shot)
    ph = env.physics
    m = ph.model
    fly_b, ghost_b = m.name2id("walker/", "body"), m.name2id("ghost/", "body")
    camera = Camera(ph, height=H, width=W, camera_id=CAMERAS[cam])
    clip = out / f"{sh['num']}_{shot}_{cam}_{SIM_FPS}"
    pipe = ffmpeg(str(clip) + ".mp4")
    rows, t_wall = [], time.time()
    for i in range(len(st["t"])):
        ph.data.qpos[:], ph.data.qvel[:] = st["qpos"][i], st["qvel"][i]
        if ph.data.act.size:
            ph.data.act[:] = st["act"][i]
        ph.data.time = float(st["time"][i])
        ph.forward()
        if i == 0 and cam == "track1":     # the replay must match the live render of the same state
            ref = np.load(str(base) + ".check.npy").astype(int)
            diff = np.abs(ph.render(camera_id=CAMERAS[cam], width=960, height=540).astype(int) - ref)
            print(f"replay check vs live render: mean abs diff {diff.mean():.3f}, max {diff.max()}", flush=True)
        img = camera.render()
        pipe.stdin.write(np.ascontiguousarray(img).tobytes())
        mat = camera.matrix
        fx, fy = project(mat, ph.data.xpos[fly_b])
        gx, gy = project(mat, ph.data.xpos[ghost_b])
        rows.append(dict(frame=i, sim_time_s=round(float(st["t"][i]), 4), controller=sh["controller"], path=sh["path"],
                         distance_to_path_mm=round(float(st["drift"][i]), 4),
                         fly_x=round(fx, 1), fly_y=round(fy, 1), ghost_x=round(gx, 1), ghost_y=round(gy, 1),
                         fly_in_frame=int(np.isfinite(fx) and 0 <= fx < W and 0 <= fy < H),
                         ghost_in_frame=int(np.isfinite(gx) and 0 <= gx < W and 0 <= gy < H)))
        if i % 100 == 0:
            print(f"{shot}/{cam}: frame {i}/{len(st['t'])}, {(time.time() - t_wall) / (i + 1):.2f} s/frame", flush=True)
    pipe.stdin.close()
    pipe.wait()
    with open(str(clip) + ".frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    beats = {k: dict(sim_time_s=v, frame=int(round((v - sh["t0"]) * SIM_FPS)), video_s=round((v - sh["t0"]) * SIM_FPS / 60, 3))
             for k, v in sh["beats"].items()}
    info = dict(file=clip.name + ".mp4", shot=shot, camera=cam, path=sh["path"], controller=sh["controller"],
                resolution=f"{W}x{H}", fps=60, sim_fps=SIM_FPS, slow_motion=round(SIM_FPS / 60, 2), frames=len(rows),
                duration_s=round(len(rows) / 60, 3),
                sim_window_s=[round(float(st["t"][0]), 4), round(float(st["t"][-1]), 4)], beats=beats,
                sidecar=clip.name + ".frames.csv", render_wall_s=round(time.time() - t_wall))
    (Path(str(clip) + ".json")).write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False), flush=True)


def stage_manifest(out):
    clips = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(out / "F*_*_*_*.json")))]
    summaries = {json.loads(Path(p).read_text())["shot"]: json.loads(Path(p).read_text())
                 for p in sorted(glob.glob(str(out / "F*.summary.json")))}
    for c in clips:
        s = summaries.get(c["shot"])
        c["flight"] = s
        t0, t1 = SHOTS[c["shot"]]["t0"], SHOTS[c["shot"]]["t1"]
        c["measured_beats"] = {f"first {k} off the path": dict(sim_time_s=v, frame=int(round((v - t0) * SIM_FPS)),
                                                                video_s=round((v - t0) * SIM_FPS / 60, 3))
                               for k, v in (s or {}).get("first_time_off_by", {}).items() if t0 <= v <= t1}
    man = dict(project="fly (fruit fly, flybody + TabPFN)", made_by="showcase/launch_fly.py",
               spec="3840x2160, 60 fps CFR, H.264 High, CRF 14, yuv420p, BT.709 tv range, no audio, no text",
               timing_reference=("Turn times are the 80° saccade's peak yaw rate (the Gaussian centre in "
                                 "fly/collect/synthetic_paths.saccade, sigma 12 ms): 0.80 s on the tour, 0.60 s on "
                                 "'smooth, then sudden turn'. Delays like '5 mm off N ms later' are from that peak, the "
                                 "same reference as the fly README's sweep ('lost within 30-80 ms'). The README "
                                 "caption '40 ms later it is 5 mm off' is a different run (16,384-step context); "
                                 "do not use it with these clips."),
               sidecar_columns=dict(distance_to_path_mm="fly CoM to the reference (ghost) CoM, the README metric",
                                    fly_x_y="fly thorax in pixels, origin top-left",
                                    ghost_x_y="the ghost (reference fly on the path) in pixels"),
               clips=clips)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"manifest: {len(clips)} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fly", "render", "manifest"])
    ap.add_argument("--shot", choices=list(SHOTS))
    ap.add_argument("--cam", choices=list(CAMERAS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "fly":
        stage_fly(a.shot, a.out)
    elif a.stage == "render":
        stage_render(a.shot, a.cam, a.out)
    else:
        stage_manifest(a.out)
