"""One continuous video: a controller flies many path types back to back, resetting after each.

    MUJOCO_GL=egl python collect/showreel.py --controller policy --video-seconds 120
    MUJOCO_GL=egl python collect/showreel.py --controller tabpfn --context-steps 4096 --video-seconds 120
    MUJOCO_GL=egl python collect/showreel.py --controller tabpfn --context-steps 4096 --policy-controls wings

Paths cycle in REEL order. A segment ends when the path ends, the fly falls, or it gets more than
--max-dev-mm from the ghost; then the next path starts. Every frame is labelled (controller, path,
time, distance). One frame per --frame-every control steps (10 = 2 ms of flight per frame, so 1 s
of flight is ~17 s of video at 30 fps). Stops once the video reaches --video-seconds.
Writes results/showreel/<tag>.mp4 and <tag>.json (one record per segment).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from preview_flight import Policy, find  # noqa: E402
from synthetic_paths import ACTION_GROUPS, DT, make_env, path_from_yaw_rate, saccade, tour_yaw_rate  # noqa: E402


def _profile(name):
    """Yaw-rate profile (rad/s per step) for each showreel path."""
    r = np.radians
    if name == "tour":
        return tour_yaw_rate()
    if name == "smooth curve":
        t = np.arange(int(2.0 / DT)) * DT
        return np.full(len(t), r(90.0))
    if name == "tight curve":
        t = np.arange(int(1.0 / DT)) * DT
        return np.full(len(t), r(360.0))
    if name == "smooth, then sudden turn":
        t = np.arange(int(1.5 / DT)) * DT
        return r(180.0) + saccade(t, 0.6)
    if name == "sudden turn, then smooth":
        t = np.arange(int(1.5 / DT)) * DT
        return r(180.0) + saccade(t, 0.15)
    if name == "S-curve":
        t = np.arange(int(2.0 / DT)) * DT
        return r(180.0) * np.tanh((1.0 - t) / 0.05)  # left for 1 s, smooth switch, right for 1 s
    raise ValueError(name)


REEL = ["tour", "S-curve", "tight curve", "sudden turn, then smooth", "smooth, then sudden turn"]


def _label(frame, lines):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=17)
    d.rectangle([0, 0, img.width, 22 * len(lines) + 10], fill=(255, 255, 255))
    for i, s in enumerate(lines):
        d.text((10, 5 + 22 * i), s, fill=(15, 23, 42), font=font)
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["policy", "tabpfn"], required=True)
    parser.add_argument("--context-steps", type=int, default=4096)
    parser.add_argument("--context-passes", nargs="+", help="recorded passes for TabPFN's context (default: all)")
    parser.add_argument("--policy-controls", nargs="+", default=[], choices=list(ACTION_GROUPS),
                        help="hand these action groups to the policy; TabPFN controls the rest")
    parser.add_argument("--video-seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-every", type=int, default=10, help="control steps per video frame")
    parser.add_argument("--max-dev-mm", type=float, default=5.0, help="reset when this far from the ghost")
    parser.add_argument("--once", action="store_true",
                        help="fly each path type once (no video-length cap); for context-size sweeps")
    parser.add_argument("--out", default="results/showreel", help="output folder")
    args = parser.parse_args()

    data = ROOT / "data"
    longest = max(len(_profile(p)) for p in REEL) * DT
    env, loader = make_env(str(find(data, "wing_pattern_fmech.npy")), longest + 0.01)
    spec = env.action_spec()
    expert = Policy(find(data, "flight", want_dir=True), spec)
    handed = [i for g in args.policy_controls for i in ACTION_GROUPS[g]]
    if args.controller == "tabpfn":
        from closed_loop import TabPFNController
        ctrl = TabPFNController(args.context_steps, "cuda", bounds=(spec.minimum, spec.maximum),
                                **({"passes": args.context_passes} if args.context_passes else {}))
        name = f"TabPFN, {args.context_steps} context steps"
        if handed:
            name += f", policy flies {' + '.join(args.policy_controls)}"
        tag = f"tabpfn_ctx{args.context_steps}" + ("_policy-" + "-".join(args.policy_controls) if handed else "")
    else:
        ctrl, name, tag = None, "Trained policy (teacher)", "policy"

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    need = float("inf") if args.once else int(args.video_seconds * args.fps)
    frames, segments, curves, k, t0 = [], [], {}, 0, time.time()
    while len(frames) < need and not (args.once and k >= len(REEL)):
        path = REEL[k % len(REEL)]
        qpos, qvel, _ = path_from_yaw_rate(_profile(path))
        pad = 10  # the task trims future_steps + 1 at the end
        loader.set_next_trajectory(np.vstack([qpos, np.repeat(qpos[-1:], pad, 0)]),
                                   np.vstack([qvel, np.repeat(qvel[-1:], pad, 0)]))
        timestep = env.reset()
        drift, n, why, ts = [], 0, "path end", time.time()
        while timestep.step_type != 2:
            obs = timestep.observation
            if ctrl is None:
                a = expert(obs)
            else:
                a = np.array(ctrl(obs), np.float32)
                if handed:
                    a[handed] = expert(obs)[handed]
            if n % args.frame_every == 0:
                d = drift[-1] if drift else 0.0
                frames.append(_label(env.physics.render(camera_id=1, width=640, height=480), [
                    name, f"Path {k + 1}: {path}", f"t = {n * DT * 1000:5.0f} ms   distance to ghost {d:5.2f} mm"]))
            timestep = env.step(a)
            drift.append(10 * float(np.linalg.norm(timestep.observation["walker/ref_displacement"][0])))
            n += 1
            if drift[-1] > args.max_dev_mm:
                why = f"more than {args.max_dev_mm:g} mm off"
                break
            if len(frames) >= need:
                why = "video full"
                break
        if timestep.step_type == 2 and not env.task._reached_traj_end:
            why = "fell"
        seg = dict(path=path, flew_ms=round(n * DT * 1000, 1), planned_ms=round(len(qpos) * DT * 1000, 1),
                   ended=why, drift_mean_mm=round(float(np.mean(drift)), 3), drift_max_mm=round(float(np.max(drift)), 3),
                   drift_at_mm={f"{ms}ms": round(drift[int(ms / 1000 / DT) - 1], 3)
                                for ms in (20, 50, 100, 200, 500, 1000) if int(ms / 1000 / DT) <= len(drift)},
                   ms_per_step=round((time.time() - ts) / n * 1000, 1))
        curves[f"{k}_{path}"] = np.array(drift, np.float32)
        segments.append(seg)
        print(f"segment {k + 1}: {path}: flew {seg['flew_ms']:.0f}/{seg['planned_ms']:.0f} ms, {why}, "
              f"drift {seg['drift_mean_mm']:.2f}/{seg['drift_max_mm']:.2f} mm, video {len(frames) / args.fps:.0f} s, "
              f"{time.time() - t0:.0f} s wall", flush=True)
        # Hold the last frame for half a second so the reset is visible.
        frames += [frames[-1]] * (args.fps // 2)
        k += 1

    import imageio.v2 as imageio
    imageio.mimwrite(out / f"{tag}.mp4", frames, fps=args.fps, macro_block_size=1)
    np.savez_compressed(out / f"{tag}_drift.npz", **curves)
    (out / f"{tag}.json").write_text(json.dumps(dict(controller=name, context_steps=args.context_steps,
                                                     segments=segments, wall_s=round(time.time() - t0)), indent=1))
    print(f"wrote {out / tag}.mp4 ({len(frames) / args.fps:.0f} s of video)", flush=True)


if __name__ == "__main__":
    main()
