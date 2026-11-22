"""Experiment E: do sudden turns break TabPFN? Fly synthetic reference paths with a controller.

    MUJOCO_GL=egl conda run -n flybody python collect/synthetic_paths.py --controller policy
    MUJOCO_GL=egl conda run -n flybody python collect/synthetic_paths.py --controller tabpfn --context-steps 512

Hypothesis: TabPFN's random context is mostly cruising, so it follows smooth paths but fails
at a sudden turn (saccade). Three 1.0 s paths (longer than any recorded flight, 0.1-0.3 s),
all at 12 cm/s, 0.7 cm height, body pitch -47.5 deg (median real values):
  smooth               constant gentle curve, 90 deg over 1 s
  smooth_then_saccade  gentle curve, an 80 deg saccade at t = 0.6 s, gentle again
  saccade_then_smooth  an 80 deg saccade at t = 0.15 s, then a gentle curve
  curve                tight constant curve, 360 deg/s (one full circle, ~1.9 cm radius, in 1 s):
                       visibly curved within the first 0.1 s, which is about how long TabPFN flies
A saccade is a Gaussian yaw-rate pulse (sigma 12 ms, peak ~2,660 deg/s), like the real ones
(80 deg turns, ~3,000 deg/s peaks). Writes results/exp_e/: per-step drift, a plot and videos.
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

DT = 2e-4
DURATION = 1.0
SPEED, HEIGHT, PITCH_DEG = 12.0, 0.7, -47.5
SACCADE = dict(turn_deg=80.0, sigma_s=0.012)
# name -> (saccade time or None, constant cruise yaw rate in deg/s)
ACTION_GROUPS = {"head": [0, 1, 2], "wings": [3, 4, 5, 6, 7, 8], "abdomen": [9, 10], "beat": [11]}
PATHS = {"smooth": (None, 90.0), "smooth_then_saccade": (0.6, 90.0), "saccade_then_smooth": (0.15, 90.0),
         "curve": (None, 360.0)}


def saccade(t, at):
    """Yaw-rate pulse (rad/s) of one SACCADE-sized turn centred at time `at`."""
    peak = np.radians(SACCADE["turn_deg"]) / (SACCADE["sigma_s"] * np.sqrt(2 * np.pi))
    return peak * np.exp(-0.5 * ((t - at) / SACCADE["sigma_s"]) ** 2)


def make_path(spec, n_steps):
    """CoM reference (qpos (n,7), qvel (n,6)) for a constant-speed path; spec = PATHS[name]."""
    saccade_t, cruise_yaw_rate = spec
    t = np.arange(n_steps) * DT
    yaw_rate = np.full(n_steps, np.radians(cruise_yaw_rate))
    if saccade_t is not None:
        yaw_rate += saccade(t, saccade_t)
    return path_from_yaw_rate(yaw_rate)


def tour_yaw_rate():
    """A 4 s path with every kind of turn: 180°/s curves, straight, S-curve, 360°/s curve, sudden left and right turns."""
    t = np.arange(int(4.0 / DT)) * DT
    r = np.radians
    deg = np.select([t < 0.8, t < 1.6, t < 2.6, t < 3.2], [180.0, 0.0, 180.0 * np.tanh((2.1 - t) / 0.05), -360.0], 180.0)
    return r(deg) + saccade(t, 0.8) - saccade(t, 2.6) + saccade(t, 3.6)


def path_from_yaw_rate(yaw_rate):
    """Constant speed and height along any yaw-rate profile (rad/s per step) -> qpos, qvel, yaw rate (deg/s)."""
    from flybody.quaternions import mult_quat
    n_steps = len(yaw_rate)
    yaw = np.concatenate([[0.0], np.cumsum(yaw_rate[:-1] * DT)])
    vel = SPEED * np.column_stack([np.cos(yaw), np.sin(yaw)])
    xy = np.concatenate([[[0.0, 0.0]], np.cumsum(vel[:-1] * DT, axis=0)])
    p = np.radians(PITCH_DEG) / 2
    pitch_q = np.array([np.cos(p), 0.0, np.sin(p), 0.0])
    quat = np.array([mult_quat(np.array([np.cos(y / 2), 0, 0, np.sin(y / 2)]), pitch_q) for y in yaw])
    qpos = np.column_stack([xy, np.full(n_steps, HEIGHT), quat])
    qvel = np.column_stack([vel, np.zeros(n_steps), np.zeros((n_steps, 2)), yaw_rate])
    return qpos, qvel, np.degrees(yaw_rate)


def make_env(wpg_path, time_limit):
    """flybody's flight_imitation(), but with a custom-path loader and a longer time limit."""
    from dm_control import composer
    from dm_control.locomotion.arenas import floors
    from flybody.fruitfly import fruitfly
    from flybody.tasks.flight_imitation import FlightImitationWBPG
    from flybody.tasks.pattern_generators import WingBeatPatternGenerator
    from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader

    loader = InferenceFlightTrajectoryLoader()
    task = FlightImitationWBPG(walker=fruitfly.FruitFly, arena=floors.Floor(),
                               wbpg=WingBeatPatternGenerator(base_pattern_path=wpg_path),
                               traj_generator=loader, terminal_com_dist=float("inf"),
                               initialize_qvel=True, force_actuators=False, disable_legs=True,
                               time_limit=time_limit, joint_filter=0.0, future_steps=5)
    env = composer.Environment(time_limit=time_limit, task=task, strip_singleton_obs_buffer_dim=True)
    return env, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["policy", "tabpfn"], required=True)
    parser.add_argument("--context-steps", type=int, default=512)
    parser.add_argument("--paths", nargs="+", default=list(PATHS), choices=list(PATHS))
    parser.add_argument("--context-passes", nargs="+", help="recorded passes for TabPFN's context (default: all)")
    parser.add_argument("--diagnose", action="store_true",
                        help="also save the fly's offset from the ghost (ahead/left/up) and, per step, "
                             "TabPFN's actions next to what the policy would do in the same state")
    parser.add_argument("--policy-controls", nargs="+", default=[], choices=list(ACTION_GROUPS),
                        help="hand these action groups to the policy; TabPFN controls the rest")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    data = ROOT / "data"
    env, loader = make_env(str(find(data, "wing_pattern_fmech.npy")), DURATION)
    if args.controller == "tabpfn":
        from closed_loop import TabPFNController
        spec = env.action_spec()
        act = ctrl = TabPFNController(args.context_steps, "cuda", bounds=(spec.minimum, spec.maximum),
                                      **({"passes": args.context_passes} if args.context_passes else {}))
        tag = f"tabpfn_ctx{args.context_steps}"
        if args.context_passes:
            tag += "_" + "-".join(p.split("_")[0] for p in args.context_passes)
    else:
        act = Policy(find(data, "flight", want_dir=True), env.action_spec())
        tag = "policy"
    if args.policy_controls:
        tag += "_policy-" + "-".join(args.policy_controls)
    handed = [i for g in args.policy_controls for i in ACTION_GROUPS[g]]
    expert = (Policy(find(data, "flight", want_dir=True), env.action_spec())
              if args.diagnose or handed else None)

    out = ROOT / "results/exp_e"
    out.mkdir(parents=True, exist_ok=True)
    n_steps = int(round(DURATION / DT)) + 10  # the task trims future_steps + 1 at the end
    summary = {}
    for name in args.paths:
        qpos, qvel, yaw_rate = make_path(PATHS[name], n_steps)
        loader.set_next_trajectory(qpos, qvel)
        timestep = env.reset()
        drift, frames, n, t0 = [], [], 0, time.time()
        diag = {"offset_mm": [], "act": [], "act_policy": []}
        while timestep.step_type != 2:
            if not args.no_video and n % 10 == 0:
                frames.append(env.physics.render(camera_id=1, width=640, height=480))
            a = act(timestep.observation)
            if expert is not None:
                label = expert(timestep.observation)
                a = np.array(a, np.float32)  # copy: env.step edits the wing actions in place
                a[handed] = label[handed]
                diag["act"].append(a.copy())
                diag["act_policy"].append(label)
            timestep = env.step(a)
            drift.append(10 * float(np.linalg.norm(timestep.observation["walker/ref_displacement"][0])))
            if expert is not None:
                # Fly minus ghost in world mm, split along the path direction (ahead), left of it, and up.
                task = env.task
                fly = task._walker.get_pose(env.physics)[0]
                ghost = task._ref_qpos[min(task._step_counter, len(task._ref_qpos) - 1), :3]
                fwd = qvel[n, :2] / np.linalg.norm(qvel[n, :2])
                d = 10 * (np.asarray(fly) - ghost)
                diag["offset_mm"].append([d[:2] @ fwd, d[:2] @ np.array([-fwd[1], fwd[0]]), d[2]])
            n += 1
            if n % 500 == 0:
                print(f"  {name}: {n} steps, drift now {drift[-1]:.2f} mm, {time.time() - t0:.0f} s", flush=True)
        drift = np.array(drift)
        st = PATHS[name][0]
        windows = {"all": slice(None)}
        if st is not None:
            s = int(st / DT)
            windows.update(before=slice(max(0, s - 500), s - 100), saccade=slice(s - 100, s + 250),
                           after=slice(s + 250, s + 1250))
        summary[name] = dict(steps=n, sim_s=n * DT, reached_end=bool(env.task._reached_traj_end),
                             saccade_t=st, wall_s=round(time.time() - t0, 1),
                             drift_mm={k: dict(mean=float(drift[w].mean()), max=float(drift[w].max()))
                                       for k, w in windows.items() if len(drift[w])})
        np.savez_compressed(out / f"{name}_{tag}.npz", drift_mm=drift, ref_yaw_rate_deg_s=yaw_rate[:n],
                            t=np.arange(n) * DT, **({k: np.array(v, np.float32) for k, v in diag.items()}
                                                   if expert is not None else {}))
        if frames:
            import imageio.v2 as imageio
            imageio.mimwrite(out / f"{name}_{tag}.mp4", frames, fps=30, macro_block_size=1)
        d = summary[name]["drift_mm"]
        print(f"{tag} {name}: {n} steps ({n * DT:.2f} s), reached end {summary[name]['reached_end']}, "
              + ", ".join(f"{k} {v['mean']:.2f}/{v['max']:.2f}" for k, v in d.items()) + " mm (mean/max)", flush=True)
    path = out / f"summary_{tag}.json"
    if path.exists():  # keep results for paths not flown this run
        summary = {**json.loads(path.read_text()), **summary}
    path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
