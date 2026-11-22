"""Run flybody's flight policy on every reference flight and log each control step.

Layout and column meanings: docs/data/collection-plan.md. Runs in the `flybody` env on CPU:

    MUJOCO_GL=egl conda run -n flybody python collect/collect_flights.py --pass-id 1
    MUJOCO_GL=egl conda run -n flybody python collect/collect_flights.py --pass-id 1 --flights 0 1 2   # quick test

Writes data/flights/pass<k>_<name>.npz (every step), pass<k>_<name>_wingbeat.npz (every
23rd step) and meta.json.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preview_flight import Policy, find  # noqa: E402

PASSES = {
    1: dict(name="clean", noise=0.0, seed=0, left_wing_gain=1.0, mass_scale=1.0),
    2: dict(name="noise_a", noise=0.5, seed=1, left_wing_gain=1.0, mass_scale=1.0),
    3: dict(name="noise_b", noise=0.5, seed=2, left_wing_gain=1.0, mass_scale=1.0),
    4: dict(name="weak_wing", noise=0.0, seed=0, left_wing_gain=0.7, mass_scale=1.0),
    5: dict(name="heavy", noise=0.0, seed=0, left_wing_gain=1.0, mass_scale=1.2),
    # Stronger noise so the fly drifts further from the ghost: recovery rows for closed-loop control.
    6: dict(name="noise_strong", noise=2.0, seed=3, left_wing_gain=1.0, mass_scale=1.0),
}
OBS_KEYS = ["walker/accelerometer", "walker/actuator_activation", "walker/gyro", "walker/joints_pos",
            "walker/joints_vel", "walker/ref_displacement", "walker/ref_root_quat",
            "walker/velocimeter", "walker/world_zaxis"]  # the policy's own (sorted) input order
ACTION_NAMES = ["head_abduct", "head_twist", "head", "wing_yaw_left", "wing_roll_left",
                "wing_pitch_left", "wing_yaw_right", "wing_roll_right", "wing_pitch_right",
                "abdomen_abduct", "abdomen", "beat_freq_mod"]
STEPS_PER_WINGBEAT = 23  # 1 / (218 Hz * 0.2 ms)

_env = _policy = _cfg = None  # per-worker globals, set in _init_worker


def _init_worker(data_dir, cfg):
    global _env, _policy, _cfg
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # TF on CPU; one thread each, we parallelise by process
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from flybody.fly_envs import flight_imitation

    data_dir = Path(data_dir)
    _env = flight_imitation(str(find(data_dir, "flight-dataset_*.hdf5")),
                            str(find(data_dir, "wing_pattern_fmech.npy")),
                            randomize_start_step=False, terminal_com_dist=float("inf"))
    _policy = Policy(find(data_dir, "flight", want_dir=True), _env.action_spec())
    _cfg = cfg


def _apply_physics_changes(physics):
    """env.reset() recompiles the model, so pass-specific physics edits go after each reset."""
    model = physics.model
    if _cfg["left_wing_gain"] != 1.0:
        for name in ("wing_yaw_left", "wing_roll_left", "wing_pitch_left"):
            model.actuator_gainprm[model.name2id(f"walker/{name}", "actuator"), 0] *= _cfg["left_wing_gain"]
    if _cfg["mass_scale"] != 1.0:
        walker = [b for b in range(model.nbody) if (model.id2name(b, "body") or "").startswith("walker/")]
        model.body_mass[walker] *= _cfg["mass_scale"]
        model.body_inertia[walker] *= _cfg["mass_scale"]


def _run_flight(flight):
    env, task = _env, _env.task
    rng = np.random.default_rng(_cfg["seed"] * 100_000 + flight)
    task.set_next_trajectory_index(flight)
    timestep = env.reset()
    physics = env.physics
    _apply_physics_changes(physics)
    root = physics.named.data
    wing_joints = physics.bind(task._wing_joints)
    low, high = env.action_spec().minimum, env.action_spec().maximum

    rows = {k: [] for k in ("obs", "action_mean", "action_std", "action_applied", "state", "ref", "outcome", "step")}
    step = 0
    while timestep.step_type != 2:  # 2 = LAST
        obs = timestep.observation
        mean, std = _policy.distribution(obs)
        applied = mean if _cfg["noise"] == 0 else np.clip(
            mean + _cfg["noise"] * std * rng.standard_normal(mean.shape), low, high)
        # State before the action, at the same time as obs.
        com = physics.named.data.subtree_com["walker/"].copy()
        quat = root.xquat["walker/thorax"].copy()
        qvel = root.qvel["walker/"].copy()  # free joint: linear (world), angular (local)
        wing_q, wing_v = wing_joints.qpos.copy(), wing_joints.qvel.copy()
        t = physics.data.time
        ref = task._ref_qpos[step]

        # flybody's before_step adds the wingbeat pattern to the wing actions IN PLACE, so pass a
        # copy: otherwise the stored action_mean/action_applied include the pattern (bug in passes 1-5).
        timestep = env.step(applied.copy())
        # Pattern-generator target and frequency used during this step.
        wpg = task._wbpg.get_last_angles().copy()
        freq = float(task._wbpg._ctrl_freq)

        rows["obs"].append(np.concatenate([np.ravel(obs[k]) for k in OBS_KEYS]))
        rows["action_mean"].append(mean)
        rows["action_std"].append(std)
        rows["action_applied"].append(applied)
        rows["state"].append(np.concatenate([com, quat, qvel[:3], qvel[3:6], wing_q, wing_v, wpg, [freq, t]]))
        rows["ref"].append(ref)
        rows["outcome"].append([float(timestep.reward or 0.0),
                                10 * float(np.linalg.norm(timestep.observation["walker/ref_displacement"][0]))])
        rows["step"].append(step)
        step += 1
    out = {k: np.asarray(v, np.int32 if k == "step" else np.float32) for k, v in rows.items()}
    out["flight"] = flight
    out["reached_end"] = bool(task._reached_traj_end)
    return out


def _flight_info(ref_path):
    """flight -> (pair, type, mirrored, length). Reflected flight k of a type mirrors original flight k."""
    import h5py
    info = {}
    with h5py.File(ref_path) as f:
        idx = {name: f["trajectory_type_indices"][name][()] for name in f["trajectory_type_indices"]}
        lengths = {int(k): f["trajectories"][k]["com_qpos"].shape[0] for k in f["trajectories"]}
    for kind, code in (("evasion", 0), ("saccade", 1)):
        for orig, refl in zip(idx[f"{kind}_original"], idx[f"{kind}_reflected"]):
            info[int(orig)] = (int(orig), code, 0, lengths[int(orig)])
            info[int(refl)] = (int(orig), code, 1, lengths[int(refl)])
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", type=int, required=True, choices=sorted(PASSES))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/flights"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--flights", type=int, nargs="*", help="subset, for testing (default: all)")
    args = parser.parse_args()

    cfg = PASSES[args.pass_id]
    ref_path = find(args.data, "flight-dataset_*.hdf5")
    info = _flight_info(ref_path)
    flights = args.flights if args.flights else sorted(info)
    flights = sorted(flights, key=lambda f: -info[f][3])
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"pass {args.pass_id} ({cfg['name']}): {len(flights)} flights on {args.workers} workers", flush=True)

    t0, results = time.time(), []
    ctx = mp.get_context("spawn")  # TensorFlow is not fork-safe
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(args.data), cfg)) as pool:
        # Longest flights first keeps all workers busy until the end.
        for i, res in enumerate(pool.imap_unordered(_run_flight, flights, chunksize=1), 1):
            results.append(res)
            if i % 20 == 0 or i == len(flights):
                steps = sum(len(r["step"]) for r in results)
                print(f"  {i}/{len(flights)} flights, {steps} steps, {time.time() - t0:.0f} s", flush=True)
    results.sort(key=lambda r: r["flight"])

    n = [len(r["step"]) for r in results]
    ids = np.concatenate([
        np.column_stack([np.full(k, r["flight"]), np.full(k, info[r["flight"]][0]), np.full(k, info[r["flight"]][1]),
                         np.full(k, info[r["flight"]][2]), r["step"], np.full(k, args.pass_id)])
        for r, k in zip(results, n)]).astype(np.int32)
    arrays = {"ids": ids}
    for key in ("obs", "action_mean", "action_std", "action_applied", "state", "ref", "outcome"):
        arrays[key] = np.concatenate([r[key] for r in results])

    stem = args.out / f"pass{args.pass_id}_{cfg['name']}"
    np.savez_compressed(f"{stem}.npz", **arrays)
    beat = ids[:, 4] % STEPS_PER_WINGBEAT == 0
    np.savez_compressed(f"{stem}_wingbeat.npz", **{k: v[beat] for k, v in arrays.items()})

    early = [r["flight"] for r in results if not r["reached_end"]]
    dist = arrays["outcome"][:, 1]
    summary = dict(rows=int(len(ids)), wingbeat_rows=int(beat.sum()), flights=len(results),
                   ended_early=early, dist_to_ghost_mm=dict(mean=float(dist.mean()), p95=float(np.percentile(dist, 95)),
                                                            max=float(dist.max())),
                   seconds=round(time.time() - t0, 1))
    _write_meta(args.out / "meta.json", args.pass_id, cfg, summary)
    print(json.dumps(summary, indent=1), flush=True)


def _write_meta(path, pass_id, cfg, summary):
    meta = json.loads(path.read_text()) if path.exists() else {}
    obs_cols = (["accelerometer_" + a for a in "xyz"] + ["gyro_" + a for a in "xyz"]
                + [f"joints_pos_{i}" for i in range(25)] + [f"joints_vel_{i}" for i in range(25)]
                + [f"ref_displacement_{s}_{a}" for s in range(6) for a in "xyz"]
                + [f"ref_root_quat_{s}_{c}" for s in range(6) for c in "wxyz"]
                + ["velocimeter_" + a for a in "xyz"] + ["world_zaxis_" + a for a in "xyz"])
    meta.update({
        "source": {"flybody_commit": "d015e9bfe441bd90ae431bac24c55cb74bdbce26", "policy": "data/policies/flight",
                   "reference": "data/flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5",
                   "control_timestep_s": 2e-4, "steps_per_wingbeat_file": STEPS_PER_WINGBEAT},
        "columns": {
            "ids": ["flight", "pair", "type", "mirrored", "step", "pass"],
            "obs": obs_cols,
            "action_mean": ACTION_NAMES, "action_std": ACTION_NAMES, "action_applied": ACTION_NAMES,
            "state": (["com_x", "com_y", "com_z", "quat_w", "quat_x", "quat_y", "quat_z",
                       "lin_vel_x", "lin_vel_y", "lin_vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z"]
                      + [f"wing_qpos_{n}" for n in ACTION_NAMES[3:9]] + [f"wing_qvel_{n}" for n in ACTION_NAMES[3:9]]
                      + [f"wpg_target_{n}" for n in ACTION_NAMES[3:9]] + ["beat_freq", "t"]),
            "ref": ["ref_root_x", "ref_root_y", "ref_root_z", "ref_quat_w", "ref_quat_x", "ref_quat_y", "ref_quat_z"],
            "outcome": ["reward", "dist_to_ghost_mm"],
        },
        "notes": {"type": "0 evasion, 1 saccade", "units": "cm, s, rad; ang_vel is in the thorax frame",
                  "row_time": "obs/state before the action at step t; reward and wpg_target for step t->t+1"},
    })
    meta.setdefault("passes", {})[str(pass_id)] = {**cfg, **summary}
    path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
