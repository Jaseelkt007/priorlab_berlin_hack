"""Experiment D: TabPFN flies the fly. Closed loop, TabPFN replaces the policy at every step.

Runs in the `flybody` conda env (needs local `tabpfn` + a CUDA GPU, and the TabPFN 3.5
licence accepted on platform.priorlabs.ai so TABPFN_TOKEN can download the weights):

    MUJOCO_GL=egl conda run -n flybody python collect/closed_loop.py --latency-only
    MUJOCO_GL=egl conda run -n flybody python collect/closed_loop.py --flights 3 --video

Controller: TabPFN 3.5 Fast, fit_mode="fit_with_cache", with DuckPFN's pooled table (policy's
104 inputs + action ID -> that action, standardised). Context: N random steps from the
training flight pairs of passes 1-3 (clean + noisy, so it has seen recoveries). Each control
step predicts the 12 actions in one call. Held-out flights use the same pair split as the
other experiments; the original policy's score on the same flight comes from pass 1.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(Path(__file__).parent))
from flydata import load, split_pairs  # noqa: E402
from preview_flight import find  # noqa: E402

OBS_KEYS = ["walker/accelerometer", "walker/actuator_activation", "walker/gyro", "walker/joints_pos",
            "walker/joints_vel", "walker/ref_displacement", "walker/ref_root_quat",
            "walker/velocimeter", "walker/world_zaxis"]


class TabPFNController:
    def __init__(self, n_steps, device, seed=0):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        obs, act = [], []
        for name in ("pass1_clean", "pass2_noise_a", "pass3_noise_b"):
            d, _ = load(name, wingbeat=False, keys=["ids", "obs", "action_mean"])
            test, self.test_pairs = split_pairs(d["ids"])
            obs.append(d["obs"][~test])
            act.append(d["action_mean"][~test])
        obs, act = np.concatenate(obs), np.concatenate(act)
        pick = np.random.default_rng(seed).choice(len(obs), n_steps, replace=False)
        obs, act = obs[pick], act[pick]
        self.mu, self.sd = act.mean(0), np.maximum(act.std(0), 1e-6)
        self.n_act = act.shape[1]
        X = np.column_stack([np.repeat(obs, self.n_act, 0), np.tile(np.arange(self.n_act), len(obs))]).astype(np.float32)
        y = ((act - self.mu) / self.sd).reshape(-1)
        self.model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5_FAST, device=device, n_estimators=1, random_state=seed,
            fit_mode="fit_with_cache", categorical_features_indices=[X.shape[1] - 1])
        t0 = time.time()
        self.model.fit(X, y)
        print(f"fitted context: {n_steps} steps = {len(X)} rows in {time.time() - t0:.1f} s", flush=True)

    def __call__(self, observation):
        o = np.concatenate([np.ravel(observation[k]) for k in OBS_KEYS]).astype(np.float32)
        X = np.column_stack([np.repeat(o[None], self.n_act, 0), np.arange(self.n_act)]).astype(np.float32)
        return np.asarray(self.model.predict(X)) * self.sd + self.mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-steps", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flights", type=int, default=3, help="how many held-out flights (shortest first)")
    parser.add_argument("--latency-only", action="store_true", help="time 50 predictions and stop")
    parser.add_argument("--video", action="store_true", help="also save an MP4 per flight")
    args = parser.parse_args()

    from flybody.fly_envs import flight_imitation
    data = ROOT / "data"
    env = flight_imitation(str(find(data, "flight-dataset_*.hdf5")), str(find(data, "wing_pattern_fmech.npy")),
                           randomize_start_step=False, terminal_com_dist=float("inf"))
    ctrl = TabPFNController(args.context_steps, args.device)
    low, high = env.action_spec().minimum, env.action_spec().maximum

    timestep = env.reset()
    t0 = time.time()
    for _ in range(50):
        ctrl(timestep.observation)
    per_step = (time.time() - t0) / 50
    print(f"latency: {per_step * 1000:.1f} ms per control step (12 rows) -> "
          f"{3000 * per_step / 60:.1f} min of compute per 0.6 s flight", flush=True)
    if args.latency_only:
        return

    clean, _ = load("pass1_clean", wingbeat=False, keys=["ids", "outcome"])
    test_flights = np.unique(clean["ids"][np.isin(clean["ids"][:, 1], ctrl.test_pairs), 0])
    lengths = {f: int((clean["ids"][:, 0] == f).sum()) for f in test_flights}
    flights = sorted(test_flights, key=lambda f: lengths[f])[: args.flights]

    out_dir = ROOT / "results/exp_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for flight in flights:
        env.task.set_next_trajectory_index(int(flight))
        timestep = env.reset()
        drift, frames, n, t0 = [], [], 0, time.time()
        while timestep.step_type != 2:
            if args.video and n % 10 == 0:
                frames.append(env.physics.render(camera_id=1, width=640, height=480))
            timestep = env.step(np.clip(ctrl(timestep.observation), low, high))
            drift.append(10 * float(np.linalg.norm(timestep.observation["walker/ref_displacement"][0])))
            n += 1
        policy = clean["outcome"][clean["ids"][:, 0] == flight, 1]
        results[int(flight)] = dict(
            steps=n, planned=lengths[flight], reached_end=bool(env.task._reached_traj_end),
            tabpfn_drift_mm=dict(mean=float(np.mean(drift)), max=float(np.max(drift))),
            policy_drift_mm=dict(mean=float(policy.mean()), max=float(policy.max())),
            wall_s=round(time.time() - t0, 1))
        print(f"flight {flight}: {n}/{lengths[flight]} steps, reached end {results[int(flight)]['reached_end']}, "
              f"drift TabPFN {np.mean(drift):.2f}/{np.max(drift):.2f} mm vs policy {policy.mean():.2f}/{policy.max():.2f} mm "
              f"(mean/max), {time.time() - t0:.0f} s", flush=True)
        if args.video:
            import imageio.v2 as imageio
            imageio.mimwrite(out_dir / f"flight{flight}_tabpfn.mp4", frames, fps=30, macro_block_size=1)
    (out_dir / "results.json").write_text(json.dumps(
        dict(context_steps=args.context_steps, latency_ms=per_step * 1000, flights=results), indent=1))


if __name__ == "__main__":
    main()
