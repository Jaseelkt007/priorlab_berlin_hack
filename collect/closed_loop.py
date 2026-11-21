"""Experiment D: TabPFN flies the fly. Closed loop, TabPFN replaces the policy at every step.

Runs in the `flybody` conda env (needs local `tabpfn` + a CUDA GPU, and the TabPFN 3.5
licence accepted on platform.priorlabs.ai so TABPFN_TOKEN can download the weights):

    MUJOCO_GL=egl conda run -n flybody python collect/closed_loop.py --latency-only
    MUJOCO_GL=egl conda run -n flybody python collect/closed_loop.py --flights 3 --video

Controller: TabPFN 3.5 Fast, fit_mode="fit_with_cache", with DuckPFN's pooled table (policy's
104 inputs + action ID -> that action, standardised). Context: N random steps from the
training flight pairs of passes 1-3 and 6 (clean + noisy + strong noise, so it has seen recoveries). Each control
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


CONTEXT_PASSES = ("pass1_clean", "pass2_noise_a", "pass3_noise_b", "pass6_noise_strong")


def recorded_steps(prev_action=False, passes=CONTEXT_PASSES):
    """All training-pair steps of passes 1-3 and 6 (clean, noisy, strong noise): obs (n, 104), policy action (n, 12).

    prev_action: append the action actually applied at the previous step (12 more columns, as
    DuckPFN's `last_action`), and drop each flight's step 0, which has no previous action.
    """
    obs, act = [], []
    for name in passes:
        d, _ = load(name, wingbeat=False, keys=["ids", "obs", "action_mean", "action_applied"])
        test, test_pairs = split_pairs(d["ids"])
        o, keep = d["obs"], ~test
        if prev_action:
            # Rows are in step order within each flight, and every flight starts at step 0.
            o = np.column_stack([o, np.roll(d["action_applied"], 1, axis=0)])
            keep &= d["ids"][:, 4] > 0
        obs.append(o[keep])
        act.append(d["action_mean"][keep])
    return np.concatenate(obs), np.concatenate(act), test_pairs


class TabPFNController:
    def __init__(self, n_steps, device, seed=0, context=None, prev_action=False, delta=False, bounds=None,
                 passes=CONTEXT_PASSES):
        """Context: `n_steps` random recorded steps, or explicit `context=(obs, act)` arrays.

        prev_action: also feed the previous applied action (call reset() at every episode start;
        the first step uses the context's mean action). bounds: (low, high) to clip outputs to,
        so the remembered previous action is the one the env really applied. delta (needs
        prev_action): predict the change from the previous action instead of the action
        (DuckPFN's target_mode="delta"), so "copy the last action" is no longer a free answer.
        """
        if delta and not prev_action:
            raise ValueError("delta needs prev_action")
        self.prev_action, self.delta, self.bounds = prev_action, delta, bounds
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        if context is None:
            obs, act, self.test_pairs = recorded_steps(prev_action, passes)
            pick = np.random.default_rng(seed).choice(len(obs), n_steps, replace=False)
            obs, act = obs[pick], act[pick]
        else:
            obs, act = context
            n_steps = len(obs)
        if delta:
            act = act - obs[:, -act.shape[1]:]  # the previous applied action is the last 12 columns
        self.mu, self.sd = act.mean(0), np.maximum(act.std(0), 1e-6)
        self.n_act = act.shape[1]
        # First step of an episode: no previous action yet, use the context's mean action.
        self.first_prev = ((act + obs[:, -self.n_act:]) if delta else act).mean(0).astype(np.float32)
        X = np.column_stack([np.repeat(obs, self.n_act, 0), np.tile(np.arange(self.n_act), len(obs))]).astype(np.float32)
        y = ((act - self.mu) / self.sd).reshape(-1)
        self.model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5_FAST, device=device, n_estimators=1, random_state=seed,
            fit_mode="fit_with_cache", categorical_features_indices=[X.shape[1] - 1])
        t0 = time.time()
        self.model.fit(X, y)
        print(f"fitted context: {n_steps} steps = {len(X)} rows x {X.shape[1]} columns "
              f"in {time.time() - t0:.1f} s", flush=True)
        self.reset()

    def reset(self):
        self.prev = self.first_prev.copy()

    def __call__(self, observation):
        return self.act_flat(np.concatenate([np.ravel(observation[k]) for k in OBS_KEYS]))

    def act_flat(self, o):
        """Same as calling the controller, from the 104 sensor values already flattened in OBS_KEYS order."""
        o = np.asarray(o, np.float32)
        if self.prev_action:
            o = np.concatenate([o, self.prev])
        X = np.column_stack([np.repeat(o[None], self.n_act, 0), np.arange(self.n_act)]).astype(np.float32)
        a = np.asarray(self.model.predict(X)) * self.sd + self.mu
        if self.delta:
            a = self.prev + a
        if self.bounds is not None:
            a = np.clip(a, *self.bounds)
        self.prev = a.astype(np.float32)
        return a


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
            imageio.mimwrite(out_dir / f"flight{flight}_tabpfn_ctx{args.context_steps}.mp4", frames, fps=30, macro_block_size=1)
    (out_dir / f"results_ctx{args.context_steps}.json").write_text(json.dumps(
        dict(context_steps=args.context_steps, latency_ms=per_step * 1000, flights=results), indent=1))


if __name__ == "__main__":
    main()
