"""Replay flybody's pretrained flight-imitation policy and save a video.

Runs in the `flybody` conda env (Python 3.10: flybody core, tensorflow 2.8, tfp 0.16; no acme).

    MUJOCO_GL=egl conda run -n flybody python collect/preview_flight.py \
        --data data --traj 0 --out flight_preview.mp4

`--data` is searched recursively for the three files flybody's own notebook uses:
the flight policy SavedModel (`.../policy/flight`), the reference flight dataset
(`flight-dataset_*.hdf5`) and the wingbeat base pattern (`wing_pattern_fmech.npy`).
"""

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def find(root: Path, pattern: str, want_dir: bool = False) -> Path:
    hits = [p for p in root.rglob(pattern) if p.is_dir() == want_dir]
    if not hits:
        raise FileNotFoundError(f"No {pattern} under {root}")
    return sorted(hits)[0]


class Policy:
    """flybody's TestPolicyWrapper without acme: batch the obs, return the mean action.

    The SavedModel outputs a canonical action in [-1, 1]; we rescale it to the env's
    bounds (what acme's CanonicalSpecWrapper(clip=True) does) and feed float32 obs
    (what SinglePrecisionWrapper does).
    """

    def __init__(self, path, action_spec):
        import tensorflow as tf
        import tensorflow_probability  # noqa: F401  (the SavedModel returns a tfp distribution)

        self.tf = tf
        self.model = tf.saved_model.load(str(path))
        self.low, self.high = action_spec.minimum, action_spec.maximum

    def distribution(self, observation):
        """Mean action (clipped, env units) and the policy's std (env units)."""
        batched = {k: self.tf.constant(np.asarray(v, np.float32)[None]) for k, v in observation.items()}
        dist = self.model(batched)
        canonical = np.clip(dist.mean()[0].numpy(), -1.0, 1.0)
        half_range = 0.5 * (self.high - self.low)
        return self.low + (canonical + 1.0) * half_range, dist.stddev()[0].numpy() * half_range

    def __call__(self, observation):
        return self.distribution(observation)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--traj", type=int, default=0, help="reference flight index")
    parser.add_argument("--out", type=Path, default=Path("flight_preview.mp4"))
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--every", type=int, default=10, help="render every Nth control step")
    args = parser.parse_args()

    # Imported here so --help works without the heavy stack.
    from flybody.fly_envs import flight_imitation

    data = args.data.expanduser()
    policy_path = find(data, "flight", want_dir=True)
    ref_path = find(data, "flight-dataset_*.hdf5")
    wpg_path = find(data, "wing_pattern_fmech.npy")
    print(f"policy {policy_path}\nref    {ref_path}\nwpg    {wpg_path}")

    env = flight_imitation(str(ref_path), str(wpg_path), terminal_com_dist=float("inf"))
    policy = Policy(policy_path, env.action_spec())

    env.task.set_next_trajectory_index(args.traj)
    timestep = env.reset()
    print("observation keys:", {k: np.shape(v) for k, v in timestep.observation.items()})
    print("action dim:", env.action_spec().shape)

    frames, steps, reward, t0 = [], 0, 0.0, time.time()
    drift = []  # distance to the ghost (reference fly), from ref_displacement[0], in cm
    while timestep.step_type != 2:  # 2 = LAST
        if steps % args.every == 0:
            frames.append(env.physics.render(camera_id=args.camera, width=640, height=480))
        timestep = env.step(policy(timestep.observation))
        reward += float(timestep.reward or 0.0)
        drift.append(float(np.linalg.norm(timestep.observation["walker/ref_displacement"][0])))
        steps += 1
    wall = time.time() - t0
    print(f"{steps} control steps ({steps * env.control_timestep():.3f} s sim) "
          f"in {wall:.1f} s wall ({steps / wall:.0f} steps/s), mean reward {reward / max(steps, 1):.3f}")
    print(f"distance to ghost: mean {np.mean(drift) * 10:.2f} mm, max {np.max(drift) * 10:.2f} mm, "
          f"final {drift[-1] * 10:.2f} mm")

    # 3000 steps at 0.2 ms; every 10th frame at 30 fps is ~10x slow motion.
    imageio.mimwrite(args.out, frames, fps=30, macro_block_size=1)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
