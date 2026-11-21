"""Watch flybody's pretrained flight policy live in the MuJoCo viewer.

    conda run -n flybody python collect/view_flight.py --traj 0 --slowmo 20

Plays one reference flight after another (starting at --traj). The ghost fly and the
cyan dots are the real fly's recorded flight; the orange fly is the policy.
--slowmo 20 means 1 s of simulated flight takes 20 s on screen (a wingbeat is 4.6 ms,
so real time is just a blur). Close the window to stop.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preview_flight import Policy, find  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--traj", type=int, default=0, help="first reference flight index")
    parser.add_argument("--slowmo", type=float, default=20.0, help="wall seconds per sim second")
    args = parser.parse_args()

    from flybody.fly_envs import flight_imitation

    data = args.data.expanduser()
    env = flight_imitation(str(find(data, "flight-dataset_*.hdf5")),
                           str(find(data, "wing_pattern_fmech.npy")),
                           terminal_com_dist=float("inf"))
    policy = Policy(find(data, "flight", want_dir=True), env.action_spec())
    dt = env.control_timestep()

    # env.reset() recompiles the physics (flybody moves the trajectory sites per flight),
    # so env.physics is a new MjModel/MjData every episode. The viewer therefore owns its
    # own copy, and we mirror the live state into it after every step.
    env.reset()
    model = copy.deepcopy(env.physics.model.ptr)
    view = mujoco.MjData(model)

    traj = args.traj
    with mujoco.viewer.launch_passive(model, view) as viewer:
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "walker/thorax")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.2
        while viewer.is_running():
            env.task.set_next_trajectory_index(traj)
            timestep = env.reset()
            with viewer.lock():
                model.site_pos[:] = env.physics.model.site_pos  # this flight's trajectory dots
            steps, drift = 0, []
            while viewer.is_running() and timestep.step_type != 2:
                t0 = time.time()
                timestep = env.step(policy(timestep.observation))
                drift.append(np.linalg.norm(timestep.observation["walker/ref_displacement"][0]) * 10)
                steps += 1
                live = env.physics.data
                with viewer.lock():
                    view.qpos[:], view.qvel[:], view.time = live.qpos, live.qvel, live.time
                    mujoco.mj_forward(model, view)
                viewer.sync()
                # Hold each control step for dt * slowmo of wall time (policy time counts).
                time.sleep(max(0.0, dt * args.slowmo - (time.time() - t0)))
            if drift:
                print(f"flight {traj}: {steps} steps ({steps * dt * 1000:.0f} ms), "
                      f"distance to ghost mean {np.mean(drift):.2f} mm, max {np.max(drift):.2f} mm",
                      flush=True)
            traj += 1


if __name__ == "__main__":
    main()
