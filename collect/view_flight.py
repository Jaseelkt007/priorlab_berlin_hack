"""Watch a controller fly live in the MuJoCo viewer: flybody's policy or TabPFN (experiment D).

    conda run -n flybody python collect/view_flight.py --flights 14 106 68
    conda run -n flybody python collect/view_flight.py --controller tabpfn --context-steps 512 --flights 14 106 68
    conda run -n flybody python collect/view_flight.py --controller tabpfn --context-steps 256 --prev-action --path smooth

Plays the given reference flights in a loop (each from its first step). The ghost fly and
the cyan dots are the real fly's recorded flight; the orange fly is the controller.
Flights 14, 106 and 68 are held out from TabPFN's context. --path flies one of experiment E's
synthetic 1 s paths instead (looped). --prev-action feeds TabPFN the previous action too.
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
from synthetic_paths import DT, DURATION, PATHS, make_env, make_path, path_from_yaw_rate, tour_yaw_rate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--flights", type=int, nargs="+", default=[14, 106, 68], help="reference flights to loop over")
    parser.add_argument("--controller", choices=["policy", "tabpfn", "remote"], default="policy",
                        help="remote: TabPFN served by collect/remote_brain.py on a GPU machine")
    parser.add_argument("--remote", default="localhost:7777", help="address of the remote_brain server (tunnelled)")
    parser.add_argument("--context-steps", type=int, default=512, help="TabPFN context size (512 fits 4 GB)")
    parser.add_argument("--context-passes", nargs="+", help="recorded passes TabPFN's context is drawn from "
                        "(default: closed_loop.CONTEXT_PASSES), e.g. pass1_clean")
    parser.add_argument("--prev-action", action="store_true", help="TabPFN also sees the previous action")
    parser.add_argument("--path", choices=[*PATHS, "tour"],
                        help="fly this synthetic path instead of --flights (tour: 4 s, every kind of turn)")
    parser.add_argument("--duration", type=float, default=DURATION, help="length of a --path flight in seconds")
    parser.add_argument("--slowmo", type=float, default=20.0, help="wall seconds per sim second")
    args = parser.parse_args()

    from flybody.fly_envs import flight_imitation

    data = args.data.expanduser()
    if args.path:
        if args.path == "tour":
            qpos, qvel, _ = path_from_yaw_rate(tour_yaw_rate())
            path = (np.vstack([qpos, np.repeat(qpos[-1:], 10, 0)]), np.vstack([qvel, np.repeat(qvel[-1:], 10, 0)]))
            args.duration = len(qpos) * DT
        else:
            path = make_path(PATHS[args.path], int(round(args.duration / DT)) + 10)[:2]
        env, loader = make_env(str(find(data, "wing_pattern_fmech.npy")), args.duration)
    else:
        env = flight_imitation(str(find(data, "flight-dataset_*.hdf5")),
                               str(find(data, "wing_pattern_fmech.npy")),
                               randomize_start_step=False, terminal_com_dist=float("inf"))
    tabpfn = None
    if args.controller == "remote":
        from remote_brain import RemoteController
        policy = tabpfn = RemoteController(args.remote)
    elif args.controller == "tabpfn":
        from closed_loop import TabPFNController
        spec = env.action_spec()
        tabpfn = TabPFNController(args.context_steps, "cuda", prev_action=args.prev_action,
                                  bounds=(spec.minimum, spec.maximum),
                                  **({"passes": args.context_passes} if args.context_passes else {}))
        policy = tabpfn
    else:
        policy = Policy(find(data, "flight", want_dir=True), env.action_spec())
    dt = env.control_timestep()

    # env.reset() recompiles the physics (flybody moves the trajectory sites per flight),
    # so env.physics is a new MjModel/MjData every episode. The viewer therefore owns its
    # own copy, and we mirror the live state into it after every step.
    env.reset()
    model = copy.deepcopy(env.physics.model.ptr)
    view = mujoco.MjData(model)

    i = 0
    with mujoco.viewer.launch_passive(model, view) as viewer:
        viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "walker/thorax")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 1.2
        while viewer.is_running():
            if args.path:
                traj = args.path
                loader.set_next_trajectory(*path)
            else:
                traj = args.flights[i % len(args.flights)]
                env.task.set_next_trajectory_index(traj)
            timestep = env.reset()
            if tabpfn is not None:
                tabpfn.reset()
            with viewer.lock():
                # This flight's trajectory dots: positions and which are visible (the viewer's
                # copy was made before the first path was loaded, so it only showed 40 ms of dots).
                model.site_pos[:] = env.physics.model.site_pos
                model.site_rgba[:] = env.physics.model.site_rgba
            steps, drift = 0, []
            while viewer.is_running() and timestep.step_type != 2:
                t0 = time.time()
                timestep = env.step(policy(timestep.observation))
                drift.append(np.linalg.norm(timestep.observation["walker/ref_displacement"][0]) * 10)
                steps += 1
                if steps % (2500 if args.controller == "policy" else 50) == 0:
                    print(f"  {steps} steps ({steps * dt * 1000:.0f} ms), distance to ghost {drift[-1]:.2f} mm",
                          flush=True)
                live = env.physics.data
                with viewer.lock():
                    view.qpos[:], view.qvel[:], view.time = live.qpos, live.qvel, live.time
                    mujoco.mj_forward(model, view)
                viewer.sync()
                # Hold each control step for dt * slowmo of wall time (policy time counts).
                time.sleep(max(0.0, dt * args.slowmo - (time.time() - t0)))
            if drift:
                why = ("window closed" if not viewer.is_running() else
                       "reached the end" if env.task._reached_traj_end else "fell (or physics blew up)")
                print(f"{args.controller} flight {traj} [{why}]: {steps} steps ({steps * dt * 1000:.0f} ms), "
                      f"distance to ghost mean {np.mean(drift):.2f} mm, max {np.max(drift):.2f} mm",
                      flush=True)
            i += 1


if __name__ == "__main__":
    main()
