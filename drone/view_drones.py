"""Fly the drones yourself in the MuJoCo viewer.

    python drone/view_drones.py              # both drones
    python drone/view_drones.py --drone x2   # only the Skydio X2
    python drone/view_drones.py --drone cf2  # only the Crazyflie 2
    python drone/view_drones.py --selftest   # no window: fly a fixed course and print where each drone ends up

Two windows open: the MuJoCo viewer, and a small "Drone controls" window. Click the
controls window, then hold keys (see KEYS). The drone you are not flying holds its position.
The flight controller is in flight.py. Setup: drone/models/README.md.
"""

import argparse
import time
import tkinter as tk
import warnings
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from flight import FlightController

MENAGERIE = Path(__file__).parent / "models" / "mujoco_menagerie"

# name -> (xml, start position x y z in metres, label)
DRONES = {
    "x2": ("skydio_x2/x2.xml", (0.0, 0.0, 1.0), "Skydio X2"),
    "cf2": ("bitcraze_crazyflie_2/cf2.xml", (0.0, 1.5, 1.0), "Crazyflie 2"),  # beside the X2, not in its path
}

# key -> (command index, value); command = forward, left, up, yaw rate
KEYS = {
    "Up": (0, 1), "Down": (0, -1),
    "Left": (1, 1), "Right": (1, -1),
    "w": (2, 1), "s": (2, -1),
    "a": (3, 1), "d": (3, -1),
}
HELP = """Click this window, then hold keys:
  Arrow keys   forward / back / left / right
  W / S        up / down
  A / D        turn left / right
  Tab          switch drone
  R            reset both drones
Mouse in the viewer: drag to rotate, scroll to zoom."""

# Sky, checker floor and lights copied from menagerie's skydio_x2/scene.xml; lights raised for 1 m flight.
# Timestep 0.002 because the Crazyflie needs a small step; air and RK4 as in the menagerie drone models.
WORLD_XML = """
<mujoco model="drones">
  <option timestep="0.002" density="1.225" viscosity="1.8e-5" integrator="RK4"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 0.5 -0.8"/>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def build(names):
    """One world with sky and floor, each drone attached with its own name prefix."""
    world = mujoco.MjSpec.from_string(WORLD_XML)
    for name in names:
        xml, pos, _ = DRONES[name]
        child = mujoco.MjSpec.from_file(str(MENAGERIE / xml))
        world.worldbody.add_frame(pos=list(pos)).attach_body(child.worldbody.first_body(), f"{name}_", "")
    with warnings.catch_warnings():  # the drones' own timestep is replaced by ours on purpose
        warnings.simplefilter("ignore")
        return world.compile()


def controllers(model, names):
    # menagerie names each drone body like its key, so after the prefix it is e.g. "x2_x2"
    return {name: FlightController(model, f"{name}_{name}", name) for name in names}


def selftest(names):
    """Each drone flies the course alone, in its own world, so they cannot crash into each other."""
    course = [("hover", 2, [0, 0, 0, 0]), ("forward", 4, [1, 0, 0, 0]), ("stop", 4, [0, 0, 0, 0]),
              ("up", 3, [0, 0, 1, 0]), ("turn left", 3, [0, 0, 0, 1]), ("left", 4, [0, 1, 0, 0]), ("stop", 4, [0, 0, 0, 0])]
    for name in names:
        model = build([name])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        pilot = controllers(model, [name])[name]
        for label, seconds, cmd in course:
            pilot.command[:] = cmd
            end = data.time + seconds
            while data.time < end:
                pilot.update(data, model.opt.timestep)
                mujoco.mj_step(model, data)
            pos, vel, rot, _ = pilot.state(data)
            yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
            print(f"{name:3s} {label:9s}: pos {np.round(pos, 2)}  speed {np.linalg.norm(vel):.2f} m/s  heading {yaw:6.1f} deg")


def fly(model, names):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pilots = controllers(model, names)
    active = [0]  # index into names
    held, presses, releases = set(), {}, {}

    viewer = mujoco.viewer.launch_passive(model, data)

    def follow():
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pilots[names[active[0]]].body
            viewer.cam.distance, viewer.cam.elevation = (3.0, -20) if names[active[0]] == "x2" else (0.8, -20)

    def reset():
        with viewer.lock():
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            for pilot in pilots.values():
                pilot.reset_target(data)

    root = tk.Tk()
    root.title("Drone controls")
    root.geometry("+40+40")
    tk.Label(root, text=HELP, justify="left", font=("monospace", 11)).pack(padx=12, pady=8)
    status = tk.Label(root, justify="left", font=("monospace", 11, "bold"), fg="#1f5fa8")
    status.pack(padx=12, pady=(0, 10), anchor="w")

    # Holding a key makes the system send repeated release+press pairs, so a release only
    # counts if no new press follows within 60 ms.
    def on_press(event):
        key = event.keysym if event.keysym in KEYS else event.keysym.lower()
        if key == "tab":
            active[0] = (active[0] + 1) % len(names)
            follow()
            return "break"
        if key == "r":
            reset()
        if key in KEYS:
            held.add(key)
            presses[key] = presses.get(key, 0) + 1

    def on_release(event):
        key = event.keysym if event.keysym in KEYS else event.keysym.lower()
        if key in KEYS:
            releases[key] = presses.get(key, 0)
            root.after(60, lambda: held.discard(key) if presses.get(key, 0) == releases[key] else None)

    root.bind("<KeyPress>", on_press)
    root.bind("<KeyRelease>", on_release)
    root.bind("<Tab>", on_press)

    follow()
    start = time.time() - data.time

    def tick():
        if not viewer.is_running():
            root.destroy()
            return
        command = np.zeros(4)
        for key in held:
            index, value = KEYS[key]
            command[index] += value
        for i, name in enumerate(names):
            pilots[name].command[:] = command if i == active[0] else 0
        with viewer.lock():
            steps = 0
            while data.time < time.time() - start and steps < 50:  # real time; skip ahead if we fall behind
                for pilot in pilots.values():
                    pilot.update(data, model.opt.timestep)
                mujoco.mj_step(model, data)
                steps += 1
        viewer.sync()
        name = names[active[0]]
        pos, vel, _, _ = pilots[name].state(data)
        status.config(text=f"Flying: {DRONES[name][2]}\nheight {pos[2]:5.2f} m   speed {np.linalg.norm(vel):4.2f} m/s")
        root.after(10, tick)

    root.after(10, tick)
    root.mainloop()
    viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone", choices=["both", "x2", "cf2"], default="both")
    parser.add_argument("--selftest", action="store_true", help="fly a fixed course without a window")
    args = parser.parse_args()
    names = ["x2", "cf2"] if args.drone == "both" else [args.drone]
    if args.selftest:
        selftest(names)
    else:
        fly(build(names), names)


if __name__ == "__main__":
    main()
