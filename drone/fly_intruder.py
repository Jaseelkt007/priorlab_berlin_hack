"""Fly the intruder Crazyflie yourself; the IMM predicts live where it will be in 1 s.

    python drone/fly_intruder.py

Two windows open: the MuJoCo viewer and a small "Intruder controls" window. Click the controls window,
then hold keys: arrows = forward / back / left / right, W / S = up / down, A / D = turn, R = restart.

You start 35 m south of the protected zone (the red cylinder), facing it, with the same speed and tilt
limits as the scripted intruder (3 m/s). The radar at the zone measures you 10 times a second with the
same noise and dropouts as in the dataset; the IMM sees only those readings, one at a time.
On screen, as in demo_imm.py: green = your real path, white = radar, blue = IMM's guess for 1 s ahead with
its 90% range, yellow = the guess it made 1 s ago for now (the yellow line is its error), and the alarm.
"""

import time
import tkinter as tk

import mujoco
import mujoco.viewer
import numpy as np

from compare import imm_breach_probability
from demo_imm import AHEAD, CHASE_CAM, CHECK, ZNEAR, draw
from flight import GAINS
from imm import HORIZONS, TUNED, Tracker
from intruder import RADAR_EVERY, SETUP, Radar, outcome_of
from sim import ZONE_RADIUS, build, controllers
from view_drones import Keyboard

START = np.array([0.0, -35.0, 10.0])  # m; facing the zone (+y)
MAX_STEPS = 6000  # radar readings kept: 10 minutes
HELP = """Click this window, then hold keys:
  Arrow keys   forward / back / left / right
  W / S        up / down
  A / D        turn left / right
  R            restart at the start point
Reach the red zone; the IMM tries to predict you.
Mouse in the viewer: drag to rotate, scroll to zoom."""


class Session:
    """One attempt: the drone at the start point, a fresh radar track and IMM."""

    def __init__(self, model, data, pilot):
        mujoco.mj_resetData(model, data)
        q = pilot.qpos
        data.qpos[q : q + 3] = START
        data.qpos[q + 3 : q + 7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # heading +y, toward the zone
        mujoco.mj_forward(model, data)
        pilot.reset_target(data)
        self.radar = Radar(np.random.default_rng())
        self.tracker = Tracker(1, TUNED)
        self.rng = np.random.default_rng(0)
        H = len(HORIZONS)
        self.mean = np.full((MAX_STEPS, H, 3), np.nan)
        self.cov = np.full((MAX_STEPS, H, 3, 3), np.nan)
        self.ok = np.zeros(MAX_STEPS, bool)
        self.p_alarm = np.zeros(MAX_STEPS)
        self.meas = np.full((MAX_STEPS, 3), np.nan)
        self.log = {"t": [], "true": []}
        self.errors = []
        self.next_radar = 0.0
        self.hi = {h: int(np.argmin(np.abs(HORIZONS - h))) for h in (AHEAD, CHECK)}
        self.pred = (self.mean, self.cov, self.ok, self.p_alarm, self.hi, round(CHECK / RADAR_EVERY))
        self.breach_time = None

    def reading(self, t, pos):
        """A radar reading at time t: measure, update the IMM, store its prediction. Returns its index."""
        k = len(self.log["t"])
        if k >= MAX_STEPS:
            return None
        z = self.radar(pos)
        mean, cov, _ = self.tracker.step(z[None])
        self.log["t"].append(t)
        self.log["true"].append(pos.copy())
        self.meas[k] = z
        if not np.isnan(mean).any():
            self.mean[k], self.cov[k], self.ok[k] = mean[0], cov[0], True
            self.p_alarm[k] = imm_breach_probability(mean, cov, self.rng)[0]
        return k


def main():
    model = build(["cf2"], zone=True, timestep=SETUP["cf2"][0])
    model.vis.map.znear = ZNEAR
    data = mujoco.MjData(model)
    pilot = controllers(model, ["cf2"], GAINS["cf2_intruder"])["cf2"]
    s = [Session(model, data, pilot)]

    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pilot.body
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = CHASE_CAM, -15, 90

    root = tk.Tk()
    root.title("Intruder controls")
    root.geometry("+20+20")
    root.attributes("-topmost", True)  # stays above the 3D window, where key presses would be lost
    tk.Label(root, text=HELP, justify="left", font=("monospace", 11)).pack(padx=12, pady=8)
    status = tk.Label(root, justify="left", font=("monospace", 11, "bold"), fg="#1f5fa8")
    status.pack(padx=12, pady=(0, 10), anchor="w")
    legend = ("green: your real path   white: radar   blue: IMM says in 1 s + 90% range   "
              "yellow: IMM's guess for now, made 1 s ago")

    def on_key(key):
        if key == "r":
            with viewer.lock():
                s[0] = Session(model, data, pilot)
            start[0] = time.time()

    keyboard = Keyboard(root, on_key)
    root.after(500, root.focus_force)  # take the keyboard once the window is up
    start = [time.time()]

    def tick():
        if not viewer.is_running():
            root.destroy()
            return
        ses = s[0]
        command = keyboard.command()
        command[:2] /= max(1.0, np.hypot(*command[:2]))  # diagonal keys: same 3 m/s limit as the intruder
        pilot.command[:] = command
        new_k = None
        with viewer.lock():
            steps = 0
            while data.time < time.time() - start[0] and steps < 100:  # real time; skip ahead if behind
                pos = pilot.state(data)[0]
                if data.time >= ses.next_radar - 1e-9:
                    new_k = ses.reading(data.time, pos)
                    ses.next_radar += RADAR_EVERY
                if ses.breach_time is None and outcome_of(pos) == "breach":
                    ses.breach_time = data.time
                pilot.update(data, model.opt.timestep)
                mujoco.mj_step(model, data)
                steps += 1
            if new_k is not None:
                log = {k: np.asarray(v) for k, v in ses.log.items()}
                texts = draw(viewer.user_scn, new_k, log, ses.meas, ses.pred, ses.errors)
                held = " ".join(sorted(keyboard.held)) or "none (click the 'Intruder controls' window)"
                texts.insert(0, f"keys: {held}")
                if ses.breach_time is not None:
                    texts.append(f"YOU BREACHED THE ZONE at {ses.breach_time:.1f} s (R to restart)")
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   "\n".join(texts), None),
                                  (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, legend, None)])
        viewer.sync()
        pos, vel, _, _ = pilot.state(data)
        err = f"{np.median(ses.errors):4.2f} m" if ses.errors else "-"
        status.config(text=f"distance to zone edge {max(np.linalg.norm(pos[:2]) - ZONE_RADIUS, 0):5.1f} m\n"
                           f"height {pos[2]:5.1f} m   speed {np.linalg.norm(vel):4.2f} m/s\n"
                           f"IMM 1 s ahead, median off by: {err}")
        root.after(10, tick)

    root.after(10, tick)
    root.mainloop()
    viewer.close()
    if s[0].errors:
        print(f"IMM 1 s ahead on your flight: median off by {np.median(s[0].errors):.2f} m "
              f"over {len(s[0].errors)} readings")


if __name__ == "__main__":
    main()
