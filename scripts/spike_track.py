"""Spike 1: does the torque-controlled arm track ruckig trajectories within Franka limits?

Moves pick-height -> lift -> above bin and back, at several alphas, with no item in hand.
Reports tracking error, torque saturation and peak TCP acceleration.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from sim.control import ArmController, Gripper
from sim.franka import BIN_XY, CONVEYOR_TOP, PICK_XY, Q_HOME, SHELF_TOP, TAU_MAX, build_cell
from sim.motion import IK, JointTrajectory, steps

m, ids = build_cell()
d = mujoco.MjData(m)
ik = IK()
pts = [np.r_[PICK_XY, CONVEYOR_TOP + 0.04], np.r_[PICK_XY, 0.45], np.r_[BIN_XY, 0.45], np.r_[BIN_XY, SHELF_TOP + 0.08]]
qs = []
for p in pts:
    qs.append(ik.solve(p, q_seed=qs[-1] if qs else None))
print("IK ok:", [np.round(q, 2).tolist() for q in qs[:2]])

for alpha in [float(a) for a in (sys.argv[1:] or ["0.7", "0.8", "0.9"])]:
    mujoco.mj_resetData(m, d)
    d.qpos[:7] = qs[0]
    d.qpos[ids["item_qpos"]:ids["item_qpos"] + 3] = [0.0, -0.6, 0.5]  # park the item out of the way
    mujoco.mj_forward(m, d)
    arm, grip = ArmController(m), Gripper(m, ids["grip_tendon"])
    grip.sync(d)
    arm.reset(d)
    path = qs + qs[-2::-1]
    err_max, sat, n, v_prev, acc = 0.0, 0, 0, None, []
    t_total = 0.0
    for q0, q1 in zip(path[:-1], path[1:]):
        tr = JointTrajectory(q0, q1, alpha)
        t_total += tr.duration
        for k in range(steps(tr.duration) + 200):
            qd, dqd, ddqd = tr.at(k * m.opt.timestep)
            tau = arm.torque(d, qd, dqd, ddqd)
            d.ctrl[:7] = tau
            d.ctrl[7] = grip.ctrl(d)
            mujoco.mj_step(m, d)
            err_max = max(err_max, np.abs(qd - d.qpos[:7]).max())
            sat += np.any(np.abs(tau) >= TAU_MAX - 1e-6)
            n += 1
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, ids["tcp"], v, 0)
            if v_prev is not None:
                acc.append((v[3:] - v_prev) / m.opt.timestep)
            v_prev = v[3:].copy()
    a = np.linalg.norm(np.array(acc), axis=1)
    a_smooth = np.convolve(a, np.ones(10) / 10, mode="valid")
    print(f"alpha {alpha:.1f}: motion time {t_total:.2f} s | max joint tracking error {np.degrees(err_max):.2f} deg | "
          f"torque-saturated steps {100 * sat / n:.1f}% | peak TCP accel {a_smooth.max():.1f} m/s^2")
