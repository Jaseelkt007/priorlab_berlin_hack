"""1 kHz controllers, shaped like the real Franka interfaces.

Arm: joint-space computed-torque (impedance) tracking of a ruckig trajectory,
torque-limited and torque-rate-limited like libfranka. The model terms come from
the robot only: the controller does not know the item's mass, just as the real
robot does not know an unconfigured payload.

Gripper: like franka::Gripper. `move(width)` drives the fingers with a capped
force; `grasp(force)` squeezes with a set normal force per finger.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .franka import DT, DTAU_MAX, HAND_FORCE_MAX, TAU_MAX


class ArmController:
    def __init__(self, model: mujoco.MjModel, wn: float = 40.0, zeta: float = 1.0):
        self.m = model
        self.kp = wn * wn
        self.kd = 2.0 * zeta * wn
        self.M = np.zeros((model.nv, model.nv))
        self.tau_prev = np.zeros(7)

    def reset(self, d: mujoco.MjData) -> None:
        """Start from the torque that holds the current pose, as the real robot does when a controller starts."""
        mujoco.mj_forward(self.m, d)
        self.tau_prev = (d.qfrc_bias[:7] - d.qfrc_passive[:7]).copy()

    def torque(self, d: mujoco.MjData, q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray) -> np.ndarray:
        mujoco.mj_fullM(self.m, d, self.M)
        M7 = self.M[:7, :7]
        q, dq = d.qpos[:7], d.qvel[:7]
        a = ddq_d + self.kp * (q_d - q) + self.kd * (dq_d - dq)
        tau = M7 @ a + d.qfrc_bias[:7] - d.qfrc_passive[:7]
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        step = DTAU_MAX * DT
        tau = np.clip(tau, self.tau_prev - step, self.tau_prev + step)
        self.tau_prev = tau
        return tau


class Gripper:
    """Tendon 'split' couples both fingers: tendon length = width / 2."""

    def __init__(self, model: mujoco.MjModel, tendon_id: int):
        self.m, self.tid = model, tendon_id
        self.mode, self.target_w, self.force = "move", 0.08, 0.0
        self.speed = 0.1  # m/s width change, Franka Hand max is about 0.1 m/s
        self._w_cmd = 0.08

    def move(self, width: float, speed: float = 0.1) -> None:
        self.mode, self.target_w, self.speed = "move", width, speed

    def grasp(self, force: float) -> None:
        self.mode, self.force = "grasp", min(force, HAND_FORCE_MAX)

    def width(self, d: mujoco.MjData) -> float:
        return 2.0 * d.ten_length[self.tid]

    def ctrl(self, d: mujoco.MjData) -> float:
        w, wdot = self.width(d), 2.0 * d.ten_velocity[self.tid]
        if self.mode == "grasp":
            # joint force = 0.5 * tendon force per finger, so tendon force = -2 * normal force; small damping
            return -2.0 * self.force - 20.0 * wdot
        step = self.speed * DT
        self._w_cmd = float(np.clip(self.target_w, self._w_cmd - step, self._w_cmd + step))
        f = 4000.0 * (self._w_cmd - w) / 2.0 - 60.0 * wdot
        return float(np.clip(f, -2.0 * 30.0, 2.0 * 30.0))  # moving force capped like the real hand

    def sync(self, d: mujoco.MjData) -> None:
        self._w_cmd = self.width(d)
