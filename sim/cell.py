"""The pick-and-place cell: item setup, 1 kHz stepping, motion primitives and measurements."""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .control import ArmController, Gripper
from .franka import CONVEYOR_TOP, DT, PICK_XY, Q_HOME, ContactParams, build_cell
from .motion import IK, JointTrajectory, steps


@dataclass
class Item:
    width: float = 0.06   # m, along the gripping axis (world x); Franka Hand max 0.08
    depth: float = 0.05   # m, world y
    height: float = 0.08  # m
    mass: float = 0.5     # kg
    mu: float = 0.5       # friction coefficient pad/item (packaging x pad wear)
    com: tuple = (0.0, 0.0, 0.0)  # centre-of-mass offset in the item frame, m


@dataclass
class Trace:
    rel0: np.ndarray | None = None      # item position in the TCP frame right after the grasp
    slip: list = field(default_factory=list)
    tilt: list = field(default_factory=list)


class Cell:
    def __init__(self, cp: ContactParams | None = None, seed: int = 0):
        self.m, self.ids = build_cell(cp)
        self.d = mujoco.MjData(self.m)
        self.ik = IK()
        self.arm = ArmController(self.m)
        self.grip = Gripper(self.m, self.ids["grip_tendon"])
        self.rng = np.random.default_rng(seed)
        self.cp = cp or ContactParams()
        self.t = 0.0
        self.trace = Trace()
        self.q_cmd = Q_HOME.copy()

    # ---------- item ----------
    def set_item(self, it: Item) -> None:
        m, ids = self.m, self.ids
        g, b = ids["item_geom"], ids["item_body"]
        half = np.array([it.width, it.depth, it.height]) / 2
        m.geom_size[g] = half
        m.geom_rbound[g] = np.linalg.norm(half)
        m.geom_aabb[g] = np.r_[0, 0, 0, half]
        m.body_mass[b] = it.mass
        w, dd, h = it.width, it.depth, it.height
        m.body_inertia[b] = it.mass / 12 * np.array([dd * dd + h * h, w * w + h * h, w * w + dd * dd])
        m.body_ipos[b] = np.array(it.com)
        m.geom_friction[g] = [it.mu, 0.005, 1e-4]
        for p in ids["pairs"]:
            m.pair_friction[p] = [it.mu, it.mu, it.mu * self.cp.torsion_radius, 1e-4, 1e-4]
        mujoco.mj_setConst(m, self.d)
        self.item = it

    def reset(self, item: Item, q0: np.ndarray | None = None) -> None:
        self.set_item(item)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:7] = Q_HOME if q0 is None else q0
        self.d.qpos[7:9] = 0.04
        qa = self.ids["item_qpos"]
        self.d.qpos[qa:qa + 3] = [PICK_XY[0], PICK_XY[1], CONVEYOR_TOP + item.height / 2 + 1e-4]
        self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.m, self.d)
        self.arm.reset(self.d)
        self.grip.move(0.08)
        self.grip.sync(self.d)
        self.q_cmd = self.d.qpos[:7].copy()
        self.t, self.trace = 0.0, Trace()

    # ---------- stepping ----------
    def _step(self, qd, dqd, ddqd, record: bool) -> None:
        self.d.ctrl[:7] = self.arm.torque(self.d, qd, dqd, ddqd)
        self.d.ctrl[7] = self.grip.ctrl(self.d)
        mujoco.mj_step(self.m, self.d)
        self.t += DT
        if record:
            self._record()

    def move(self, q_target: np.ndarray, alpha: float, record: bool = False, settle: float = 0.05) -> float:
        tr = JointTrajectory(self.q_cmd, q_target, alpha)
        for k in range(steps(tr.duration)):
            self._step(*tr.at(k * DT), record)
        for _ in range(steps(settle)):
            self._step(q_target, np.zeros(7), np.zeros(7), record)
        self.q_cmd = q_target.copy()
        return tr.duration

    def hold(self, seconds: float, record: bool = False) -> None:
        z = np.zeros(7)
        for _ in range(steps(seconds)):
            self._step(self.q_cmd, z, z, record)

    def move_tcp(self, pos, alpha: float, record: bool = False, settle: float = 0.05) -> float:
        q = self.ik.solve(pos, q_seed=self.q_cmd)
        return self.move(q, alpha, record, settle)

    # ---------- measurements ----------
    def tcp_pose(self):
        sid = self.ids["tcp"]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def item_pose(self):
        b = self.ids["item_body"]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def item_in_tcp(self):
        p_t, R_t = self.tcp_pose()
        p_i, R_i = self.item_pose()
        return R_t.T @ (p_i - p_t), R_t.T @ R_i

    def mark_grasp(self) -> None:
        self.trace.rel0 = self.item_in_tcp()

    def _record(self) -> None:
        if self.trace.rel0 is None:
            return
        p, R = self.item_in_tcp()
        p0, R0 = self.trace.rel0
        self.trace.slip.append(np.linalg.norm(p - p0))
        dR = R0.T @ R
        self.trace.tilt.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))

    def finger_normal_force(self) -> float:
        """Sum of contact normal forces between finger pads and the item."""
        total, f = 0.0, np.zeros(6)
        pads = set(self.ids["pad_geoms"])
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if (c.geom1 in pads and c.geom2 == self.ids["item_geom"]) or (c.geom2 in pads and c.geom1 == self.ids["item_geom"]):
                mujoco.mj_contactForce(self.m, self.d, i, f)
                total += f[0]
        return total

    def ft(self) -> np.ndarray:
        return self.d.sensordata[:6].copy()
