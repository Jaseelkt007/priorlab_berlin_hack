"""Motion planning: mink inverse kinematics + ruckig jerk-limited joint trajectories.

`alpha` scales Franka's velocity, acceleration and jerk limits together, like
franky's relative dynamics factor. alpha = 1 means the robot's full limits.
"""
from __future__ import annotations

import mink
import mujoco
import numpy as np
import ruckig

from .franka import DDDQ_MAX, DDQ_MAX, DQ_MAX, DT, Q_HOME, build_ik_model

# Top-down grasp: hand z points down; fingers close along world x.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


class IK:
    def __init__(self):
        self.model = build_ik_model()
        self.cfg = mink.Configuration(self.model)
        self.task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-6)
        self.posture = mink.PostureTask(self.model, cost=1e-3)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def solve(self, pos, R=R_DOWN, q_seed=None, iters: int = 300, tol: float = 1e-4) -> np.ndarray:
        q0 = np.r_[Q_HOME if q_seed is None else q_seed, 0.04, 0.04]
        self.cfg.update(q0)
        self.posture.set_target(np.r_[Q_HOME, 0.04, 0.04])
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(R)), np.asarray(pos)))
        for _ in range(iters):
            v = mink.solve_ik(self.cfg, [self.task, self.posture], 0.05, "daqp", damping=1e-3, limits=self.limits)
            self.cfg.integrate_inplace(v, 0.05)
            err = self.task.compute_error(self.cfg)
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 1e-3:
                break
        else:
            raise RuntimeError(f"IK did not converge for {pos}: err {np.linalg.norm(err[:3]):.4f} m")
        return self.cfg.q[:7].copy()


class JointTrajectory:
    """Rest-to-rest jerk-limited move between two joint configurations."""

    def __init__(self, q0, q1, alpha: float):
        a = float(np.clip(alpha, 0.05, 1.0))
        otg = ruckig.Ruckig(7)
        inp = ruckig.InputParameter(7)
        inp.current_position = list(q0)
        inp.current_velocity = [0.0] * 7
        inp.current_acceleration = [0.0] * 7
        inp.target_position = list(q1)
        inp.target_velocity = [0.0] * 7
        inp.target_acceleration = [0.0] * 7
        # 0.95 safety margin on the hardware limits, as recommended for real motion generators
        inp.max_velocity = list(0.95 * DQ_MAX * a)
        inp.max_acceleration = list(0.95 * DDQ_MAX * a)
        inp.max_jerk = list(0.95 * DDDQ_MAX * a)
        self.traj = ruckig.Trajectory(7)
        res = otg.calculate(inp, self.traj)
        if res not in (ruckig.Result.Working, ruckig.Result.Finished):
            raise RuntimeError(f"ruckig failed: {res}")
        self.duration = self.traj.duration

    def at(self, t: float):
        p, v, a = self.traj.at_time(min(max(t, 0.0), self.duration))
        return np.array(p), np.array(v), np.array(a)


def steps(duration: float) -> int:
    return int(np.ceil(duration / DT))
