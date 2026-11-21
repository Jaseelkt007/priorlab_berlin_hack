"""Flight controller for the MuJoCo Menagerie drones (Skydio X2, Crazyflie 2).

You give it a velocity command (forward/left in the drone's heading, up, yaw rate);
it holds position when the command is zero. Inside, a cascade:

    velocity command -> moving setpoint -> desired acceleration (position PD)
    -> desired thrust direction + heading -> rotation error (geometric attitude control, Lee et al. 2010)
    -> wanted [thrust, roll/pitch/yaw torque] -> motor commands (mixer built from the model itself)

The mixer reads each actuator's site and gear from the compiled model, so the same code drives
the X2 (4 rotor thrusts) and the Crazyflie (1 thrust + 3 moments).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

G = 9.81


@dataclass
class Gains:
    max_speed: float  # m/s, horizontal
    max_climb: float  # m/s
    max_yaw_rate: float  # rad/s
    max_tilt: float  # rad
    pos_w: float  # position loop natural frequency, rad/s
    att_w: float  # attitude loop natural frequency, rad/s
    pos_zeta: float = 1.0  # position loop damping ratio


# The Crazyflie model's moment actuators are weak (1e-5 N m max), so it tilts slowly:
# it gets a lower speed and tilt limit than the X2 (tuned in a headless sweep).
GAINS = {
    "x2": Gains(max_speed=3.0, max_climb=1.5, max_yaw_rate=1.5, max_tilt=np.radians(25), pos_w=2.0, att_w=10.0),
    "cf2": Gains(max_speed=0.5, max_climb=0.5, max_yaw_rate=0.3, max_tilt=np.radians(8), pos_w=0.8, att_w=5.0),
    # The X2 as a fast intruder (the "x2" entry is kept gentle for keyboard flying).
    "x2_intruder": Gains(max_speed=10.0, max_climb=4.0, max_yaw_rate=1.5, max_tilt=np.radians(35), pos_w=2.0, att_w=10.0),
}


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _vee(s):
    return np.array([s[2, 1], s[0, 2], s[1, 0]])


class FlightController:
    """Flies one drone body. Call `update(data, dt)` every physics step.

    Steer it with either `command` (keyboard style: forward, left, up, yaw rate, each -1..1, in the
    drone's heading) or `velocity` (a world-frame velocity in m/s; overrides `command` when set).
    """

    def __init__(self, model, body_name, gains_key, gains=None):
        self.model = model
        self.body = model.body(body_name).id
        joint = model.body_jntadr[self.body]
        self.qpos = model.jnt_qposadr[joint]
        self.dof = model.jnt_dofadr[joint]
        self.g = gains or GAINS[gains_key]

        self.mass = model.body_subtreemass[self.body]
        rot = _quat_to_mat(model.body_iquat[self.body])
        self.inertia = rot @ np.diag(model.body_inertia[self.body]) @ rot.T  # in the body frame

        self.actuators = [a for a in range(model.nu) if model.actuator_trnid[a, 0] in self._sites()]
        self.mixer = np.linalg.pinv(self._allocation())  # [thrust, tx, ty, tz] -> ctrl
        self.ctrl_low = model.actuator_ctrlrange[self.actuators, 0]
        self.ctrl_high = model.actuator_ctrlrange[self.actuators, 1]

        self.command = np.zeros(4)  # forward, left, up, yaw rate; each -1..1
        self.velocity = None  # world-frame m/s
        self.target = None  # position setpoint, set on first update
        self.target_yaw = 0.0

    def _sites(self):
        return {s for s in range(self.model.nsite) if self.model.site_bodyid[s] == self.body}

    def _allocation(self):
        """Body-frame [thrust along z, torque x, y, z about the centre of mass] per unit of each control."""
        com = self.model.body_ipos[self.body]
        cols = []
        for a in self.actuators:
            site = self.model.actuator_trnid[a, 0]
            rot = _quat_to_mat(self.model.site_quat[site])
            gear = self.model.actuator_gear[a]
            force = rot @ gear[:3]
            torque = rot @ gear[3:] + np.cross(self.model.site_pos[site] - com, force)
            cols.append([force[2], *torque])
        return np.array(cols).T

    def state(self, data):
        pos = data.qpos[self.qpos : self.qpos + 3].copy()
        vel = data.qvel[self.dof : self.dof + 3].copy()
        rot = data.xmat[self.body].reshape(3, 3).copy()
        omega = data.qvel[self.dof + 3 : self.dof + 6].copy()  # body frame
        return pos, vel, rot, omega

    def reset_target(self, data):
        pos, _, rot, _ = self.state(data)
        self.target = pos
        self.target_yaw = np.arctan2(rot[1, 0], rot[0, 0])

    def update(self, data, dt):
        g = self.g
        if self.target is None:
            self.reset_target(data)
        pos, vel, rot, omega = self.state(data)

        # 1. Velocity command in the world frame, then move the setpoint along it.
        fwd, left, up, yaw = np.clip(self.command, -1, 1)
        if self.velocity is not None:
            v_cmd = np.asarray(self.velocity, dtype=float).copy()
            h = np.linalg.norm(v_cmd[:2])
            if h > g.max_speed:
                v_cmd[:2] *= g.max_speed / h
            v_cmd[2] = np.clip(v_cmd[2], -g.max_climb, g.max_climb)
        else:
            c, s = np.cos(self.target_yaw), np.sin(self.target_yaw)
            v_cmd = np.array([g.max_speed * (c * fwd - s * left), g.max_speed * (s * fwd + c * left), g.max_climb * up])
        self.target_yaw += g.max_yaw_rate * yaw * dt
        self.target = self.target + v_cmd * dt
        self.target = pos + np.clip(self.target - pos, -1.0, 1.0)  # no runaway if the drone lags
        self.target[2] = max(self.target[2], 0.3)  # stay above the floor

        # 2. Position PD -> desired acceleration, with tilt-limited horizontal part.
        acc = g.pos_w**2 * (self.target - pos) + 2 * g.pos_zeta * g.pos_w * (v_cmd - vel)
        acc[2] = max(acc[2], -0.6 * G)  # never ask for negative thrust: that would flip the drone
        max_h = G * np.tan(g.max_tilt)
        h = np.linalg.norm(acc[:2])
        if h > max_h:
            acc[:2] *= max_h / h
        force = self.mass * (acc + [0, 0, G])

        # 3. Desired orientation: body z along the force, body x toward the target heading.
        zb = force / np.linalg.norm(force)
        heading = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw), 0.0])
        yb = np.cross(zb, heading)
        yb /= np.linalg.norm(yb)
        rot_des = np.column_stack([np.cross(yb, zb), yb, zb])

        # 4. Geometric attitude control.
        err_r = 0.5 * _vee(rot_des.T @ rot - rot.T @ rot_des)
        alpha = -(g.att_w**2) * err_r - 2 * g.att_w * omega
        torque = self.inertia @ alpha + np.cross(omega, self.inertia @ omega)
        thrust = force @ rot[:, 2]

        # 5. Mixer -> motor commands.
        ctrl = self.mixer @ np.concatenate([[thrust], torque])
        data.ctrl[self.actuators] = np.clip(ctrl, self.ctrl_low, self.ctrl_high)
