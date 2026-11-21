"""Franka Panda limits and the pick-and-place cell model.

The robot is the official MuJoCo Menagerie `franka_emika_panda` MJCF (fetched by
scripts/fetch_assets.sh). We change it the way the real robot is driven:
torque motors on the 7 joints (libfranka torque interface, 1 kHz) and a
force-limited gripper, then add the cell: conveyor, shelf with bin, item,
a wrist force/torque site and explicit finger-pad/item contacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"

# Franka Panda joint-space limits (FCI docs, "Robot and interface specifications").
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
DQ_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
DDQ_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
DDDQ_MAX = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
DTAU_MAX = 1000.0  # Nm/s, torque rate limit
CART_ACC_MAX = 13.0  # m/s^2, Cartesian translational acceleration limit

# Franka Hand.
HAND_STROKE = 0.08  # m, fully open width
HAND_FORCE_CONT = 70.0  # N, continuous grasp force
HAND_FORCE_MAX = 140.0  # N, peak
TCP_OFFSET = 0.1034  # m from the hand body origin along hand z

Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
DT = 0.001  # 1 kHz, like the real control loop

# Cell layout (world = robot base frame, z up, table/floor at z = 0).
CONVEYOR_TOP = 0.02
PICK_XY = np.array([0.50, -0.35])
SHELF_TOP = 0.30
BIN_XY = np.array([0.45, 0.42])
BIN_INNER = 0.20
BIN_WALL_H = 0.06


@dataclass
class ContactParams:
    noslip_iterations: int = 0
    impratio: float = 10.0
    pad_solref: tuple = (0.004, 1.0)
    pad_solimp: tuple = (0.95, 0.99, 0.001, 0.5, 2.0)
    torsion_radius: float = 0.006  # m, effective contact-patch radius for torsional friction


def _set_motor(act, limit: float) -> None:
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm = np.r_[1.0, np.zeros(9)]
    act.biastype = mujoco.mjtBias.mjBIAS_NONE
    act.biasprm = np.zeros(10)
    act.ctrlrange = np.array([-limit, limit])
    act.forcerange = np.array([-limit, limit])
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


def _robot_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MENAGERIE / "scene.xml"))
    for key in list(spec.keys):
        spec.delete(key)
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0.005, 0.005])
    spec.body("hand").add_site(name="ft", pos=[0, 0, 0], size=[0.005, 0.005, 0.005])
    return spec


def build_ik_model() -> mujoco.MjModel:
    """Robot only, original actuators: used by mink for inverse kinematics."""
    return _robot_spec().compile()


def build_cell(cp: ContactParams | None = None) -> tuple[mujoco.MjModel, dict]:
    cp = cp or ContactParams()
    spec = _robot_spec()
    spec.option.timestep = DT
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = cp.impratio
    spec.option.noslip_iterations = cp.noslip_iterations

    for i, act in enumerate(spec.actuators):
        if i < 7:
            _set_motor(act, TAU_MAX[i])
        else:  # gripper tendon "split": joint force = 0.5 * tendon force on each finger
            _set_motor(act, 2.0 * HAND_FORCE_MAX)

    wb = spec.worldbody
    BOX = mujoco.mjtGeom.mjGEOM_BOX
    wb.add_geom(name="conveyor", type=BOX, size=[0.12, 0.25, CONVEYOR_TOP / 2],
                pos=[PICK_XY[0], -0.40, CONVEYOR_TOP / 2], rgba=[0.25, 0.27, 0.3, 1])
    wb.add_geom(name="shelf", type=BOX, size=[0.14, 0.14, SHELF_TOP / 2],
                pos=[BIN_XY[0], BIN_XY[1], SHELF_TOP / 2], rgba=[0.75, 0.75, 0.72, 1])
    h, t, s = BIN_WALL_H, 0.005, BIN_INNER / 2
    for name, size, off in [("bin_n", [s + t, t, h / 2], [0, s + t, h / 2]), ("bin_s", [s + t, t, h / 2], [0, -s - t, h / 2]),
                            ("bin_e", [t, s, h / 2], [s + t, 0, h / 2]), ("bin_w", [t, s, h / 2], [-s - t, 0, h / 2])]:
        wb.add_geom(name=name, type=BOX, size=size, pos=[BIN_XY[0] + off[0], BIN_XY[1] + off[1], SHELF_TOP + off[2]],
                    rgba=[0.8, 0.66, 0.45, 1])

    item = wb.add_body(name="item", pos=[PICK_XY[0], PICK_XY[1], CONVEYOR_TOP + 0.04])
    item.add_freejoint(name="item_free")
    # explicit inertial with an off-origin CoM so mass and CoM can be changed per item at run time
    item.explicitinertial = True
    item.mass, item.ipos, item.inertia = 0.5, [1e-4, 1e-4, 1e-4], [1e-3, 1e-3, 1e-3]
    item.add_geom(name="item_geom", type=BOX, size=[0.03, 0.04, 0.04], density=0, rgba=[0.84, 0.72, 0.55, 1])

    # Finger contacts with the item go through explicit pairs so friction can be set exactly.
    spec.add_exclude(bodyname1="left_finger", bodyname2="item")
    spec.add_exclude(bodyname1="right_finger", bodyname2="item")
    pad_names = []
    for side in ("left_finger", "right_finger"):
        for j, g in enumerate(spec.body(side).geoms):
            if g.type == BOX:  # the five fingertip pad boxes
                g.name = f"{side[0]}pad{j}"
                pad_names.append(g.name)
    for n in pad_names:
        spec.add_pair(name=f"pair_{n}", geomname1=n, geomname2="item_geom", condim=4,
                      friction=[0.5, 0.5, 0.5 * cp.torsion_radius, 1e-4, 1e-4],
                      solref=list(cp.pad_solref), solimp=list(cp.pad_solimp))

    spec.add_sensor(name="ft_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
    spec.add_sensor(name="ft_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")

    model = spec.compile()
    ids = {
        "item_body": model.body("item").id,
        "item_geom": model.geom("item_geom").id,
        "item_qpos": model.jnt_qposadr[model.joint("item_free").id],
        "item_dof": model.jnt_dofadr[model.joint("item_free").id],
        "tcp": model.site("tcp").id,
        "pairs": [model.pair(f"pair_{n}").id for n in pad_names],
        "pad_geoms": [model.geom(n).id for n in pad_names],
        "left_finger": model.body("left_finger").id,
        "right_finger": model.body("right_finger").id,
        "hand": model.body("hand").id,
        "grip_tendon": model.tendon("split").id,
    }
    return model, ids
