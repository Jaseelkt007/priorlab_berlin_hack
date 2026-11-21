"""Suction-cup physics for the delta cell (MuJoCo).

The effector is a point mass on x/y/z slide joints, force-driven to follow a reference
trajectory exactly (industrial deltas track their path very stiffly; arm dynamics are not
modelled). The suction cup lip is a ring of small contact spheres at the cup radius, and
MuJoCo's adhesion actuator supplies the vacuum holding force at those contacts. Friction
between lip and product is set exactly through explicit contact pairs.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

DT = 0.0005
N_LIP = 8
LIP_R = 0.0015  # m, radius of each lip contact sphere


@dataclass
class Product:
    size: tuple = (0.06, 0.045, 0.008)  # m, full extents x, y, z
    mass: float = 0.012                  # kg
    mu: float = 0.6                      # lip/product friction
    com_z: float = 0.0                   # CoM offset along z from the box centre (m), negative = lower


def build(cup_d: float = 0.015, prod: Product | None = None, solref=(0.002, 1.0)) -> tuple[mujoco.MjModel, dict]:
    prod = prod or Product()
    r = cup_d / 2
    lips = "\n".join(
        f'<geom name="lip{i}" type="sphere" size="{LIP_R}" pos="{r * np.cos(2 * np.pi * i / N_LIP):.5f} {r * np.sin(2 * np.pi * i / N_LIP):.5f} 0" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>'
        for i in range(N_LIP))
    pairs = "\n".join(
        f'<pair geom1="lip{i}" geom2="prod" condim="3" friction="{prod.mu} {prod.mu} 0.005 0.0001 0.0001" solref="{solref[0]} {solref[1]}" solimp="0.95 0.99 0.0005"/>'
        for i in range(N_LIP))
    sx, sy, sz = (s / 2 for s in prod.size)
    xml = f"""
<mujoco model="suction">
  <option timestep="{DT}" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" friction="0.5 0.005 0.0001"/>
    <body name="eff" pos="0 0 {prod.size[2] + 0.002 + LIP_R}">
      <joint name="ex" type="slide" axis="1 0 0" damping="0"/>
      <joint name="ey" type="slide" axis="0 1 0" damping="0"/>
      <joint name="ez" type="slide" axis="0 0 1" damping="0"/>
      <inertial pos="0 0 0.02" mass="0.5" diaginertia="1e-3 1e-3 1e-3"/>
      <geom type="cylinder" size="{r + 0.002} 0.004" pos="0 0 {0.004 + LIP_R}" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      {lips}
    </body>
    <body name="prod" pos="0 0 {sz}">
      <freejoint name="prod_free"/>
      <inertial pos="0 0 {prod.com_z}" mass="{prod.mass}" diaginertia="{prod.mass / 12 * (prod.size[1]**2 + prod.size[2]**2):.3e} {prod.mass / 12 * (prod.size[0]**2 + prod.size[2]**2):.3e} {prod.mass / 12 * (prod.size[0]**2 + prod.size[1]**2):.3e}"/>
      <geom name="prod" type="box" size="{sx} {sy} {sz}" rgba="0.85 0.66 0.36 1"/>
    </body>
  </worldbody>
  <contact>{pairs}</contact>
  <actuator>
    <motor name="fx" joint="ex" gear="1" ctrlrange="-1000 1000"/>
    <motor name="fy" joint="ey" gear="1" ctrlrange="-1000 1000"/>
    <motor name="fz" joint="ez" gear="1" ctrlrange="-1000 1000"/>
    <adhesion name="vac" body="eff" ctrlrange="0 1" gain="1"/>
  </actuator>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    ids = {"eff": m.body("eff").id, "prod": m.body("prod").id, "prod_geom": m.geom("prod").id,
           "lips": [m.geom(f"lip{i}").id for i in range(N_LIP)], "prod_qpos": m.jnt_qposadr[m.joint("prod_free").id]}
    return m, ids


class Effector:
    """Exact tracking of a reference (p, v, a) for the effector point mass: computed force + stiff PD."""

    def __init__(self, m: mujoco.MjModel, mass: float = 0.5, kp: float = 4e5, kd: float = 1.2e3):
        self.m, self.mass, self.kp, self.kd = m, mass, kp, kd

    def force(self, d: mujoco.MjData, p, v, a) -> np.ndarray:
        q, qd = d.qpos[:3], d.qvel[:3]
        return self.mass * (np.asarray(a) + np.array([0, 0, 9.81])) + self.kp * (np.asarray(p) - q) + self.kd * (np.asarray(v) - qd)


def lip_contacts(m: mujoco.MjModel, d: mujoco.MjData, ids: dict) -> int:
    lips, pg, n = set(ids["lips"]), ids["prod_geom"], set()
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in lips and c.geom2 == pg) or (c.geom2 in lips and c.geom1 == pg):
            n.add(c.geom1 if c.geom1 in lips else c.geom2)
    return len(n)
