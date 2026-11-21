"""The simulated world: sky, floor, the protected zone, and the menagerie drones.

Shared by the keyboard viewer (view_drones.py) and the intruder dataset (intruder.py).
Setup: drone/models/README.md.
"""

import warnings
from pathlib import Path

import mujoco

from flight import FlightController

MENAGERIE = Path(__file__).parent / "models" / "mujoco_menagerie"

# name -> (xml, start position x y z in metres, label)
DRONES = {
    "x2": ("skydio_x2/x2.xml", (0.0, 0.0, 1.0), "Skydio X2"),
    "cf2": ("bitcraze_crazyflie_2/cf2.xml", (0.0, 1.5, 1.0), "Crazyflie 2"),  # beside the X2, not in its path
}

# The protected site: a cylinder at the origin. The radar sits at its centre.
ZONE_RADIUS = 10.0  # m
ZONE_HEIGHT = 30.0  # m
ARENA_RADIUS = 60.0  # m, intruders start on this circle

# Sky, checker floor and lights copied from menagerie's skydio_x2/scene.xml.
# Air density/viscosity and RK4 as in the menagerie drone models.
WORLD_XML = """
<mujoco model="drones">
  <option timestep="{timestep}" density="1.225" viscosity="1.8e-5" integrator="RK4"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <map zfar="500"/>
    <global offwidth="1280" offheight="720"/>
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

# Visual only (no collisions): red zone cylinder, faint arena disc, the radar post.
ZONE_XML = f"""
<mujoco>
  <worldbody>
    <geom name="zone" type="cylinder" size="{ZONE_RADIUS} {ZONE_HEIGHT / 2}" pos="0 0 {ZONE_HEIGHT / 2}"
      rgba="1 0.15 0.15 0.18" contype="0" conaffinity="0"/>
    <geom name="arena" type="cylinder" size="{ARENA_RADIUS} 0.01" pos="0 0 0.01"
      rgba="1 1 0.4 0.06" contype="0" conaffinity="0"/>
    <geom name="radar" type="box" size="0.4 0.4 1.5" pos="0 0 1.5" rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def build(names, zone=False, timestep=0.002):
    """One world with sky and floor (and the zone if asked), each drone attached with its own name prefix.

    The Crazyflie needs timestep 0.002; the X2 alone runs fine at 0.005.
    """
    world = mujoco.MjSpec.from_string(WORLD_XML.format(timestep=timestep))
    if zone:
        world.worldbody.add_frame().attach_body(mujoco.MjSpec.from_string(ZONE_XML).worldbody, "zone_", "")
    for name in names:
        xml, pos, _ = DRONES[name]
        child = mujoco.MjSpec.from_file(str(MENAGERIE / xml))
        world.worldbody.add_frame(pos=list(pos)).attach_body(child.worldbody.first_body(), f"{name}_", "")
    with warnings.catch_warnings():  # the drones' own timestep is replaced by ours on purpose
        warnings.simplefilter("ignore")
        return world.compile()


def controllers(model, names, gains=None):
    # menagerie names each drone body like its key, so after the prefix it is e.g. "x2_x2"
    return {name: FlightController(model, f"{name}_{name}", name, gains) for name in names}
