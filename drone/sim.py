"""The simulated world: sky, floor, the protected zone, and the menagerie drones.

Shared by the keyboard viewer (view_drones.py) and the intruder dataset (intruder.py).
Setup: drone/models/README.md.
"""

import os
import warnings
from pathlib import Path

# WSL on laptops with two GPUs: OpenGL otherwise draws on the built-in GPU, where the viewer window is slow
# and crashes (X error BadDrawable). This picks the NVIDIA GPU; it only affects WSL's D3D12 graphics driver.
os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

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
    <headlight diffuse="0.35 0.35 0.35" ambient="0.35 0.35 0.38" specular="0 0 0"/>
    <rgba haze="0.75 0.82 0.9 1"/>
    <map zfar="500" znear="0.0005"/>
    <quality shadowsize="8192"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.72 0.92" rgb2="0.95 0.97 1" width="512" height="3072"/>
    <texture type="2d" name="grass" builtin="flat" mark="random" random="0.35" rgb1="0.33 0.48 0.22"
      markrgb="0.26 0.38 0.17" width="512" height="512"/>
    <material name="grass" texture="grass" texuniform="true" texrepeat="40 40" reflectance="0"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 120" dir="-0.35 -0.25 -0.9" directional="true" castshadow="true"
      diffuse="0.85 0.82 0.75" specular="0.1 0.1 0.1"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grass"/>
  </worldbody>
</mujoco>
"""


POLE_GROUP = 3  # geom group of the zone's marker poles: the zoomed radar view hides them (they block the drone)


def _scenery():
    """Visual only (no collisions): the protected building with its radar, the zone, and the surroundings.
    Inside the arena only flat or low things: intruders fly as low as 3 m anywhere in it. Trees and other
    buildings stand beyond 75 m, where no intruder ever flies (max 70 m from the centre)."""
    vis = 'contype="0" conaffinity="0"'
    g = [
        # the zone: a red ring on the ground and thin red poles up to its ceiling (a see-through cylinder
        # tints everything and casts a shadow)
        f'<geom name="ring" type="cylinder" size="{ZONE_RADIUS} 0.01" pos="0 0 0.02" rgba="0.85 0.12 0.12 1" {vis}/>',
        f'<geom name="pad" type="cylinder" size="{ZONE_RADIUS - 0.6} 0.01" pos="0 0 0.03" rgba="0.62 0.62 0.6 1" {vis}/>',
        *[f'<geom name="pole{i}" type="cylinder" size="0.06 {ZONE_HEIGHT / 2}" '
          f'pos="{ZONE_RADIUS * np.cos(a):.2f} {ZONE_RADIUS * np.sin(a):.2f} {ZONE_HEIGHT / 2}" '
          f'rgba="0.9 0.15 0.15 0.8" group="{POLE_GROUP}" {vis}/>' for i, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False))],
        # the protected building, 8 x 8 x 10 m, with window bands and a roof edge
        f'<geom name="building" type="box" size="4 4 5" pos="0 0 5" rgba="0.78 0.76 0.72 1" {vis}/>',
        *[f'<geom name="windows{i}" type="box" size="4.02 4.02 0.45" pos="0 0 {2 + 2.6 * i}" '
          f'rgba="0.25 0.35 0.45 1" {vis}/>' for i in range(3)],
        f'<geom name="roof" type="box" size="4.2 4.2 0.15" pos="0 0 10.1" rgba="0.45 0.45 0.45 1" {vis}/>',
        # the radar: a mast on the roof with a dish on top
        f'<geom name="mast" type="cylinder" size="0.18 1.8" pos="0 0 12" rgba="0.85 0.85 0.85 1" {vis}/>',
        f'<geom name="dish" type="cylinder" size="1.1 0.08" pos="0 0 13.9" euler="70 0 0" rgba="0.95 0.95 0.95 1" {vis}/>',
        f'<geom name="feed" type="sphere" size="0.22" pos="0 -0.35 14.1" rgba="0.3 0.3 0.3 1" {vis}/>',
        # an access road from the building out of the arena
        f'<geom name="road" type="box" size="55 3 0.01" pos="62 0 0.015" rgba="0.3 0.3 0.31 1" {vis}/>',
        f'<geom name="roadline" type="box" size="55 0.12 0.011" pos="62 0 0.016" rgba="0.92 0.9 0.8 1" {vis}/>',
    ]
    rng = np.random.default_rng(7)  # fixed: the same scenery every time
    for i, a in enumerate(np.linspace(0, 2 * np.pi, 70, endpoint=False) + rng.uniform(-0.04, 0.04, 70)):
        if abs(np.sin(a)) < 0.06 and np.cos(a) > 0:
            continue  # leave the road free
        r = rng.uniform(78, 95)
        h = rng.uniform(4, 9)
        x, y = r * np.cos(a), r * np.sin(a)
        g += [f'<geom name="trunk{i}" type="cylinder" size="0.35 {h / 3}" pos="{x:.1f} {y:.1f} {h / 3:.2f}" '
              f'rgba="0.38 0.27 0.18 1" {vis}/>',
              f'<geom name="crown{i}" type="ellipsoid" size="{h / 3:.2f} {h / 3:.2f} {h / 2.2:.2f}" '
              f'pos="{x:.1f} {y:.1f} {h * 0.9:.2f}" rgba="{rng.uniform(0.15, 0.25):.2f} {rng.uniform(0.35, 0.5):.2f} 0.15 1" {vis}/>']
    for i, a in enumerate(np.linspace(0.4, 2 * np.pi - 0.4, 9)):
        r, w, d, h = rng.uniform(105, 125), rng.uniform(8, 16), rng.uniform(8, 16), rng.uniform(8, 28)
        g.append(f'<geom name="house{i}" type="box" size="{w / 2:.1f} {d / 2:.1f} {h / 2:.1f}" '
                 f'pos="{r * np.cos(a):.1f} {r * np.sin(a):.1f} {h / 2:.1f}" euler="0 0 {np.degrees(a):.0f}" '
                 f'rgba="{rng.uniform(0.6, 0.8):.2f} {rng.uniform(0.6, 0.75):.2f} {rng.uniform(0.55, 0.7):.2f} 1" {vis}/>')
    return "<mujoco><worldbody>" + "\n".join(g) + "</worldbody></mujoco>"


ZONE_XML = _scenery()
RADAR_TOP = (0.0, 0.0, 15.4)  # m: just above the radar dish (its rim reaches 14.9 m) on the roof


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
