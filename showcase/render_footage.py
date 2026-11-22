"""Clean footage for the launch video, to the editor's spec (clip-studio): 4K/60 CFR H.264, no overlays, slow motion
by sim_fps (frames per simulated second: 60 real time, 250 = 4.2x slow, 1000 = 16.7x slow), with sidecars:
  <clip>.frames.csv   one row per frame: pick, phase, seal, risk per speed, chosen speed, outcome, running counters,
                      screen-space cup/product positions and product bbox (in this clip's pixels)
  <clip>.events.json  frame-exact per-pick events (touch, seal_measured, decision, move_start, accel_reversal, release,
                      placed, drop_start, drop_impact), story events, per-frame camera (pos, lookat, fovy)
and footage/manifest.json (one entry per clip, with the sim code's git commit).

Physics: every pick is one sim.line.PickCell run (closed-chain delta, compliant cup, pneumatics, belt), recorded at
1 ms, then released (vacuum off 40 ms after arrival). A stream is the picks back to back: each pick starts and ends at
the place point, so the robot motion is continuous. Display-only geometry (declared in the manifest): the queue of
upcoming items on the infeed belt, placed items carried off on the outfeed, the robot's support frame, the studio.
Every number in the sidecars comes from the same runs that drew the frames.

Usage (server): .venv/bin/python showcase/render_footage.py PLAN.json [--workers 6]
A plan lists the picks (stream, idx, policy) and the clips (name, camera, window, sim_fps); see make_plan_test().
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import multiprocessing as mp

import numpy as np
import pandas as pd

MS = 1e-3
RELEASE_T = 0.30                      # s recorded after arrival (vacuum off at +40 ms, product settles)
ALPHAS = np.round(np.arange(0.2, 1.01, 0.1), 2)
V_OUT = 0.45                          # m/s outfeed belt (display only): ~0.28 m between placed bars
SKU_RGBA = {"bar 45 g": [0.9098, 0.4627, 0.1686, 1.0], "protein bar 60 g": [0.1216, 0.6392, 0.6039, 1.0]}   # plain one-colour wraps


# ================================================================================================ tasks
def task_for(stream: str, idx: int) -> dict:
    if stream == "shift:bar":
        from showcase.line_shift import shift_tasks
        return shift_tasks()[idx]
    from sim.scenarios import stream_tasks
    scen, prod = stream.split(":")
    return {t["idx"]: t for t in stream_tasks(scen, prod)}[idx]


# ================================================================================================ pass 1: physics
def simulate(job: dict) -> dict:
    """One pick at the chosen speed, recorded every 1 ms (qpos + what the sidecars need), then released."""
    import mujoco
    from sim.delta import plant_friction
    from sim.line import PickCell
    from sim.picklog import instance_setup
    task = task_for(job["stream"], job["idx"])
    setup, logged, hidden = instance_setup(task, job["alpha"])
    cell = PickCell(setup)
    m, d, P, cup = cell.m, cell.d, cell.P, cell.cup
    every = int(round(MS / P.dt))
    prod_g = m.geom("prod").id
    pad_g, belt_g, floor_g = m.geom("pad").id, m.geom("belt").id, m.geom("floor").id
    lip_g = set(int(g) for g in cup.lip_geoms)
    rec = dict(qpos=[], dp=[], sealed=[], contact=[], vz=[])

    def sample(c, t):
        if int(round(t / P.dt)) % every:
            return
        rec["qpos"].append(d.qpos.copy())
        rec["dp"].append(float(cup.dp))
        rec["sealed"].append(int((cup.gaps < 50e-6).sum()))
        hit = ""
        for k in range(d.ncon):
            g1, g2 = int(d.contact[k].geom1), int(d.contact[k].geom2)
            if prod_g in (g1, g2):
                o = g2 if g1 == prod_g else g1
                if o not in lip_g:
                    hit = {pad_g: "pad", belt_g: "belt", floor_g: "floor"}.get(o, "other")
        rec["contact"].append(hit)
        rec["vz"].append(float(d.qvel[m.jnt_dofadr[m.joint("prod_free").id] + 2]))
    out = cell.run(sample)
    for k in range(int(RELEASE_T / P.dt)):
        t = cell.T_total + k * P.dt
        if k == int(0.04 / P.dt):
            cup.vac.on = False
        d.qfrc_applied[:] = 0
        d.ctrl[:3] = cell.ctrl.torque(d, cell.T_total)
        plant_friction(m, d, P, cell.ctrl.vadr)
        cup.apply()
        mujoco.mj_step(m, d)
        cup.update(t, P.dt)
        sample(cell, t + P.dt)
    n = len(rec["qpos"])
    ts = np.arange(n) * MS
    # transfer acceleration reversal: the planned platform acceleration along the pick->place direction turns negative
    ax = np.array([cell.ref(t)[2][0] for t in ts])
    move = (ts >= cell.t_lift) & (ts <= cell.t_arrive)
    pos_acc = np.where(move & (ax > 1.0))[0]
    rev = None
    if len(pos_acc):
        after = np.where(move & (ts > ts[pos_acc[0]]) & (ax < 0))[0]
        rev = float(ts[after[0]]) if len(after) else None
    sealed = np.array(rec["sealed"])
    lost = np.where((ts > cell.t_lift) & (ts < cell.t_arrive + 0.04) & (sealed == 0))[0]
    drop_start = float(ts[lost[0]]) if out["dropped"] and len(lost) else None
    impact = None
    if drop_start is not None:
        hits = np.where((ts > drop_start) & (np.array(rec["contact"]) != ""))[0]
        impact = float(ts[hits[0]]) if len(hits) else None
    placed = None
    if not out["dropped"]:
        rel = cell.t_arrive + 0.04
        ok = np.where((ts > rel) & (np.array(rec["contact"]) == "pad") & (np.abs(np.array(rec["vz"])) < 0.02))[0]
        placed = float(ts[ok[0]]) if len(ok) else rel
    ev = dict(touch=cell.t_touch, seal_measured=cell.t_lift, decision=cell.t_lift, move_start=cell.t_lift,
              accel_reversal=rev, release=cell.t_arrive + 0.04, placed=placed, drop_start=drop_start,
              drop_impact=impact)
    return dict(job, setup=setup, logged=logged, out=out, model=m, qpos=np.array(rec["qpos"]),
                dp=np.array(rec["dp"]), dur=n * MS, events=ev, t_touch=cell.t_touch, t_lift=cell.t_lift,
                t_arrive=cell.t_arrive, y0=float(cell.prod0[1]), z0=float(cell.prod0[2]), vb=float(setup.v_belt),
                gsize=m.geom_size[prod_g].copy(), gtype=int(m.geom_type[prod_g]))


# ================================================================================================ look
def style_model(m, W: int, H: int, fovy: float, ortho: bool):
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.global_.fovy = fovy
    m.vis.global_.orthographic = int(ortho)
    m.vis.quality.offsamples = 8
    m.vis.quality.shadowsize = 8192
    m.vis.map.shadowclip = 3.0
    m.vis.map.znear, m.vis.map.zfar = 0.01, 50.0
    m.vis.quality.numslices = 64                      # smooth cylinders/ellipsoids at 4K
    m.vis.quality.numstacks = 48
    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = np.array([0.08, 0.08, 0.085]) * EXPOSURE
    m.vis.headlight.diffuse[:] = np.array([0.05, 0.05, 0.05]) * EXPOSURE
    m.vis.headlight.specular[:] = [0.0, 0.0, 0.0]


# look (clip-studio): dark studio, satin light-grey robot, carbon rods, charcoal belts, product colour the only colour
ALU = dict(rgba=[0.851, 0.8627, 0.8784, 1.0], specular=0.5, shininess=0.55)          # housing, upper arms, platform: satin grey
CARBON = dict(rgba=[0.1059, 0.1137, 0.1255, 1.0], specular=0.25, shininess=0.4)      # parallel rods
SILICONE = dict(rgba=[0.2275, 0.2392, 0.2588, 1.0], specular=0.12, shininess=0.2)   # cup rubber (matte)
BELT = dict(rgba=[0.1647, 0.1765, 0.1922, 1.0], specular=0.05, shininess=0.05)       # matte charcoal, lighter than the floor
RAIL = dict(rgba=[0.2902, 0.3059, 0.3294, 1.0], specular=0.3, shininess=0.4)         # belt side rails: thin edge highlight
FLOOR = dict(rgba=[0.0392, 0.0431, 0.051, 1.0], specular=0.35, shininess=0.6)       # near-black glossy floor (reflectance below)
FLOOR_REFLECTANCE = 0.3
BG_TOP, BG_BOTTOM = np.array([0.0863, 0.098, 0.1176]), np.array([0.0431, 0.051, 0.0627])   # background gradient (sRGB)
WHITE_CAP = 240.0                                                        # no clipped highlights
EXPOSURE = 0.6       # lights render at 0.6x so OpenGL never clamps; post restores it with a highlight shoulder
KNEE = 0.75          # display level above which highlights roll off smoothly toward white (below: unchanged)


def _style(g, s, rgba=None):
    g.rgba[:] = rgba if rgba is not None else s["rgba"]
    g.specular, g.shininess = s["specular"], s["shininess"]


def restyle_scene(m, scn, sku_rgba):
    import mujoco
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g.objid) or ""
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g.objid]) or ""
        if name in ("belt", "pad"):
            _style(g, BELT)
        elif name == "floor":
            _style(g, FLOOR)
            g.reflectance = FLOOR_REFLECTANCE
            g.size[0] = g.size[1] = 40.0          # no visible floor edge (the fog takes the far floor)
        elif name == "prod":
            _style(g, dict(WRAP, rgba=sku_rgba), sku_rgba)
        elif body.startswith("arm") or name == "base" or body == "plat":
            _style(g, ALU)
        elif body.startswith("s") and body[1:].isdigit():
            _style(g, CARBON)
            g.size[0] = 0.011                     # visual only: the real carbon rods are ~22 mm (model capsule 12 mm)
        elif body == "tool" and m.geom_type[g.objid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            g.rgba[3] = 0.0                       # the fitting stand-in: replaced by the drawn suction cup
        elif body.startswith("lip"):
            g.rgba[3] = 0.0                       # contact points of the lip ring: the drawn cup skirt is the visual
        elif body == "tool":
            _style(g, SILICONE)


HOUSING = ALU
WRAP = dict(specular=0.7, shininess=0.8)                                 # foil-wrap highlight
METAL = dict(rgba=[0.7216, 0.7373, 0.7608, 1.0], specular=1.0, shininess=0.9)
KINDS = ["housing", "belt", "cup", "queue", "outfeed", "wrap_seal"]


def _item(kind, gtype, size, pos, mat, style, rgba=None):
    return dict(kind=kind, type=int(gtype), size=np.r_[np.asarray(size, float), np.zeros(3)][:3],
                pos=np.asarray(pos, float), mat=np.asarray(mat, float).reshape(9), rgba=np.asarray(rgba if rgba is not None else style["rgba"], float),
                specular=style["specular"], shininess=style["shininess"])


def _bar_items(kind, gtype, size, pos, mat, rgba):
    """A product, plus (for flow-wrapped bars) the crimped seal fins at both ends (display only: thin film)."""
    import mujoco
    out = [_item(kind, gtype, size, pos, mat, WRAP, rgba)]
    size = np.asarray(size, float)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX and size[0] > 2.5 * size[1]:
        R = np.asarray(mat, float).reshape(3, 3)
        fin_rgba = np.minimum(np.array(rgba, float) * [1.12, 1.12, 1.12, 1], 1.0)
        for sgn in (-1, 1):
            out.append(_item("wrap_seal", mujoco.mjtGeom.mjGEOM_BOX, [0.005, size[1] * 1.02, 0.0010],
                             np.asarray(pos) + R @ [sgn * (size[0] + 0.005), 0, 0], R, WRAP, fin_rgba))
    return out


def display_items(m, d, p, picks, k, t, tl, pitch, camd):
    """Everything drawn that is not a body of the simulated model (all display only), in world coordinates:
    base housing, belt rails, outfeed belt, the drawn suction cup, the bar's seal fins, the queue of
    upcoming items, placed items riding the outfeed, dropped items where they came to rest."""
    import mujoco
    from sim.line import X_PICK, X_PLACE, Z_BELT
    B, C, E = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID
    eye = np.eye(3)
    it = []
    # belt side rails (infeed belt half-width 80 mm, outfeed 70 mm): a thin lighter edge so the line reads
    for xb, hw, top in ((X_PICK, 0.08, Z_BELT), (X_PLACE, 0.07, Z_BELT - 0.0025)):
        for sgn in (-1, 1):
            it.append(_item("belt", B, [0.004, 0.6, 0.008], [xb + sgn * (hw + 0.004), 0.0, top - 0.004], eye, RAIL))
    # base housing over the three motors (the real robot's wide top housing), rounded lid
    it.append(_item("housing", C, [0.30, 0.055], [0.0, 0.0, 0.035], eye, HOUSING))
    it.append(_item("housing", E, [0.30, 0.30, 0.05], [0.0, 0.0, 0.09], eye, HOUSING))
    # drive section (visual only): a gearbox on each motor axis (radius r_b, z_m below the base frame, axis along the
    # arm's hinge) and a post up to the housing, so the arms visibly attach to the body
    from sim.delta import PHI, tdir, udir
    r_b, z_m = DRIVE_R, DRIVE_Z
    for ph in PHI:
        pv = r_b * udir(ph) + np.array([0.0, 0.0, z_m])
        hinge = tdir(ph)
        ax = np.array([hinge[0], hinge[1], 0.0]) / np.linalg.norm(hinge[:2])
        b1 = np.cross([0.0, 0.0, 1.0], ax)
        R = np.stack([b1, np.cross(ax, b1), ax], axis=1)          # columns: x, y, z(= hinge axis)
        it.append(_item("housing", C, [0.055, 0.065], pv, R, HOUSING))
        top = np.array([pv[0], pv[1], -0.02])
        it.append(_item("housing", C, [0.038, (top[2] - pv[2]) / 2], (top + pv) / 2, eye, HOUSING))
    it.append(_item("belt", B, [0.07, 0.6, 0.004], [X_PLACE, 0.0, Z_BELT - 0.0065], eye, BELT))
    # the suction cup on the tool flange: flat cup (the bar line's Ø25 flat cup), skirt + body + fitting
    c = d.site_xpos[m.site("cupc").id]
    r_lip = float(p["setup"].cup.r_lip)
    it.append(_item("cup", E, [r_lip + 0.0008, r_lip + 0.0008, 0.0045], c + [0, 0, 0.0030], eye, SILICONE))
    it.append(_item("cup", C, [0.62 * r_lip, 0.004], c + [0, 0, 0.0085], eye, SILICONE))
    it.append(_item("cup", C, [0.0055, 0.010], c + [0, 0, 0.022], eye, METAL))
    # seal fins on the simulated product
    g = m.geom("prod").id
    it += _bar_items("wrap_seal", m.geom_type[g], m.geom_size[g], d.geom_xpos[g], d.geom_xmat[g], p["rgba"])[1:]
    if camd.get("ghosts", True):    # not in the side view: it looks along the belts and would see them end-on
        for j, q in enumerate(picks[k + 1:k + 6], start=1):
            yq = p["y0"] + p["vb"] * tl - j * pitch[k]
            if yq > -0.6:
                it += _bar_items("queue", q["gtype"], q["gsize"], [X_PICK, yq, q["z0"]], eye, q["rgba"])
        for q in picks[:k]:
            pos = np.array(q["final_pos"])
            if not q["out"]["dropped"]:
                pos[1] += V_OUT * max(t - q["t_release_g"], 0.0)
                if pos[1] > 0.6:
                    continue
            it += _bar_items("outfeed", q["gtype"], q["gsize"], pos, q["final_mat"], q["rgba"])
    return it


def draw_items(scn, items):
    import mujoco
    for x in items:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, x["type"], x["size"], x["pos"], x["mat"], x["rgba"].astype(np.float32))
        g.specular, g.shininess = x["specular"], x["shininess"]
        scn.ngeom += 1


LIGHT_AZ = 150.0            # the rig is placed relative to the hero (A) camera's heading, the same for every angle


def add_lights(scn):
    """One rig for all angles. Key: soft spot, low and to the side (as seen from the hero camera), neutral-warm, the
    only shadow caster. Rim: behind/above, cool white, strong, no shadows. Fill: very low. Pool: a narrow downward
    spot under the robot (no shadows) that pools on the glossy floor and belts and falls off into the dark."""
    import mujoco
    phi = np.radians(LIGHT_AZ)
    fwd = np.array([np.cos(phi), np.sin(phi), 0.0])     # hero camera's horizontal viewing direction
    right = np.array([np.sin(phi), -np.cos(phi), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    tgt = np.array([0.0, 0.0, -0.85])
    spec = [  # pos, colour, specular, castshadow, cutoff, exponent, target
        (tgt - 1.2 * fwd - 2.3 * right + 1.1 * z, [0.95, 0.90, 0.83], 0.35, 1, 38.0, 4.0, tgt),    # key: low, to the side
        (tgt + 2.2 * fwd + 0.4 * right + 1.4 * z, [1.25, 1.36, 1.56], 0.5, 0, 40.0, 4.0, tgt),     # rim: doubled
        (tgt - 2.6 * fwd + 0.8 * right - 0.1 * z, [0.08, 0.08, 0.085], 0.0, 0, 70.0, 4.0, tgt),    # fill: halved
        (np.array([0.0, 0.0, 0.6]), [0.24, 0.235, 0.225], 0.0, 0, 24.0, 12.0,                       # pool under the robot
         np.array([0.0, 0.0, -1.3])),
    ]
    scn.nlight = 0
    for pos, col, spc, sh, cut, ex, aim in spec:
        L = scn.lights[scn.nlight]
        L.type = mujoco.mjtLightType.mjLIGHT_SPOT
        L.pos[:] = pos
        dvec = aim - np.asarray(pos)
        L.dir[:] = dvec / np.linalg.norm(dvec)
        L.ambient[:] = [0.0, 0.0, 0.0]
        L.diffuse[:] = np.asarray(col) * EXPOSURE
        L.specular[:] = [spc * EXPOSURE] * 3
        L.castshadow = sh
        L.cutoff, L.exponent = cut, ex
        L.attenuation[:] = [1.0, 0.0, 0.0]
        L.headlight = 0
        L.softness = 0.6 if sh else 0.0
        scn.nlight += 1


def tonemap(lin: np.ndarray) -> np.ndarray:
    """lin: the motion-blur average in linear light (gamma 2.2 decode of the rendered frames). Back to display values,
    then the exposure is restored in DISPLAY space (MuJoCo writes its lighting result directly as display values,
    so the lights at EXPOSURE scale those values), then a soft knee above KNEE (slope 1 at the knee, asymptote
    1). Mid-tones match a render at full exposure; highlights roll off instead of clipping."""
    x = (lin ** (1 / 2.2)) / EXPOSURE
    hi = x > KNEE
    x[hi] = KNEE + (1 - KNEE) * (1 - np.exp(-(x[hi] - KNEE) / (1 - KNEE)))
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def composite(img: np.ndarray, depth: np.ndarray, fog_start: float, fog_end: float) -> np.ndarray:
    """Haze toward the background gradient by distance, background where nothing was drawn, highlight cap.
    sRGB in, sRGB out (the blend is done in sRGB: simple and matches a display-referred grade)."""
    H, W = depth.shape
    bg = (BG_TOP[None, :] + (BG_BOTTOM - BG_TOP)[None, :] * np.linspace(0, 1, H)[:, None]) * 255.0  # (H, 3)
    f = np.clip((depth - fog_start) / max(fog_end - fog_start, 1e-6), 0.0, 1.0)[:, :, None]
    out = img.astype(np.float32) * (1 - f) + bg[:, None, :] * f
    return np.clip(out * (WHITE_CAP / 255.0), 0, 255).astype(np.uint8)


# ================================================================================================ cameras
def ease(u):
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


CAMERAS = {  # lookat (None: tracked), distance, azimuth, elevation, fovy (deg; view height in m if ortho)
    # A: low, close, long lens; robot + belts fill ~60 % of the frame; no support frame (its legs would sit in front)
    # A: the whole robot (housing to belts, ~1.14 m) fills ~70 % of the frame height, with headroom
    "A-wide": dict(lookat=[0.0, 0.0, -0.42], dist=4.2, az=150, el=-4, fovy=22, orbit=0.0, depth=True),
    "A-wide-orbit": dict(lookat=[0.0, 0.0, -0.42], dist=4.2, az=143, el=-4, fovy=22, orbit=14.0, depth=True),
    "B-top": dict(lookat=[0.0, 0.0, -1.0], dist=0.95, az=0.0, el=-89.9, fovy=55, orbit=0.0),
    # A2: 3/4 mid shot, elevated, platform + cup + both belts + products; rods run out of the top of frame
    "A2-mid": dict(lookat=[-0.07, 0.03, -0.98], dist=1.15, az=150, el=-28, fovy=30, orbit=0.0, push=0.05),
    "A2-mid-lock": dict(lookat=[-0.07, 0.03, -0.98], dist=1.15, az=150, el=-28, fovy=30, orbit=0.0),
    # C: follows the cup (Gaussian-smoothed, sigma 40 ms): cup in the upper third, the bar in the lower third
    "C-close": dict(lookat=None, track="cup", offset=[0.0, 0.0, -0.016], sigma=0.04, dist=0.30, az=158, el=-3,
                    fovy=18, orbit=0.0, depth=True),
    "D-track": dict(lookat=None, track="tool", offset=[0.0, 0.0, 0.0], sigma=0.12, dist=0.75, az=150, el=-12, fovy=24,
                    orbit=0.0),
    # E: pick point to place point (0.305 m) spans ~70 % of the width: view height 0.245 m (orthographic)
    "E-side": dict(lookat=[0.0, 0.0, -0.975], dist=3.0, az=90, el=-1.0, fovy=0.245, orbit=0.0, ortho=True, ghosts=False),
}


# ================================================================================================ pass 2: render
def project(cam, p, W, H):
    """World point -> pixel (u, v) and depth, using the scene's GL camera frustum (perspective or orthographic)."""
    pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    dv = np.asarray(p) - pos
    z = float(dv @ fwd)
    x, y = float(dv @ right), float(dv @ up)
    if not cam.orthographic:
        if z <= 1e-6:
            return np.nan, np.nan, z
        x, y = x * cam.frustum_near / z, y * cam.frustum_near / z
    # half-width of the frustum: MuJoCo derives it from the viewport aspect at draw time (0 in the scene copy)
    hw = cam.frustum_width if cam.frustum_width > 0 else 0.5 * (cam.frustum_top - cam.frustum_bottom) * W / H
    u = (x - (cam.frustum_center - hw)) / (2 * hw) * W
    v = (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * H
    return u, v, z


def _bbox(uv):
    """Screen bbox of the projected corners; NaN when every corner is behind the camera."""
    ok = np.isfinite(uv).all(axis=1)
    if not ok.any():
        return dict(bbox_x0=np.nan, bbox_y0=np.nan, bbox_x1=np.nan, bbox_y1=np.nan)
    q = uv[ok]
    return dict(bbox_x0=round(float(q[:, 0].min()), 1), bbox_y0=round(float(q[:, 1].min()), 1),
                bbox_x1=round(float(q[:, 0].max()), 1), bbox_y1=round(float(q[:, 1].max()), 1))


def render_clip(job: dict) -> dict:
    import mujoco
    from sim.line import X_PICK
    picks, clip = job["picks"], job["clip"]
    W, H, fps_sim = clip["W"], clip["H"], clip["sim_fps"]
    camd = CAMERAS[clip["camera"]]
    starts = np.cumsum([0.0] + [p["dur"] for p in picks[:-1]])
    t0, t1 = clip["t0"], clip["t1"]
    n_frames = int(np.floor((t1 - t0) * fps_sim)) + 1
    # tracking target (D): the tool position on the frame grid, smoothed zero-phase (Gaussian, sigma 0.12 s)
    renderers, datas = {}, {}

    def state(t):
        k = int(np.searchsorted(starts, t, side="right") - 1)
        k = min(max(k, 0), len(picks) - 1)
        i = min(int(round((t - starts[k]) / MS)), len(picks[k]["qpos"]) - 1)
        return k, max(i, 0)
    track = None
    if camd["lookat"] is None:          # tracked camera: target on the frame grid, smoothed zero-phase (Gaussian)
        pts = []
        for f in range(n_frames):
            k, i = state(t0 + f / fps_sim)
            m = picks[k]["model"]
            d = datas.setdefault(k, mujoco.MjData(m))
            d.qpos[:] = picks[k]["qpos"][i]
            mujoco.mj_kinematics(m, d)
            pts.append(d.site_xpos[m.site("cupc").id].copy() if camd["track"] == "cup" else d.xpos[m.body("tool").id].copy())
        pts = np.array(pts) + np.array(camd.get("offset", [0, 0, 0]))
        sig = max(camd.get("sigma", 0.12) * fps_sim, 1.0)
        ker = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
        ker /= ker.sum()
        pad = np.pad(pts, ((len(ker) // 2, len(ker) // 2), (0, 0)), mode="edge")
        track = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(3)], axis=1)
    path = job["path"]
    stills = job.get("still_frames")          # stills mode: only these frames, PNG, no video or sidecars
    ff = None if stills else subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60",
         "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-profile:v", "high",
         "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-g", "60", "-keyint_min", "60",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv",
         "-movflags", "+faststart", "-an", path], stdin=subprocess.PIPE)
    rows, cams = [], []
    poses = dict(xpos=[], xquat=[], prod_size=[], items=[])
    depth_dir = path[:-4] + ".depth"
    if camd.get("depth") and job.get("depth", True):
        Path(depth_dir).mkdir(exist_ok=True)
    nsub = {60: 8, 250: 2}.get(int(fps_sim), 1)     # motion blur: 180° shutter on the 1 ms grid (none at 1000)
    lut = ((np.arange(256) / 255.0) ** 2.2).astype(np.float32)   # sRGB-ish -> linear for the average
    fog_start = camd.get("fog", (camd["dist"] + 0.6, camd["dist"] + 3.5))[0]
    fog_end = camd.get("fog", (camd["dist"] + 0.6, camd["dist"] + 3.5))[1]
    for f in (stills or range(n_frames)):
        t = t0 + f / fps_sim
        u = ease(f / max(n_frames - 1, 1))
        acc = None
        for jsub in reversed(range(nsub)):           # sub-states t-(nsub-1) ms ... t; the frame-time state is last
            k, i = state(max(t - jsub * MS, 0.0))
            p = picks[k]
            m = p["model"]
            if k not in renderers:
                # one GL context per pick model; at 4K with 8x MSAA each holds ~0.5 GB of GPU memory, so keep only
                # the current pick and the previous one (motion-blur sub-states can straddle a pick boundary)
                for kk in [kk for kk in renderers if kk < k - 1]:
                    renderers.pop(kk).close()
                style_model(m, W, H, camd["fovy"], camd.get("ortho", False))
                renderers[k] = mujoco.Renderer(m, H, W)
                datas.setdefault(k, mujoco.MjData(m))
            r, d = renderers[k], datas[k]
            d.qpos[:] = p["qpos"][i]
            mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera()                 # the camera is held at the output frame's pose
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = track[f] if track is not None else camd["lookat"]
            cam.distance = camd["dist"] * (1.0 - camd.get("push", 0.0) * u)          # eased push-in
            cam.azimuth, cam.elevation = camd["az"] + camd["orbit"] * u, camd["el"]
            cam.orthographic = int(camd.get("ortho", False))   # camera-level switch: fovy is the view height (m)
            r.update_scene(d, cam)
            scn = r.scene
            restyle_scene(m, scn, p["rgba"])
            tl = i * MS
            items = display_items(m, d, p, picks, k, t - jsub * MS, tl, job["pitch"], camd)
            draw_items(scn, items)
            add_lights(scn)
            img = r.render()
            acc = lut[img] if acc is None else acc + lut[img]
        img = tonemap(acc / nsub)
        # depth of the frame-time state: haze + background, and the depth pass for A and C
        r.enable_depth_rendering()
        dep = r.render()
        r.disable_depth_rendering()
        img = composite(img, dep, fog_start, fog_end)
        if stills:
            from PIL import Image
            Image.fromarray(img).save(f"{path[:-4]}_f{f}.png")
            continue
        ff.stdin.write(img.tobytes())
        if camd.get("depth") and job.get("depth", True):     # 16-bit PNG in mm, half resolution
            mm = np.clip(dep[::2, ::2] * 1000.0, 0, 65535).astype(np.uint16)
            from PIL import Image
            Image.fromarray(mm).save(f"{depth_dir}/{f:05d}.png")
        # pose export (same state and items as this frame)
        poses["xpos"].append(d.xpos.copy())
        poses["xquat"].append(d.xquat.copy())
        poses["prod_size"].append(m.geom_size[m.geom("prod").id].copy())
        poses["items"].append(items)
        # sidecar row
        gl = scn.camera[0]
        g = m.geom("prod").id
        cpos = d.site_xpos[m.site("cupc").id]
        ppos, pmat, psz = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        cu, cv, _ = project(gl, cpos, W, H)
        pu, pv, pz = project(gl, ppos, W, H)
        corners = [ppos + pmat @ (np.array(s) * psz) for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)]
        uv = np.array([project(gl, c, W, H)[:2] for c in corners])
        vis = bool(pz > 0 and np.isfinite(pu) and 0 <= pu < W and 0 <= pv < H)
        tg = starts[k] + tl
        ev = p["events_g"]
        dec = p["decision"]
        known = tg >= ev["decision"]
        if p["out"]["dropped"] and ev["drop_start"] is not None and tg >= ev["drop_start"]:
            outcome = "dropped"
        elif (not p["out"]["dropped"]) and ev["placed"] is not None and tg >= ev["placed"]:
            outcome = "placed"
        else:
            outcome = "pending"
        cnt = p["counters_after"] if outcome != "pending" else p["counters_before"]
        phase = ("approach" if tl < p["t_touch"] else "touch" if tl < p["t_touch"] + 0.005 else
                 "dwell" if tl < p["t_lift"] else "move" if tl < p["t_arrive"] else "release")
        row = dict(frame=f, sim_time_s=round(tg, 6), pick_id=p["idx"], product=p["sku"], phase=phase,
                   seal_kpa=round(p["out"]["seal_kpa"], 2) if tg >= ev["seal_measured"] else np.nan,
                   vac_kpa_live=round(p["dp"][i] / 1e3, 2), evac_ms=p["out"]["evac_ms"] if tg >= ev["seal_measured"] else np.nan,
                   cup_cycles=int(p["logged"]["cup_cycles"]), mass_g=round(p["logged"]["weight_g"], 1),
                   temp_c=round(p["logged"]["prod_t"], 1), rh_pct=round(p["logged"]["room_rh"], 1),
                   context_rows=dec.get("n_ctx", np.nan) if job["policy"].startswith("tabpfn") else np.nan)
        for a, rk in zip(ALPHAS, dec.get("risk") or [np.nan] * 9):
            row[f"risk_{int(round(a * 100))}"] = round(rk, 4) if (known and job["policy"].startswith("tabpfn")) else np.nan
        row.update(chosen_speed=p["alpha"] if known else np.nan, policy=job["policy"],
                   tabpfn_ms=round(dec.get("tabpfn_ms", np.nan), 2) if (known and job["policy"].startswith("tabpfn")) else np.nan,
                   outcome=outcome,
                   placement_err_mm=round(p["out"]["place_offset_mm"], 2) if outcome == "placed" else np.nan,
                   picks=cnt["picks"], placed=cnt["placed"], dropped=cnt["dropped"], good_per_min=round(cnt["gpm"], 1),
                   cup_x=round(cu, 1), cup_y=round(cv, 1), product_x=round(pu, 1), product_y=round(pv, 1),
                   **_bbox(uv), visible=int(vis))
        rows.append(row)
        cams.append(dict(frame=f, pos=[round(float(x), 5) for x in gl.pos], forward=[round(float(x), 5) for x in gl.forward],
                         up=[round(float(x), 5) for x in gl.up], lookat=[round(float(x), 5) for x in cam.lookat],
                         fovy=camd["fovy"], orthographic=bool(camd.get("ortho", False))))
    if stills:
        return dict(file=Path(path).name, frames=len(stills), sim_t0=t0, sim_t1=t1, stills=True)
    ff.stdin.close()
    ff.wait()
    base = path[:-4]
    pd.DataFrame(rows).to_csv(base + ".frames.csv", index=False)
    write_poses(base + ".poses.npz", picks[0]["model"], poses, cams, fps_sim, t0)
    # events: every per-pick event inside the clip, frame-exact (nearest frame on this clip's sim_fps grid)
    events = []
    for k, p in enumerate(picks):
        for name, tg in p["events_g"].items():
            if tg is None or not (t0 <= tg <= t1):
                continue
            e = dict(event=name, pick_id=p["idx"], sim_time=round(tg, 6), frame=int(round((tg - t0) * fps_sim)))
            if name == "decision":
                e.update(speed=p["alpha"], policy=job["policy"])
                if job["policy"].startswith("tabpfn"):
                    e.update(risk=p["decision"].get("risk"), latency_ms=p["decision"].get("tabpfn_ms"),
                             context_rows=p["decision"].get("n_ctx"))
            events.append(e)
    for s in job.get("story", []):
        if t0 <= s["sim_time"] <= t1:
            events.append(dict(s, frame=int(round((s["sim_time"] - t0) * fps_sim))))
    events.sort(key=lambda e: (e["sim_time"], e["event"]))
    Path(base + ".events.json").write_text(json.dumps(dict(clip=Path(path).name, sim_fps=fps_sim, video_fps=60,
                                                            sim_t0=t0, sim_t1=t1, events=events, camera=cams), indent=1, default=_num))
    return dict(file=Path(path).name, frames=n_frames, sim_t0=t0, sim_t1=t1)


def write_poses(path, m0, poses, cams, fps_sim, t0):
    """Everything a re-render (e.g. Blender) needs, per frame: world pose of every body, the geom table (static,
    body-local), the display-only items (world), and the camera. Units: metres, quaternions w-x-y-z (MuJoCo)."""
    import mujoco
    F = len(poses["xpos"])
    nmax = max(len(x) for x in poses["items"])
    it_type = np.full((F, nmax), -1, np.int16)
    it_kind = np.full((F, nmax), -1, np.int8)
    it_size, it_pos = np.zeros((F, nmax, 3), np.float32), np.zeros((F, nmax, 3), np.float32)
    it_mat, it_rgba = np.zeros((F, nmax, 9), np.float32), np.zeros((F, nmax, 4), np.float32)
    for f, items in enumerate(poses["items"]):
        for j, x in enumerate(items):
            it_type[f, j], it_kind[f, j] = x["type"], KINDS.index(x["kind"])
            it_size[f, j], it_pos[f, j], it_mat[f, j], it_rgba[f, j] = x["size"], x["pos"], x["mat"], x["rgba"]
    body_names = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in range(m0.nbody)]
    geom_names = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(m0.ngeom)]
    np.savez_compressed(
        path, frame=np.arange(F), sim_time_s=t0 + np.arange(F) / fps_sim, sim_fps=fps_sim,
        body_names=np.array(body_names), xpos=np.array(poses["xpos"], np.float32), xquat=np.array(poses["xquat"], np.float32),
        geom_names=np.array(geom_names), geom_type=m0.geom_type.copy(), geom_size=m0.geom_size.copy(),
        geom_body=m0.geom_bodyid.copy(), geom_pos=m0.geom_pos.copy(), geom_quat=m0.geom_quat.copy(),
        geom_rgba=m0.geom_rgba.copy(), geom_mesh=np.array([""] * m0.ngeom),
        prod_size=np.array(poses["prod_size"], np.float32),
        item_kinds=np.array(KINDS), item_kind=it_kind, item_type=it_type, item_size=it_size, item_pos=it_pos,
        item_mat=it_mat, item_rgba=it_rgba,
        cam_pos=np.array([c["pos"] for c in cams], np.float32), cam_forward=np.array([c["forward"] for c in cams], np.float32),
        cam_up=np.array([c["up"] for c in cams], np.float32), cam_lookat=np.array([c["lookat"] for c in cams], np.float32),
        cam_fovy=np.array([c["fovy"] for c in cams], np.float32), cam_orthographic=np.array([c["orthographic"] for c in cams]),
        geom_type_names=np.array([mujoco.mjtGeom(i).name for i in range(8)]),
        notes=np.array("geom_type/item_type: MuJoCo mjtGeom (2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box); "
                       "size: MuJoCo half-sizes (cylinder: radius, half-height); item_mat: 3x3 world rotation row-major; "
                       "the model geom 'prod' size varies per pick: use prod_size; the tool cylinder is hidden in "
                       "the render (replaced by the 'cup' items)."))


def _drive_geometry():
    from sim.delta import DeltaParams
    P = DeltaParams()
    return P.r_b, P.z_m


DRIVE_R, DRIVE_Z = _drive_geometry()   # motor axes of the simulated robot (sim.delta.DeltaParams r_b, z_m)
POLICY_LABEL = {"tabpfn_jax": "tabpfn", "fixed": "fixed30", "fixed50": "fixed50", "recipe50": "recipe50",
                "full100": "full100"}   # decisions key -> file label (no underscores: the name field separator)


# ================================================================================================ plan
def build_stream(stream: str, idxs: list[int], policy: str, decisions: pd.DataFrame | None, alpha_fixed=None,
                 workers: int = 8) -> list[dict]:
    """Simulate the picks of a stream at the policy's chosen speeds; attach decisions, counters, global events."""
    jobs = []
    for i in idxs:
        dec = {}
        if decisions is not None:
            r = decisions[(decisions["policy_key"] == policy) & (decisions["idx"] == i)].iloc[0]
            dec = dict(alpha=float(r["alpha"]), risk=json.loads(r["risk"]) if isinstance(r["risk"], str) and r["risk"] != "[]" else None,
                       n_ctx=int(r["n_ctx"]), tabpfn_ms=float(r["tabpfn_ms"]) if "tabpfn_ms" in r and pd.notna(r["tabpfn_ms"]) else np.nan,
                       dropped=bool(r["dropped"]))
        alpha = dec.get("alpha", alpha_fixed)
        sku = task_for(stream, i).get("sku", "bar 45 g")
        jobs.append(dict(stream=stream, idx=i, alpha=alpha, decision=dec, sku=sku, rgba=SKU_RGBA.get(sku, SKU_RGBA["bar 45 g"])))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        picks = list(ex.map(simulate, jobs))
    t = 0.0
    held = drops = 0
    cyc = 0.0
    for k, p in enumerate(picks):
        if "dropped" in p["decision"]:
            assert p["decision"]["dropped"] == bool(p["out"]["dropped"]), f"pick {p['idx']}: re-simulation differs from the bank"
        p["counters_before"] = dict(picks=held + drops, placed=held, dropped=drops, gpm=60 * held / cyc if cyc else 0.0)
        held += int(not p["out"]["dropped"])
        drops += int(p["out"]["dropped"])
        cyc += float(p["out"]["cycle_s"])
        p["counters_after"] = dict(picks=held + drops, placed=held, dropped=drops, gpm=60 * held / cyc)
        p["events_g"] = {k_: (t + v if v is not None else None) for k_, v in p["events"].items()}
        p["t_release_g"] = t + p["t_arrive"] + 0.04
        p["t_start_g"] = t
        import mujoco
        m = p["model"]
        d = mujoco.MjData(m)
        d.qpos[:] = p["qpos"][-1]
        mujoco.mj_kinematics(m, d)
        g = m.geom("prod").id
        p["final_pos"], p["final_mat"] = d.geom_xpos[g].copy(), d.geom_xmat[g].copy()
        t += p["dur"]
    return picks


def queue_pitch(picks):
    out = []
    for k, p in enumerate(picks):
        nxt = picks[k + 1] if k + 1 < len(picks) else None
        out.append(max((p["y0"] + p["vb"] * p["dur"] - nxt["y0"]) if nxt else 0.2, 0.12))
    return out


def _num(x):
    """JSON for numpy scalars and arrays."""
    return x.tolist() if hasattr(x, "tolist") else float(x)


def git_commit():
    if os.environ.get("SIM_COMMIT"):                 # the server copy has no .git: the push passes the laptop's HEAD
        return os.environ["SIM_COMMIT"]
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_plan_test():
    """Test pack: shift picks 0–8 under TabPFN; ~5 s per angle A–E around the middle picks."""
    return dict(
        name="test", out="/home/nebius/delta_line/footage", stream="shift:bar", policy="tabpfn_jax", idxs=list(range(0, 9)),
        clips=[
            dict(scene="T00", label="cell", camera="A-wide", picks=[2, 7], sim_fps=60),
            dict(scene="T00", label="cell", camera="A-wide-orbit", picks=[2, 7], sim_fps=60),
            dict(scene="T00", label="cell", camera="B-top", picks=[2, 7], sim_fps=60),
            dict(scene="T00", label="touch", camera="C-close", pick=4, window="touch", sim_fps=250),
            dict(scene="T00", label="move", camera="D-track", pick=4, window="move", sim_fps=250),
            dict(scene="T00", label="move", camera="E-side", pick=4, window="move", sim_fps=250),
        ])


def make_plan_full():
    """The full batch (clip-studio's shot list), in delivery order. Each sub-plan simulates one stream at one policy
    once (the full 120 picks for the shift, so running counters and sim times match across clips) and renders
    several clips. S02 overrides the speed of picks 3–6 (5 is placed at 50 % and dropped at 100 % in the bank)."""
    out = "/home/nebius/delta_line/footage"
    events = [("heavy", [30, 33]), ("filmroll", [60, 63]), ("worncup", [90, 93]), ("cupchange", [110, 113])]
    shift = list(range(120))
    ev_clips = [dict(scene="S04", label=ev, camera="D-track", picks=w, sim_fps=250) for ev, w in events]
    long_clips = [dict(scene="S04", label="shift", camera=c, picks=[0, 119], sim_fps=60) for c in ("A-wide", "A2-mid-lock", "B-top")]
    return [
        dict(name="recipe50", out=out, stream="shift:bar", policy="recipe50", alpha=0.5, idxs=[3, 4, 5, 6], clips=[
            dict(scene="S02", label="problem", camera="C-close", pick=5, window="touch", sim_fps=250),
            dict(scene="S02", label="problem", camera="D-track", pick=5, window="move", sim_fps=250),
            dict(scene="S02", label="problem", camera="E-side", pick=5, window="move", sim_fps=250)]),
        dict(name="full100", out=out, stream="shift:bar", policy="full100", alpha=1.0, idxs=[3, 4, 5, 6], clips=[
            dict(scene="S02", label="problem", camera="C-close", pick=5, window="touch", sim_fps=250),
            dict(scene="S02", label="problem", camera="D-track", pick=5, window="move", sim_fps=1000),
            dict(scene="S02", label="problem", camera="E-side", pick=5, window="move", sim_fps=1000)]),
        dict(name="tabpfn_a", out=out, stream="shift:bar", policy="tabpfn_jax", idxs=shift, clips=[
            dict(scene="S03", label="decision", camera="C-close", pick=4, window="touch", sim_fps=1000),
            dict(scene="S03", label="decision", camera="A-wide", pick=4, window="touch", sim_fps=1000)] + [
            dict(scene="S01", label="hero", camera=c, picks=[1, 14], sim_fps=60)
            for c in ("A-wide", "A-wide-orbit", "A2-mid", "B-top", "C-close")]),
        dict(name="tabpfn_events", out=out, stream="shift:bar", policy="tabpfn_jax", idxs=shift, clips=ev_clips),
        dict(name="fixed50_events", out=out, stream="shift:bar", policy="fixed50", idxs=shift, clips=ev_clips),
        dict(name="tabpfn_long", out=out, stream="shift:bar", policy="tabpfn_jax", idxs=shift, clips=long_clips),
        dict(name="fixed50_long", out=out, stream="shift:bar", policy="fixed50", idxs=shift, clips=long_clips),
    ]


def clip_window(picks, c):
    by = {p["idx"]: p for p in picks}
    if "picks" in c:
        a, b = by[c["picks"][0]], by[c["picks"][1]]
        return max(a["t_start_g"] - 1.0, 0.0), min(b["t_start_g"] + b["dur"] + 1.0, picks[-1]["t_start_g"] + picks[-1]["dur"] - MS)
    p = by[c["pick"]]
    e = p["events_g"]
    if c["window"] == "touch":      # from 1 s before touch until the cup has carried the bar ~20 cm, + 1 s
        import mujoco
        m, d = p["model"], mujoco.MjData(p["model"])
        tool = []
        for q in p["qpos"]:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            tool.append(d.site_xpos[m.site("cupc").id].copy())
        tool = np.array(tool)
        i_t = int(round((e["touch"] - p["t_start_g"]) / MS))
        far = np.where(np.linalg.norm(tool[i_t:] - tool[i_t], axis=1) > 0.20)[0]
        t_far = p["t_start_g"] + (i_t + (far[0] if len(far) else len(tool) - 1 - i_t)) * MS
        return max(e["touch"] - 1.0, 0.0), t_far + 1.0
    end = e["drop_impact"] or e["placed"] or e["release"]
    return max(e["move_start"] - 1.0, 0.0), end + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", default="test", nargs="?")
    ap.add_argument("--W", type=int, default=3840)
    ap.add_argument("--H", type=int, default=2160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="override the plan's output folder")
    ap.add_argument("--only", default="", help="comma-separated cameras to render (default: all in the plan)")
    ap.add_argument("--only-plan", dest="only_plan", default="", help="comma-separated sub-plan names (full plan)")
    ap.add_argument("--stills", action="store_true", help="render one look frame per clip (PNG in <out>/look/)")
    a = ap.parse_args()
    plans = {"test": lambda: [make_plan_test()], "full": make_plan_full}[a.plan]() if a.plan in ("test", "full") \
        else json.loads(Path(a.plan).read_text())
    if a.only_plan:
        plans = [p for p in plans if p.get("name") in a.only_plan.split(",")]
    for plan in plans:
        run_plan(plan, a)


def run_plan(plan, a):
    if a.only:
        plan["clips"] = [c for c in plan["clips"] if c["camera"] in a.only.split(",")]
    if not plan["clips"]:
        return
    out = Path(a.out or plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    decisions = None
    if plan["stream"] == "shift:bar" and "alpha" not in plan:
        # shift decisions: TabPFN answered in JAX (policy key tabpfn_jax), fixed recipes fixed (30 %) and fixed50 (50 %)
        decisions = pd.read_csv(ROOT / "results/shift/decisions_jax.csv")
    picks = build_stream(plan["stream"], plan["idxs"], plan["policy"], decisions, plan.get("alpha"), a.workers)
    pitch = queue_pitch(picks)
    story = []
    if plan["stream"] == "shift:bar":
        for i, name in ((30, "heavy_variant"), (60, "new_batch"), (90, "worn_cup"), (110, "cup_change")):
            for p in picks:
                if p["idx"] == i:
                    story.append(dict(event=name, pick_id=i, sim_time=round(p["t_start_g"], 6)))
    jobs = []
    label = POLICY_LABEL.get(plan["policy"], plan["policy"])
    for c in plan["clips"]:
        t0, t1 = clip_window(picks, c)
        name = f"{c['scene']}_{c['label']}_{c['camera']}_{label}_{c['sim_fps']}.mp4"
        job = dict(picks=picks, pitch=pitch, policy=label, story=story, path=str(out / name),
                   clip=dict(camera=c["camera"], t0=t0, t1=t1, sim_fps=c["sim_fps"], W=a.W, H=a.H))
        if a.stills:      # one look frame: C 8 frames after touch, the others at the acceleration reversal
            by = {p["idx"]: p for p in picks}
            pk = by[c["pick"]] if "pick" in c else next(p for p in picks if p["idx"] >= c["picks"][0] + 1)
            ev, off = ("touch", 8) if c["camera"].startswith("C") else ("accel_reversal", 0)
            job["still_frames"] = [int(round((pk["events_g"][ev] - t0) * c["sim_fps"])) + off]
            job["path"] = str(out / "look" / name)
            (out / "look").mkdir(exist_ok=True)
        jobs.append(job)
    with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(render_clip, jobs))
    if a.stills:
        for rres in results:
            print(f"still {rres['file'][:-4]}: written", flush=True)
        return
    import fcntl
    lock = open(out / ".manifest.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)                 # concurrent sub-plan jobs share one manifest
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else []
    commit = git_commit()
    for c, rres in zip(plan["clips"], results):
        man = [e for e in man if e["file"] != rres["file"]]
        man.append(dict(file=rres["file"], scene=c["scene"], camera=c["camera"], policy=label,
                        sim_fps=c["sim_fps"], video_fps=60, resolution=f"{a.W}x{a.H}", frames=rres["frames"],
                        sim_t0=rres["sim_t0"], sim_t1=rres["sim_t1"], stream=plan["stream"],
                        speed_override=plan.get("alpha"),
                        pick_ids=[p["idx"] for p in picks if p["t_start_g"] + p["dur"] >= rres["sim_t0"] and p["t_start_g"] <= rres["sim_t1"]],
                        sim_commit=commit, motion_blur_substates={60: 8, 250: 2}.get(c["sim_fps"], 1),
                        display_only=["infeed queue", "outfeed carry-off", "base housing", "drawn suction cup",
                                      "wrap seal fins", "belt rails"]))
        print(f"{rres['file']}: {rres['frames']} frames, sim {rres['sim_t0']:.3f}-{rres['sim_t1']:.3f} s", flush=True)
    man_p.write_text(json.dumps(man, indent=1, default=_num))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


if __name__ == "__main__":
    main()
