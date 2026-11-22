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
SKU_RGBA = {"bar 45 g": [0.86, 0.36, 0.12, 1], "protein bar 60 g": [0.10, 0.48, 0.62, 1]}   # plain one-colour wraps


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
    m.vis.headlight.active = 1
    m.vis.headlight.ambient[:] = [0.28, 0.28, 0.29]
    m.vis.headlight.diffuse[:] = [0.10, 0.10, 0.10]
    m.vis.headlight.specular[:] = [0.0, 0.0, 0.0]


ALU = dict(rgba=[0.80, 0.81, 0.83, 1], specular=0.9, shininess=0.75)
CARBON = dict(rgba=[0.11, 0.11, 0.12, 1], specular=0.45, shininess=0.55)
SILICONE = dict(rgba=[0.08, 0.08, 0.08, 1], specular=0.15, shininess=0.2)
BELT = dict(rgba=[0.12, 0.12, 0.13, 1], specular=0.04, shininess=0.05)
STUDIO = dict(rgba=[0.84, 0.84, 0.85, 1], specular=0.0, shininess=0.0)


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
            _style(g, STUDIO)
        elif name == "prod":
            _style(g, dict(specular=0.55, shininess=0.7, rgba=sku_rgba), sku_rgba)
        elif body.startswith("arm") or name == "base" or body == "plat":
            _style(g, ALU)
        elif body.startswith("s") and body[1:].isdigit():
            _style(g, CARBON)
        elif body == "tool" and m.geom_type[g.objid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            g.rgba[3] = 0.0                       # the fitting stand-in: replaced by the drawn suction cup
        elif body.startswith("lip"):
            g.rgba[3] = 0.0                       # contact points of the lip ring: the drawn cup skirt is the visual
        elif body == "tool":
            _style(g, SILICONE)


HOUSING = dict(rgba=[0.88, 0.88, 0.89, 1], specular=0.5, shininess=0.5)
WRAP = dict(specular=0.55, shininess=0.7)
METAL = dict(rgba=[0.55, 0.56, 0.58, 1], specular=0.8, shininess=0.7)
KINDS = ["studio", "frame", "housing", "belt", "cup", "queue", "outfeed", "wrap_seal"]


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
        fin_rgba = np.array(rgba, float) * [0.85, 0.85, 0.85, 1]
        for sgn in (-1, 1):
            out.append(_item("wrap_seal", mujoco.mjtGeom.mjGEOM_BOX, [0.005, size[1] * 1.02, 0.0010],
                             np.asarray(pos) + R @ [sgn * (size[0] + 0.005), 0, 0], R, WRAP, fin_rgba))
    return out


def display_items(m, d, p, picks, k, t, tl, pitch, camd):
    """Everything drawn that is not a body of the simulated model (all display only), in world coordinates:
    studio, support frame, base housing, outfeed belt, the drawn suction cup, the bar's seal fins, the queue of
    upcoming items, placed items riding the outfeed, dropped items where they came to rest."""
    import mujoco
    from sim.line import X_PICK, X_PLACE, Z_BELT
    B, C, E = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID
    eye = np.eye(3)
    it = []
    for pos, size in (([-4, 0, 0], [0.02, 4, 3]), ([4, 0, 0], [0.02, 4, 3]), ([0, 4, 0], [4, 0.02, 3]),
                      ([0, -4, 0], [4, 0.02, 3]), ([0, 0, 2.6], [4, 4, 0.02])):
        it.append(_item("studio", B, size, pos, eye, STUDIO))
    if camd.get("frame", True):     # support frame outside the belts: 4 posts, top rails, the beam carrying the base
        for x, y in ((0.55, 0.75), (-0.55, 0.75), (0.55, -0.75), (-0.55, -0.75)):
            it.append(_item("frame", C, [0.022, 0.665], [x, y, -0.635], eye, ALU))
        for x in (0.55, -0.55):
            it.append(_item("frame", B, [0.022, 0.75, 0.022], [x, 0.0, 0.03], eye, ALU))
        for y in (0.75, -0.75):
            it.append(_item("frame", B, [0.55, 0.022, 0.022], [0.0, y, 0.03], eye, ALU))
        it.append(_item("frame", B, [0.55, 0.06, 0.022], [0.0, 0.0, 0.03], eye, ALU))
    # base housing over the three motors (the real robot's wide top housing), rounded lid
    it.append(_item("housing", C, [0.30, 0.055], [0.0, 0.0, 0.035], eye, HOUSING))
    it.append(_item("housing", E, [0.30, 0.30, 0.05], [0.0, 0.0, 0.09], eye, HOUSING))
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


def add_lights(scn):
    """Soft key (shadows), rim and fill; replaces the model's single light."""
    import mujoco
    spec = [  # pos, target, diffuse, specular, castshadow, cutoff
        ([1.6, -1.4, 1.2], [0.0, 0.0, -0.9], 0.75, 0.35, 1, 40.0),
        ([-1.4, 1.7, 0.6], [0.0, 0.0, -0.8], 0.45, 0.45, 0, 45.0),
        ([0.4, -2.2, -0.4], [0.0, 0.0, -0.9], 0.22, 0.0, 0, 60.0),
    ]
    scn.nlight = 0
    for pos, tgt, dif, spc, sh, cut in spec:
        L = scn.lights[scn.nlight]
        L.type = mujoco.mjtLightType.mjLIGHT_SPOT
        L.pos[:] = pos
        dvec = np.array(tgt) - np.array(pos)
        L.dir[:] = dvec / np.linalg.norm(dvec)
        L.ambient[:] = [0.0, 0.0, 0.0]
        L.diffuse[:] = [dif] * 3
        L.specular[:] = [spc] * 3
        L.castshadow = sh
        L.cutoff, L.exponent = cut, 2.0
        L.attenuation[:] = [1.0, 0.0, 0.0]
        L.headlight = 0
        try:
            L.softness = 0.35
        except Exception:
            pass
        scn.nlight += 1


# ================================================================================================ cameras
def ease(u):
    u = float(np.clip(u, 0, 1))
    return u * u * (3 - 2 * u)


CAMERAS = {  # lookat (None: tracked), distance, azimuth, elevation, fovy (deg; view height in m if ortho)
    # A: low, close, long lens; robot + belts fill ~60 % of the frame; no support frame (its legs would sit in front)
    "A-wide": dict(lookat=[0.0, 0.0, -0.50], dist=3.0, az=150, el=-4, fovy=22, orbit=0.0, frame=False, depth=True),
    "A-wide-orbit": dict(lookat=[0.0, 0.0, -0.50], dist=3.0, az=143, el=-4, fovy=22, orbit=14.0, frame=False, depth=True),
    "B-top": dict(lookat=[0.0, 0.0, -1.0], dist=0.95, az=0.0, el=-89.9, fovy=55, orbit=0.0),
    # C: follows the cup (Gaussian-smoothed, sigma 40 ms): cup in the upper third, the bar in the lower third
    "C-close": dict(lookat=None, track="cup", offset=[0.0, 0.0, -0.016], sigma=0.04, dist=0.30, az=158, el=-3,
                    fovy=18, orbit=0.0, depth=True),
    "D-track": dict(lookat=None, track="tool", offset=[0.0, 0.0, 0.0], sigma=0.12, dist=0.75, az=150, el=-12, fovy=24,
                    orbit=0.0),
    "E-side": dict(lookat=[0.0, 0.0, -0.965], dist=3.0, az=90, el=-1.0, fovy=0.52, orbit=0.0, ortho=True, ghosts=False),
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
    ff = subprocess.Popen(
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
    for f in range(n_frames):
        t = t0 + f / fps_sim
        k, i = state(t)
        p = picks[k]
        m = p["model"]
        if k not in renderers:
            style_model(m, W, H, camd["fovy"], camd.get("ortho", False))
            renderers[k] = mujoco.Renderer(m, H, W)
            datas.setdefault(k, mujoco.MjData(m))
        r, d = renderers[k], datas[k]
        d.qpos[:] = p["qpos"][i]
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        u = ease(f / max(n_frames - 1, 1))
        cam.lookat[:] = track[f] if track is not None else camd["lookat"]
        cam.distance, cam.azimuth, cam.elevation = camd["dist"], camd["az"] + camd["orbit"] * u, camd["el"]
        cam.orthographic = int(camd.get("ortho", False))     # camera-level switch: fovy is then the view height (m)
        r.update_scene(d, cam)
        scn = r.scene
        restyle_scene(m, scn, p["rgba"])
        tl = i * MS
        items = display_items(m, d, p, picks, k, t, tl, job["pitch"], camd)
        draw_items(scn, items)
        add_lights(scn)
        img = r.render()
        ff.stdin.write(img.tobytes())
        if camd.get("depth") and job.get("depth", True):     # depth pass (metres -> 16-bit PNG in mm, half resolution)
            r.enable_depth_rendering()
            dep = r.render()
            r.disable_depth_rendering()
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
                   bbox_x0=round(np.nanmin(uv[:, 0]), 1), bbox_y0=round(np.nanmin(uv[:, 1]), 1),
                   bbox_x1=round(np.nanmax(uv[:, 0]), 1), bbox_y1=round(np.nanmax(uv[:, 1]), 1), visible=int(vis))
        rows.append(row)
        cams.append(dict(frame=f, pos=[round(float(x), 5) for x in gl.pos], forward=[round(float(x), 5) for x in gl.forward],
                         up=[round(float(x), 5) for x in gl.up], lookat=[round(float(x), 5) for x in cam.lookat],
                         fovy=camd["fovy"], orthographic=bool(camd.get("ortho", False))))
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
    return e["move_start"] - 1.0, end + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", default="test", nargs="?")
    ap.add_argument("--W", type=int, default=3840)
    ap.add_argument("--H", type=int, default=2160)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="override the plan's output folder")
    ap.add_argument("--only", default="", help="comma-separated cameras to render (default: all in the plan)")
    a = ap.parse_args()
    plan = make_plan_test() if a.plan == "test" else json.loads(Path(a.plan).read_text())
    if a.only:
        plan["clips"] = [c for c in plan["clips"] if c["camera"] in a.only.split(",")]
    out = Path(a.out or plan["out"])
    out.mkdir(parents=True, exist_ok=True)
    # shift decisions: TabPFN answered in JAX (policy key tabpfn_jax), fixed recipes fixed (30 %) and fixed50 (50 %)
    decisions = pd.read_csv(ROOT / "results/shift/decisions_jax.csv") if plan["stream"] == "shift:bar" else None
    picks = build_stream(plan["stream"], plan["idxs"], plan["policy"], decisions, plan.get("alpha"), a.workers)
    pitch = queue_pitch(picks)
    story = []
    if plan["stream"] == "shift:bar":
        for i, name in ((30, "heavy_variant"), (60, "new_batch"), (90, "worn_cup"), (110, "cup_change")):
            for p in picks:
                if p["idx"] == i:
                    story.append(dict(event=name, pick_id=i, sim_time=round(p["t_start_g"], 6)))
    jobs = []
    for c in plan["clips"]:
        t0, t1 = clip_window(picks, c)
        pol = plan["policy"]
        name = f"{c['scene']}_{c['label']}_{c['camera']}_{pol}_{c['sim_fps']}.mp4"
        jobs.append(dict(picks=picks, pitch=pitch, policy=pol, story=story, path=str(out / name),
                         clip=dict(camera=c["camera"], t0=t0, t1=t1, sim_fps=c["sim_fps"], W=a.W, H=a.H)))
    with ProcessPoolExecutor(min(a.workers, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(render_clip, jobs))
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else []
    commit = git_commit()
    for c, rres, j in zip(plan["clips"], results, jobs):
        man = [e for e in man if e["file"] != rres["file"]]
        man.append(dict(file=rres["file"], scene=c["scene"], camera=c["camera"], policy=plan["policy"],
                        sim_fps=c["sim_fps"], video_fps=60, resolution=f"{a.W}x{a.H}", frames=rres["frames"],
                        sim_t0=rres["sim_t0"], sim_t1=rres["sim_t1"], stream=plan["stream"],
                        pick_ids=[p["idx"] for p in picks if p["t_start_g"] + p["dur"] >= rres["sim_t0"] and p["t_start_g"] <= rres["sim_t1"]],
                        sim_commit=commit, display_only=["infeed queue", "outfeed carry-off", "support frame", "studio"]))
        print(f"{rres['file']}: {rres['frames']} frames, sim {rres['sim_t0']:.3f}-{rres['sim_t1']:.3f} s", flush=True)
    man_p.write_text(json.dumps(man, indent=1, default=_num))


if __name__ == "__main__":
    main()
