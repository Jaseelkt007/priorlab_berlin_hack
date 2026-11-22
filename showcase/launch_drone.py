"""Clean launch-film footage for the drone project (drone/), without touching its code: imports its simulation,
saved predictions and scene, and renders text-free 4K/60 shots with per-frame sidecars for the editor.

The case the drone README headlines: pilot b (the evasive pilot no model has seen), test flight 78, an attack that
breaches the protected zone at 24.32 s. Drawn: the real path (green), the IMM (what radars use, blue) and
IMM + TabPFN (purple, the project's best on the new pilot). The flight is flown exactly as recorded, then played
back. Display only (declared in the manifest): the drone drawn 4x bigger (as in the project's own pitch videos), a
night grade (dark sky, grass and haze, lit windows, glowing zone, a soft light on the drone), the overlays.

Ranges: both models' 1 s-ahead range is the 5 %-95 % interval per direction (forward, left, up in the drone's
heading frame), the cone compare.py scores; drawn as an ellipsoid with those half-widths around the 1 s-ahead point.
IMM: 1.645 sigma of its covariance per direction. IMM + TabPFN: half the 5-95 % quantile spread.

Shots (one per process):
  world   D01 1x   chase from behind the drone toward the protected building: the attack, start to breach
  compare D02 0.5x chase beside the drone: real path vs both models' predicted paths, 1 s-ahead points and ranges
  alarm   D03 0.5x fixed side view of the zone edge: IMM + TabPFN's 1 s-ahead point and range reach the zone,
          the zone glows while its alarm (P breach within 2 s > 0.5) is on, then the drone breaches

  MUJOCO_GL=egl ~/priorlab_drone/.venv/bin/python showcase/launch_drone.py --drone-dir ~/priorlab_drone/drone \\
      --shot compare --out ~/delta_line/footage_drone
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

PILOT, FLIGHT = "b", 78
LAG = 10                 # radar readings in 1 s
Z90 = 1.645              # 5-95 % of a normal: mean +- 1.645 sigma (compare.py)
SCALE = 4.0              # drone drawn 4x bigger (display only)
GREEN, BLUE, PURPLE = (0.10, 0.85, 0.25), (0.15, 0.45, 1.00), (0.75, 0.25, 0.90)   # pitch_video.py's colours
SKY_HORIZON, SKY_ZENITH = np.array([0.085, 0.10, 0.145]), np.array([0.012, 0.016, 0.03])
GRASS = np.array([0.36, 0.42, 0.48])                      # multiplies the grass texture
DARKEN = {"building": 0.45, "roof": 0.40, "mast": 0.55, "dish": 0.55, "feed": 0.5, "pad": 0.35, "road": 0.40,
          "roadline": 0.45, "trunk": 0.30, "crown": 0.30, "house": 0.28}
ZONE_GLOW, ALARM_GLOW = 0.35, 1.0


def ffmpeg_cmd(exe, W, H, out):
    return [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60",
            "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-profile:v", "high",
            "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p", "-g", "60", "-keyint_min", "60",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv",
            "-movflags", "+faststart", "-an", str(out)]


def project(cam, p, W, H):
    """World point -> pixel (u, v) from the scene's GL camera (perspective); NaN if behind the camera."""
    pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
    dv = np.asarray(p, float) - pos
    z = float(dv @ fwd)
    if z <= 1e-6 or not np.isfinite(dv).all():
        return np.nan, np.nan
    x, y = float(dv @ np.cross(fwd, up)) * cam.frustum_near / z, float(dv @ up) * cam.frustum_near / z
    hw = cam.frustum_width if cam.frustum_width > 0 else 0.5 * (cam.frustum_top - cam.frustum_bottom) * W / H
    return ((x - (cam.frustum_center - hw)) / (2 * hw) * W,
            (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * H)


def px_radius(cam, centre, r_m, W, H):
    """Screen size (px) of a world length r_m at centre, along the camera's right axis."""
    if not (np.isfinite(r_m) and np.isfinite(centre).all()):
        return np.nan
    right = np.cross(np.array(cam.forward), np.array(cam.up))
    u0, v0 = project(cam, centre, W, H)
    u1, v1 = project(cam, np.asarray(centre) + r_m * right, W, H)
    return float(np.hypot(u1 - u0, v1 - v0))


def heading_frame(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])   # columns: forward, left, up in world


def night(model, mujoco, drone_body):
    """Night grade on the compiled model (display only; before the Renderer uploads textures). Returns geom id sets
    restyled per frame: windows (lit), zone ring + poles (glow), drone (lighter, faint emission)."""
    for t in range(model.ntex):
        adr, w, h, c = model.tex_adr[t], model.tex_width[t], model.tex_height[t], model.tex_nchannel[t]
        px = model.tex_data[adr: adr + w * h * c].reshape(-1, c).astype(float) / 255.0
        if model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            k = np.clip((0.95 - px[:, :1]) / 0.40, 0, 1)     # 0 at the day horizon colour, 1 at the day zenith
            px[:, :3] = (1 - k) * SKY_HORIZON + k * SKY_ZENITH
        else:
            px[:, :3] = px[:, :3] * GRASS
        model.tex_data[adr: adr + w * h * c] = np.clip(px * 255 + 0.5, 0, 255).astype(np.uint8).ravel()
    windows, zone = set(), set()
    for g in range(model.ngeom):
        name = model.geom(g).name.removeprefix("zone_")
        base = name.rstrip("0123456789")
        if base in DARKEN:
            model.geom_rgba[g, :3] *= DARKEN[base]
        elif base == "windows":
            model.geom_rgba[g, :3] = [0.98, 0.78, 0.48]
            windows.add(g)
        elif base in ("ring", "pole"):
            model.geom_rgba[g, :3] = [0.95, 0.12, 0.10]
            zone.add(g)
    model.vis.rgba.haze[:] = [0.035, 0.045, 0.07, 1]
    model.vis.headlight.ambient[:] = [0.05, 0.055, 0.07]
    model.vis.headlight.diffuse[:] = [0.09, 0.09, 0.10]
    model.vis.headlight.specular[:] = [0, 0, 0]
    sun = model.light("sun").id
    model.light_diffuse[sun] = [0.26, 0.30, 0.42]            # moonlight
    model.light_specular[sun] = [0.05, 0.05, 0.06]
    root = model.body_rootid[drone_body]
    drone = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root}
    return windows, zone, drone


def spot(scn, mujoco, pos, aim, rgb, cutoff, exponent, quad=0.0):
    if scn.nlight >= len(scn.lights):
        return
    L = scn.lights[scn.nlight]
    L.type = mujoco.mjtLightType.mjLIGHT_SPOT
    L.pos[:] = pos
    d = np.asarray(aim, float) - np.asarray(pos, float)
    L.dir[:] = d / np.linalg.norm(d)
    L.ambient[:], L.diffuse[:], L.specular[:] = [0, 0, 0], rgb, [0.3 * x for x in rgb]
    L.castshadow, L.headlight = 0, 0
    L.cutoff, L.exponent = cutoff, exponent
    L.attenuation[:] = [1.0, 0.0, quad]
    scn.nlight += 1


def episodes(p, t):
    """[(on, off)] sim times where p > 0.5 (NaN = off); off is None if still on at the end."""
    on = np.nan_to_num(p, nan=0.0) > 0.5
    out, start = [], None
    for k in range(len(on)):
        if on[k] and start is None:
            start = float(t[k])
        if not on[k] and start is not None:
            out.append((start, float(t[k])))
            start = None
    if start is not None:
        out.append((start, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone-dir", required=True, help="the drone project's code folder (its data/ next to it)")
    ap.add_argument("--shot", choices=["world", "compare", "alarm"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--W", type=int, default=3840)
    ap.add_argument("--H", type=int, default=2160)
    a = ap.parse_args()
    drone_dir = Path(a.drone_dir).expanduser().resolve()
    sys.path.insert(0, str(drone_dir))
    import imageio_ffmpeg
    import mujoco
    from demo_imm import add
    from features import DATA
    from intruder import PILOT_SEED, SETUP, Flight
    from pitch_video import model_track, polyline
    from sim import POLE_GROUP, ZONE_RADIUS, build

    out_dir = Path(a.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = a.W, a.H
    seed = SETUP["cf2"][2] + PILOT_SEED[PILOT] + FLIGHT

    # 1. fly the flight exactly as recorded; keep the state every 1/120 s (60 fps at 0.5x)
    flight = Flight(seed, "cf2", pilot=PILOT)
    frames, nxt = [], 0.0
    while flight.step():
        if flight.data.time >= nxt:
            frames.append((flight.data.time, flight.data.qpos.copy(), len(flight.log["t"]) - 1))
            nxt += 1 / 120
    frames.append((flight.data.time, flight.data.qpos.copy(), len(flight.log["t"]) - 1))
    log = {k: np.asarray(v) for k, v in flight.log.items()}
    true, n, tk = log["true"], len(log["t"]), log["t"]
    t_end = float(flight.data.time)
    times = np.array([f[0] for f in frames])

    # 2. the saved predictions this flight was scored on, and both models' 5-95 % half-widths per direction
    tracks = {m: model_track(m, PILOT, seed, n) for m in ("imm", "combo")}
    table = np.load(DATA / "table_cf2_pilotb.npz")
    rows = np.where(table["flight"] == seed)[0]
    kk = np.rint(table["t"][rows] / 0.1).astype(int)
    ok = kk < n
    psi = np.full(n, np.nan)
    psi[kk[ok]] = table["psi"][rows[ok]]
    imm = np.load(DATA / "imm_cf2_pilotb.npz")
    hi = int(np.argmin(np.abs(imm["horizons"] - 1.0)))
    half = {m: np.full((n, 3), np.nan) for m in tracks}
    for k, row in zip(kk[ok], rows[ok]):
        cov = imm["cov"][row, hi].astype(float)
        if np.isfinite(cov).all() and np.isfinite(psi[k]):
            R = heading_frame(psi[k])
            half["imm"][k] = Z90 * np.sqrt(np.maximum(np.diag(R.T @ cov @ R), 0.0))
    combo = np.load(DATA / "tabpfn_cf2_pilotb_combo.npz")
    tr = combo["test_rows"]
    mine = table["flight"][tr] == seed
    kq = np.rint(table["t"][tr[mine]] / 0.1).astype(int)
    q = combo["B_x2_plus_cf2_quantiles"][mine]              # (rows, target fwd/left/up, quantile 5/50/95 %)
    half["combo"][kq[kq < n]] = ((q[:, :3, 2] - q[:, :3, 0]) / 2.0)[kq < n]

    # 3. beats
    breach_t = t_end if flight.outcome == "breach" else None
    alarm_eps = {m: episodes(tracks[m][1], tk) for m in tracks}
    rad = lambda p: float(np.hypot(p[0], p[1]))
    imm_ahead_in = next((float(tk[k]) for k in range(n) if np.isfinite(tracks["imm"][0][k]).all()
                         and rad(tracks["imm"][0][k]) < ZONE_RADIUS), None)
    combo_range_in = next((float(tk[k]) for k in range(n) if np.isfinite(tracks["combo"][0][k]).all()
                           and np.isfinite(half["combo"][k]).all()
                           and rad(tracks["combo"][0][k]) - half["combo"][k][:2].max() < ZONE_RADIUS), None)
    combo_last = max((float(tk[k]) for k in range(n) if np.isfinite(tracks["combo"][0][k]).all()), default=None)
    beats = dict(breach=breach_t, imm_alarm_episodes=alarm_eps["imm"], combo_alarm_episodes=alarm_eps["combo"],
                 imm_1s_point_enters_zone=imm_ahead_in, combo_1s_range_reaches_zone=combo_range_in,
                 combo_1s_prediction_last=combo_last)

    # 4. the shot
    first_alarm = alarm_eps["combo"][0][0] if alarm_eps["combo"] else t_end - 6
    name, slow, t0 = {"world": ("D01_world_chase_60", 1.0, 0.0),
                      "compare": ("D02_compare_chase_60", 0.5, max(t_end - 9.0, 0.0)),
                      "alarm": ("D03_alarm_side_60", 0.5, max(first_alarm - 3.0, 0.0))}[a.shot]
    t1 = t_end + 1.0                                        # 1 s handle: the drone holds after the breach
    step = slow / 60.0
    grid = np.arange(t0, t1 + 1e-9, step)
    models_drawn = {"world": (), "compare": ("imm", "combo"), "alarm": ("combo",)}[a.shot]

    model = build(["cf2"], zone=True, timestep=0.002, scale=SCALE)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.quality.offsamples = 8
    body = model.body("cf2_cf2").id
    windows, zone, drone_geoms = night(model, mujoco, body)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[POLE_GROUP] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def drone_at(i):
        data.qpos[:] = frames[i][1]
        mujoco.mj_forward(model, data)
        return data.xpos[body].copy()

    if a.shot == "alarm":                                  # fixed side view of the approach to the breach point
        A = drone_at(min(int(np.searchsorted(times, t0)), len(frames) - 1))
        B = drone_at(len(frames) - 1)
        ab = (B - A)[:2]
        L_ab = float(np.linalg.norm(ab))
        nrm = np.array([-ab[1], ab[0]]) / max(L_ab, 1e-6)
        M = A + 0.7 * (B - A)
        if nrm @ M[:2] < 0:
            nrm = -nrm                                      # camera on the far side from the building: it sits behind
        side = dict(M=np.array([M[0], M[1], 3.5]), dist=0.5 * L_ab + 7.0,
                    az=float(np.degrees(np.arctan2(-nrm[1], -nrm[0]))))

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    ff = subprocess.Popen(ffmpeg_cmd(exe, W, H, out_dir / f"{name}.mp4"), stdin=subprocess.PIPE)
    import time
    rows_out, cams, look, glow = [], [], None, ZONE_GLOW
    spent = dict(scene=0.0, overlay=0.0, render=0.0, write=0.0, sidecar=0.0)
    clock = time.perf_counter()
    for f, t in enumerate(grid):
        tic = time.perf_counter()
        ts = min(t, t_end)
        _, qpos, k = frames[min(int(np.searchsorted(times, ts)), len(frames) - 1)]
        k = min(k, n - 1)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pos = data.xpos[body].copy()

        if a.shot == "world":                               # behind the drone, looking at the building
            az_t = np.arctan2(-pos[1], -pos[0])
            look = [az_t, pos.copy()] if look is None else look
            look[0] += 0.03 * np.angle(np.exp(1j * (az_t - look[0])))
            look[1] += 0.15 * (pos - look[1])
            fwd = np.array([np.cos(look[0]), np.sin(look[0]), 0.0])
            cam.lookat[:] = look[1] + 4.0 * fwd + np.array([0, 0, 0.8])
            cam.distance, cam.elevation, cam.azimuth = 11.0, -8.0, float(np.degrees(look[0]))
        elif a.shot == "compare":                           # beside the drone (pitch_video.set_camera), a bit closer
            v = log["vel"][k][:2]
            h = np.arctan2(v[1], v[0]) if np.hypot(*v) > 0.3 else None
            look = [h if h is not None else 0.0] if look is None else look
            if h is not None:
                look[0] += 0.05 * np.angle(np.exp(1j * (h - look[0])))
            fwd = np.array([np.cos(look[0]), np.sin(look[0]), 0.0])
            cam.lookat[:] = pos - fwd * 2.5
            cam.distance, cam.elevation, cam.azimuth = 11.0, -22.0, float(np.degrees(look[0])) + 90.0
        else:
            tgt = side["M"] + 0.25 * (pos - side["M"])
            look = tgt.copy() if look is None else look + 0.05 * (tgt - look)
            cam.lookat[:] = look
            cam.distance, cam.elevation, cam.azimuth = side["dist"], -16.0, side["az"]

        r.update_scene(data, cam, scene_option=opt)
        scn = r.scene
        gl = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])   # the mono camera (camera[0] is the left eye)
        alarm_on = bool(np.isfinite(tracks["combo"][1][k]) and tracks["combo"][1][k] > 0.5)
        if a.shot == "alarm":
            glow += 0.25 * ((ALARM_GLOW if alarm_on else ZONE_GLOW) - glow)
        lift = (glow - ZONE_GLOW) / (ALARM_GLOW - ZONE_GLOW)
        for i in range(scn.ngeom):                          # night grade on the model's own geoms
            g = scn.geoms[i]
            if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                continue
            if g.objid in windows:
                g.emission = 0.6
            elif g.objid in zone:
                g.emission = glow
                g.rgba[:3] = np.array([0.95, 0.12, 0.10]) + lift * np.array([0.05, 0.13, 0.02])
            elif g.objid in drone_geoms:
                g.rgba[:3] = 0.5 * np.array(g.rgba[:3]) + 0.5 * np.array([0.85, 0.85, 0.88])
                g.emission = 0.25
        cam_pos = np.array(gl.pos)
        toward = cam_pos - pos
        toward[2] = 0
        toward /= max(np.linalg.norm(toward), 1e-6)
        spot(scn, mujoco, pos + np.array([0, 0, 3.0]) + 2.0 * toward, pos, [0.75, 0.72, 0.68], 14.0, 8.0, 0.04)
        spot(scn, mujoco, [0.0, 0.0, 34.0], [0.0, 0.0, 0.0], [0.38, 0.34, 0.28], 19.0, 2.0)

        spent["scene"] += time.perf_counter() - tic
        tic = time.perf_counter()
        n_model = scn.ngeom                                 # overlays from here (display only)
        def clear(points):                                  # no capsule right in front of the lens
            pts = np.array(points, float)
            pts[np.linalg.norm(pts - cam_pos, axis=1) < 4.0] = np.nan
            return pts

        trail = true[max(0, k - 30): k + 1] if a.shot == "world" else true[: k + 1]   # D01: the last 3 s
        polyline(scn, clear(np.vstack([trail, pos])), 0.04 if a.shot == "world" else 0.03, [*GREEN, 1])
        info = {}
        for m in ("imm", "combo"):
            col = BLUE if m == "imm" else PURPLE
            pred, alarm = tracks[m]
            said = pred[k - LAG] if k >= LAG else np.full(3, np.nan)
            ahead, hw = pred[k], half[m][k]
            if m in models_drawn:
                if k >= LAG:
                    polyline(scn, clear(pred[: k - LAG + 1]), 0.025, [*col, 1])
                if a.shot == "compare" and np.isfinite(said).all():
                    add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.10] * 3, said, [*col, 1])
                if np.isfinite(ahead).all():
                    add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.16] * 3, ahead, [*col, 1])
                    if np.isfinite(hw).all() and np.isfinite(psi[k]):
                        add(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.maximum(hw, 0.05), ahead, [*col, 0.16],
                            heading_frame(psi[k]).ravel())
            info[m] = dict(said=said, ahead=ahead, rad=float(np.max(hw[:2])) if np.isfinite(hw).all() else np.nan,
                           p=float(alarm[k]) if np.isfinite(alarm[k]) else np.nan)
        for i in range(n_model, scn.ngeom):
            g = scn.geoms[i]
            g.castshadow = 0
            g.emission = 0.25 if g.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID else 0.7

        spent["overlay"] += time.perf_counter() - tic
        tic = time.perf_counter()
        img = r.render()
        spent["render"] += time.perf_counter() - tic
        tic = time.perf_counter()
        ff.stdin.write(img.tobytes())
        spent["write"] += time.perf_counter() - tic
        tic = time.perf_counter()

        du, dv = project(gl, pos, W, H)
        row = dict(frame=f, video_time_s=round(f / 60.0, 4), sim_time_s=round(float(ts), 4), radar_step=int(k),
                   drone_x_m=round(float(pos[0]), 3), drone_y_m=round(float(pos[1]), 3), drone_z_m=round(float(pos[2]), 3),
                   drone_px=round(du, 1), drone_py=round(dv, 1),
                   drone_visible=int(np.isfinite(du) and 0 <= du < W and 0 <= dv < H),
                   drone_in_zone=int(breach_t is not None and t >= breach_t - 1e-9))
        for m, s in info.items():
            u1, v1 = project(gl, s["said"], W, H)
            u2, v2 = project(gl, s["ahead"], W, H)
            fin = np.isfinite(s["ahead"]).all()
            row.update({f"{m}_drawn": int(m in models_drawn),
                        f"{m}_guess_now_px": round(u1, 1), f"{m}_guess_now_py": round(v1, 1),
                        f"{m}_error_now_m": round(float(np.linalg.norm(s["said"] - true[k])), 3)
                        if np.isfinite(s["said"]).all() else np.nan,
                        f"{m}_ahead1s_x_m": round(float(s["ahead"][0]), 3) if fin else np.nan,
                        f"{m}_ahead1s_y_m": round(float(s["ahead"][1]), 3) if fin else np.nan,
                        f"{m}_ahead1s_z_m": round(float(s["ahead"][2]), 3) if fin else np.nan,
                        f"{m}_ahead1s_px": round(u2, 1), f"{m}_ahead1s_py": round(v2, 1),
                        f"{m}_range90_halfwidth_m": round(s["rad"], 3) if np.isfinite(s["rad"]) else np.nan,
                        f"{m}_range90_halfwidth_px": round(px_radius(gl, s["ahead"], s["rad"], W, H), 1),
                        f"{m}_alarm_p": round(s["p"], 3) if np.isfinite(s["p"]) else np.nan,
                        f"{m}_alarm": int(np.isfinite(s["p"]) and s["p"] > 0.5)})
        row["zone_glow"] = round(glow, 3)
        rows_out.append(row)
        spent["sidecar"] += time.perf_counter() - tic
        if (f + 1) % 300 == 0:
            el = time.perf_counter() - clock
            print(f"{name}: frame {f + 1}/{len(grid)}, {(f + 1) / el:.1f} fps; ms/frame "
                  + " ".join(f"{k} {1000 * v / (f + 1):.0f}" for k, v in spent.items()), flush=True)
        cams.append(dict(frame=f, pos=[round(float(x), 4) for x in gl.pos],
                         forward=[round(float(x), 4) for x in gl.forward], up=[round(float(x), 4) for x in gl.up]))
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {name}")

    with open(out_dir / f"{name}.frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    frame_of = lambda tt: int(round((tt - t0) / step)) if tt is not None and t0 <= tt <= t1 else None
    events = []
    for m in tracks:
        for on, off in alarm_eps[m]:
            events.append(dict(event=f"{m}_alarm_on", sim_time=on, frame=frame_of(on)))
            if off is not None:
                events.append(dict(event=f"{m}_alarm_off", sim_time=off, frame=frame_of(off)))
    for e in ("breach", "imm_1s_point_enters_zone", "combo_1s_range_reaches_zone", "combo_1s_prediction_last"):
        events.append(dict(event=e, sim_time=beats[e], frame=frame_of(beats[e])))
    events.sort(key=lambda e: (e["sim_time"] is None, e["sim_time"] or 0.0))
    (out_dir / f"{name}.events.json").write_text(json.dumps(dict(
        clip=f"{name}.mp4", video_fps=60, playback_speed=slow, sim_t0=t0, sim_t1=t1, events=events,
        camera_per_frame=cams), indent=1))

    import fcntl
    commit = subprocess.run(["git", "-C", str(drone_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(out_dir / ".manifest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)                    # the three shots run in parallel and share one manifest
        man_p = out_dir / "manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else []
        man = [e for e in man if e["file"] != f"{name}.mp4"]
        man.append(dict(file=f"{name}.mp4", project="drone", scene=name.split("_")[0], shot=a.shot,
                        camera=name.split("_")[2], pilot=PILOT, flight=FLIGHT, resolution=f"{W}x{H}", video_fps=60,
                        frames=len(grid), duration_s=round(len(grid) / 60.0, 3), playback_speed=slow,
                        sim_t0=round(t0, 3), sim_t1=round(t1, 3), models_drawn=list(models_drawn),
                        colours=dict(real="green", imm="blue", imm_plus_tabpfn="purple"),
                        events=[e for e in events if e["frame"] is not None], drone_commit=commit,
                        sidecar=f"{name}.frames.csv (pixels from the top-left at {W}x{H})",
                        display_only=["drone drawn 4x bigger", "night grade (sky, grass, haze, lit windows, glowing "
                                      "zone, soft light on the drone)", "paths, 1 s-ahead points and 5-95 % ranges",
                                      "zone glows brighter while the IMM + TabPFN alarm is on (D03)"]))
        man.sort(key=lambda e: e["file"])
        man_p.write_text(json.dumps(man, indent=1))
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"{name}: {len(grid)} frames, {W}x{H}, sim {t0:.2f}-{t1:.2f} s at {slow}x", flush=True)
    print("beats:", json.dumps(beats), flush=True)


if __name__ == "__main__":
    main()
