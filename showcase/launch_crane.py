"""Crane footage for the launch film: clean renders of the crane project's own experiments (crane/, unmodified).

The crane project: an overhead crane's load swings after every start and stop; industry anti-sway (ZV input shaping)
only cancels it if it knows the pendulum, including a sling length someone types in. TabPFN reads the crane's own
sensor rows (12 numbers every 0.1 s; 3,000 fleet rows + the load's own rows as context) and predicts the load 1 s
ahead with a 5/50/95 % range; in the closed loop it scores 28 (speed x anti-sway tuning) candidates per move and takes
the fastest whose 99th-percentile swing stays inside the 0.45 m rack clearance, adding each outcome as a row.

Stages (the numbers in every sidecar come from these runs; the crane code is imported, not copied or edited):
  forecast  crane/mujoco_demo.py's experiment, same seed: fleet of 30 jobs, held-out 400 kg beam on 2.6 m slings,
            TabPFN refit every 5 s on fleet + this load's known rows -> results/crane_launch/forecast.json
  closed    crane/mujoco_closed_loop.py's experiment, same seed: 150 commissioning moves, 12 new loads, industry
            anti-sway (tuned for 0.5 m) vs TabPFN's choice -> results/crane_launch/closed.json
  render    clean shots (no text) at 60 fps, dark studio, sidecar CSVs, manifest -> --out
Usage (server): MUJOCO_GL=egl .venv/bin/python showcase/launch_crane.py forecast|closed|render [--out DIR] [--mode video|preview|sidecars]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "crane"))
sys.path.insert(0, str(ROOT / "showcase"))
import multiprocessing as mp

import mujoco
import numpy as np

import mujoco_crane as mc

DATA = ROOT / "results" / "crane_launch"
FPS = 60
W, H = 3840, 2160


# ================================================================================================ experiments
def stage_forecast():
    """crane/mujoco_demo.py main(), same seed and steps, without its video; predictions kept per row."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    import mujoco_demo as md
    from tabpfn import TabPFNRegressor
    rng = random.Random(1)
    fleet = []
    for j in range(30):
        cond = dict(L1=rng.uniform(3, 7), L2=rng.uniform(0.5, 1.5), m_hook=250.0, m_load=rng.uniform(1000, 3000),
                    beam=rng.uniform(1.0, 2.0))
        plan = mc.Plan([rng.uniform(0.5, 1.0) for _ in range(4)])
        fleet += md.rows_from_log(mc.run(cond, plan), plan, cond, j)
    base = rng.sample(fleet, 3000)
    plan = mc.Plan(md.DEMO_FRACS)
    log = mc.run(md.DEMO, plan)
    job = md.rows_from_log(log, plan, md.DEMO, "demo")
    preds, T = {}, job[0]["t"]
    while T <= job[-1]["t"]:
        known = [r for r in job if r["t"] <= T - md.H]
        ctx = base + known
        m = TabPFNRegressor(n_estimators=1, fit_mode="fit_with_cache")
        m.fit(md.X(ctx), np.array([r["y"] for r in ctx]))
        batch = [r for r in job if T <= r["t"] < T + md.REFIT]
        if batch:
            lo, med, hi = (np.asarray(v) for v in m.predict(md.X(batch), output_type="quantiles", quantiles=[0.05, 0.5, 0.95]))
            for r, l, mm, h in zip(batch, lo, med, hi):
                preds[r["i"]] = (float(l), float(mm), float(h), len(known))
        T += md.REFIT
    rows = []
    for r in job:
        l, mm, h, n = preds[r["i"]]
        rows.append(dict(t=round(r["t"], 3), off=r["f"]["off"], y=r["y"], p05=l, p50=mm, p95=h, own_rows=n,
                         phys=md.physics_pred(r, plan, md.DEMO), x=r["xt"]))
    seg1 = plan.seg[1][0]
    first = [r for r in rows if r["t"] < seg1]
    later = [r for r in rows if r["t"] >= seg1]
    summ = {k: dict(tabpfn_cm=round(100 * np.mean([abs(r["p50"] - r["y"]) for r in g]), 1),
                    physics_cm=round(100 * np.mean([abs(r["phys"] - r["y"]) for r in g]), 1))
            for k, g in (("first_move", first), ("later_moves", later))}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "forecast.json").write_text(json.dumps(dict(cond=md.DEMO, fracs=md.DEMO_FRACS, entered_sling=md.ENTERED_SLING,
                                                        horizon_s=md.H, summary=summ, rows=rows,
                                                        segments=[list(s[:2]) for s in plan.seg]), indent=0))
    print("forecast:", json.dumps(summ), "(published: crane/results/mujoco_demo.json)", flush=True)


def stage_closed():
    """crane/mujoco_closed_loop.py main(), same seed and decisions, industry + TabPFN lanes (no replay frames)."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    import mujoco_closed_loop as cl
    rng = random.Random(7)
    fleet = []
    for _ in range(150):
        c = dict(L1=rng.uniform(3, 6), L2=rng.uniform(0.5, 1.0), m_hook=250.0, m_load=rng.uniform(1000, 3000), beam=rng.uniform(1.0, 2.0))
        meas = cl.period_true(c) * (1 + 0.03 * rng.gauss(0, 1))
        f, t = rng.choice(cl.SPEEDS), rng.choice(cl.TUNES)
        o = cl.simulate_move(c, f, t)
        fleet.append(dict(x=cl.feats(c, meas, f, t), amp=o["amp"], settle=o["settle_s"]))
    jobs = []
    for _ in range(12):
        c = dict(L1=rng.uniform(3, 6), L2=rng.uniform(1.8, 3.0), m_hook=250.0, m_load=rng.uniform(300, 800), beam=rng.uniform(1.5, 2.5))
        jobs.append(dict(c=c, meas=cl.period_true(c) * (1 + 0.03 * rng.gauss(0, 1))))
    industry = [dict(frac=1.0, tune=cl.ENTERED, **{k: v for k, v in cl.simulate_move(j["c"], 1.0, cl.ENTERED).items() if k != "frames"})
                for j in jobs]
    ctx, lane = list(fleet), []
    for k, j in enumerate(jobs):
        best, rows, wall = cl.decide(ctx, j["c"], j["meas"])
        o = {kk: v for kk, v in cl.simulate_move(j["c"], best["frac"], best["tune"]).items() if kk != "frames"}
        lane.append(dict(frac=best["frac"], tune=best["tune"], pred_amp50=best["amp50"], pred_amp99=best["amp99"],
                         pred_settle=best["settle"], decide_s=wall, own_rows=k, **o))
        ctx.append(dict(x=cl.feats(j["c"], j["meas"], best["frac"], best["tune"]), amp=o["amp"], settle=o["settle_s"]))
    summ = {name: dict(total_s=round(sum(r["cycle_s"] for r in L), 1), hits=int(sum(r["hit"] for r in L)))
            for name, L in (("industry", industry), ("tabpfn", lane))}
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "closed.json").write_text(json.dumps(dict(jobs=jobs, industry=industry, tabpfn=lane, summary=summ), indent=0))
    print("closed:", json.dumps(summ), "(published: industry 403.4 s, tabpfn 260.6 s)", flush=True)
    for k, (a, b) in enumerate(zip(industry, lane)):
        print(f"  move {k + 1:2d}: sling {jobs[k]['c']['L2']:.2f}  industry settle {a['settle_s']:5.1f} s amp {a['amp']:.2f} | "
              f"tabpfn {int(b['frac'] * 100)}%/{b['tune']} settle {b['settle_s']:5.1f} s amp {b['amp']:.2f}", flush=True)


# ================================================================================================ look
BG_TOP, BG_BOTTOM = np.array([22, 25, 30]), np.array([11, 13, 16])     # the delta footage's background gradient
FLOOR_A, FLOOR_B = np.array([17, 19, 22]), np.array([23, 25, 29])     # a faint dark checker


def darken(m):
    """Dark studio, matching the delta footage: skybox to the dark blue-grey gradient, the floor checker to near
    black (before the GL context is made, so the textures upload dark). The crane keeps its own colours."""
    for t in range(m.ntex):
        a, c = int(m.tex_adr[t]), int(m.tex_nchannel[t])
        n = (int(m.tex_adr[t + 1]) if t + 1 < m.ntex else len(m.tex_data)) - a
        px = m.tex_data[a:a + n].reshape(-1, c).astype(np.float32)
        lum = px[:, :3].mean(axis=1, keepdims=True)
        u = (lum - lum.min()) / max(float(np.ptp(lum)), 1.0)
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            px[:, :3] = BG_BOTTOM + u * (BG_TOP - BG_BOTTOM)
        else:
            px[:, :3] = np.where(u > 0.5, FLOOR_B, FLOOR_A)
        m.tex_data[a:a + n] = np.clip(px, 0, 255).astype(np.uint8).ravel()
    m.vis.headlight.ambient[:] = [0.18, 0.18, 0.19]
    m.vis.headlight.diffuse[:] = [0.35, 0.35, 0.36]
    m.light_castshadow[:] = 0                # hard black shadow stripes on the dark floor read as artefacts
    m.vis.quality.offsamples = 8
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H


def add_box(scn, size, pos, rgba, emission=0.0):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    g.emission = emission
    scn.ngeom += 1


GHOST, ORANGE = (0.55, 1.0, 0.70), (1.0, 0.55, 0.15)   # TabPFN ghost (mint, the README's "green ghost"), physics


# ================================================================================================ simulation with frames
def sim_frames(cond, ref_state, t_end):
    """mujoco_crane's control law (feedforward + PD on the trolley) stepping at DT; one qpos per 1/60 s (the first
    step at or after each frame time). ref_state(t) -> (x, v, a) absolute trolley reference."""
    m, d = mc.make(cond)
    tid = m.body("trolley").id
    M = float(m.body_subtreemass[tid])
    kp, kv = M * 25.0, M * 10.0
    frames = []
    while len(frames) / FPS < t_end:
        while d.time + 1e-9 < len(frames) / FPS:
            xr, vr, ar = ref_state(d.time)
            d.ctrl[0] = M * ar + kp * (xr - d.qpos[0]) + kv * (vr - d.qvel[0])
            mujoco.mj_step(m, d)
        frames.append(d.qpos.copy())
    return m, np.array(frames)


# ================================================================================================ render
def project(cam, p):
    from render_footage import project as pr
    return pr(cam, p, W, H)


def encode(path):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-c:v", "libx264", "-profile:v", "high",
         "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p", "-g", "60", "-keyint_min", "60",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=tv",
         "-movflags", "+faststart", "-an", path], stdin=subprocess.PIPE)


def render_shot(job):
    """job: name, cond, frames (qpos per 1/60 s), camera ('side' or dict free cam), overlay(t) -> list of ghost boxes,
    row(t) -> sidecar fields, mode: video (<name>.mp4 + <name>.frames.csv), preview (3 PNG stills at 1/4, 1/2, 3/4)
    or sidecars (only <name>.frames.csv; the poses are deterministic, so it matches an already rendered video)."""
    import pandas as pd
    m, _ = mc.make(job["cond"])
    darken(m)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    mode = job["mode"]
    preview = mode == "preview"
    keep = {len(job["frames"]) * i // 4 for i in (1, 2, 3)}
    ff = encode(job["path"]) if mode == "video" else None
    lid, tid = m.body("load").id, m.body("trolley").id
    L_tot = job["cond"]["L1"] + 0.3 + job["cond"]["L2"] + 0.18
    rows = []
    for f, q in enumerate(job["frames"]):
        if preview and f not in keep:
            continue
        t = job["t0"] + f / FPS
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam = job["camera"]
        if cam == "side":
            r.update_scene(d, camera="side")
        else:
            c = mujoco.MjvCamera()
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = [job["track"][f], 0.0, cam["z"]]
            c.distance, c.azimuth, c.elevation = cam["dist"], 90.0, cam["el"]
            m.vis.global_.fovy = cam["fovy"]
            r.update_scene(d, c)
        scn = r.scene
        boxes, extra = job["overlay"](t, d)
        for size, pos, rgba, em in boxes:
            add_box(scn, size, pos, rgba, em)
        if mode != "sidecars":
            img = r.render()
            if preview:
                from PIL import Image
                Image.fromarray(img).save(job["path"][:-4] + f"_f{f:05d}.png")
            else:
                ff.stdin.write(img.tobytes())
        # the mono image is rendered from the average of the scene's two stereo eyes (camera[0] alone is the left
        # eye, 34 mm off), so project with that average camera
        gl = mujoco.mjv_averageCamera(scn.camera[0], scn.camera[1])
        lpos = d.site_xpos[0].copy()
        lu, lv, lz = project(gl, lpos)
        off = lpos[0] - d.xpos[tid][0]
        row = dict(frame=f, sim_time_s=round(t, 4), controller=job["controller"], trolley_x_m=round(float(d.xpos[tid][0]), 4),
                   load_x_m=round(float(lpos[0]), 4), load_offset_m=round(float(off), 4),
                   swing_deg=round(math.degrees(math.asin(max(-1, min(1, off / L_tot)))), 3),
                   load_screen_x=round(lu, 1), load_screen_y=round(lv, 1), load_visible=int(lz > 0 and 0 <= lu < W and 0 <= lv < H),
                   cam_x_m=round(float(gl.pos[0]), 4), cam_y_m=round(float(gl.pos[1]), 4), cam_z_m=round(float(gl.pos[2]), 4))
        for key, p in extra.pop("_screen", {}).items():
            u, v, z = project(gl, p)
            row[f"{key}_screen_x"], row[f"{key}_screen_y"] = round(u, 1), round(v, 1)
        row.update(extra)
        rows.append(row)
    if preview:
        return dict(file=Path(job["path"]).name, frames=len(job["frames"]))
    if ff is not None:
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {job['path']}")
    pd.DataFrame(rows).to_csv(job["path"][:-4] + ".frames.csv", index=False)
    return dict(file=Path(job["path"]).name, frames=len(job["frames"]))


_JOBS: list = []


def _render_index(i):
    return render_shot(_JOBS[i])


def smooth(x, sigma_frames):
    k = np.exp(-0.5 * (np.arange(-int(3 * sigma_frames), int(3 * sigma_frames) + 1) / sigma_frames) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, (len(k) // 2, len(k) // 2), mode="edge"), k, mode="valid")


def stage_render(out: Path, workers: int, mode: str = "video"):
    import mujoco_closed_loop as cl
    import mujoco_demo as md
    out.mkdir(parents=True, exist_ok=True)
    fc = json.loads((DATA / "forecast.json").read_text())
    cd = json.loads((DATA / "closed.json").read_text())
    jobs = []

    def zl(cond):                       # height of the load's centre when hanging still
        return mc.RAIL_Z - 0.15 - cond["L1"] - 0.3 - cond["L2"] - 0.18

    def close(cond, frames):
        """Free camera from the side camera's direction, framing hook + load, tracking the trolley (smoothed)."""
        mm, dd = mc.make(cond)
        xs = []
        for q in frames:
            dd.qpos[:] = q
            mujoco.mj_kinematics(mm, dd)
            xs.append(dd.xpos[mm.body("trolley").id][0])
        return dict(z=zl(cond) + 1.4, dist=8.5, el=-4.0, fovy=40.0), smooth(np.array(xs), 0.4 * FPS)

    # C01 the problem: the demo load (400 kg beam, 2.6 m slings), one full-speed move with no anti-sway, then the swing
    plan1 = mc.Plan([1.0], pause=10.0)
    _, fr1 = sim_frames(md.DEMO, lambda t: plan1.state(t - 1.0) if t >= 1.0 else (mc.BAYS[0], 0.0, 0.0), t_end=1.0 + plan1.T)
    jobs.append(dict(name="C01_problem_side_60", cond=md.DEMO, frames=fr1, camera="side", t0=0.0, controller="plain move (no anti-sway)",
                     overlay=lambda t, d: ([], {})))

    # C02 the live forecast: the held-out load's 3rd move (TabPFN has 291-490 of this load's rows by then). Ghost =
    # TabPFN's predicted load position 1 s ahead (median, translucent mint), its 90 % range as a faint wider box and a
    # floor band; the vs-physics variant adds the physics model (pendulum with the typed-in 0.5 m sling) in orange
    plan = mc.Plan(md.DEMO_FRACS)
    rows = {round(r["t"], 1): r for r in fc["rows"]}
    s0, s1 = fc["segments"][2][0], fc["segments"][3][0]          # 3rd move start, 4th move start
    t0c, t1c = s0 - 1.0, s1 + 1.0
    _, fr_all = sim_frames(md.DEMO, plan.state, t_end=t1c)
    fr2 = fr_all[int(round(t0c * FPS)):]
    beam, z_load = md.DEMO["beam"], zl(md.DEMO)

    def ov_forecast(t, d, physics=False):
        row = rows.get(round(math.floor(t * 10) / 10, 1))
        if row is None:
            return [], dict(own_rows=np.nan)
        xf = plan.ref(row["t"] + md.H)
        gx, px = xf + row["p50"], xf + row["phys"]
        half_rng = beam / 2 + (row["p95"] - row["p05"]) / 2
        mid_rng = xf + (row["p05"] + row["p95"]) / 2
        boxes = [([half_rng, 1.1, 0.02], [mid_rng, 0, 0.02], (*GHOST, 0.16), 0.3),            # 90 % range on the floor
                 ([half_rng, 0.30, 0.23], [mid_rng, 0, z_load], (*GHOST, 0.07), 0.2)]         # 90 % range at the load
        if physics:
            boxes.append(([beam / 2, 0.27, 0.20], [px, 0, z_load], (*ORANGE, 0.30), 0.35))   # physics, 1 s ahead
        boxes.append(([beam / 2, 0.27, 0.20], [gx, 0, z_load], (*GHOST, 0.34), 0.45))        # TabPFN median, 1 s ahead
        scr = dict(ghost_tabpfn=[gx, 0, z_load])
        if physics:
            scr["ghost_physics"] = [px, 0, z_load]
        return boxes, dict(_screen=scr, ghost_tabpfn_x_m=round(gx, 4), physics_shown=int(physics), ghost_physics_x_m=round(px, 4),
                           band_lo_x_m=round(xf + row["p05"], 4), band_hi_x_m=round(xf + row["p95"], 4), forecast_row_t=row["t"],
                           forecast_p05_m=round(row["p05"], 4), forecast_p50_m=round(row["p50"], 4), forecast_p95_m=round(row["p95"], 4),
                           physics_forecast_m=round(row["phys"], 4), actual_offset_1s_m=round(row["y"], 4), own_rows=row["own_rows"])
    cam_c, track_c = close(md.DEMO, fr2)
    for name, cam, phys in (("C02_forecast_side_60", "side", False), ("C02_forecast_close_60", cam_c, False),
                            ("C02_vsphysics_close_60", cam_c, True)):
        jobs.append(dict(name=name, cond=md.DEMO, frames=fr2, camera=cam, track=track_c, t0=t0c,
                         controller="TabPFN forecast 1 s ahead" + (" vs physics (0.5 m sling typed in)" if phys else ""),
                         overlay=lambda t, d, phys=phys: ov_forecast(t, d, phys)))

    # C03 the closed loop: the same new load under industry anti-sway (tuned for the typed-in 0.5 m sling) and under
    # TabPFN's choice; identical cameras, both start their move 1 s into the clip. The move shown: after TabPFN's first
    # 4 moves on new loads, the same trolley speed in both lanes (only the anti-sway setting differs), and the largest
    # difference in settle time
    same = [i for i in range(4, len(cd["jobs"])) if cd["industry"][i]["frac"] == cd["tabpfn"][i]["frac"]]
    k = max(same, key=lambda i: cd["industry"][i]["settle_s"] - cd["tabpfn"][i]["settle_s"])
    c = cd["jobs"][k]["c"]
    # both lanes run to the same length: the later arrival + 15 s of settling (the industry lane's full wait is in
    # the sidecar; the 15 s cap keeps the clip usable), so they cut side by side frame for frame
    t_end = 1.0 + max(cd[ln][k]["move_s"] for ln in ("industry", "tabpfn")) + 15.0
    for lane in ("industry", "tabpfn"):
        rec = cd[lane][k]
        tune_L = c["L1"] + cl.HOOK_DROP + rec["tune"] + cl.LOAD_HALF
        _, Tm = cl.shaped_state(0.0, rec["frac"], tune_L)

        def ref(t, rec=rec, tune_L=tune_L, Tm=Tm):
            s = t - 1.0
            if s < 0:
                return mc.BAYS[0], 0.0, 0.0
            if s < Tm:
                x, v, a = cl.shaped_state(s, rec["frac"], tune_L)[0]
                return mc.BAYS[0] + x, v, a
            return mc.BAYS[0] + cl.D, 0.0, 0.0
        _, frc = sim_frames(c, ref, t_end=t_end)
        info = dict(move=k + 1, sling_m=round(c["L2"], 2), speed_frac=rec["frac"], tune_sling_m=rec["tune"],
                    move_s=round(rec["move_s"], 2), settle_s=round(rec["settle_s"], 2), amp_m=round(rec["amp"], 3))
        if lane == "tabpfn":
            info.update(pred_amp50_m=round(rec["pred_amp50"], 3), pred_amp99_m=round(rec["pred_amp99"], 3), own_rows=rec["own_rows"])

        def ov_closed(t, d, info=info, Tm=Tm, settle=rec["settle_s"]):
            s = t - 1.0
            phase = "before" if s < 0 else ("moving" if s < Tm else ("settling" if s < Tm + settle else "settled"))
            return [], dict(info, phase=phase)
        if lane == "industry":
            cam3, track3 = close(c, frc)           # both lanes use the industry lane's camera path: identical cameras
        for cam_name, cam in (("side", "side"), ("close", cam3)):
            jobs.append(dict(name=f"C03_closedloop_{cam_name}_{lane}_60", cond=c, frames=frc, camera=cam, track=track3, t0=0.0,
                             controller="industry anti-sway (set for 0.5 m sling)" if lane == "industry" else "TabPFN chooses speed + anti-sway",
                             overlay=ov_closed))
    for j in jobs:
        j["path"] = str(out / f"{j['name']}.mp4")
        j["mode"] = mode
    global _JOBS
    _JOBS = jobs                       # forked workers inherit the jobs (closures cannot be pickled); they get indices
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_render_index, range(len(jobs))))
    if mode != "video":
        return
    man = []
    for j, rr in zip(jobs, res):
        man.append(dict(file=rr["file"], project="crane", frames=rr["frames"], video_fps=FPS, sim_fps=FPS, resolution=f"{W}x{H}",
                        duration_s=round(rr["frames"] / FPS, 2), camera="side (the crane's fixed side camera)" if j["camera"] == "side" else "close (free camera, tracks the trolley, smoothed 0.4 s)",
                        controller=j["controller"], source="crane/ (unmodified) via showcase/launch_crane.py; data results/crane_launch/",
                        sim_commit=os.environ.get("SIM_COMMIT", "unknown")))
        print(f"{rr['file']}: {rr['frames']} frames ({rr['frames'] / FPS:.1f} s)", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(
        notes="Crane project shots (MuJoCo overhead crane). No text. Ghost boxes: translucent teal = TabPFN's predicted "
              "load position 1 s ahead (median), teal floor band = its 90 % range, translucent orange = the physics model "
              "(pendulum with the typed-in 0.5 m sling). C03: same load, industry anti-sway vs TabPFN's choice; both moves "
              "start 1 s into the clip. Sidecar values come from the runs in results/crane_launch/.",
        closed_loop_move=k + 1, forecast_summary=fc["summary"], closed_summary=cd["summary"], clips=man), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["forecast", "closed", "render"])
    ap.add_argument("--out", default=str(Path.home() / "delta_line" / "footage_crane"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--mode", choices=["video", "preview", "sidecars"], default="video",
                    help="render: the videos + sidecars, 3 PNG stills per shot, or only the sidecars")
    a = ap.parse_args()
    {"forecast": stage_forecast, "closed": stage_closed}.get(a.stage, lambda: stage_render(Path(a.out), a.workers, a.mode))()
