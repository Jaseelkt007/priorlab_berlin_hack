"""Live 3D demo: a Crazyflie test flight with the IMM's predictions, actual vs predicted, in real time.

    python drone/demo_imm.py                 # test flight 54 (an attack that breaches the zone)
    python drone/demo_imm.py --flight 61 --speed 0.5
    python drone/demo_imm.py --view radar    # from the protected site, looking at the incoming drone
    python drone/demo_imm.py --view radar --record drone/data/demo_radar_54.mp4   # save a video instead of showing it
    python drone/demo_imm.py --view radar --tabpfn     # also TabPFN B (pink), from drone/data/tabpfn_cf2.npz

What you see:
  the drone + green trail          the real Crazyflie model (MuJoCo physics, intruder pilot) and where it flew
  white dots                       the radar readings (noisy, some missing): all the IMM gets
  blue ball + blue bubble          IMM: where it will be in 1 s; the bubble = its 90% range
  yellow ball + yellow line        what the IMM predicted 1 s ago for *now*; the line to the drone is the error
Top left: real x, y, z vs predicted x, y, z, how far off, and the alarm ("enters the zone within 2 s").

The flight is computed first, then played back. The IMM runs only on the radar readings, one at a time,\nso at every moment it knows only the past.
Flight numbers 50-299 are Crazyflie test flights (0-49 are kept for adapting TabPFN).
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from compare import imm_breach_probability
from features import DATA
from imm import HORIZONS, TUNED, run
from intruder import BEHAVIOURS, SETUP, Flight
from sim import POLE_GROUP, RADAR_TOP

AHEAD = 1.0  # s: the prediction drawn (blue)
CHECK = 1.0  # yellow: the prediction made this long ago for now
TRAIL = 300  # green trail length (radar steps)
RADAR_DOTS = 30  # radar dots shown (last 3 s)
CONE = 2.5  # 90% of a 3D normal lies within 2.5 standard deviations
FRAME = 0.02  # s between played-back frames (50 per second)
CHASE_CAM = 5.0  # m: camera distance in the chase view (close enough to see the 10 cm Crazyflie itself)
INSET = (400, 300)  # px: close-up of the drone in recorded videos
SITE_CAMERA = np.array(RADAR_TOP)  # m: the radar view's camera, on top of the radar on the building
SITE_SPAN = 4.0  # m of the scene shown around the drone in the radar view (the camera zooms)

PINK = [0.91, 0.48, 0.64, 1.0]  # TabPFN
GREEN, WHITE, BLUE, BUBBLE, YELLOW = (
    [0.1, 0.9, 0.2, 0.9], [1, 1, 1, 0.9], [0.2, 0.5, 1.0, 0.9], [0.3, 0.6, 1.0, 0.12], [1.0, 0.85, 0.1, 1.0])
EYE = np.eye(3).ravel()


def tabpfn_track(seed, n):
    """TabPFN B's 1 s predictions (world x, y, z) for each radar reading of this flight; NaN where none."""
    table = np.load(DATA / "table_cf2.npz")
    tab = np.load(DATA / "tabpfn_cf2.npz")
    rows = tab["test_rows"]
    mine = table["flight"][rows] == seed
    if not mine.any():
        raise SystemExit("no TabPFN predictions for this flight (run tabpfn_model.py with --every 1)")
    world = np.full((n, 3), np.nan)
    k = np.rint(table["t"][rows[mine]] / 0.1).astype(int)
    keep = k < n
    world[k[keep]] = tab["B_x2_plus_cf2_world"][mine][keep, 0]
    return world


def flight_radar(seed):
    """The recorded radar readings of this flight (the same seed replays the same flight)."""
    raw = np.load(DATA / "intruder_cf2.npz")
    return raw["meas"][raw["flight"] == seed]


def add(scn, kind, size, pos, rgba, mat=EYE):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def line(scn, a, b, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), EYE, np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def bubble(scn, mean, cov, rgba):
    """The 90% range as an ellipsoid: axes along the covariance's eigenvectors."""
    vals, vecs = np.linalg.eigh(cov)
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] *= -1
    add(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, CONE * np.sqrt(np.maximum(vals, 1e-6)), mean, rgba, vecs.ravel())


def fmt(p):
    return f"{p[0]:6.1f} {p[1]:6.1f} {p[2]:5.1f}"


def draw(scn, k, log, meas, pred, errors, start=0, drone=None, tab=None):
    """Draw radar step k into the scene; return the text lines. pred = (mean, cov, ok, p_alarm, hi, lag).
    drone: where the drone is drawn right now (frames come faster than radar readings); lines start there.
    tab: optional (TabPFN's 1 s predictions per radar reading (K, 3), NaN where none; its error list)."""
    mean, cov, ok, p_alarm, hi, lag = pred
    true = np.array(log["true"])
    now = true[k]
    anchor = now if drone is None else drone
    scn.ngeom = start  # the viewer's user scene starts empty; an offscreen render keeps the model's geoms
    for p in true[max(0, k - TRAIL) : k : 2]:
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025] * 3, p, GREEN)
    for p in meas[max(0, k - RADAR_DOTS) : k + 1]:
        if not np.isnan(p).any():
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.04] * 3, p, WHITE)
    if ok[k]:
        ahead = mean[k, hi[AHEAD]]
        line(scn, anchor, ahead, 0.012, BLUE)
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.08] * 3, ahead, BLUE)
        bubble(scn, ahead, cov[k, hi[AHEAD]], BUBBLE)
    texts = [f"t = {log['t'][k]:5.1f} s   doing: {BEHAVIOURS[log['behaviour'][k]]}",
             f"REAL now            x y z: {fmt(now)}"]
    if k >= lag and ok[k - lag]:
        said = mean[k - lag, hi[CHECK]]
        err = float(np.linalg.norm(said - now))
        errors.append(err)
        add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.08] * 3, said, YELLOW)
        line(scn, said, anchor, 0.015, YELLOW)
        texts += [f"IMM said 1 s ago    x y z: {fmt(said)}",
                  f"IMM off by: {err:4.2f} m   (flight median so far {np.median(errors):4.2f} m)"]
    if ok[k]:
        texts.append(f"IMM says in 1 s     x y z: {fmt(mean[k, hi[AHEAD]])}")
    if tab is not None:
        world, tab_errors = tab
        if not np.isnan(world[k]).any():
            line(scn, anchor, world[k], 0.012, PINK)
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.08] * 3, world[k], PINK)
        if k >= lag and not np.isnan(world[k - lag]).any():
            said = world[k - lag]
            err = float(np.linalg.norm(said - now))
            tab_errors.append(err)
            add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.07] * 3, said, PINK)
            line(scn, said, anchor, 0.015, PINK)
            texts += [f"TabPFN said 1 s ago x y z: {fmt(said)}",
                      f"TabPFN off by: {err:4.2f} m   (flight median so far {np.median(tab_errors):4.2f} m)"]
        if not np.isnan(world[k]).any():
            texts.append(f"TabPFN says in 1 s  x y z: {fmt(world[k])}")
    alarm = p_alarm[k]
    texts.append(f"ALARM: breach within 2 s, {alarm:.0%}" if alarm > 0.5 else f"alarm: breach within 2 s, {alarm:.0%}")
    return texts


def predictions(meas):
    """Run the IMM over the whole radar track (causal: step k uses readings 0..k only)."""
    mean, cov, _ = run(meas[None], np.array([len(meas)]), TUNED)
    mean, cov = mean[0].astype(float), cov[0].astype(float)
    hi = {h: int(np.argmin(np.abs(HORIZONS - h))) for h in (AHEAD, CHECK)}
    ok = ~np.isnan(mean).any((1, 2))
    p_alarm = np.zeros(len(meas))
    p_alarm[ok] = imm_breach_probability(mean[ok], cov[ok], np.random.default_rng(0))
    return mean, cov, ok, p_alarm, hi, round(CHECK / 0.1)


def radar_view(cam, model, drone):
    """A camera fixed on the radar post at the zone centre, turning and zooming to follow the drone,
    so the drone and the 1 s prediction around it fill about SITE_SPAN metres of the picture."""
    d = drone - SITE_CAMERA
    dist = np.linalg.norm(d)
    f = d / dist
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = drone
    cam.distance = dist
    cam.azimuth = np.degrees(np.arctan2(f[1], f[0]))
    cam.elevation = np.degrees(np.arcsin(f[2]))
    model.vis.global_.fovy = np.clip(np.degrees(2 * np.arctan(SITE_SPAN / 2 / dist)), 2.0, 45.0)


def set_camera(cam, view, flight):
    if view == "chase":
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = flight.pilot.body
        cam.distance, cam.elevation, cam.azimuth = CHASE_CAM, -20, 135
    else:
        radar_view(cam, flight.model, flight.data.xpos[flight.pilot.body])


def legend(with_tabpfn):
    if with_tabpfn:
        return ("white: radar   blue ball / yellow ball: IMM in 1 s / its guess for now   "
                "pink ball / pink cube: TabPFN in 1 s / its guess for now")
    return ("drone + green trail: real   white: radar   blue: IMM says in 1 s + 90% range   "
            "yellow: IMM's guess for now, made 1 s ago")


def record(flight, frames, log, meas, pred, view, out, fps=30, size=(1280, 720), tab_world=None):
    """Render the same scene offscreen and save it as an MP4 video (needs imageio-ffmpeg)."""
    import imageio_ffmpeg
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    mono = font_manager.findfont("DejaVu Sans Mono")
    font, small = ImageFont.truetype(mono, 20), ImageFont.truetype(mono, 15)
    renderer = mujoco.Renderer(flight.model, size[1], size[0])
    inset = mujoco.Renderer(flight.model, INSET[1], INSET[0])  # close-up of the real drone, no markers
    close = mujoco.MjvCamera()
    close.type, close.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, flight.pilot.body
    close.distance, close.elevation, close.azimuth = 0.3, -20, 135
    cam, errors, tab_errors, k_drawn = mujoco.MjvCamera(), [], [], -1
    options = mujoco.MjvOption()
    options.geomgroup[POLE_GROUP] = view != "radar"
    times = np.array([f[0] for f in frames])
    n = min(len(meas), len(log["t"]))
    writer = imageio_ffmpeg.write_frames(out, size, fps=fps, quality=8, macro_block_size=8)
    writer.send(None)
    count = 0
    for t in np.arange(0, times[-1] + 2.0, 1 / fps):  # hold the last frame 2 s
        _, qpos, k = frames[min(int(np.searchsorted(times, t)), len(frames) - 1)]
        flight.data.qpos[:] = qpos
        mujoco.mj_forward(flight.model, flight.data)
        set_camera(cam, view, flight)
        renderer.update_scene(flight.data, cam, scene_option=options)
        k = min(k, n - 1)
        # draw() appends to `errors` once per radar step, as the live viewer does
        new = k != k_drawn
        tab = None if tab_world is None else (tab_world, tab_errors if new else list(tab_errors))
        texts = draw(renderer.scene, k, log, meas, pred, errors if new else list(errors),
                     start=renderer.scene.ngeom, drone=flight.data.xpos[flight.pilot.body].copy(), tab=tab)
        k_drawn = k
        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        if view == "chase":  # the chase camera is too far to see the 10 cm drone: add a close-up
            inset.update_scene(flight.data, close)
            x0, y0 = size[0] - INSET[0] - 14, size[1] - INSET[1] - 44
            img.paste(Image.fromarray(inset.render()), (x0, y0))
            d.rectangle([x0 - 2, y0 - 2, x0 + INSET[0] + 1, y0 + INSET[1] + 1], outline=(255, 255, 255), width=2)
            d.text((x0 + 6, y0 + 4), "close-up: the real Crazyflie (10 cm)", font=small, fill=(255, 255, 255))
        box = d.multiline_textbbox((14, 14), "\n".join(texts), font=font)
        d.rectangle([box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8], fill=(0, 0, 0))
        d.multiline_text((14, 14), "\n".join(texts), font=font,
                         fill=(255, 80, 80) if texts[-1].startswith("ALARM") else (255, 255, 255))
        d.text((14, size[1] - 28), legend(tab_world is not None), font=small, fill=(255, 255, 255))
        writer.send(np.ascontiguousarray(np.asarray(img)))
        count += 1
    writer.close()
    print(f"{count} frames ({count / fps:.0f} s) -> {out}")


def fly(seed, pilot="a"):
    """Fly the whole flight first (exactly as when the dataset was made), keeping frames to play back.
    Playing back, rather than simulating inside the viewer, keeps the viewer from touching the physics."""
    flight = Flight(seed, "cf2", pilot=pilot)
    frames, next_frame = [], 0.0
    while flight.step():
        if flight.data.time >= next_frame:
            frames.append((flight.data.time, flight.data.qpos.copy(), len(flight.log["t"]) - 1))
            next_frame += FRAME
    frames.append((flight.data.time, flight.data.qpos.copy(), len(flight.log["t"]) - 1))
    return flight, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight", type=int, default=54, help="Crazyflie flight number (test flights: 50-299)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = half speed)")
    parser.add_argument("--view", choices=["chase", "radar"], default="chase",
                        help="chase: follow the drone; radar: from the protected site toward the drone")
    parser.add_argument("--record", help="save an MP4 video to this path instead of opening the viewer")
    parser.add_argument("--tabpfn", action="store_true", help="also show TabPFN B's predictions (pink)")
    args = parser.parse_args()

    seed = SETUP["cf2"][2] + args.flight
    meas = flight_radar(seed)
    if len(meas) == 0:
        raise SystemExit(f"flight {args.flight} not in drone/data/intruder_cf2.npz (run intruder.py --drone cf2 first)")
    pred = predictions(meas)
    tab_world = tabpfn_track(seed, len(meas)) if args.tabpfn else None

    flight, frames = fly(seed)
    log = {k: np.asarray(v) for k, v in flight.log.items()}
    n = min(len(meas), len(log["t"]))
    if not np.allclose(log["meas"][:n], meas[:n], equal_nan=True):
        print("WARNING: this flight differs from the recorded one; predictions may not match")
    print(f"Crazyflie test flight {args.flight}: intent {flight.policy.intent}, outcome {flight.outcome} at "
          f"{flight.data.time:.1f} s.")
    if args.record:
        record(flight, frames, log, meas, pred, args.view, args.record, tab_world=tab_world)
        return
    errors, tab_errors = [], []
    with mujoco.viewer.launch_passive(flight.model, flight.data, show_left_ui=False, show_right_ui=False) as viewer:
        set_camera(viewer.cam, args.view, flight)
        viewer.opt.geomgroup[POLE_GROUP] = args.view != "radar"
        times = np.array([f[0] for f in frames])
        start, k_drawn, replay = time.time(), -1, 1
        while viewer.is_running():
            if (time.time() - start) * args.speed > times[-1] + 2.0:  # loop: restart 2 s after the end
                start, k_drawn, errors, tab_errors, replay = time.time(), -1, [], [], replay + 1
                print(f"replay {replay}", flush=True)
            # pick the frame by the clock, so a slow window skips frames instead of falling behind
            i = min(int(np.searchsorted(times, (time.time() - start) * args.speed)), len(frames) - 1)
            t, qpos, k = frames[i]
            with viewer.lock():
                flight.data.qpos[:] = qpos
                mujoco.mj_forward(flight.model, flight.data)
                if args.view == "radar":
                    set_camera(viewer.cam, "radar", flight)
                texts = None
                if k < n:
                    new = k != k_drawn
                    tab = None if tab_world is None else (tab_world, tab_errors if new else list(tab_errors))
                    texts = draw(viewer.user_scn, k, log, meas, pred, errors if new else list(errors),
                                 drone=flight.data.xpos[flight.pilot.body].copy(), tab=tab)
                    texts[0] += f"   (replay {replay}, loops until you close the window)"
                    k_drawn = k
            if texts:  # set_texts outside the lock: inside it, it waits for the lock forever (deadlock)
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   "\n".join(texts), None),
                                  (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, legend(tab_world is not None), None)])
            viewer.sync()
            if i == len(frames) - 1 and errors:
                print(f"{flight.outcome} at {t:.1f} s; IMM 1 s ahead was off by {np.median(errors):.2f} m "
                      f"(median over this flight); replaying in 2 s.", flush=True)
                errors = []
            time.sleep(FRAME / 2)


if __name__ == "__main__":
    main()
