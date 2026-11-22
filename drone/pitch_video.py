"""Clean slow-motion videos for the pitch: the real path against one or more models' predicted paths.

    python drone/pitch_video.py --pilot b --flight 78 --model combo --out drone/results/videos/combo.mp4
    python drone/pitch_video.py --pilot b --flight 78 --model imm,tabpfn,combo --out drone/results/videos/compare.mp4 \
        --gif drone/results/compare.gif     # also a short GIF of the busiest part, for the README
    drone/make_videos.sh              # every video in drone/results/videos/, as listed in drone/results/README.md

Models: imm, imm_retuned, tabpfn_a, tabpfn (= TabPFN B), combo (IMM + TabPFN), live (the real-time replay).
combo and live exist only for pilot b. Needs drone/data/ with the saved predictions (drone/README.md, "Reproduce every result").

Renders offscreen (MUJOCO_GL=egl), so it runs on a server GPU. The flight is flown first exactly as recorded,
then played back with the drone drawn SCALE times bigger (display only). On screen:
  green line  = where the drone really went
  blue line   = where the model said it would be, 1 s earlier (so both lines cover the same moments)
  blue ball   = the model's guess for right now, made 1 s ago; the white line to the drone is its error
  light ball  = where the model says the drone will be 1 s from now
No radar dots, no close-up window; the camera flies beside the drone. The box at the top shows the numbers
(real x y z, the model's guess, its error, the alarm) and the colour key.
With several models (comma-separated) each gets its own colour (MULTI): its path, its guess for now and a thin
line to the drone; the 1 s-ahead balls are left out so the picture stays readable.
"""

import argparse

import mujoco
import numpy as np

from compare import imm_breach_probability, realtime_predictions
from demo_imm import add, fly, fmt, line
from features import DATA
from intruder import BEHAVIOURS, PILOT_SEED, SETUP
from sim import POLE_GROUP, build

SCALE = 4.0  # the 10 cm Crazyflie drawn 40 cm wide
SLOW = 0.5  # playback speed: half speed
FPS = 30
LAG = 10  # radar steps in 1 s
GREEN, BLUE, LIGHT, WHITE = [0.1, 0.85, 0.25, 1], [0.15, 0.45, 1.0, 1], [0.55, 0.75, 1.0, 0.55], [1, 1, 1, 1]
NAMES = {"imm": "IMM (radar standard, tuned on the big drone)",
         "imm_retuned": "IMM re-tuned on the small drone",
         "tabpfn_a": "TabPFN A (big-drone examples only)",
         "tabpfn": "TabPFN B (no training, examples only)",
         "combo": "IMM + TabPFN (TabPFN corrects the IMM)",
         "live": "LIVE: IMM every 0.1 s + TabPFN correction every 0.3 s"}
MULTI = {"imm": [0.15, 0.45, 1.0, 1], "tabpfn": [0.95, 0.45, 0.1, 1], "combo": [0.75, 0.25, 0.9, 1],
         "imm_retuned": [0.93, 0.63, 0.0, 1], "tabpfn_a": [0.6, 0.6, 0.6, 1], "live": [0.95, 0.3, 0.6, 1]}
SHORT = {"imm": "IMM", "imm_retuned": "IMM re-tuned", "tabpfn_a": "TabPFN A", "tabpfn": "TabPFN B",
         "combo": "IMM+TabPFN", "live": "Live"}


def model_track(model, pilot, seed, n):
    """The model's saved 1 s predictions (world x, y, z) and alarm (P breach within 2 s) for this flight, at every
    radar reading; NaN where it has none. Read from the same files the results were scored on."""
    ds = "cf2" if pilot == "a" else "cf2_pilotb"
    table = np.load(DATA / f"table_{ds}.npz")
    world, alarm = np.full((n, 3), np.nan), np.full(n, np.nan)

    def put(out, rows, values):
        mine = table["flight"][rows] == seed
        k = np.rint(table["t"][rows[mine]] / 0.1).astype(int)
        out[k[k < n]] = values[mine][k < n]

    if model in ("imm", "imm_retuned"):
        imm = np.load(DATA / (f"imm_{ds}.npz" if model == "imm" else f"imm_{ds}_cf2tuned.npz"))
        rows = np.where(table["flight"] == seed)[0]
        mean, cov = imm["mean"][rows].astype(float), imm["cov"][rows].astype(float)
        ok = ~np.isnan(mean).any((1, 2))
        hi = int(np.argmin(np.abs(imm["horizons"] - 1.0)))
        p = np.full(len(rows), np.nan)
        p[ok] = imm_breach_probability(mean[ok], cov[ok], np.random.default_rng(0))
        put(world, rows, mean[:, hi])
        put(alarm, rows, p)
        return world, alarm
    if model == "live":  # the real-time replay of the fast IMM + TabPFN (compare.realtime_predictions)
        if pilot != "b":
            raise SystemExit("the live setup was only run on pilot b (--pilot b)")
        full = dict(table)
        imm = np.load(DATA / "imm_cf2_pilotb.npz")
        hi = int(np.argmin(np.abs(imm["horizons"] - 1.0)))
        fast = np.load(DATA / "tabpfn_cf2_pilotb_combo_fast.npz")
        imm_w = imm["mean"][:, hi].astype(float)
        fast_w = np.full_like(imm_w, np.nan)
        fast_w[fast["test_rows"]] = fast["B_x2_plus_cf2_world"][:, 0]
        p_all = np.full(len(imm_w), np.nan)
        p_all[fast["alarm_rows"]] = fast["B_x2_plus_cf2_p_breach"]
        rt, used, src = realtime_predictions(full, imm_w, fast_w, every=3)
        rows = np.arange(len(imm_w))
        put(world, rows, rt)
        put(alarm, rows, np.where(used, p_all[src], np.nan))  # the alarm arrives with the same delay
        return world, alarm
    file = {"a": "tabpfn_cf2.npz", "b": "tabpfn_cf2_pilotb.npz"}[pilot] if model != "combo" else "tabpfn_cf2_pilotb_combo.npz"
    if model == "combo" and pilot != "b":
        raise SystemExit("IMM + TabPFN was only run on pilot b (--pilot b)")
    tab = np.load(DATA / file)
    key = "A_x2_only" if model == "tabpfn_a" else "B_x2_plus_cf2"
    put(world, tab["test_rows"], tab[f"{key}_world"][:, 0])
    put(alarm, tab["alarm_rows"], tab[f"{key}_p_breach"])
    return world, alarm


def polyline(scn, points, width, rgba, step=2):
    """Connect consecutive points; a gap (NaN) breaks the line."""
    pts = points[::step]
    for a, b in zip(pts[:-1], pts[1:]):
        if not (np.isnan(a).any() or np.isnan(b).any()):
            line(scn, a, b, width, rgba)


def set_camera(cam, pos, heading):
    """Beside and above the drone, so the paths behind it run across the picture."""
    f = np.array([np.cos(heading), np.sin(heading), 0.0])
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.elevation = pos - f * 2.5, 14.0, -24
    cam.azimuth = np.degrees(heading) + 90


def to_gif(video, gif, start, seconds, width=800, fps=10):
    """A short looping GIF cut from a video (ffmpeg from imageio-ffmpeg; two passes for a good palette)."""
    import subprocess

    import imageio_ffmpeg

    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds),
                    "-i", video, "-vf", f"{scale},split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    "-loop", "0", gif], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", choices=["a", "b"], default="a", help="b: the pilot no model has seen")
    parser.add_argument("--flight", type=int, default=54, help="flight number (pilot a: test flights 50-299; pilot b: 0-249)")
    parser.add_argument("--model", default="tabpfn", help=f"one of {', '.join(NAMES)}, or several separated by commas")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gif", help="also write a GIF of part of the video here")
    parser.add_argument("--gif-start", type=float, default=10.0, help="GIF start, seconds into the video")
    parser.add_argument("--gif-seconds", type=float, default=14.0)
    args = parser.parse_args()
    models = args.model.split(",")
    for m in models:
        if m not in NAMES:
            parser.error(f"unknown model {m}; choose from {', '.join(NAMES)}")

    import imageio_ffmpeg
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    seed = SETUP["cf2"][2] + PILOT_SEED[args.pilot] + args.flight
    flight, frames = fly(seed, args.pilot)
    log = {k: np.asarray(v) for k, v in flight.log.items()}
    true, n = log["true"], len(log["t"])
    tracks = {m: model_track(m, args.pilot, seed, n) for m in models}
    single = len(models) == 1
    colour = {m: BLUE for m in models} if single else {m: MULTI[m] for m in models}
    model = build(["cf2"], zone=True, timestep=0.002, scale=SCALE)  # same joints: play the flown qpos back
    data = mujoco.MjData(model)
    body = model.body("cf2_cf2").id

    size = (1280, 720)
    renderer = mujoco.Renderer(model, size[1], size[0])
    options = mujoco.MjvOption()
    options.geomgroup[POLE_GROUP] = 1
    cam = mujoco.MjvCamera()
    mono = font_manager.findfont("DejaVu Sans Mono")
    font, small = ImageFont.truetype(mono, 20), ImageFont.truetype(mono, 16)
    ink = {"fill": (255, 255, 255), "stroke_width": 2, "stroke_fill": (0, 0, 0)}  # no box: reads on grass and sky
    width = max(len(SHORT[m]) for m in models)
    writer = imageio_ffmpeg.write_frames(args.out, size, fps=FPS, quality=8, macro_block_size=8)
    writer.send(None)

    def swatch(d, x, y, rgba):
        d.rectangle((x, y + 3, x + 17, y + 15), fill=tuple(int(255 * c) for c in rgba[:3]), outline=(0, 0, 0), width=2)

    times = np.array([f[0] for f in frames])
    heading, errors, k_seen = None, {m: [] for m in models}, -1
    for t in np.arange(0, times[-1] + 2 * SLOW, SLOW / FPS):  # hold the end about 2 s of video
        _, qpos, k = frames[min(int(np.searchsorted(times, t)), len(frames) - 1)]
        k = min(k, n - 1)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pos = data.xpos[body].copy()
        v = log["vel"][k][:2]
        target = np.arctan2(v[1], v[0]) if np.hypot(*v) > 0.3 else heading
        if heading is None:
            heading = target if target is not None else 0.0
        elif target is not None:  # turn the camera smoothly
            heading += 0.08 * np.angle(np.exp(1j * (target - heading)))
        set_camera(cam, pos, heading)
        renderer.update_scene(data, cam, scene_option=options)
        scn = renderer.scene
        polyline(scn, np.vstack([true[: k + 1], pos]), 0.03, GREEN)

        lines = []  # (text, colour square or None)
        for m in models:
            pred, alarm = tracks[m]
            name = SHORT[m]
            if k >= LAG:
                polyline(scn, pred[: k - LAG + 1], 0.03 if single else 0.025, colour[m])
            said = pred[k - LAG] if k >= LAG else np.full(3, np.nan)
            err = None
            if not np.isnan(said).any():
                err = float(np.linalg.norm(said - true[k]))
                if k != k_seen:
                    errors[m].append(err)
                add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.12] * 3, said, colour[m])
                line(scn, said, pos, 0.015 if single else 0.01, WHITE if single else colour[m])
            alarm_text = (f"{'ALARM' if alarm[k] > 0.5 else 'alarm'}: breach within 2 s, {alarm[k]:.0%}"
                          if not np.isnan(alarm[k]) else None)
            if single:
                label = f"{name} said 1 s ago"
                lines.append((f"{'REAL now':{len(label)}s} x y z: {fmt(true[k])}", None))
                if err is not None:
                    lines.append((f"{label} x y z: {fmt(said)}", None))
                if not np.isnan(pred[k]).any():
                    add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.12] * 3, pred[k], LIGHT)
                    lines.append((f"{f'{name} says in 1 s':{len(label)}s} x y z: {fmt(pred[k])}", None))
                if err is not None:
                    lines.append((f"{name} off by: {err:4.2f} m   (flight median so far {np.median(errors[m]):4.2f} m)", None))
                if alarm_text:
                    lines.append((alarm_text, None))
            else:
                row = f"{name:{width}s} said 1 s ago: "
                row += (f"{fmt(said)}   off by {err:4.2f} m (median {np.median(errors[m]):4.2f})" if err is not None
                        else "  (no guess yet)")
                if alarm_text:
                    row += f"   {'ALARM' if alarm[k] > 0.5 else 'alarm'} {alarm[k]:3.0%}"
                lines.append((row, colour[m]))
        if not single:
            lines.insert(0, (f"{'REAL':{width}s} now:          {fmt(true[k])}", GREEN))
        k_seen = k

        img = Image.fromarray(renderer.render())
        d = ImageDraw.Draw(img)
        title = NAMES[models[0]] if single else "Best models side by side"
        head = [f"{title}   (0.5x slow motion, pilot {args.pilot} flight {args.flight})",
                f"t = {log['t'][k]:5.1f} s   doing: {BEHAVIOURS[log['behaviour'][k]]}"]
        y = 10
        for s in head:
            d.text((14, y), s, font=font, **ink)
            y += 26
        for s, sq in lines:
            if sq is not None:
                swatch(d, 14, y + 1, sq)
            d.text((14 if sq is None else 38, y), s, font=font, **ink)
            y += 26
        y += 6
        if single:
            name = SHORT[models[0]]
            key = [("green line", "real path", GREEN), ("blue line", f"{name}'s predicted path", BLUE),
                   ("blue ball", f"{name}'s guess for now, made 1 s ago (white line = its error)", BLUE),
                   ("light ball", f"{name}'s guess for 1 s ahead", LIGHT)]
            for a, b, rgba in key:
                swatch(d, 14, y, rgba)
                d.text((38, y), f"{a}: {b}", font=small, **ink)
                y += 21
        else:
            d.text((14, y), "lines: each model's predicted path (green = real)   balls: each model's guess for now, "
                            "made 1 s ago", font=small, **ink)
            d.text((14, y + 21), "alarm: how sure each model is that the drone breaches within 2 s", font=small, **ink)
        writer.send(np.asarray(img).tobytes())
    writer.close()
    print(", ".join(f"{SHORT[m]} median error {np.median(errors[m]):.2f} m" for m in models),
          f"(pilot {args.pilot} flight {args.flight}) -> {args.out}")
    if args.gif:
        to_gif(args.out, args.gif, args.gif_start, args.gif_seconds)
        print(f"-> {args.gif}")


if __name__ == "__main__":
    main()
