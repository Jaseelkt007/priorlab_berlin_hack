"""Random intruder flights toward the protected zone, recorded like a radar would see them.

    python drone/intruder.py --view 7                          # watch X2 flight 7 live in the viewer
    python drone/intruder.py --flights 2000                    # X2 training set -> data/intruder_x2.npz
    python drone/intruder.py --drone cf2 --view 7              # watch a Crazyflie intruder
    python drone/intruder.py --drone cf2 --flights 300         # Crazyflie test set -> data/intruder_cf2.npz

Every flight is a different random intruder (the "policy"):
- an intent: attack (heads into the zone), flyby (passes near it) or loiter (circles around it),
- a cruise height, a speed level (aggression),
- and every 1-6 s a new manoeuvre: straight, zigzag, orbit, hover, dash, dive, climb,
- plus wind: a steady push and random gusts.
The radar at the zone centre samples the position 10 times a second, with noise that grows with
range and with dropouts (NaN). The true state is stored next to it so we can score predictions.
"""

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

from flight import GAINS
from sim import ARENA_RADIUS, POLE_GROUP, ZONE_HEIGHT, ZONE_RADIUS, build, controllers

# drone -> (physics timestep in s, controller gains, first seed). The Crazyflie needs the small step.
# Crazyflie seeds start at 100000 so its test flights never reuse an X2 training flight's random plan.
SETUP = {"x2": (0.005, "x2_intruder", 0), "cf2": (0.002, "cf2_intruder", 100_000)}
X2_MASS = 1.325  # kg; wind is scaled by drone mass so every drone gets the same push per kg
POLICY_EVERY = 0.02  # s, the intruder decides 50 times a second
RADAR_EVERY = 0.1  # s, 10 Hz radar
MAX_TIME = 60.0  # s

BEHAVIOURS = ["straight", "zigzag", "orbit", "hover", "dash", "dive", "climb"]
INTENTS = ["attack", "flyby", "loiter"]
OUTCOMES = ["breach", "left", "timeout", "crash"]
# How often each intent picks each behaviour (same order as BEHAVIOURS).
WEIGHTS = {
    "attack": [0.30, 0.25, 0.05, 0.10, 0.15, 0.10, 0.05],
    "flyby": [0.40, 0.20, 0.10, 0.05, 0.05, 0.10, 0.10],
    "loiter": [0.20, 0.15, 0.30, 0.25, 0.00, 0.05, 0.05],
}
INTENT_P = [0.5, 0.3, 0.2]
# Pilot styles. "a" made every flight so far (examples and tests). "b" is a different, more evasive pilot that no
# model has seen: other manoeuvre mix, switches about twice as often, sharper zigzags, bigger height changes.
# It tests whether a model learned drones or only pilot a's habits (EXPERIMENTS.md experiment 14).
PILOTS = {
    "a": dict(weights=WEIGHTS, segment=(2, 6), hover=(1, 3), zig_angle=(30, 70), zig_period=(0.8, 2.5), height=(5, 15)),
    "b": dict(weights={"attack": [0.10, 0.30, 0.05, 0.05, 0.20, 0.15, 0.15],
                       "flyby": [0.15, 0.30, 0.10, 0.05, 0.15, 0.10, 0.15],
                       "loiter": [0.10, 0.20, 0.30, 0.20, 0.05, 0.10, 0.05]},
              segment=(1, 3.5), hover=(0.5, 2), zig_angle=(40, 80), zig_period=(0.5, 1.5), height=(8, 20)),
}
PILOT_SEED = {"a": 0, "b": 100_000}  # added to the drone's first seed: pilot b's flights never reuse a seed

RADAR_NOISE = 0.3  # m, plus 1% of range, per axis
DROPOUT_P = 0.05  # single missing samples
BURST_P = 0.01  # chance per sample that a 3-10 sample gap starts


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


class IntruderPolicy:
    """Gives a world-frame velocity command from the drone's position. All randomness comes from rng."""

    def __init__(self, rng, max_speed, pilot="a"):
        self.rng = rng
        self.style = PILOTS[pilot]
        self.intent = INTENTS[rng.choice(3, p=INTENT_P)]
        self.max_speed = max_speed
        self.aggression = rng.uniform(0.4, 1.0)
        self.altitude = rng.uniform(5, 28)
        angle = rng.uniform(0, 2 * np.pi)
        self.start = np.array([ARENA_RADIUS * np.cos(angle), ARENA_RADIUS * np.sin(angle), self.altitude])
        if self.intent == "flyby":  # a straight line from the start to the goal passes 15-40 m from the centre
            side = rng.choice([-1, 1]) * rng.uniform(15, 40)
            normal = np.array([-np.sin(angle), np.cos(angle), 0])
            self.goal = normal * side * 2.5 - self.start * 1.5
        else:
            self.goal = np.zeros(3)
        self.loiter_radius = rng.uniform(18, 35)
        self.loiter_dir = rng.choice([-1, 1])
        self.segment_end = 0.0
        self.behaviour = "straight"

    def _new_segment(self, t):
        rng = self.rng
        st = self.style
        self.behaviour = BEHAVIOURS[rng.choice(len(BEHAVIOURS), p=st["weights"][self.intent])]
        self.segment_end = t + rng.uniform(*(st["hover"] if self.behaviour == "hover" else st["segment"]))
        self.speed = self.max_speed * self.aggression * rng.uniform(0.5, 1.0)
        self.zig_angle = np.radians(rng.uniform(*st["zig_angle"]))
        self.zig_period = rng.uniform(*st["zig_period"])
        self.orbit_rate = rng.choice([-1, 1]) * self.speed / rng.uniform(5, 20)  # rad/s
        self.orbit_phase = 0.0
        if self.behaviour == "dive":
            self.altitude = max(3.0, self.altitude - rng.uniform(*st["height"]))
        elif self.behaviour == "climb":
            self.altitude = min(40.0, self.altitude + rng.uniform(*st["height"]))

    def _base_direction(self, pos):
        """Horizontal direction the intent wants to go."""
        if self.intent == "loiter":
            radial = unit(pos[:2])
            tangent = self.loiter_dir * np.array([-radial[1], radial[0]])
            r = np.linalg.norm(pos[:2])
            return unit(tangent + radial * np.clip((self.loiter_radius - r) / 10, -1, 1))
        return unit((self.goal - pos)[:2])

    def velocity(self, t, pos, dt):
        if t >= self.segment_end:
            self._new_segment(t)
        base = self._base_direction(pos)
        b = self.behaviour
        if b == "hover":
            horizontal = np.zeros(2)
        elif b == "zigzag":
            a = self.zig_angle * np.sign(np.sin(2 * np.pi * t / self.zig_period))
            c, s = np.cos(a), np.sin(a)
            horizontal = self.speed * np.array([c * base[0] - s * base[1], s * base[0] + c * base[1]])
        elif b == "orbit":
            self.orbit_phase += self.orbit_rate * dt
            c, s = np.cos(self.orbit_phase), np.sin(self.orbit_phase)
            horizontal = self.speed * np.array([c * base[0] - s * base[1], s * base[0] + c * base[1]])
        elif b == "dash":
            horizontal = self.max_speed * base
        else:  # straight, dive, climb
            horizontal = self.speed * base
        climb = np.clip(1.0 * (self.altitude - pos[2]), -4.0, 4.0)
        return np.array([horizontal[0], horizontal[1], climb])


class Wind:
    """Steady push plus gusts (an Ornstein-Uhlenbeck process), as a horizontal force in newtons.

    Sized for the X2 (up to 1.5 N steady); `scale` shrinks it for lighter drones.
    """

    def __init__(self, rng, scale=1.0):
        self.rng = rng
        angle = rng.uniform(0, 2 * np.pi)
        self.mean = scale * rng.uniform(0, 1.5) * np.array([np.cos(angle), np.sin(angle)])
        self.gust = np.zeros(2)
        self.sigma, self.tau = scale * rng.uniform(0.2, 1.0), 2.0

    def step(self, dt):
        self.gust += -self.gust / self.tau * dt + self.sigma * np.sqrt(2 * dt / self.tau) * self.rng.normal(size=2)
        return self.mean + self.gust


def outcome_of(pos):
    r = np.linalg.norm(pos[:2])
    if r < ZONE_RADIUS and pos[2] < ZONE_HEIGHT:
        return "breach"
    if r > ARENA_RADIUS + 10:
        return "left"
    if pos[2] < 0.3:
        return "crash"
    return None


GUARD_POS = np.array([7.0, -7.0, 12.0])  # m: the defender X2 hovers here, beside the building, above the roof


class Radar:
    """The radar at the zone centre: noisy position readings with single dropouts and blackout bursts."""

    def __init__(self, rng):
        self.rng, self._burst = rng, 0

    def __call__(self, pos):
        rng = self.rng
        if self._burst == 0 and rng.random() < BURST_P:
            self._burst = rng.integers(3, 11)
        if self._burst > 0:
            self._burst -= 1
            return np.full(3, np.nan)
        if rng.random() < DROPOUT_P:
            return np.full(3, np.nan)
        sigma = RADAR_NOISE + 0.01 * np.linalg.norm(pos)
        return pos + rng.normal(0, sigma, 3)


class Flight:
    """One intruder flight in its own MuJoCo world. Step it, read its state; `record` keeps the radar log."""

    def __init__(self, seed, drone="x2", guard=False, pilot="a"):
        """guard: also put the X2 (the defender) in the world, hovering in place next to the building.
        Only for watching: the guard never touches the intruder, but the dataset is made without it."""
        timestep, gains, _ = SETUP[drone]
        self.rng = np.random.default_rng(seed)
        names = [drone] + (["x2"] if guard and drone != "x2" else [])
        self.model = build(names, zone=True, timestep=timestep)
        self.data = mujoco.MjData(self.model)
        self.pilot = controllers(self.model, [drone], GAINS[gains])[drone]
        self.guard = controllers(self.model, ["x2"])["x2"] if len(names) > 1 else None
        self.policy = IntruderPolicy(self.rng, self.pilot.g.max_speed, pilot)
        self.wind = Wind(self.rng, self.pilot.mass / X2_MASS)

        start = self.policy.start
        q, v = self.pilot.qpos, self.pilot.dof
        self.data.qpos[q : q + 3] = start
        self.data.qvel[v : v + 3] = [*(unit(-start[:2]) * self.policy.max_speed * self.policy.aggression * 0.6), 0]
        if self.guard:
            g = self.guard.qpos
            self.data.qpos[g : g + 3] = GUARD_POS
        mujoco.mj_forward(self.model, self.data)
        self.pilot.reset_target(self.data)
        if self.guard:
            self.guard.reset_target(self.data)  # holds this position: no command, so it hovers
        self.outcome = None
        self.log = {k: [] for k in ["t", "true", "vel", "meas", "behaviour", "wind"]}
        self._next_policy = 0.0
        self._next_radar = 0.0
        self.radar = Radar(self.rng)  # shares the flight's random stream, so recorded flights replay exactly

    def step(self):
        """Advance one physics step. Returns False once the flight has ended."""
        d, m = self.data, self.model
        pos, vel, _, _ = self.pilot.state(d)
        if d.time >= self._next_policy:
            self.pilot.velocity = self.policy.velocity(d.time, pos, POLICY_EVERY)
            d.xfrc_applied[self.pilot.body, :2] = self.wind.step(POLICY_EVERY)
            self._next_policy += POLICY_EVERY
        if d.time >= self._next_radar - 1e-9:
            log = self.log
            log["t"].append(d.time)
            log["true"].append(pos)
            log["vel"].append(vel)
            log["meas"].append(self.radar(pos))
            log["behaviour"].append(BEHAVIOURS.index(self.policy.behaviour))
            log["wind"].append(d.xfrc_applied[self.pilot.body, :2].copy())
            self._next_radar += RADAR_EVERY
        self.pilot.update(d, m.opt.timestep)
        if self.guard:
            self.guard.update(d, m.opt.timestep)
        mujoco.mj_step(m, d)
        self.outcome = outcome_of(self.pilot.state(d)[0]) or ("timeout" if d.time >= MAX_TIME else None)
        return self.outcome is None


def simulate(seed, drone, pilot="a"):
    flight = Flight(seed, drone, pilot=pilot)
    while flight.step():
        pass
    log = {k: np.asarray(v) for k, v in flight.log.items()}
    p = flight.policy
    info = (seed, INTENTS.index(p.intent), OUTCOMES.index(flight.outcome), p.aggression, np.linalg.norm(flight.wind.mean))
    return log, info


def make_dataset(n, drone, out, workers, pilot="a"):
    t0 = time.time()
    first = SETUP[drone][2] + PILOT_SEED[pilot]
    with Pool(workers) as pool:
        results = pool.starmap(simulate, [(first + i, drone, pilot) for i in range(n)], chunksize=4)
    arrays = {k: [] for k in ["t", "true", "vel", "meas", "behaviour", "wind"]}
    flight_ids = []
    for (log, info) in results:
        for k in arrays:
            arrays[k].append(log[k])
        flight_ids.append(np.full(len(log["t"]), info[0]))
    info = np.array([r[1] for r in results])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        flight=np.concatenate(flight_ids),
        **{k: np.concatenate(v) for k, v in arrays.items()},
        flight_seed=info[:, 0].astype(int), flight_intent=info[:, 1].astype(int),
        flight_outcome=info[:, 2].astype(int), flight_aggression=info[:, 3], flight_wind=info[:, 4],
        behaviours=np.array(BEHAVIOURS), intents=np.array(INTENTS), outcomes=np.array(OUTCOMES),
        drone=np.array(drone), pilot=np.array(pilot),
    )
    outcomes = np.bincount(info[:, 2].astype(int), minlength=len(OUTCOMES))
    rows = sum(len(r[0]["t"]) for r in results)
    print(f"{n} flights, {rows} radar rows, {time.time() - t0:.0f} s -> {out}")
    print("outcomes:", dict(zip(OUTCOMES, outcomes.tolist())))
    meas = np.concatenate(arrays["meas"])
    print(f"radar dropouts: {np.isnan(meas[:, 0]).mean():.1%}")


def view(index, drone):
    """Watch flight number `index` of this drone's dataset (same seed, so the same flight)."""
    import mujoco.viewer

    seed = SETUP[drone][2] + index
    flight = Flight(seed, drone, guard=True)
    p = flight.policy
    print(f"{drone} flight {index} (seed {seed}): intent {p.intent}, aggression {p.aggression:.2f}, wind {np.linalg.norm(flight.wind.mean):.1f} N")
    with mujoco.viewer.launch_passive(flight.model, flight.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = flight.pilot.body
        viewer.cam.distance, viewer.cam.elevation = (25.0 if drone == "x2" else 2.5), -12
        viewer.opt.geomgroup[POLE_GROUP] = 1  # the zone poles: MuJoCo hides group 3 unless asked
        start = time.time()
        last = None
        while viewer.is_running():
            alive = flight.step() if flight.outcome is None else False
            pos = flight.pilot.state(flight.data)[0]
            with viewer.lock():  # behind the drone, looking toward the building
                viewer.cam.azimuth = np.degrees(np.arctan2(-pos[1], -pos[0]))
            if p.behaviour != last and alive:
                last = p.behaviour
                print(f"  t={flight.data.time:5.1f}s  {last}")
            if not alive and last != "END":
                last = "END"
                print(f"  t={flight.data.time:5.1f}s  outcome: {flight.outcome} (close the window)")
            viewer.sync()
            lag = flight.data.time - (time.time() - start)
            if lag > 0 or not alive:
                time.sleep(max(lag, 0.02))


def record_view(index, drone, out, fps=30, size=(1280, 720)):
    """The same as --view, saved as an MP4 instead of a window (renders offscreen: works on a server GPU)."""
    import imageio_ffmpeg
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    seed = SETUP[drone][2] + index
    flight = Flight(seed, drone, guard=True)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), 22)
    renderer = mujoco.Renderer(flight.model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, flight.pilot.body
    cam.distance, cam.elevation = (25.0 if drone == "x2" else 2.5), -12
    options = mujoco.MjvOption()
    options.geomgroup[POLE_GROUP] = 1  # the zone poles: MuJoCo hides group 3 unless asked
    writer = imageio_ffmpeg.write_frames(str(out), size, fps=fps, quality=8, macro_block_size=8)
    writer.send(None)
    frames, held = 0, 0
    while held < 2 * fps:  # after the end, hold the last picture 2 s
        if flight.outcome is None:
            while flight.data.time < frames / fps and flight.step():
                pass
        else:
            held += 1
        pos = flight.pilot.state(flight.data)[0]
        cam.azimuth = np.degrees(np.arctan2(-pos[1], -pos[0]))  # behind the drone, looking at the building
        renderer.update_scene(flight.data, cam, scene_option=options)
        img = Image.fromarray(renderer.render())
        text = f"t = {flight.data.time:4.1f} s   intruder: {flight.policy.behaviour}"
        if flight.outcome:
            text += f"   -> {flight.outcome.upper()}"
        d = ImageDraw.Draw(img)
        d.rectangle((6, 6, 20 + d.textlength(text, font=font), 42), fill=(0, 0, 0))
        d.text((14, 10), text, font=font, fill=(255, 255, 255))
        writer.send(np.asarray(img).tobytes())
        frames += 1
    writer.close()
    print(f"{drone} flight {index}: {flight.outcome} at {flight.data.time:.1f} s, {frames} frames -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drone", choices=list(SETUP), default="x2")
    parser.add_argument("--view", type=int, metavar="N", help="watch flight N of the dataset live")
    parser.add_argument("--record", type=Path, help="with --view: save it as an MP4 instead (no window needed)")
    parser.add_argument("--flights", type=int, default=2000)
    parser.add_argument("--out", type=Path, help="default: data/intruder_<drone>.npz")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", choices=list(PILOTS), default="a",
                        help="b: a different, more evasive pilot (writes data/intruder_<drone>_pilotb.npz)")
    args = parser.parse_args()
    if args.view is not None and args.record:
        record_view(args.view, args.drone, args.record)
    elif args.view is not None:
        view(args.view, args.drone)
    else:
        name = args.drone + ("" if args.pilot == "a" else f"_pilot{args.pilot}")
        out = args.out or Path(__file__).resolve().parent.parent / "data" / f"intruder_{name}.npz"
        make_dataset(args.flights, args.drone, out, args.workers, args.pilot)


if __name__ == "__main__":
    main()
