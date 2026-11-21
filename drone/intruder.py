"""Random intruder flights toward the protected zone, recorded like a radar would see them.

    python drone/intruder.py --view 7               # watch flight number 7 live in the viewer
    python drone/intruder.py --flights 2000         # make the dataset -> data/intruder_x2.npz

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
from sim import ARENA_RADIUS, ZONE_HEIGHT, ZONE_RADIUS, build, controllers

DRONE = "x2"
TIMESTEP = 0.005  # s, physics; the X2 alone is stable at this step
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

RADAR_NOISE = 0.3  # m, plus 1% of range, per axis
DROPOUT_P = 0.05  # single missing samples
BURST_P = 0.01  # chance per sample that a 3-10 sample gap starts


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


class IntruderPolicy:
    """Gives a world-frame velocity command from the drone's position. All randomness comes from rng."""

    def __init__(self, rng, max_speed):
        self.rng = rng
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
        self.behaviour = BEHAVIOURS[rng.choice(len(BEHAVIOURS), p=WEIGHTS[self.intent])]
        self.segment_end = t + (rng.uniform(1, 3) if self.behaviour == "hover" else rng.uniform(2, 6))
        self.speed = self.max_speed * self.aggression * rng.uniform(0.5, 1.0)
        self.zig_angle = np.radians(rng.uniform(30, 70))
        self.zig_period = rng.uniform(0.8, 2.5)
        self.orbit_rate = rng.choice([-1, 1]) * self.speed / rng.uniform(5, 20)  # rad/s
        self.orbit_phase = 0.0
        if self.behaviour == "dive":
            self.altitude = max(3.0, self.altitude - rng.uniform(5, 15))
        elif self.behaviour == "climb":
            self.altitude = min(40.0, self.altitude + rng.uniform(5, 15))

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
    """Steady push plus gusts (an Ornstein-Uhlenbeck process), as a horizontal force in newtons."""

    def __init__(self, rng):
        self.rng = rng
        angle = rng.uniform(0, 2 * np.pi)
        self.mean = rng.uniform(0, 1.5) * np.array([np.cos(angle), np.sin(angle)])
        self.gust = np.zeros(2)
        self.sigma, self.tau = rng.uniform(0.2, 1.0), 2.0

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


class Flight:
    """One intruder flight in its own MuJoCo world. Step it, read its state; `record` keeps the radar log."""

    def __init__(self, seed, model=None):
        self.rng = np.random.default_rng(seed)
        self.model = model or build([DRONE], zone=True, timestep=TIMESTEP)
        self.data = mujoco.MjData(self.model)
        self.pilot = controllers(self.model, [DRONE], GAINS["x2_intruder"])[DRONE]
        self.policy = IntruderPolicy(self.rng, self.pilot.g.max_speed)
        self.wind = Wind(self.rng)

        start = self.policy.start
        q, v = self.pilot.qpos, self.pilot.dof
        self.data.qpos[q : q + 3] = start
        self.data.qvel[v : v + 3] = [*(unit(-start[:2]) * self.policy.max_speed * self.policy.aggression * 0.6), 0]
        mujoco.mj_forward(self.model, self.data)
        self.pilot.reset_target(self.data)
        self.outcome = None
        self.log = {k: [] for k in ["t", "true", "vel", "meas", "behaviour", "wind"]}
        self._next_policy = 0.0
        self._next_radar = 0.0
        self._burst = 0

    def _radar(self, pos):
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
            log["meas"].append(self._radar(pos))
            log["behaviour"].append(BEHAVIOURS.index(self.policy.behaviour))
            log["wind"].append(d.xfrc_applied[self.pilot.body, :2].copy())
            self._next_radar += RADAR_EVERY
        self.pilot.update(d, m.opt.timestep)
        mujoco.mj_step(m, d)
        self.outcome = outcome_of(self.pilot.state(d)[0]) or ("timeout" if d.time >= MAX_TIME else None)
        return self.outcome is None


def simulate(seed):
    flight = Flight(seed)
    while flight.step():
        pass
    log = {k: np.asarray(v) for k, v in flight.log.items()}
    p = flight.policy
    info = (seed, INTENTS.index(p.intent), OUTCOMES.index(flight.outcome), p.aggression, np.linalg.norm(flight.wind.mean))
    return log, info


def make_dataset(n, out, workers):
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(simulate, range(n), chunksize=8)
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
    )
    outcomes = np.bincount(info[:, 2].astype(int), minlength=len(OUTCOMES))
    rows = sum(len(r[0]["t"]) for r in results)
    print(f"{n} flights, {rows} radar rows, {time.time() - t0:.0f} s -> {out}")
    print("outcomes:", dict(zip(OUTCOMES, outcomes.tolist())))
    meas = np.concatenate(arrays["meas"])
    print(f"radar dropouts: {np.isnan(meas[:, 0]).mean():.1%}")


def view(seed):
    import mujoco.viewer

    flight = Flight(seed)
    p = flight.policy
    print(f"flight {seed}: intent {p.intent}, aggression {p.aggression:.2f}, wind {np.linalg.norm(flight.wind.mean):.1f} N")
    with mujoco.viewer.launch_passive(flight.model, flight.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = flight.pilot.body
        viewer.cam.distance, viewer.cam.elevation = 25.0, -25
        start = time.time()
        last = None
        while viewer.is_running():
            alive = flight.step() if flight.outcome is None else False
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", type=int, metavar="SEED", help="watch one flight live")
    parser.add_argument("--flights", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "intruder_x2.npz")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.view is not None:
        view(args.view)
    else:
        make_dataset(args.flights, args.out, args.workers)


if __name__ == "__main__":
    main()
