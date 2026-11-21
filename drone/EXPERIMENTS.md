# Drone track: details and experiments

The short version is in [`README.md`](README.md). This file has the detail and every experiment.

## 1. Data

Every flight is a random intruder flown by a script (`intruder.py`): an **intent** (attack 50%, flyby 30%, loiter 20%), a new **manoeuvre** every 1-6 s (straight, zigzag, orbit, hover, dash, dive, climb), a random speed and height, and **wind**. A **radar** at the zone centre records 10 positions a second with noise (0.3 m + 1% of range) and gaps (~11% missing).

| | X2 (learn from) | Crazyflie (test on) |
|---|---|---|
| Flights | 2,000 | 300 |
| Radar readings | 469,025 | 149,688 |
| Time to make | 16.5 min | 15 min |
| Breach / left / timeout / crash | 1,127 / 597 / 271 / 5 | 135 / 7 / 158 / 0 |
| Alarm positives (breach within 5 s) | 12.8% | 4.7% |

Crashed flights are dropped. Crazyflie flights use different random seeds, so they never repeat an X2 flight's plan.

## 2. The table

One row per radar reading, from **radar data only** (never the true position, never the flight clock).

**Inputs (16):**

| Group | Features |
|---|---|
| Where it is | `range`, `height`, `zone_fwd`, `zone_left` (where the zone is, seen from the drone's heading) |
| How it moves | `speed`, `climb`, `acc_fwd`, `acc_left`, `acc_up`, `turn_rate` |
| Short memory | `speed_1s`, `turn_rate_1s`, `climb_1s`, `turn_rate_2s` (1 s and 2 s ago) |
| Radar quality | `n_missing` (gaps in the last 1 s), `fit_error` (noise) |

**Outputs:**

| | |
|---|---|
| `x1 y1 z1`, `x3 y3 z3`, `x5 y5 z5` | where the drone really is 1, 3, 5 s later (world coordinates, used for all scoring) |
| `breach_5s` | 1 if it enters the zone within 5 s (the alarm) |

Models may learn the same answers in the drone's own direction (forward, left, up), which is easier to learn; `to_world` converts back exactly.

**Why no time input:** every flight starts at the arena edge, so "time since start" would let a model cheat, and a real radar does not know when a flight started.

**Split:** always by flight, never by row.

## 3. Results

Median error (m) between the predicted position and where the drone really was, on the Crazyflie test flights. Lower is better.

| Model | 1 s ahead | 3 s ahead | 5 s ahead |
|---|---|---|---|
| IMM (tuned on X2 flights) | *running* | | |
| TabPFN | *needs token* | | |

## 4. Experiments

Each experiment in four lines: what happened, why, what we changed, and the result.

### 1. The small drone wobbled
- **What happened:** when told to fly at a speed, the Crazyflie overshot by 60% and kept drifting after stopping.
- **Why:** its motors for tilting are very weak, and our controller corrected its tilt too slowly.
- **Changed:** let the controller correct the tilt faster. (Adding more braking made it worse, so that was not the fix.)
- **Result:** it now flies at the asked speed and stops within 8 cm.

### 2. Two bugs in the intruder pilot
- **What happened:** some drones crashed in dives, and drones meant to fly *past* the zone flew *into* it (4 of 6).
- **Why:** in a steep dive the controller asked for negative thrust, which flipped the drone; and a maths slip aimed fly-bys 6-16 m from the centre instead of 15-40 m.
- **Changed:** thrust can never go below a safe minimum; fixed the fly-by aim.
- **Result:** in 100 test flights: no crashes, attacks 43/43 reach the zone, fly-bys 34/35 pass by.

### 3. How fast can the small drone fly?
- **What we wanted:** the Crazyflie as a real attacker, faster than its 0.5 m/s keyboard setting.
- **Tried:** speeds of 2-5 m/s with different tilt limits, on a fast run, a sharp reversal and a zigzag.
- **Found:** above about 12° of tilt it cannot catch itself and flips over.
- **Result:** **3 m/s with 12° tilt** is the fastest it flies safely. The big drone's data was not affected.

| Speed and tilt | Outcome |
|---|---|
| 2 m/s, 10° | flies fine |
| **3 m/s, 12°** | **flies fine (chosen)** |
| 4-5 m/s, 15-20° | flips and falls |

### 4. Getting the speed right from noisy radar
- **What happened:** the first way of reading speed from the radar dots gave speeds about 1.6 m/s too high and a fake "speeding up" signal.
- **Why:** it fitted a curve to only 1 second of noisy dots and read the speed at the very last dot, where the noise is worst.
- **Changed:** compared four methods against the true values on 200 flights and picked the best: a straight line through the last 1.5 s for speed, a curve through the last 2 s for acceleration.
- **Result:** speed error down from 4.9 to 1.5 m/s.

| Method | Speed error | Fake speeding-up |
|---|---|---|
| Curve, 1 s (first try) | 4.9 m/s | +6.2 m/s² |
| Line 1 s + curve 2 s | 1.8 | +1.1 |
| Curve 2 s | 2.4 | +1.6 |
| **Line 1.5 s + curve 2 s (chosen)** | **1.5** | **+0.9** |

### 5. Nonsense at the start of each flight
- **What happened:** the first rows of every flight had impossible values (e.g. -69 m/s² acceleration).
- **Why:** right after the radar first sees a drone, there are only 3-4 dots, too few to measure speed or turning.
- **Changed:** a row is only made once there are enough dots (after about 1 s), as real trackers also wait to confirm a new target.
- **Result:** clean tables; about 9% fewer rows.

### 6. The IMM forgot accelerations and turns
- **What happened:** the IMM did worse than it should. On a made-up track with steady acceleration it was 20 m off after 5 s.
- **Why:** the IMM mixes its three trackers every step. The "straight" tracker always assumes zero acceleration, and that zero was mixed into the "speeding up" tracker until it forgot the acceleration. The same happened to turning.
- **Changed:** each tracker now only shares what it actually tracks (the standard fix).
- **Result:** on made-up tracks, each kind of motion is now picked up by the right tracker.

| Made-up track | Tracker that takes over | Error after 5 s |
|---|---|---|
| straight | straight (54%) | 1.1 m |
| speeding up | speeding up (52%) | 10.0 m (was 20) |
| turning | turning (73%) | 8.4 m |

### 7. Tuning the IMM
- **What:** the IMM has five settings (how jumpy each tracker expects the drone to be, how often it switches). We search them on 150 big-drone flights for the smallest error 3 s ahead. It never sees the small-drone test flights.
- **Result:** *running.*

## 5. Known limits

- All data is simulated; no real radar or drone.
- The radar gives x, y, z with simple noise; a real radar measures distance and angles.
- The Crazyflie is limited to 3 m/s by its weak turning motors (model as menagerie ships it).
- At 3 m/s, Crazyflie flybys rarely leave the arena in 60 s, so they end as timeouts.
