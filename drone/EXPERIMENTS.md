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

Median error (m) between the prediction and where the drone really was. Lower is better.

| Model | X2: 1 s | 3 s | 5 s | Crazyflie: 1 s | 3 s | 5 s |
|---|---|---|---|---|---|---|
| Straight-line guess | 2.3 | 7.7 | 13.6 | 1.7 | 4.5 | 7.2 |
| IMM (tuned on X2) | | | | *running* | | |
| TabPFN | | | | *needs token* | | |

The straight-line guess is worst on dashes (X2, 3 s: 11.7 m) and orbits (10.3 m), and best on straight flight (5.5 m). The Crazyflie's errors are smaller only because it is slower; relative to how far it moves it is harder.

## 4. Experiments

Each one: what went wrong or what we wanted to know, what we did, what we decided.

### 1. Crazyflie wobbles
- **Problem:** its turning motors are very weak, so with gentle settings it overshot its speed by 60%.
- **Tried:** more damping (worse), then a faster attitude loop.
- **Decision:** faster attitude loop. Now it holds the commanded speed and stops within 8 cm.

### 2. Intruder bugs
- **Problem:** in steep dives the controller asked for negative thrust and flipped the drone; flybys aimed too close and 4 of 6 hit the zone.
- **Fix:** minimum thrust; correct flyby aim.
- **Result:** 0 crashes in 100 flights; attacks 43/43 breach, flybys 34/35 pass.

### 3. Crazyflie top speed
- **Question:** how fast can it fly as an intruder?

| Limit | Result |
|---|---|
| 2 m/s, 10° tilt | stable |
| **3 m/s, 12° tilt** | **stable, chosen** |
| 4-5 m/s, 15-20° tilt | flips over and falls |

### 4. Estimating speed from noisy radar
- **Problem:** the first method (curve fit over 1 s) was very noisy and invented a +6 m/s² acceleration.
- **Tried four methods** on 200 flights against the true values:

| Method | Speed error | Fake acceleration |
|---|---|---|
| A: curve, 1 s (first) | 4.9 m/s | +6.2 m/s² |
| B: line 1 s, curve 2 s | 1.8 | +1.1 |
| C: curve 2 s | 2.4 | +1.6 |
| **D: line 1.5 s, curve 2 s (chosen)** | **1.5** | **+0.9** |

### 5. Garbage at the start of each track
- **Problem:** the first rows of each flight had nonsense values (acceleration -69 m/s²) from only 3-4 readings.
- **Fix:** a row needs enough readings first (tracks start after ~1.1 s, like a real tracker confirming a target).

### 6. IMM tracker loses its acceleration
- **Problem:** before tuning, the IMM did worse than the straight-line guess. On a made-up accelerating track it was 20 m off at 5 s, as if it ignored acceleration.
- **Cause:** when the three trackers are blended, the "straight" tracker's built-in zero acceleration erased the others' acceleration and turn rate within a second.
- **Fix:** a tracker only contributes to what it actually tracks.
- **Check on made-up tracks:** each motion is now recognised by the right tracker.

| Track | Leading tracker | Error at 5 s |
|---|---|---|
| straight | straight (54%) | 1.1 m |
| accelerating | accelerating (52%) | 10.0 m |
| turning | turning (73%) | 8.4 m |

### 7. IMM tuning
- **What:** search the IMM's noise settings on 150 X2 flights (training data only) for the lowest 3 s error.
- **Result:** *running.*

## 5. Known limits

- All data is simulated; no real radar or drone.
- The radar gives x, y, z with simple noise; a real radar measures distance and angles.
- The Crazyflie is limited to 3 m/s by its weak turning motors (model as menagerie ships it).
- At 3 m/s, Crazyflie flybys rarely leave the arena in 60 s, so they end as timeouts.
