# Drone track: details and experiments

The short version is in `README.md`. This file keeps the full detail: data, features, results by manoeuvre, and every experiment with what we changed.

## How the data is made

Every flight is a random intruder (`intruder.py`). Nothing is flown by hand.

| Random per flight | Values |
|---|---|
| Intent | attack 50% (heads into the zone), flyby 30% (passes 15-40 m away), loiter 20% (circles at 18-35 m) |
| Manoeuvre, new every 1-6 s | straight, zigzag, orbit, hover, dash, dive, climb |
| Speed level, cruise height | 40-100% of max speed; 5-28 m |
| Wind | steady push plus random gusts, scaled by drone mass (X2: up to 1.5 N) |

A flight ends when the drone enters the zone (breach), leaves the arena, hits 60 s, or crashes. Flights are reproducible: flight N always uses seed N (X2) or 100000 + N (Crazyflie), so the Crazyflie test flights never reuse an X2 training flight's random plan.

| Dataset | Flights | Radar rows | Time to make (4 CPU cores) | Why this many |
|---|---|---|---|---|
| X2 (training) | 2,000 | 469,025 | 16.5 min | covers every intent x manoeuvre mix many times; more than TabPFN needs per prediction (it uses a few thousand rows as context), but the MLP and fine-tuning want plenty |
| Crazyflie (test) | 300 | 149,688 | 15 min (smaller physics step, slower drone, longer flights) | enough flights for confidence intervals on every metric |

X2 outcomes: 1,127 breach, 597 left, 271 timeout, 5 crash (dropped). Variety: 11,141 separate manoeuvres; the rarest, dash, still has 747. Alarm positives (breach within 5 s): 12%.

Crazyflie outcomes: 135 breach, 7 left, 158 timeout, 0 crash. By intent: attack 131/151 breach (19 run out of time), flyby 83/92 timeout (too slow to leave), loiter 56/57 timeout. Alarm positives: 4.6%.

Stored per radar reading: time, true position and velocity, radar measurement (NaN when missing), current manoeuvre, wind. Per flight: intent, outcome, aggression, wind strength.

## Inputs and outputs (full spec)

One table row = one radar reading, built **only from radar measurements up to that moment** (what a real system would have; never the true state and never the flight clock, see below). Code: `features.py`.

**Motion estimate:** position and velocity from a straight-line fit to the last 1.5 s of readings; acceleration from a quadratic fit to the last 2 s (why: Experiment 4). Heading ψ = direction of the velocity. Horizontal vectors are rotated into the drone's frame (forward, left), so an attack from the north looks the same as one from the south.

**Inputs (16 features)**

| Feature | Meaning | Why |
|---|---|---|
| `range`, `height` | distance from the zone centre, height | where it is relative to what we protect |
| `zone_fwd`, `zone_left` | where the zone centre is, seen from the drone's heading | is it heading at the zone or past it |
| `speed`, `climb` | horizontal speed, vertical speed | |
| `acc_fwd`, `acc_left`, `acc_up` | acceleration in the drone frame | speeding up, turning, diving |
| `turn_rate` | rad/s | zigzags and orbits |
| `speed_1s`, `turn_rate_1s`, `climb_1s`, `turn_rate_2s` | the same, 1 s and 2 s earlier (NaN at the start of a flight) | short memory, to recognise patterns like a zigzag |
| `n_missing` | missing readings in the last 1 s | how much to trust the row |
| `fit_error` | RMS residual of the fit | noise level, grows with range |

**Time is not an input.** It is used to compute features and targets, but "time since the flight started" would let the model cheat (every flight starts at the arena edge, loiterers always stop at 60 s), and a real radar does not know when a flight started.

**Outputs**

| Output | Meaning |
|---|---|
| `d1_fwd`, `d1_left`, `d1_up` … `d5_*` (9 targets) | true position after 1, 3, 5 s minus the current estimate, in the drone frame; NaN when the flight ends first |
| 5% / 50% / 95% quantiles of each | the prediction cone (TabPFN gives these directly) |
| `breach_5s` | 1 if the drone enters the zone within 5 s: the alarm (classifier probability) |

**Split rule:** always by flight, never by row (neighbouring rows of one flight are near copies).

## Results (full tables)

### Baseline: straight-line guess on X2 (all 1,995 flights)

The table files are numpy archives: `X` (inputs), `Y` (position targets), `breach_5s` (alarm), plus bookkeeping (`flight`, `t`, `behaviour`, `intent`, `pos`, `psi`, `true`) and the name lists (`features`, `targets`). To look at them by eye: `python drone/features.py --preview 2000` writes `data/table_x2_preview.csv` with named columns (`X_*` inputs, `Y_*` outputs).

3D error in metres between the guess "keep the current velocity" and the true future position.

| Horizon | Median | 90th percentile | straight | zigzag | orbit | hover | dash | dive | climb |
|---|---|---|---|---|---|---|---|---|---|
| 1 s | 2.34 | 5.74 | 1.7 | 2.4 | 2.9 | 2.8 | 3.0 | 2.8 | 2.8 |
| 3 s | 7.68 | 17.52 | 5.5 | 7.0 | 10.3 | 9.3 | 11.7 | 8.9 | 9.2 |
| 5 s | 13.62 | 29.52 | 10.4 | 12.4 | 17.4 | 15.7 | 19.4 | 15.7 | 16.4 |

Errors grow with the horizon and are worst for dashes, orbits and hover-then-go: the manoeuvres a learned model should handle better. This is the bar every model has to beat.

### Baseline: straight-line guess on the Crazyflie (all 300 flights)

| Horizon | Median | 90th percentile | straight | zigzag | orbit | hover | dash | dive | climb |
|---|---|---|---|---|---|---|---|---|---|
| 1 s | 1.71 | 3.12 | 1.6 | 1.8 | 1.7 | 1.9 | 1.6 | 1.8 | 1.8 |
| 3 s | 4.50 | 8.41 | 4.0 | 4.9 | 4.2 | 5.2 | 4.5 | 5.1 | 4.9 |
| 5 s | 7.19 | 12.98 | 6.6 | 7.4 | 6.7 | 7.7 | 7.5 | 8.2 | 8.1 |

Smaller in metres than the X2 only because the Crazyflie is slower (median 1.6 m/s vs 4.3). It is not an easier target: relative to how far it moves, the straight-line guess does worse. Its slow tilting also makes the manoeuvres more even in difficulty. When comparing models across drones, compare each model to this baseline, not raw metres.

## Experiments log

What we tried, what we found, what we changed. Newest last.

**1. Flight controller tuning (Crazyflie).** The Crazyflie model's moment motors give at most 1e-5 N m, so it tilts slowly. With gentle settings (position loop 0.5 rad/s, attitude 1.5 rad/s) it overshot its speed by 60% and drifted 0.35 m after stopping. More damping made it worse; the limit was the slow attitude loop. A faster attitude loop fixed it: position 0.8, attitude 5.0 rad/s gives the commanded speed, stops within 8 cm, holds position to 7 mm. The X2 flew well with the first settings (position 2, attitude 10 rad/s).

**2. Two intruder bugs (100-flight trial).** (a) In steep dives the controller asked for negative thrust, which flips the drone: 1 crash in 20 flights. Fix: never ask for less than 40% of gravity's worth of thrust. (b) Flybys aimed only 6-16 m from the centre instead of 15-40 m, so 4 of 6 breached. After both fixes: attack 43/43 breach, flyby 34/35 leave, loiter mostly circles, 0 crashes. Orbits also made up 38% of all rows, so the loiter mix was rebalanced (orbit 45% → 30%).

**3. Crazyflie top speed.** Tested 2-5 m/s with 10-20° tilt limits on a fast run, a hard reversal and a zigzag. At 15° tilt or more it flips (tilt 94-180°, then falls): its moment motors cannot catch a big tilt. **3 m/s at 12° is the safe limit** (reaches 2.84 m/s, clean zigzag, stays upright). A 20-flight trial at this setting: 10 breach, 10 timeout, 0 crash, median speed 1.6 m/s. The X2 flights are bit-for-bit unchanged by this work, so the X2 dataset stays valid.

**4. Motion estimate from radar.** First version: a quadratic fit over the last 1 s. On 200 flights it was badly noisy and biased: velocity error 4.9 m/s, speed overestimated by 1.6 m/s, and a fake +6.2 m/s² forward acceleration (noise in the fitted velocity and acceleration at the newest point is correlated). Compared four estimators against the true state:

| Estimator | Position error | Velocity error | Accel error | Speed bias | Forward-accel bias |
|---|---|---|---|---|---|
| A: quadratic, 1 s (first version) | 0.92 m | 4.94 m/s | 11.2 m/s² | +1.60 | +6.22 |
| B: line 1 s + quadratic 2 s for accel | 0.71 | 1.82 | 4.0 | +0.10 | +1.14 |
| C: quadratic 2 s for everything | 0.76 | 2.39 | 4.0 | +0.61 | +1.64 |
| **D: line 1.5 s + quadratic 2 s for accel (chosen)** | **0.70** | **1.53** | **4.0** | **-0.01** | **+0.89** |

With D the table's speed median is 4.3 m/s (true flights: 4.6) instead of 6.1. Building the X2 table also went from 7 min to 27 s by reading the compressed file once instead of once per flight.

**5. Track start-up.** The CSV preview showed nonsense in the first rows of each flight (climb -10.8 m/s, acceleration -69 m/s² at t = 0.3 s): with only 3-4 readings the fits are meaningless. Now a row needs 8 of the last 15 readings for position/velocity and 12 of the last 20 for acceleration, so tracks start after about 1.1 s, like a real tracker confirming a new track. Rows: X2 460,693 → 418,050, Crazyflie 148,520 → 137,088. Baselines moved slightly (X2 3 s median 7.89 → 7.68 m); the tables above are the current numbers.

## Known limits

- All data is simulated; there is no real radar or real drone in the loop.
- The Crazyflie model's turning motors are very weak, so it tilts slowly and is limited to 3 m/s. We keep the model as menagerie ships it.
- 5 of 2,000 X2 flights crash in low dives; we drop them.
- At 3 m/s, Crazyflie flybys rarely leave the arena within 60 s, so they end as timeouts.
- The radar is a simple noise model (Gaussian noise growing with range, random dropouts), not a physical radar simulation.
