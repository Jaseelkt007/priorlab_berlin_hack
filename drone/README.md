# Drone track: predicting where an intruder drone is going

Part of our Prior Labs entry ("TabPFN for spatial awareness"). Branch: `drone`.

## The idea in one line

A drone approaches a protected site. From noisy radar positions, **TabPFN predicts where it will be in 1, 3 and 5 seconds, as an area rather than a point**, and raises the alarm before it gets there. It **learns a new kind of drone from a few observations, without retraining**, and we compare it against the industry standard (a Kalman filter).

## Status

| Step | State |
|---|---|
| Flight controller for both drones, keyboard viewer | done |
| Protected zone, random intruder pilot, radar simulation | done |
| X2 dataset: 2,000 flights | done (`data/intruder_x2.npz`, local only) |
| Crazyflie as a faster intruder + its test dataset (~300 flights) | next |
| Tables, Kalman vs TabPFN vs MLP | planned (needs `TABPFN_TOKEN`) |
| Fine-tuned TabPFN | planned, probably on a rented GPU (Brev); our laptop GPU has 4 GB |
| Demo: prediction cone + zone alarm in the viewer | planned |
| Stretch: X2 defender intercepts the Crazyflie | only if everything above works |

## The scene

| Thing | Size | Why |
|---|---|---|
| Protected zone | cylinder, 10 m radius, 30 m high, at the centre | a small site, e.g. a substation |
| Arena | 60 m radius; intruders start on its edge | about 10 s or more of approach |
| Radar | at the zone centre, 10 readings a second | typical update rate for a short-range radar |
| Radar noise | 0.3 m + 1% of range, per axis | far targets are measured less precisely |
| Radar dropouts | 5% single misses + bursts of 3-10 missing readings (~11% in total) | real radars lose targets |

## The drones and their roles

| Drone | Size | Role | Why |
|---|---|---|---|
| **Skydio X2** | 1.3 kg, up to 10 m/s | **Training data.** Later the **defender** | big, fast, stable: rich training flights; faster than the Crazyflie, so it can intercept |
| **Crazyflie 2** | 27 g, slow to tilt | **The attacking intruder in tests and the demo** | small drones are the real threat and the hardest to track; it moves differently from the X2, so it is a true "never seen this drone" test |

Both are from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie); see `models/README.md`.

## How the data is made

Every flight is a random intruder (`intruder.py`). Nothing is flown by hand.

| Random per flight | Values |
|---|---|
| Intent | attack 50% (heads into the zone), flyby 30% (passes 15-40 m away), loiter 20% (circles at 18-35 m) |
| Manoeuvre, new every 1-6 s | straight, zigzag, orbit, hover, dash, dive, climb |
| Speed level, cruise height | 40-100% of max speed; 5-28 m |
| Wind | steady push up to 1.5 N plus random gusts |

A flight ends when the drone enters the zone (breach), leaves the arena, hits 60 s, or crashes.

| Dataset | Flights | Radar rows | Time to make (4 CPU cores) | Why this many |
|---|---|---|---|---|
| X2 (training) | 2,000 | 469,025 | 16.5 min | covers every intent x manoeuvre mix many times; more than TabPFN needs per prediction (it uses a few thousand rows as context), but the MLP and fine-tuning want plenty |
| Crazyflie (test) | ~300 | estimate ~60k | estimate ~15 min | enough flights for confidence intervals on every metric |

Outcomes in the X2 set: 1,127 breach, 597 left, 271 timeout, 5 crash (dropped before training).

Stored per radar reading: time, true position and velocity, radar measurement (NaN when missing), current manoeuvre, wind. Per flight: intent, outcome, aggression, wind strength.

## Inputs and outputs (planned)

One table row = one moment in a flight, built from **the last 1 s of radar readings only** (what a real system would have).

**Inputs**

| Feature | Why |
|---|---|
| distance and height relative to the zone | where it is relative to what we protect |
| velocity, acceleration (from the last readings) | where it is heading and whether it is speeding up |
| turn rate, climb rate | zigzags and dives show up here |
| number of missing readings in the window | tells the model how much to trust the inputs |

Positions and motion are expressed **in the drone's own heading frame** (forward / sideways / up), so an attack from the north looks the same as one from the south. That makes the patterns learnable from fewer rows.

**Outputs**

| Output | Example |
|---|---|
| movement after 1, 3 and 5 s (forward, sideways, up) | "+24 m forward, +3 m left, -2 m up in 3 s" |
| a 5%-95% range for each | the prediction cone |
| alarm | "zone breach likely within 5 s (87%)" |

## What we compare

| Model | What it is | Trained on our data? |
|---|---|---|
| Straight-line guess | keep the current velocity | no |
| **Kalman filter** | industry standard for tracking | no (hand-set physics) |
| Small MLP | a plain neural network | yes |
| **TabPFN** (foundation) | learns from rows in its context | **no, 0 gradient steps** |
| TabPFN fine-tuned | TabPFN further trained on X2 flights | yes |

**Main test:** every model learns from X2 flights only, then predicts **Crazyflie** attacks. TabPFN additionally gets the first few seconds of Crazyflie readings in its context, which is how it adapts without retraining.

**Metrics:** error at 1/3/5 s; how often the true position lands inside the 90% range (should be ~90%); **seconds of warning before a breach**; false alarms on flybys and loiterers; how many Crazyflie readings TabPFN needs to adapt.

## Files

| File | What it does |
|---|---|
| `sim.py` | the world: sky, floor, zone, arena, radar post; builds it with any drones |
| `flight.py` | flight controller: turns "fly this way" into motor commands, works for both drones |
| `intruder.py` | random intruder pilot, wind, radar recording, dataset generator |
| `view_drones.py` | fly the drones yourself with the keyboard |
| `models/README.md` | how to download the two drone models |

## Commands

Run from the repo root in the `priorlab` conda env (Python 3.12, `pip install mujoco tabpfn python-dotenv`).

```sh
python drone/view_drones.py                 # fly both drones with the keyboard
python drone/view_drones.py --selftest      # check the flight controller, no window
python drone/intruder.py --view 8           # watch intruder flight 8 live (any number)
python drone/intruder.py --flights 2000     # make the X2 dataset -> data/intruder_x2.npz
```

## Known limits

- All data is simulated; there is no real radar or real drone in the loop.
- The Crazyflie model's turning motors are very weak, so it tilts slowly. We keep the model as menagerie ships it.
- 5 of 2,000 X2 flights crash in low dives; we drop them.
