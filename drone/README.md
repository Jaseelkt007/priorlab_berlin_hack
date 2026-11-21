# Drone track: where is the intruder drone going?

Part of our Prior Labs entry ("TabPFN for spatial awareness"). Branch: `drone`.
Full detail and every experiment: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## The idea

A small drone flies toward a protected site. A radar sees it, but the readings are noisy and sometimes missing.

**TabPFN predicts where the drone will be in 1, 3 and 5 seconds, with a range ("it will be somewhere in this cone"), and raises the alarm before it reaches the site.** It has never seen this kind of drone before; it adapts from a few radar readings, without retraining.

We compare it with what real radar trackers use: an **IMM**, three Kalman-type filters (straight, speeding up, turning) that switch automatically.

## How it works

1. **A simulated sky** (MuJoCo): a red protected zone in the middle, a radar at its centre.
2. **Intruder flights**: a script flies drones toward the zone, randomly attacking, passing by or circling, with zigzags, dives, dashes and wind.
3. **Radar recording**: 10 positions a second, with noise and gaps.
4. **Tables**: each radar reading becomes one row: what the radar sees now (inputs) and where the drone really went next (answers).
5. **Models** learn from the big drone's flights and are tested on the small drone's.

| Drone | Role |
|---|---|
| Skydio X2 (big, 10 m/s) | flights to learn from (2,000); later the defender |
| Crazyflie (small, 3 m/s) | the attacker we test on (300 flights) |

## Inputs and outputs

**Inputs** (16 numbers, from the radar only): where the drone is relative to the zone, how fast it goes, whether it is turning, climbing or speeding up, the same a second or two ago, and how many radar readings were missing.

**Outputs**: where it will be after 1, 3 and 5 s, as x, y, z (with a range), and **alarm**: will it enter the zone within 5 s?

## Status

| | |
|---|---|
| Simulation, both drones, intruder flights | done |
| Data: 2,000 X2 flights, 300 Crazyflie flights | done |
| Tables | done |
| IMM tracker (what real radars use) | done: tuned on X2, tested on Crazyflie |
| TabPFN | **next** (needs `TABPFN_TOKEN` in `.env`) |
| TabPFN fine-tuned | if the GPU allows |
| Demo: prediction cone + alarm in the viewer | planned |

## Results so far

Median error between the predicted and the real position, on the small-drone test flights:

| Model | 1 s ahead | 3 s ahead | 5 s ahead |
|---|---|---|---|
| IMM (what real radars use) | 1.55 m | 4.28 m | 7.04 m |
| TabPFN | *needs token* | | |

## Try it

From the repo root, in the `priorlab` conda env. First get the drone models: see `models/README.md`.

```sh
python drone/view_drones.py                 # fly the two drones with the keyboard
python drone/intruder.py --view 8           # watch an intruder flight (any number)
python drone/intruder.py --drone cf2 --view 8
```

Make the data again (it is not in git):

```sh
python drone/intruder.py --flights 2000              # X2 flights, 16 min
python drone/intruder.py --drone cf2 --flights 300   # Crazyflie flights, 15 min
python drone/features.py                             # tables, 30 s each
python drone/features.py --drone cf2
python drone/features.py --preview 2000              # a CSV you can open in Excel
```

## Files

| File | What it does |
|---|---|
| `sim.py` | the world: sky, floor, zone, radar |
| `flight.py` | flies a drone: "go this way" → motor commands |
| `intruder.py` | the random intruder, wind, radar, data recording |
| `features.py` | radar readings → tables |
| `imm.py` | the IMM tracker (the baseline to beat) |
| `view_drones.py` | keyboard flying |
| `EXPERIMENTS.md` | full detail, results by manoeuvre, everything we tried |
