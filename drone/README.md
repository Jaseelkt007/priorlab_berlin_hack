# Drone track: where is the intruder drone going?

Part of our Prior Labs entry ("TabPFN for spatial awareness"). Branch: `drone`.
Full detail and every experiment: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## The idea

A small drone flies toward a protected site. A radar sees it, but the readings are noisy and sometimes missing.

**TabPFN predicts where the drone will be in 1 second, with a range ("it will be somewhere in this cone"), and raises the alarm before it reaches the site.** It has never seen this kind of drone before; it adapts from a few radar readings, without retraining.

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

**Output**: where it will be in 1 s, as x, y, z (with a range), and **alarm**: will it enter the zone within 2 s?

## Status

| | |
|---|---|
| Simulation, both drones, intruder flights | done |
| Data: 2,000 X2 flights, 300 Crazyflie flights | done |
| Tables | done |
| IMM tracker (what real radars use) | done: tuned on X2, tested on Crazyflie |
| TabPFN | code ready (`tabpfn_model.py`); runs once `TABPFN_TOKEN` is in `.env` |
| TabPFN fine-tuned | if the GPU allows |
| 3D demo: real vs predicted x, y, z + alarm, live or as video | done for the IMM; TabPFN next |

## Results so far

Small-drone test flights. Error = how far the prediction for 1 s ahead is from where the drone really was (median):

| Model | Error 1 s ahead | Alarm: breaches caught |
|---|---|---|
| IMM (what real radars use) | 1.54 m | 99% |
| TabPFN A (learned from X2 only) | *needs token* | |
| TabPFN B (X2 + a few Crazyflie flights) | *needs token* | |

The IMM is a strong opponent: its 90% range is honest (89%), and its 2 s alarm catches 99% of breaches at 1 false alarm per 20 safe moments. `python drone/compare.py` scores every model on the same test rows.

## Try it

From the repo root, in the `priorlab` conda env. First get the drone models: see `models/README.md`.

```sh
python drone/view_drones.py                 # fly the two drones with the keyboard
python drone/intruder.py --view 8           # watch an intruder flight (any number)
python drone/intruder.py --drone cf2 --view 8
python drone/demo_imm.py --view radar       # a test flight: real vs IMM-predicted position, live
python drone/demo_imm.py --view radar --record data/demo_radar_54.mp4   # the same as a video  (needs: pip install imageio-ffmpeg)
```

Make the data again (it is not in git):

```sh
python drone/intruder.py --flights 2000              # X2 flights, 16 min
python drone/intruder.py --drone cf2 --flights 300   # Crazyflie flights, 15 min
python drone/features.py                             # tables, 30 s each
python drone/features.py --drone cf2
python drone/features.py --preview 2000              # a CSV you can open in Excel
python drone/imm.py --drone cf2                      # IMM predictions, 25 s
python drone/tabpfn_model.py                         # TabPFN predictions (needs TABPFN_TOKEN)
python drone/compare.py                              # the results table
```

## Files

| File | What it does |
|---|---|
| `sim.py` | the world: sky, floor, zone, radar |
| `flight.py` | flies a drone: "go this way" → motor commands |
| `intruder.py` | the random intruder, wind, radar, data recording |
| `features.py` | radar readings → tables |
| `imm.py` | the IMM tracker (the baseline to beat) |
| `tabpfn_model.py` | TabPFN: predicts positions with a range, and the alarm |
| `compare.py` | scores IMM and TabPFN on the same test rows |
| `demo_imm.py` | the 3D demo: real vs predicted position, alarm, video recording |
| `view_drones.py` | keyboard flying |
| `EXPERIMENTS.md` | full detail, results by manoeuvre, everything we tried |
