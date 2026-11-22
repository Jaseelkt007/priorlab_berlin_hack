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
| Alarm positives (breach within 2 s) | 5.3% | 2.0% |

Crashed flights are dropped. Crazyflie flights use different random seeds, so they never repeat an X2 flight's plan.

## 2. The table: every input and output

Each radar reading becomes **one row**. A row answers: *"given what the radar has seen up to now, where will the drone be 1 s from now, and will it enter the zone within 2 s?"*

**The example row** used below is a real one: small-drone test flight #100000 at 10.2 s, in the middle of a zigzag, 48 m from the zone. The radar had just lost it for most of the last second (7 of 10 readings missing).

### How the radar dots become numbers

The radar gives only a dot (x, y, z) every 0.1 s, noisy and sometimes missing. From the recent dots we estimate:
- **position and velocity**: a straight line through the last 1.5 s of dots (why: experiment 4);
- **acceleration**: a curve through the last 2 s of dots;
- **heading**: the direction the drone is flying.

Many inputs are measured **from the drone's point of view**: *forward* = the way it is flying, *left* = to its left, *up* = up. Then an attack from the north and one from the south look the same, so a model learns from fewer examples.

### Inputs (16)

| # | Name | What it means | How it is computed | Unit | Example | Why it helps |
|---|---|---|---|---|---|---|
| 1 | `range` | how far the drone is from the zone centre, along the ground | distance from (0, 0) to the estimated position | m | 47.8 | close drones matter more and are measured more precisely |
| 2 | `height` | how high it flies | estimated z | m | 20.6 | the zone is 30 m high; above it the drone cannot breach |
| 3 | `zone_fwd` | how far ahead of the drone the zone centre is | zone centre in the drone's view, forward part | m | 46.7 | positive and large = flying toward the zone |
| 4 | `zone_left` | how far to the drone's left the zone centre is | zone centre in the drone's view, sideways part | m | -10.1 (to its right) | near zero = aimed straight at the zone; large = passing by |
| 5 | `speed` | how fast it flies over the ground | size of the horizontal velocity | m/s | 2.66 | the main driver of where it will be |
| 6 | `climb` | how fast it goes up (+) or down (-) | vertical velocity | m/s | +1.73 | dives and climbs |
| 7 | `acc_fwd` | is it speeding up (+) or slowing down (-) | acceleration along its heading | m/s² | +1.94 | a dash starts with this |
| 8 | `acc_left` | is it being pushed sideways, i.e. turning | acceleration to its left | m/s² | +0.26 | the start of a turn or zigzag leg |
| 9 | `acc_up` | is its climb speeding up or slowing down | vertical acceleration | m/s² | -1.82 | a climb or dive about to stop |
| 10 | `turn_rate` | how fast its heading is turning | sideways acceleration ÷ speed | rad/s | 0.10 | circling and zigzags |
| 11 | `speed_1s` | its speed 1 s ago | `speed` of the row 1 s earlier | m/s | 1.45 | speed now vs then = speeding up or not |
| 12 | `turn_rate_1s` | its turn rate 1 s ago | `turn_rate` 1 s earlier | rad/s | 0.03 | |
| 13 | `climb_1s` | its climb 1 s ago | `climb` 1 s earlier | m/s | 3.02 | climb is fading (3.0 → 1.7) |
| 14 | `turn_rate_2s` | its turn rate 2 s ago | `turn_rate` 2 s earlier | rad/s | -1.01 | a turn the other way 2 s ago = a zigzag rhythm |
| 15 | `n_missing` | how many of the last 10 radar readings were lost | count of gaps in the last 1 s | 0-10 | 7 | tells the model how much to trust the other numbers |
| 16 | `fit_error` | how noisy the recent dots are | spread of the dots around the fitted line | m | 0.72 | noise grows with distance |

Inputs 11-14 are empty (NaN) in the first seconds of a track, when there is no "1 s ago" yet. TabPFN handles empty cells.

**Not an input: time.** Every flight starts at the arena edge, so "seconds since the flight started" would let a model cheat (it would learn the script, not the motion), and a real radar does not know when a flight started.

**Not an input: anything true.** The true position, true speed, the manoeuvre name and the intent are known in the simulation but never given to a model. They are only used for scoring.

### Outputs

| Name | What it means | Unit | Example (real) |
|---|---|---|---|
| `x_1s`, `y_1s`, `z_1s` | where the drone really was 1 s later | m | 19.6, -42.6, 22.3 |
| `breach_2s` | 1 if it entered the zone within 2 s, else 0 (the alarm) | 0 / 1 | 0 |

In the example the drone was at (19.9, -43.4, 20.6) and still climbing, so 1 s later it is 1.7 m higher and has barely moved sideways.

**What a model gives back:** for each position, a best guess plus a 5%-95% range (the cone); for the alarm, a probability (e.g. "8% chance of a breach within 2 s").

**How models learn the positions:** as movement from the current estimate in the drone's view (forward, left, up), e.g. here 0.89 m forward, 0.23 m right, 1.62 m up after 1 s. That is easier to learn; it converts back to x, y, z exactly (`to_world`), and all scoring is done in x, y, z.

### Rules

- **Split by flight, never by row:** rows of one flight are near copies, so mixing them between learning and testing would leak answers.
- **Learn on the big drone, test on the small one.**
- To look at the table yourself: `python drone/features.py --drone cf2 --preview 2000` writes a CSV with named columns.

## 3. Results

We predict **one thing: where the drone will be 1 s from now (x, y, z)**, plus the alarm (experiment 9 says why).

**Test set:** the 250 Crazyflie flights after the first 50 (those 50 are kept for TabPFN B's examples and the IMM's fairness tuning). Every radar reading with a known future: 112,501 moments. Every model is scored on exactly these moments (`python drone/report.py`).

| Model | Error 1 s ahead (median) | Truth inside its 90% range | Alarm AUC | Breaches caught (1 false alarm per 20) |
|---|---|---|---|---|
| IMM (tuned on X2 flights) | 1.54 m | 89% | 0.991 | 99% |
| IMM re-tuned on Crazyflie flights 0-49 (experiment 12) | 1.50 m | 82% | 0.987 | 98% |
| TabPFN A (learned from X2 only) | 1.70 m | 93% | 0.989 | 95% |
| **TabPFN B (X2 + rows from 50 other Crazyflie flights)** | **1.37 m** | 93% | 0.993 | 98% |

TabPFN B against the re-tuned IMM: **8% better** (95% range over flights: 7.1-9.5%), better on 208 of 250 flights. On a new pilot (experiment 14): 1.67 vs 1.80 m, still 8% better. Full analysis: `results/README.md`.

**Error** = distance between the predicted and the real position. **Range:** how often the real position is inside the 5%-95% range, per direction (should be 90%): the IMM's range is honest.

**Alarm** ("will it enter the zone within 2 s?"): AUC (1.0 = perfect, 0.5 = coin flip) and the share of breaches caught when we accept 1 false alarm per 20 safe moments. The alarm is nearly solved by plain physics: 2 s ahead, a slow drone cannot surprise anyone much. TabPFN can at best match it here.

**Where the IMM goes wrong:** in single flights it is often off right after the drone changes what it is doing (dive → hover → climb). Over all test rows, though, rows with a manoeuvre change in the next second are *not* worse (1.48 vs 1.54 m): many changes barely change the motion within 2 s (e.g. straight → zigzag). So TabPFN's chance is not one obvious weak spot; it has to be a little better almost everywhere.

**See it:** `python drone/demo_imm.py --view radar` plays a test flight in 3D with the real drone, the IMM's prediction and its error on screen (experiment 10).

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
- **What:** the IMM has five settings (how jumpy each tracker expects the drone to be, how often it switches). We searched them on 150 big-drone flights for the smallest error. It never sees the small-drone test flights.
- **What happened:** the first search (for 3 s ahead) ended with several best values at the edge of the range we tried, so we searched again past those edges.
- **Result:** error 3 s ahead on the big drone went 9.7 → 8.0 → 7.7 m; the last widening only gained 0.3 m, so the settings were close to the best.

### 8. Shorter prediction times
- **Why:** predicting 3-5 s ahead means guessing a drone's next decision, which no method can do well; a defence system acts on the next 1-2 s. So we switched from 1 / 3 / 5 s to **1 / 1.5 / 2 s**, and the alarm from "within 5 s" to "**within 2 s**".
- **Changed:** rebuilt both tables; re-tuned the IMM for 1.5 s ahead, to be fair to it.
- **Result:** re-tuning barely changed it (big drone, 1.5 s: 3.30 → 3.28 m), so the earlier settings already suited short times. On the small drone: 1.55 / 2.19 / 2.88 m.

### 9. One answer: where is it in 1 second?
- **Why:** three horizons (1, 1.5, 2 s) made results and demo harder to read. One clear question: *where will the drone be in 1 second?* The radar-only IMM answers it first; TabPFN then answers the same question from the table.
- **Changed:** the table's answer is now just `x_1s, y_1s, z_1s` (plus the 2 s alarm). The IMM was re-tuned for 1 s ahead.
- **Result:** re-tuning picked the same settings (big drone 2.16 m), so the IMM stays as it was. Small drone: **1.54 m**.

### 10. Seeing it: the 3D demo
- **What:** `demo_imm.py` replays a small-drone test flight in MuJoCo with the real drone (green), the radar dots (white), the IMM's guess for 1 s ahead with its 90% range (blue), and the guess it made 1 s ago for *now* (yellow; the yellow line is the error). Real and predicted x, y, z and the alarm are written on screen. Two views: `chase` (behind the drone) and `radar` (from the protected site). `--record file.mp4` saves a video.
- **What happened:** flying the drone live inside the viewer gave a slightly different flight from the recorded one: the viewer touches the force we use for wind. The predictions then did not line up.
- **Changed:** the flight is computed first, exactly as in the dataset, then played back. Checked: all 216 radar readings of flight 54 match the recording.
- **Result:** flight 54 (an attack): the IMM is about 1.5-2 m off 1 s ahead (1.68 m median) and the alarm comes on 2.4 s before the breach.

### 11. TabPFN against the IMM
- **What:** TabPFN v3.5 (package 10.2.0), no training. Context: 4,000 random X2 rows (A), plus 1,000 rows from 50 Crazyflie flights that are never tested on (B). One regressor per movement (forward, left, up) with 5/50/95% quantiles, one classifier for the alarm.
- **What happened:** on the laptop GPU (GTX 1650, 4 GB) it ran out of memory and took 3.5 min per model for a tenth of the data. Moved to an H100: the full run takes about 1 minute.
- **Result:** A 1.70 m (10% worse than the IMM), **B 1.37 m (11% better)**; B is closer than the IMM at 62% of moments, better on straight, zigzag, orbit, hover and dash, slightly worse on dive and climb. Its 90% range holds the truth 93% of the time. Alarm: about a tie.
- **Not yet fair in one respect:** B saw Crazyflie data, the IMM was tuned only on X2. Experiment 12 fixes that.

### 12. Fairness check: the IMM gets the same Crazyflie data
- **What:** re-tuned the IMM's 5 settings on Crazyflie flights 0-49, the flights TabPFN B takes its examples from (`imm.py --tune --drone cf2`). The first search ended on the edge of the grid for 3 settings, so the grid was widened and searched again.
- **Result:** settings `q_cv 0.3, q_ca 0.0001, q_ct 0.001, q_turn 0.0001, p_stay 0.9997`. Test error **1.54 → 1.50 m**. TabPFN B (1.37 m) is still **8% better**; over 1,000 resamples of the 250 flights the gain stays between 7.1% and 9.5%, and B is better on 208 of 250 flights.
- **Side effect:** the re-tuned IMM is overconfident (its 90% range holds the truth 82% of the time) and its alarm warns a little later (3.5 vs 3.9 s).

### 13. How many examples of the new drone does TabPFN need?
- **What:** TabPFN B with 0, 100, 250, 500, 1,000 and 2,000 Crazyflie rows in its context, the same 4,000 X2 rows, every 5th test reading (22,501 moments).
- **Result:** 1.70, 1.56, **1.45**, 1.40, 1.37, 1.37 m. From **250 rows (about 25 s of radar)** it beats both IMMs (1.54 and 1.50 m); it stops improving at about 1,000.
- **Careful:** the rows are scattered over 50 flights; 25 s of one continuous track has less variety.

### 14. A pilot no model has seen
- **Why:** every flight so far was flown by the same scripted pilot, so TabPFN could have learned that pilot's habits rather than how drones move.
- **What:** pilot b (`intruder.py --pilot b`): other manoeuvre mix (more zigzag, dash, dive, climb), switches about twice as often, sharper zigzags, bigger height changes. 250 new Crazyflie flights. Nothing re-tuned: TabPFN keeps its pilot-a examples, the IMMs their settings.
- **Result:** everyone gets worse (harder pilot). IMM 1.80 m, re-tuned IMM 1.82 m (its tuning fitted pilot a), TabPFN A 2.25 m, **TabPFN B 1.67 m: still 8% better** (6.6-8.6% over resamples, better on 217 of 250 flights); its 90% range holds the truth 90% of the time. Alarm: IMM 100%, TabPFN B 96% of breaches caught.
- **So:** the examples of the **drone** matter most; TabPFN's gain is not only the pilot's habits.

### 15. IMM + TabPFN: TabPFN corrects the radar's tracker
- **Idea:** keep the IMM a radar already runs, and let TabPFN correct it. The IMM's answer at each reading goes into the row as 6 extra inputs: its predicted move in 1 s (forward, left, up, in the drone frame) and its 3 model probabilities (straight / accelerating / turning). TabPFN learns from the examples when the IMM is right and how it is usually wrong. Still no training.
- **Not cheating:** the IMM only uses readings up to now, and its settings were tuned on other flights (X2). It is the same information TabPFN already has, worked through physics first.
- **How:** `imm.py --drone <x2|cf2|cf2_pilotb>` (now also saves the model probabilities), then `tabpfn_model.py --versions B --imm-features --test cf2_pilotb --every 1 --out data/tabpfn_cf2_pilotb_combo.npz`. Same examples as TabPFN B; tested only on **pilot b**, the pilot no model has seen.
- **Result:** IMM 1.80 m, TabPFN B 1.67 m, **IMM + TabPFN 1.62 m**: **10% better than the IMM** (9.2-10.9% over flight resamples, better on 232 of 250 flights), **2.7% better than TabPFN B** (2.2-3.1%). Better or equal on every manoeuvre except climb (2.06 vs the IMM's 1.96 m); on dive it now ties the IMM. 90% range holds the truth 89% of the time; alarm 97% of breaches caught.
- **So:** the gain over TabPFN alone is small but real; the strongest claim is against the IMM: "TabPFN upgrades the tracker you already run by 10%, on a pilot it never saw".

### 16. Is it fast enough for live use?
- **Budget:** the radar gives a reading every 100 ms; all 4 TabPFN models (forward, left, up, alarm) must answer within that. Measured with `python drone/latency.py` on an H100, IMM + TabPFN, one reading at a time (and 10 drones at once).
- **As used so far: 5.2 s per reading.** In batch it looked fast (about 0.5 ms per row), but every call re-reads all example rows, for each of the 4 models.
- **Speed-ups (error on pilot b, every 10th reading):**

| Setting | Delay per reading | Error |
|---|---|---|
| as used so far (5,000 examples) | 5,166 ms | 1.62 m |
| + example cache (`fit_mode="fit_with_cache"`) | 620 ms | 1.62 m |
| + 2 internal passes (`n_estimators=2`) | 320 ms | 1.62 m |
| + 1 pass | 256 ms | 1.63 m |
| + 1 pass, 1,000 X2 + 1,000 Crazyflie examples | **252 ms** | **1.56 m** |

- **So:** 20x faster with no accuracy lost. Fewer X2 examples even helped (the Crazyflie rows are then half of the examples); that was measured on pilot b only. Still **not under 100 ms**: what is left is about 60 ms of fixed cost per model call, 4 calls per reading, and 10 drones cost the same as 1.
- **Live design that fits today:** the IMM tracks every 100 ms (microseconds), and TabPFN updates its correction every 0.3 s (it needs 252 ms). Measured in experiment 17. Untested ways to reach 100 ms: run the 4 models in parallel (separate GPU streams or GPUs), or one model for all 3 directions.

### 17. Live replay: the IMM every 0.1 s, TabPFN every 0.3 s
- **Question:** experiment 15 had a TabPFN answer at every reading, which takes 252 ms at best. What is left when the system runs live?
- **How:** the fast setting from experiment 16 (example cache, 1 pass, 1,000 X2 + 1,000 Crazyflie examples) for every pilot-b reading: `tabpfn_model.py --versions B --imm-features --fast --context 1000 --adapt-rows 1000 --test cf2_pilotb --every 1 --out data/tabpfn_cf2_pilotb_combo_fast.npz` (H100, about 1 minute). Then `compare.realtime_predictions` replays it as a live system: the IMM answers at every reading; every 0.3 s TabPFN starts on the newest reading and its answer arrives 0.3 s later; until the next one arrives, its correction (TabPFN's 1 s prediction minus the IMM's, for the reading it started from) is added to the IMM's fresh prediction. Scored by `report.py` (`results/realtime.csv`).
- **Result (pilot b, 116,914 moments):**

| Setup | Error 1 s ahead |
|---|---|
| IMM alone | 1.80 m |
| IMM + TabPFN at every reading, slow (5.2 s each, not live) | 1.62 m |
| IMM + TabPFN at every reading, fast (252 ms each, not live) | 1.56 m |
| **Live: IMM every 0.1 s + TabPFN correction every 0.3 s** | **1.72 m: 4% better than the IMM** (3.6-5.0%, better on 201 of 250 flights) |

- **So:** it works live, but the 0.3-0.6 s old correction keeps only about a third of the gain (13% instant, 4% live). The limit is speed, not accuracy. Tried and not kept: fewer examples (200) were no faster (about 250 ms is a fixed cost per call) and less accurate.

### 18. Pitch videos
- `pitch_video.py` replays a flight with the saved predictions (the same files the results come from): the real path (green), one model's predicted path, its guess for now made 1 s ago, its guess for 1 s ahead, the numbers and the alarm; or several models in one video, one colour each. `make_videos.sh` makes every video in `results/videos/` and `results/compare.gif`; the list is in `results/README.md`.

## 5. Known limits

- All data is simulated; no real radar or drone.
- Scripted pilots only (two styles, experiment 14); no human pilots.
- Close range: intruders start 60 m out; a real counter-drone radar sees 1-5 km.
- The examples' answers are perfect: "where the drone really was 1 s later" comes from the simulator. In real use it would come from radar tracks of past flights smoothed afterwards (readings before and after each point), which is noisier, so real accuracy would be somewhat lower. The live drone's future is never shown to any model.
- The radar gives x, y, z with simple noise; a real radar measures distance and angles.
- Live, TabPFN answers in 252 ms on an H100, so its correction is 0.3-0.6 s old when used (experiment 17).
- The Crazyflie is limited to 3 m/s by its weak turning motors (model as menagerie ships it).
- At 3 m/s, Crazyflie flybys rarely leave the arena in 60 s, so they end as timeouts.
