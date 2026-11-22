# Fly flight data for TabPFN: collection plan

We run flybody's pretrained flight policy on every recorded real-fly flight in MuJoCo and log every control step. The tables TabPFN sees are built later from these logs.

- **Source:** TuragaLab/flybody (Vaxenburg et al., Nature 2025), commit `d015e9b`. The policy is `data/policies/flight`, a DMPO actor, 104 inputs → 12 actions, about 165k parameters. The reference flights are `data/flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5`.
- **Reference flights:** 272 in total, 264,120 steps (52.8 s of flight). They are 92 evasions plus 92 mirrored copies, and 44 saccades plus 44 mirrored copies. They run 269 to 2,881 steps, with a median of 886.
- **Timing:** one control step is 0.2 ms (physics runs at 0.05 ms). One wingbeat at about 218 Hz takes about 23 steps.
- **Script:** `collect/collect_flights.py` runs in the `flybody` conda env on CPU. There is one process per core and one env per process.

## Passes

Every flight runs from its first step to its last, without a random start, so the same flight lines up across passes. The episode only ends early if the fly drops below 0.2 cm or the simulation blows up. Distance to the ghost does not end it.

| Pass | File | Change | Used for |
|---|---|---|---|
| 1 | `pass1_clean` | none; the policy's mean action is applied | forward model (where it goes next), copying the policy offline |
| 2 | `pass2_noise_a` | Gaussian noise on the applied action, 0.5 × the policy's own std, seed 1 | closed-loop copying: shows how to recover from small mistakes |
| 3 | `pass3_noise_b` | same, seed 2 | same |
| 4 | `pass4_weak_wing` | left wing yaw, roll and pitch actuator gains × 0.7 | adaptation with no retraining |
| 5 | `pass5_heavy` | all body masses × 1.2 | adaptation with no retraining |
| 6 | `pass6_noise_strong` | Gaussian noise on the applied action, 2.0 × the policy's own std, seed 3 | closed-loop copying: the fly drifts 1–11 mm from the ghost (21% of steps beyond 1 mm, vs 3% in passes 2–3), so the context shows recoveries from bigger mistakes. 2.5× crashes most flights. |

**Re-collected 21 Nov 2026 (evening).** The first collection had a bug: flybody's `before_step` adds the wingbeat pattern to the six wing actions *in place*, and the collector stored the arrays after `env.step()`. So passes 1, 4 and 5 stored policy + pattern as `action_mean`, and passes 2–3 stored it in `action_applied`. The collector now passes a copy to `env.step()`. The buggy files have been deleted.

The exact settings for each pass are written to `meta.json`.

## Files (`data/flights/`, gitignored)

```
pass<k>_<name>.npz            full rate: one row per 0.2 ms control step (~264k rows)
pass<k>_<name>_wingbeat.npz   one row per wingbeat, every 23rd step (~11k rows); the copy to share
meta.json                     column names, units, pass settings, versions
```

Each `.npz` holds named float32 (or int32) arrays that all have the same number of rows N. Load one with `np.load(path)`. `meta.json` names every column.

## Columns: one row = one control step t

The state and observation are taken **before** the action at step t. The action is the one taken at step t. The reward and the wing target belong to the step from t to t+1.

### `ids` (int32, 6)
| Column | Meaning |
|---|---|
| `flight` | reference flight index, 0–271 |
| `pair` | index of the original flight. The mirrored copy shares it. **Split train/test on this.** |
| `type` | 0 = evasion, 1 = saccade |
| `mirrored` | 0 = original, 1 = mirrored |
| `step` | step number within the flight, starting at 0 |
| `pass` | 1–5 |

### `obs` (104): exactly the policy's input, in its key order
| Block | Size | Content |
|---|---|---|
| `accelerometer` | 3 | body acceleration |
| `actuator_activation` | 0 | empty in this task |
| `gyro` | 3 | body angular velocity |
| `joints_pos` | 25 | head, wing and abdomen joint angles |
| `joints_vel` | 25 | joint velocities |
| `ref_displacement` | 6×3 | ghost position relative to the fly for the current step and the next 5 (the policy sees 1 ms ahead) |
| `ref_root_quat` | 6×4 | ghost orientation relative to the fly for the same steps |
| `velocimeter` | 3 | body velocity in the body frame |
| `world_zaxis` | 3 | the "up" direction as seen by the body |

### `action_mean`, `action_std`, `action_applied` (12 each, env units)
The 12 actions are head (3), wings (6: yaw, roll and pitch for each side), abdomen (2) and beat frequency (1). `action_mean` is the policy's clean action and is **the target for copying the policy**. `action_std` is the policy's own uncertainty, converted to env units. `action_applied` is what was actually sent to the sim. It equals the mean except in the noise passes.

### `state` (33): the true body state in world coordinates; targets for "where will it be"
| Column | Size | Meaning |
|---|---|---|
| `com_pos` | 3 | fly centre of mass (cm) |
| `root_quat` | 4 | thorax orientation (w, x, y, z) |
| `lin_vel` | 3 | thorax linear velocity, world frame (cm/s) |
| `ang_vel` | 3 | thorax angular velocity, thorax frame (rad/s) |
| `wing_qpos` | 6 | wing joint angles |
| `wing_qvel` | 6 | wing joint velocities |
| `wpg_target` | 6 | the wing angles the pattern generator drives toward during this step. Wingbeat phase is derived from these. |
| `beat_freq` | 1 | current wingbeat frequency (Hz) |
| `t` | 1 | simulated time (s) |

### `ref` (7): the ghost (the real fly's recorded flight)
`ref_root_pos` (3) and `ref_quat` (4) give the reference pose at step t.

### `outcome` (2)
`reward` is the task reward for the step. `dist_to_ghost` is the fly–ghost distance in mm.

## Built at table time, not stored
- **Future targets.** Δ position or heading 1, 3 or 10 wingbeats ahead are taken from later rows of the same flight. Rows without that much future left are dropped.
- **Wingbeat phase** (0 to 2π), computed from `wpg_target`.
- **Train/test split by `pair`.** 20% of the originals plus their mirrored copies form the test set, stratified by `type`.

## Sizes and where the data goes
- **Full-rate passes:** about 264k rows × about 200 values, roughly 200 MB raw per pass (smaller compressed). These stay on the collecting PC.
- **Wingbeat files:** about 11k rows, a few MB per pass. These go to the GPU machines (RTX 4050 6 GB for the main TabPFN runs, GTX 1650 4 GB for the second) over the LAN, a drive or USB.
- **Context size for TabPFN Fast:** an estimated 10–30k rows on the 4050 and 5–10k on the 1650. Measure before relying on these.

## Results so far

| Pass | Rows (full / wingbeat) | Time on 8 workers | Distance to ghost: mean / p95 / max | Ended early |
|---|---|---|---|---|
| 1 clean | 260,006 / 11,431 | 11 min | 0.33 / 0.78 / 7.5 mm | flights 100 and 140 |

- **Size:** 175 MB on disk (compressed) for the full file and 7.6 MB for the wingbeat file. Throughput is about 400 steps/s in total, below the planned ~1,000 because these 8 logical cores share 4 physical ones and every step is logged.
- **Early endings are real failures and are kept in the data.** Flights 100 and 140 are both mirrored evasions. The fly sank below flybody's 2 mm minimum height (after 1,677 of 1,700 steps and 416 of 2,875 steps). Their originals, flights 8 and 48, fly the whole path. They are listed under `ended_early` in `meta.json`.
- **Checks passed:** column counts match `meta.json`, every value is finite, steps are contiguous within each flight, mirrored copies match their originals, quaternions are unit length, the timestep is exactly 0.2 ms and the beat frequency is 216.6–219.5 Hz.
