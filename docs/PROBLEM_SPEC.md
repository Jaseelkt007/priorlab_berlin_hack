# Problem spec: per-item motion settings for a Franka pick-and-place cell

Status: draft v1 (2026-11-21). The spec the simulation and the TabPFN models are built from. Change it here first, then in code.

## 1. The decision we make

A Franka arm moves items from a conveyor into a bin. A classical controller (motion planner + Franka's built-in controllers) does the moving. **Our job is one decision per item, taken after the grasp and a short test lift, before the transfer:**

> choose the **acceleration scaling** α and the **grip force** F for this item's transfer

**Objective:** finish as fast as possible (throughput), subject to:
- P(drop) ≤ 1%
- 90th-percentile slip ≤ 5 mm
- F ≤ the item's allowed grip force (fragility)

If no candidate meets the constraints, the item goes to a **manual lane**.

**Why grip force is not simply always "max":** high force crushes soft or fragile packaging and wears the pads faster. Real item master data carries a fragility limit, which makes the trade-off real.

**Cycle time is computed, not predicted.** For a given α, the transfer time follows from the motion profile. We only predict what is uncertain.

## 2. Real Franka numbers the simulation must respect

Values are for Franka Emika Panda / FR3; **check them against the datasheet of the exact robot we use.**

| Quantity | Value | Used for |
|---|---|---|
| Payload | 3 kg | item mass range (keep items ≤ 2 kg with the hand attached) |
| Reach | 855 mm | cell layout |
| Cartesian limits | v 1.7 m/s · a 13 m/s² · jerk 6500 m/s³ | α scales these (α = 1 means the limit) |
| Joint torque sensors | all 7 joints, 1 kHz | test-lift weight and centre-of-mass estimate |
| External wrench estimate | `O_F_ext_hat_K` in libfranka `RobotState` (6D force/torque at the hand) | `est_mass_kg`, `com_offset_mm` |
| Franka Hand stroke | 0–80 mm (2 × 40 mm) | **items must be < ~75 mm wide** on the gripping axis |
| Franka Hand force | 70 N continuous, 140 N max | grip force range 20–70 N |
| Gripper API | `grasp(width, speed, force, ε_in, ε_out)`; state has `width`, `is_grasped` | grasp, width readout, drop detection |
| Safety reflexes | collision thresholds (`setCollisionBehavior`) stop the arm on large unexpected forces | optional output: reflex stop |

## 3. Physics that creates the problem

A parallel grip holds by friction. A translational slip starts when

    m · |g + a(t)|  >  2 · μ · F

and a rotational slip (the item pivots between the fingers) starts when the gravity torque from an off-centre centre of mass exceeds the torsional friction of the small contact patches.

- **Visible before the transfer:** size (camera); mass and centre-of-mass offset (test lift, via the wrench estimate).
- **Hidden:** friction μ, which depends on packaging material, finger-pad wear and dust.

The hidden part is why a fixed rule or a policy trained on assumed values fails, and why learning from the log helps.

## 4. Columns

### Inputs (known before the transfer)

| # | Column | Unit | Source on the real robot | Source in MuJoCo |
|---|---|---|---|---|
| 1 | `width_mm` | mm | gripper width after `grasp` | finger joint positions |
| 2 | `height_mm` | mm | camera / item master data | geom size |
| 3 | `est_mass_kg` | kg | test lift: F_z of `O_F_ext_hat_K` / g | F/T sensor at the hand + noise (σ ≈ 1 N) |
| 4 | `com_offset_mm` | mm | test lift: ‖(T_x, T_y)‖ / (m·g) | same sensor |
| 5 | `probe_slip_mm` | mm | item shift during the 2 cm test lift (camera; optional) | box pose relative to the hand |
| 6 | `packaging` | category | item master data (cardboard / glossy / polybag) | maps to a μ range |
| 7 | `lifts_since_pad_change` | count | cell counter | drives the pad-wear model |
| 8 | `max_grip_N` | N | item master data (fragility) | sampled per item |
| 9 | `alpha` | 0.1–0.8 | **action**: scales velocity, acceleration and jerk limits | ruckig trajectory generator (α = 0.8 already reaches the 13 m/s² Cartesian limit) |
| 10 | `grip_N` | 20–70 | **action**: grasp force | gripper actuator force limit |

**Optional physics hint (ablation):** `load_ratio = est_mass · (9.81 + alpha · 13) / (2 · grip_N)`. The friction coefficient is unknown, so this is a hint, not the answer. Test with and without.

### Outputs (measured after the transfer)

| Column | Type | TabPFN model | Real robot | MuJoCo |
|---|---|---|---|---|
| `dropped` | 0/1 | **classifier** → P(drop) | gripper width collapses to ~0, or the wrench estimate loses the item's weight; camera at the bin | box–finger contact lost and box below the hand |
| `slip_max_mm` | mm | **regressor** → quantiles (q50, q90) | camera tracking the item relative to the fingers, or tactile pads (add-on) | max displacement of the box in the hand frame |
| `tilt_max_deg` | deg | regressor (optional) | camera | max rotation of the box in the hand frame |
| `place_err_mm` | mm | regressor (optional) | camera over the bin | box pose vs. target |

Slip is a **near-miss signal**: it grows before drops start, which lets the planner slow down before the first failure.

## 5. Rows: how many, and where from

| Set | Rows | How |
|---|---|---|
| **Calibration context** (start of a shift / new cell) | 200–500 | varied α, F and items, sampled densely near the slip boundary. Sim: minutes. Real: ~500 lifts × ~6 s ≈ 1 h |
| **Online context** | +1 per item | sliding window of the last 500–2,000 rows so drift (pad wear) shows up |
| **Drop rows in context** | ≥ 30 | the classifier needs positives; calibration aims for them on purpose |
| **Evaluation** | ≥ 1,000 items per condition × 5 seeds | sim only; held-out masses, packaging and wear levels |
| **RL baseline training** | millions of sim steps | MJX/JAX on GPU, randomized mass, friction and centre of mass |

**Learning-curve experiment:** decision quality at context sizes 25, 50, 100, 200, 500, 1,000 and 2,000. This shows how few real lifts are needed.

TabPFN-3.5 limits (1M rows, thousands of features) are far above all of these.

## 6. One decision, at run time

1. Grasp with the default F₀, then do the test lift (2 cm, hold 0.3 s) → inputs 1–8.
2. Candidate actions: α ∈ {0.2, 0.3, …, 1.0} × F ∈ {20, 30, …, 70} with F ≤ `max_grip_N`, giving at most 54 rows.
3. One TabPFN classifier call gives P(drop); one regressor call gives slip quantiles. Both are batched with a cached context and must fit inside a ~5 s cycle.
4. Pick the fastest α among the safe candidates, or send the item to the manual lane.
5. Execute, measure the outputs, and append the row.

## 7. Hidden variables in simulation (never columns)

| Hidden variable | Range | Which approach it hurts |
|---|---|---|
| Friction μ per packaging | cardboard 0.5–0.7 · glossy 0.25–0.4 · polybag 0.3–0.6 (to calibrate) | any fixed rule |
| Pad wear | μ multiplier falls from 1.0 to ~0.6 over N lifts | RL trained with fresh pads |
| Contents shifting (liquids, loose parts) | centre-of-mass jitter during the transfer | rules using the static centre of mass |
| Estimate noise | mass ±0.1 kg, centre of mass ±5 mm | everyone, honestly |

## 8. Baselines and metrics

- **Baselines:**
  - fixed safe settings (α = 0.3, F = 70)
  - fixed fast settings (α = 1.0, F = 40)
  - analytic rule using an assumed μ
  - PPO with the same inputs, trained with randomized mass, μ and centre of mass (the strong baseline)
  - MLP / XGBoost world model on the same rows
- **Metrics:** throughput (items/h), drop rate, near-miss rate (slip > 5 mm), calibration (coverage of the q90 slip band), decision latency.
- **Key scenarios:**
  - **in-distribution:** RL should do well here
  - **unseen packaging μ**
  - **pad wear partway through the run:** weight readings look normal, the grip weakens; the scenario RL handles worst

## 9. MuJoCo build notes and risks

- **Model:** MuJoCo Menagerie `franka_emika_panda` (arm + hand). Grip force = the hand actuator's force limit, with the width command set to closed.
- **Motion:** IK (e.g. the `mink` library or damped least squares) plus a trapezoidal Cartesian profile with peak acceleration α · 13 m/s², tracked by the joint position actuators.
- **Contacts:** finger pads with `condim="4"` or `"6"` (torsional friction; with condim 3 the item pivots freely). Tune `solref`/`solimp`; consider `noslip_iterations`.
- **Biggest risk: soft-contact creep.** MuJoCo can show slow slip even under a grip that should hold. **First spike (≤ 2 h):** a box held still must not creep, and it must start slipping close to m·(g + a) = 2μF. If that fails, fall back to the pushing scenario, which needs only sliding friction.
- **Measurements:** box pose in the hand frame each step (slip, tilt); finger–box contacts (drop); F/T sensor at the hand site (test-lift wrench, plus noise to mimic the real estimate).

## 10. Validated in simulation (2026-11-21)

Scripts: `scripts/spike_track.py`, `scripts/spike_grip.py`, `scripts/spike_dynamic.py`.

- **Arm tracking:**
  - α 0.3–0.8: max joint error ≤ 1°, no torque saturation; peak TCP acceleration 5.2 / 10.8 / 12.7 m/s² at α 0.3 / 0.7 / 0.8.
  - α ≥ 0.9: exceeds the 13 m/s² Cartesian limit and tracking breaks down, as a real Franka would abort.
  - **Usable action range: α ∈ [0.1, 0.8].**
- **Static grip:** the item drops at 0.95× the theoretical critical squeeze force (m·g / 2μ) and holds at 1.05×. The measured pad normal force equals the commanded force. Creep while holding: about 0.025 mm per 2 s, negligible. No-slip post-processing is not needed.
- **Dynamic grip** (1 kg item, μ 0.5, 15 mm CoM offset):
  - slip and tilt grow with α at a fixed grip;
  - at 1.4× critical force the item holds at α 0.2–0.5 and drops at α 0.8; at 3× it holds everywhere;
  - **rotation about the grip axis (tilt) from the CoM offset is the dominant failure mode**, as with real parallel grippers.
- **Test-lift estimates:** mass within ±0.15 kg, CoM offset ±10 mm with Franka-like noise.
- **Throughput trade-off:** transfer takes 6.6 / 2.9 / 2.0 s at α 0.2 / 0.5 / 0.8.
- **Speed:** about 0.7 s wall time per episode on one core, so 500 rows take about 6 min.
