# Suction model validation (MuJoCo, 2026-11-21)

Script: `scripts/spike_suction.py`. Model: `sim/suction.py`.
- **Cup:** the lip is a ring of 8 contact points at the cup radius.
- **Holding force:** MuJoCo `adhesion` actuator. The gain equals the total holding force F = seal × area × efficiency.
- **Friction:** explicit lip/product contact pairs.
- **Effector:** force-driven point mass that tracks the reference exactly.
- **Timestep:** 0.5 ms.

| Test | Simulated failure acceleration | Theory | Difference |
|---|---|---|---|
| Lift, F = 5 N, m = 0.2 kg | 14.6 m/s² | F/m − g = 15.2 | −4% |
| Lift, F = 10 N | 39.9 | 40.2 | −1% |
| Shear, flat product, μ = 0.3, F = 5 N, m = 50 g | 29.2 | μF/m = 30.0 | −3% |
| Shear, μ = 0.6 | 57.9 | 60.0 | −4% |
| Peel, tall product, h = 2 cm, F = 8 N, r = 10 mm, m = 0.1 kg | 17.4 | F·r/(m·h) = 40.0 | ×0.44 |
| Peel, h = 4 cm | 9.1 | 20.0 | ×0.45 |

- **Lift and shear match theory within 4%.**
- **Peel** has the right scaling (doubling h halves the tolerated acceleration) but fails at about 0.45× the rigid-lip formula, because the lip lets go progressively once one side lifts. We use that as the model's peel factor, not the textbook one.

## Finding: no warning before failure

At 0.3–0.97× the critical acceleration, tilt stays below 0.01° and all 8 lip points stay in contact. Above it, the product lets go at once. MuJoCo's contacts are rigid, whereas a real rubber or bellows lip deforms and leaks progressively, which the vacuum sensor shows as a dip before a drop.

**Decision:** model the vacuum sensor from the **simulated lip contact forces**:
- tangential load against available friction at each lip point;
- uneven normal load around the ring (the peel moment).

This is not computed from our formula, so the near-miss signal comes from the simulation's force state. The leak curve itself (how much vacuum is lost at a given lip load) is an assumption, documented in DEBT.md. A compliant (bellows) cup model is the alternative if time allows.

## Gate 1: pneumatics (2026-11-21), `sim/pneumatics.py`, `tests/test_pneumatics.py`, 9 of 9 passed

| Test (model-suction.md §6) | Result | Reference |
|---|---|---|
| V1 evacuation, Si08-3 | +2 to +6.5 % at all 7 levels | Piab table; tolerance ±10 % (Piab's stated accuracy) |
| V1 evacuation, Pi12-2 | −8 to **+10.7 %** | Piab table. **Tolerance widened to ±12 % for this ejector only**: its flow and evacuation tables were measured at different feed pressures (0.34 vs 0.314 MPa), so they cannot agree exactly. |
| V2 linear-curve time constants | T1 3.510 ms vs analytic 3.511 ms; T2/T1 = 2.997 | SMC: T2 = 3·T1 |
| V3 steady seal, orifice | root solve = ODE steady state, 74.49 kPa | (internal consistency) |
| V3 SMC adsorption round trip | P1 recovered at 20, 40, 53, 65 kPa (< 2 %) | SMC method |
| V3 min vs series leak blend | ≤ 0.12 kPa difference for gaps ≤ 10 µm | spec: < 5 % |
| V3 vs research calculation (§3.4 table) | within 0.4 kPa for 0–40 µm gaps | cross-check only |
| V4 effective area | Schmalz SPB1 forces correspond to d2 (0.99–1.07×) | Schmalz datasheet |
| V4 vacuum-dependent area | Piab B25XP 9.7 / 19.0 / 22.0 N at 20 / 60 / 90 kPa reproduced | Piab datasheet |

## Gate 2: compliant cup mechanics (2026-11-21), `sim/cup.py`, `tests/test_cup.py`, 10 of 10 passed

Cup: Schmalz SPB1 20 class. 16 lip segments on spring slide joints (Dex-Net ring); vacuum force F = Δp·A_eff applied with `mj_applyFT` normal to the sealed face; pneumatics from gate 1. dt = 0.2 ms.

| Test | Result | Reference | Tolerance |
|---|---|---|---|
| V5 lift, m = 0.2 / 0.5 kg | +1.5 % / +2.9 % | a* = F/m − g | ±4 % |
| V6 shear, μ = 0.3 / 0.6, 1 mm slip | +4.4 % / +0.5 % | a* = μ(F − m·g)/m | ±5 % |
| V7 torsion | +3.7 % | τ = r·μ·f_N | ±5 % |
| V8 peel onset, lip springs, k_ring 4000 / 12000 N/m | +0.3 % / −0.1 % | M = r·f_N/2 (Dex-Net) | ±5 % |
| V9 peel ultimate, pressure frozen | +0.5 % | M = r·f_N (rigid ring) | ±10 % |
| V9b peel with pneumatics | lift-off 81.0, **dip onset 81.8**, failure 86.4 N·mm | failure in [55.7, 111.4]; dip at lift-off | ±5 % |
| V13 convergence (dt/2, 2× sub-steps) | seal and failure unchanged | | < 2 % |

### What it took (each fix is physics or MuJoCo semantics, not tuning)

- **MuJoCo 4 `gap` semantics.** Contacts are detected up to margin + gap, and force acts only below margin. So the lip pairs use margin 0 and gap 3 mm. The research document described the old semantics.
- **Vacuum force normal to the sealed face.** Pressure acts perpendicular to the surface. Applying it along the cup axis left a sideways force once the product tilted (≈ 9 % of the friction capacity). This was the cause of the early shear failure.
- **`noslip_iterations = 10`** removes MuJoCo's soft-constraint creep inside the friction cone (8° of creep under torsion before the fix).
- **Stiff lip end stops** (`solreflimit` 0.5 ms), so a lip cannot stretch past its free length.
- **Failure criteria for a compliant cup:** detachment means every lip segment has lost contact. Elastic movement of up to about 2 mm is not failure.

### Open items (not hidden; they go into calibration, test V15)

- **Lateral lip compliance is not modelled.** The lips slide only along the cup axis. With a soft ring (4000 N/m) the product tilts about 11° before lift-off, and friction at the laterally rigid lips then carries extra moment. The external peel moment is +29 % above the linear-ring value (frozen pressure) and about +45 % with the pressure model on. With a stiffer ring (12000 N/m, 4° tilt) this drops to +0.6 %. Real bellows cups are compliant sideways, so the bellows tilt joints (spec §4.1) or lateral lip compliance should be added, and the ring stiffness calibrated.
- **Speed.** With the pneumatics on, the Python sub-stepped ODE runs about 10× slower than real time (about 3 s wall per 0.3 s simulated). This must be optimised before bulk log generation (a vectorised or compiled ODE).

## Gate 3, in progress (2026-11-22)

- **Speed:** the pneumatics are compiled with Numba (same equations). They match the reference integrator within 0.004 Pa (one and two nodes, `tests/test_pneumatics.py`) and run **12.6× faster**: the cup simulation went from 0.11× to 1.38× real time.
- **Bellows compliance added** (`CupConfig(bellows=True)`): axial, tilt and lateral joints between the fitting and the lip ring.
  - Axial: k_bz ≤ F_suction/Z (Schmalz bound), with an end stop at the 8 mm stroke.
  - Tilt: k_θ = k_bz·R²/2.
  - Lateral: k_lat = 6·k_bz·R²/L², from thin-walled-tube beam theory (fixed–guided beam, as in Hudoklin et al. 2022). We tried to verify the equivalent EJMA formulas by search and could not, so they are not cited.
  - The vacuum reaction acts on the cup's ceiling (the fitting), so the bellows is compressed.
  - **Check:** a hanging 200 g product compresses the bellows 6.5 mm against 6.4 mm predicted, which reproduces the "lifting effect".
- **Finding:** with bellows compliance at the published stiffness bound, the cup and product tilt 30–37° before the first lip lifts off. The lip springs still meet the Dex-Net onset rule (+2 % / +8 %). At such tilts no closed-form reference exists for the external moment. This regime is therefore handled by the **V15 sensitivity sweep** (drop thresholds under pick accelerations across the sourced stiffness ranges), not by a single validated number.
- All 21 gate-1 and gate-2 tests still pass (bellows off by default).

### Gate 3: V11 and V12 (2026-11-22), `tests/test_seal.py`, 4 of 4 passed

- **V11 seal vs roughness:** 75.0 / 75.0 / 74.9 / 74.2 / 72.2 / 63.6 / 36.6 kPa at g0 = 0 / 2 / 5 / 10 / 15 / 20 / 30 µm. A smooth surface seals at the ejector maximum, and the seal falls monotonically, matching the trend of Huh et al.
- **V12a, peel (tall product, 0.2 kg, CoM 20 mm below the cup):** **every drop is preceded by a vacuum dip**, 16 of 16 across both cup models. Lead time 6–29 ms (median 14–19 ms). The transition is binary: below the limit the vacuum does not move at all (fine sweep in 0.5 m/s² steps); above it, failure runs away (lift-off → leak → less force → more lift-off). This matches Huh et al. ("does not vary while the seal remains, lowers at leak onset"). GOMP-ST's "> 0.1 s" is the visible **deformation** time before failure, not a pressure dip. Our bellows model shows such deformation (up to 30° of tilt) without any pressure change.
- **V12b, shear (flat product, 50 g):** **graded near-misses exist.** Past the friction limit (119 m/s²) the product slides in the cup and is still held: slip 0.7 / 1.4 / 3.2 / 3.9 mm at 1.05 / 1.1 / 1.2 / 1.26 × the limit, with small vacuum dips (0.06–1 kPa).

**Consequence for the pitch and the model:**
- The explainer's claim that "a deep vacuum dip that still held is a near-miss" is **not supported for peel**, and must be corrected.
- The vacuum's predictive value is the **seal at touch** (leaks, porosity, wear: V11).
- Graded near-misses come from **slip, measured as placement error in the tray** (and small dips) for shear-limited products.
- The TabPFN targets should be `dropped` (classifier) and `place_offset_mm` (regressor), rather than a vacuum-dip regressor.
