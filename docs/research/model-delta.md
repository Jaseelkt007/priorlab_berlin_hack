# Delta robot model for the pick cell (phase 0c/0d research)

Researched 2026-11-21 for MuJoCo 4.0.1 (Python). Scope: how to model an ABB IRB 360 FlexPicker-class delta robot in MuJoCo, with real geometry and limits, how real lines time-parameterize and track picks, how to model conveyors, and how to validate the model.

**Labels used below.**
- **[cited]**: stated in the linked source. Where the source is primary (vendor spec, paper, official docs), it says so.
- **[3rd party]**: from a non-vendor source that we could not check against the vendor, for example a university's ROS driver.
- **[inferred]**: our own derivation or calculation from cited numbers. The working is shown.
- **[measured here]**: from a throw-away feasibility prototype we ran in MuJoCo 4.0.1 during this research (section 4.5). Its masses are placeholders, so only the modelling conclusions carry over, not the torque values.

## 0. Summary and recommendation

1. **Model the robot class IRB 360-1/1130**:
   - 1 kg payload, Ø1130 mm working range.
   - 150 m/s² max acceleration, 10 m/s max speed.
   - 25/305/25 cycle: 0.30 s at 0.1 kg, about 0.35 s at 1 kg.
   This is the variant whose numbers our pitch already uses, and it suits 12–125 g products.
2. **Geometry** (all lengths between joint centres):
   - base radius (centre to motor axis) 200 mm;
   - motor axes 275 mm below the base frame;
   - upper arm 350 mm;
   - forearm (strut) 800 mm;
   - platform radius 45 mm;
   - parallelogram width 100 mm;
   - TCP about 50 mm below the platform joint plane.

   The 275 mm comes from ABB's own drawing. The other values come from a third-party ROS driver for a real IRB 360. Together they reproduce ABB's published working envelope within that driver's joint limits (section 2.3). ABB does not publish link lengths.
3. **Dynamics.**
   - The only published measured parameter set for any IRB 360 is Gnad et al. 2024, for the 6/1600 (masses, inertias, friction).
   - Use it, scaled, as the prior for the 1/1130.
   - Calibrate the motor torque limit, which ABB does not publish, so that the simulated **25/305/25 cycle times** match ABB's table.
   - Keep the **90/400/90 cycle times** as a held-out check.
4. **MuJoCo structure.** Use a closed chain built with `equality/connect` (site1/site2), following the official docs and the Menagerie closed-chain models (Robotiq 2F-85, Cassie):
   - 3 hinged upper arms;
   - 6 struts on 2-hinge universal joints at the elbow;
   - a free platform;
   - 6 `connect` constraints at the platform ball joints.

   **Correction to PLAN.md:** the official `slider_crank` example does *not* use equality constraints. It demonstrates the `slider-crank` actuator transmission (section 4.1).
5. **Controller.** Use computed torque:
   - the feedforward comes from MuJoCo's own `mj_inverse`, projected onto the 3 motors through the equality-constraint Jacobian;
   - add joint PD;
   - torque limits go on the motor actuators.

   **Motion generation** follows what ABB documents for its QuickMove controller: the time-optimal speed profile along a fixed path, under Cartesian caps (speed, acceleration, jerk) and motor-torque limits.
6. **Feasibility [measured here].** At 150 m/s², dt = 0.5 ms, and equality `solref="0.001 1" solimp="0.99 0.999 0.001"`:
   - the closed chain tracks a 300 mm move within **0.044 mm**;
   - the loop-closure error is **22 µm**;
   - the platform tilts by **0.1 mrad**.

   MuJoCo's default constraint softness is too soft: it gives 0.24 mm of violation.
7. **The effector-only reduction (the current `sim/suction.py`) is valid for grip physics** if test V8 passes: the full model's platform acceleration must match the reference within 2 % RMS and 5 % peak. The arm model's real value is that it makes **cycle time and achievable acceleration depend on payload and torque limits**. Keep the closed chain as the reference cell. Use the reduced effector driven by the closed chain's platform trajectories only if log generation needs the speed: about 26k steps/s for the chain against about 200k for a reduced model.
8. **Conveyors.** Use `geom/surfacevel` (added in MuJoCo 3.11.0, present in 4.0.1). A static box with `surfacevel="v 0 0 0 0 0"` transports products by friction. We checked that a 12 g product reaches 0.500 m/s belt speed in about 0.1 s.

## 1. Kinematics and dynamics

### 1.1 Architecture

- Clavel's delta [cited: Clavel 1988, 1991; US patent 4,976,582] has:
  - three rotary actuators fixed to the base, each driving an **upper arm**;
  - each upper arm connected to the moving platform by a **parallelogram**: two parallel struts with ball joints at both ends;
  - the parallelograms keep the platform parallel to the base, so it has 3 translational DOF;
  - a fourth, telescopic shaft with universal joints (axis 4) that passes the rotation of a base motor to the tool [cited: Williams 2016; ABB IRB 360 product spec, axes figure].
- On the IRB 360 the ball joints are **clamped spring-loaded ball sockets**. Gnad et al. report that bulky loads with a large CoM offset can make the struts' longitudinal force exceed the spring clamp force, and the socket then dislocates ("the mechanism falls apart") [cited: Gnad et al. 2024 RAS, §1]. This is irrelevant for 12–125 g products. It shows that the strut force (about ±10–15 N in normal motion without load, same source §6.4) is a real limit.

### 1.2 Inverse kinematics (closed form)

Notation, in the frame of arm *i*, rotated by φᵢ = 0°, 120°, 240° about the vertical axis:
- motor axis at radius r_b, at height 0;
- upper arm length L, forearm length l, platform joint radius r_p;
- θᵢ is the upper-arm angle, positive *downward* from horizontal;
- the platform centre p = (x, y, z) is in the motor-plane frame, with z < 0 below the motors;
- (xᵢ, yᵢ, z) is p rotated into arm i's frame.

```
elbow:        E_i = ( r_b + L cos θ_i , 0 , −L sin θ_i )
platform jt:  P_i = ( x_i + r_p , y_i , z )
constraint:   |E_i − P_i|² = l²
with a_i = x_i + r_p − r_b :
   −a_i cos θ_i + z sin θ_i = K_i ,   K_i = (l² − L² − a_i² − y_i² − z²) / (2L)
   θ_i = atan2(z, −a_i) ± arccos( K_i / sqrt(a_i² + z²) )
```

- This is the standard closed form [cited: Williams 2016, "IPK solution"; Clavel 1991].
- Choose the "elbow-out" branch, the one with larger cos θᵢ.
- There is no solution when |Kᵢ| > sqrt(aᵢ² + z²). That defines the workspace boundary.
- The parallelogram width w does not enter the position kinematics.

### 1.3 Forward kinematics

The platform centre lies at the intersection of three spheres:
- each sphere is centred at the elbow Eᵢ shifted inward by r_p (Eᵢ − r_p·uᵢ, where uᵢ is the radial unit vector);
- each sphere has radius l.

Take the root with the platform below the base [cited: Williams 2016, "Three-spheres intersection algorithm", App. A/B]. In MuJoCo, FK is free: it is the assembled simulation state.

### 1.4 Velocity Jacobian

Differentiating each constraint gives dᵢ·(ṗ − Ėᵢ) = 0, where dᵢ = Pᵢ − Eᵢ is the forearm vector:

```
J_x ṗ = J_q θ̇ ,   J_x rows = d_iᵀ ,   J_q = diag( d_i · ∂E_i/∂θ_i )
```

- Singularities occur where J_x or J_q loses rank.
- We checked the Jacobian against finite differences of the IK, and they agree to all printed digits [measured here].
- Joint accelerations follow from θ̈ = J_q⁻¹ (J_x p̈ + J̇_x ṗ − J̇_q θ̇).

### 1.5 Dynamics: full and simplified models

- **Full multibody model.** It has 3 upper arms, 6 struts, the platform and the telescopic shaft. It is the reference. It is what MuJoCo simulates when we build the closed chain.
- **Simplified (Codourey-type) model.** This is the standard real-time model for computed-torque control of the Delta [cited: Codourey 1998, IJRR 17(12):1325–1336, which develops a virtual-work dynamic model of the DELTA for computed-torque control; the lumping details below come from papers that build on it, we did not read the original]:
  - neglect the struts' rotational inertia;
  - lump each strut's mass at its two ends: a fraction r at the elbow, (1 − r) at the platform;
  - the platform translates only.

  ```
  τ = I_b θ̈ + G_b(θ) + Jᵀ m_n ( p̈ − g⃗ )        with  ṗ = J θ̇ ,  J = J_x⁻¹ J_q
  I_b = I_arm + i_G² I_rotor + 2 r m_s L²          (per arm, about the motor axis)
  G_b = −(m_arm c_arm + 2 r m_s L) g cos θ          (arm gravity)
  m_n = m_platform + m_payload + 6 (1 − r) m_s
  ```

  Secondary sources disagree on r: some use 2/3 at the elbow and 1/3 at the platform, others ½ and ½. Our reading [inferred, from elementary mechanics]:
  - ½/½ keeps the strut's mass and centre of mass;
  - 2/3 at the elbow keeps its moment of inertia about the elbow.

  Neither is exact. In our prototype, ½/½ matched the full multibody model better: feedforward error 2.8 % of peak torque against 5.0 % for 2/3 (section 4.5). Choose r with test V4 rather than by citation.
- **Physically consistent identification of an IRB 360.** Gnad et al. 2024 (RA-L and RAS, open access, ABB IRB 360-6/1600) model:
  - the upper arms by first moments (m·c_x, m·c_z) and Θ_yy about the joint, which is exact for a 1-DOF rotation;
  - the struts as slender rods with the CoM at mid-length;
  - the platform by its mass only, since it only translates;
  - the telescopic shaft as two cylinders.

  They identify these parameters from motor-torque measurements, and verify strut forces with a force-sensing strut [cited: Gnad et al. 2024 RAS §3, §6]. Their values are in table 5.1. The same paper reports that the manufacturer's CAD has **no volumetric information**, so inertias cannot come from ABB CAD.

### 1.6 Motors and gearboxes

- **Not published by ABB:** motor type, gear ratio, rated or peak torque, rotor inertia.
- **Published facts:**
  - ABB uses "maintenance-free AC motors" and oil-filled gearboxes [cited: ABB PS §1.5.1].
  - Axis 4 reaches 2880 °/s and 1200 rad/s² [cited: ABB PS §1.6.2–1.6.3]. The axis-4 motor reaches 3200 °/s at the gearbox output [cited: Gnad 2024 RAS §6.3.1].
  - Identified friction at the gearbox output of arms 1–3: Coulomb 4.4–6.5 N·m and viscous 1.4–2.1 N·m·s/rad (assembled robot) [cited: Gnad 2024 RAS Table 4; RA-L §VIII-B].
  - Lumped axis-4 inertia 2.87×10⁻³ kg·m² [cited: RA-L].
  - The verification RMS torque error was 1.5–1.7 N·m [cited: RAS Fig. 10].
- **Limits Gnad et al. used for time-optimal planning:**
  - joint speed 1088 °/s, acceleration 5000 °/s², jerk 100 000 °/s³ ("motor torques and speeds set according to the data sheet, accelerations and jerks up to the user");
  - a 10 % safety margin on torques and forces;
  - strut force limit 120 N.

  Their time-optimal 800 mm × 180 mm U-turn with 2.5 kg took about 0.48 s [cited: RAS §6.6].
- **Consequence for us:** the torque limit must be **calibrated** from ABB's payload-dependent cycle times (section 5.5). Do not invent it.

## 2. Real geometry and limits

### 2.1 ABB IRB 360 performance (primary: product specification 3HAC029963-001 rev. Z, 2021, and datasheet ROB0082EN 2022)

| Quantity | IRB 360-1/1130 | IRB 360-3/1130 | IRB 360-8/1130 | IRB 360-1/1600 | IRB 360-6/1600 | Source |
|---|---|---|---|---|---|---|
| Payload | 1 kg | 3 kg | 8 kg | 1 kg | 6 kg | PS §1.1.2 |
| Working range | Ø1130 × 250 mm (TCP z 865→1115 below base frame), cone to z 1165 at R 483.5 | same | Ø750 (R 375) × 250, z 892→1142 (cone to 1242) | Ø1600 (R 720) × 300, z 1112→1412 (cone to 1462) | R 600, z 1107.5→1412.5 (cone to 1567.5) | PS §1.6.1 figure |
| Max TCP speed | 10 m/s | 10 m/s | 10 m/s | 10 m/s | 10 m/s | PS §1.6.2 |
| Max TCP acceleration | **150 m/s²** | 100 m/s² | 100 m/s² | not given | 100 m/s² | PS §1.6.3 |
| Axis 4 | 2880 °/s, 1200 rad/s² | same | same | same | same | PS §1.6.2–3 |
| 25/305/25 cycle, ppm (s) | 200 @0.1 kg (0.300), 170 @1 kg (0.353) | 150 @0.1 (0.400), 150 @1 (0.400), 115 @3 (0.522) | 160 @1 (0.375), 140 @4 (0.429), 100 @8 (0.600) | 170 @0.1 (0.353), 150 @1 (0.400) | 140 @1 (0.429), 125 @3 (0.480), 100 @6 (0.600) | PS §1.7.1 |
| 90/400/90 cycle, ppm (s) | 135 @0.1 (0.444), 120 @1 (0.500) | 100, 100, 80 (0.60, 0.60, 0.75) | 110, 90, 65 | 120, 110 | 105, 95, 75 | PS §1.7.1 |
| Pose repeatability RP | 0.10 mm @0.1 kg, 0.09 @1, 0.06 @3, 0.07 @8 (1130 family) | | | | | PS §1.6.2 (ISO 9283, 0.8 m/s) |
| Path accuracy AT | 0.51 mm @0.1 kg, 0.52 @1, 1.00 @3, 2.32 @8 | | | | | PS §1.6.2 |
| Pose stabilization time Pst (to 0.2 mm) | 0.03 s @1 kg, 0.05 @3/8 | | | | | PS §1.6.2 |
| Base loads in operation | Fx ±330 N, Fy ±260 N, Fz −1500 ± 170 N, Mx ±200, My ±230, Mz ±100 N·m (1/800, 1/1130, 1/1600, 3/1130) | | ±550/±500/−1500 ± 460 N; ±380/±440/±180 N·m (8/1130, 6/1600) | | | PS §1.3.3 |
| Robot frame natural frequency | > 17 Hz recommended | | > 40 Hz | | | PS §1.3.3 |
| Manipulator weight | 120 kg (145 kg stainless) | | | | | PS §1.1.3 |
| Supplementary load | 350 g on upper arm, 350 g on lower arm | | | | | datasheet |
| Integrated vacuum supply | max 7 bar / **max vacuum 0.75 bar** | | | | | datasheet (relevant to the suction model) |

Notes:
- **Cycle definition** [cited: PS §1.7.1]:
  - Cycle 1 is a 25-305-25 mm movement (up 25, across 305, down 25, and back) with a 90° rotation of axis 4.
  - Cycle 2 is 90-400-90.
  - "Both cycles incorporate an air activation time of 35 ms for picking and 35 ms for placing". ABB measures the cycles "under real conditions" and calls them approximate.
  - Stainless wash-down robots reach 90–95 % of this performance.
- **Older and other ABB documents give slightly different figures.** The datasheet gives 0.36 s at 1 kg for the 1/1130, and an older product guide gives 0.54 s at 3 kg for the 3/1130 [cited: datasheet; ROB0082EN_B]. The 2015 upgrade-kit note says "up to 65 cycles/min at 8 kg for 90-400-90" [cited: ROB0322].
- **ABB conveyor tracking (IRB 360-1/1130 with PickMaster)** [cited: datasheet]:

  | Conveyor type | Speed | Tracking repeatability |
  |---|---|---|
  | Constant | 200 mm/s | 1.0 mm |
  | Constant | 350–750 mm/s | 1.5 mm |
  | Constant | 800–1400 mm/s | 5.0 mm |
  | Start/stop | 500 mm/s, start/stop in 0.2 s | 3.5 mm |
  | Indexing conveyor control | 3.5 g acceleration/deceleration | 2.0 mm |

### 2.2 Link geometry: what is known

ABB does not publish link lengths. Available evidence:

| Source | Base radius r_b | Motor axis depth | Upper arm L | Forearm l | Platform r_p | Status |
|---|---|---|---|---|---|---|
| ABB PS §1.6.1 "Extreme position" figure | — | **F = 275 mm** below base frame (all variants) | dimension R = 366 (1130), 447 (1/1600), 515 (6/1600), 251 (1/800): the radius of the circle the elbow sweeps about the motor axis | — | — | [cited]. Reading R as the swing radius to the outer elbow is [inferred] from the figure. |
| TU Gdańsk MecHaRo-Lab ROS 2 driver for a real IRB 360 (`irb360.urdf`, `irb360_kinematics_plugin.cpp`) | 200 mm (`f = 200·2√3`) | 275 mm (URDF motor origins z = −0.275) | 350 mm | 800 mm | 45 mm (`e = 45·2√3`) | [3rd party]. Joint limits −0.805…+1.745 rad. Elbow ball pair about 100 mm apart. Strut CoM in the CAD-export URDF at 0.400 m along the strut, consistent with l = 800. |
| Gnad et al. 2024 (IRB 360-6/1600, measured) | — | — | carbon part about 500 mm (RA-L regularization) | **1106 mm** strut, measured | — | [cited] |
| `harshalaptraise/delta-hil` (claims "solved against real CAD", variant unstated, probably 1600) | 175 | 263 | 508 | 1046 | 76 | [3rd party], unverified |
| Williams 2016 notes (IRB 360-1/1600, "scaled from a figure") | 164 (side s_B = 567) | — | 524 | 1244 | 22 (s_P = 76) | [cited], but only a rough estimate. It fails our workspace check (below). |

### 2.3 Workspace check of the geometry [inferred, computed with the IK above]

We sampled ABB's 1130 working cylinder:
- r ≤ 565 mm;
- TCP z from 865 to 1115 mm below the base frame, that is 590–840 mm below the motor plane;
- 468 points.

Results:

| Geometry | TCP offset below platform joints | Required θ range |
|---|---|---|
| TU Gdańsk (r_b 200, L 350, l 800, r_p 45) | 0 mm | −36.8° … +106.4° |
| same | **50 mm** | **−47.6° … +99.0°** |
| same | 100 mm | −62.2° … +93.6° |
| `delta-hil` | 0 / 50 mm | −64° … +55° / −84° … +53° |
| Williams | any | the top of the envelope is unreachable |

- The TU Gdańsk geometry with a 50 mm TCP offset needs almost exactly the driver's joint limits (−46.1° … +100.0°). This is independent support that it is a 1130-class geometry.
- An inconsistency remains. ABB's dimension R = 366 mm is 16 mm longer than L = 350. Our position estimate of the elbow in ABB's figure gives r_b ≈ 185 mm, not 200. Both differences could come from how the drawing dimensions the elbow (ball housing against ball centre). **Open question** (section 6).
- The forearm/upper-arm ratio is 800/350 = 2.29. The measured 6/1600 has about 1106/500 = 2.2. These are consistent.

### 2.4 Other delta robots for comparison

| Robot | Payload | Working range | Cycle figures | Source |
|---|---|---|---|---|
| Omron (Adept) Quattro 650H | 6 kg (P30 platform 15 kg) | Ø1300 × 500 mm | Adept cycle **25/305/25**: 0.30 s @0.1 kg, 0.36 @1, 0.37 @2, 0.41 @4, 0.43 @6 ("sustained, 20 °C"). Extended 25/700/25: 0.46–0.61 s. Repeatability ±0.10 mm | Omron datasheet I265E-E3-02 (primary) |
| Yaskawa Motoman MPP3H | 3 kg | Ø1300, 500 mm stroke (recommended Ø1040 × 300) | 25-305-25: 230 cpm @1 kg, 170 @3 kg (footnote: "cycle time without picking time and product"). 300-400-100: 100/80. 70-700-70: 130/95. Conveyor 800/1300/2000 mm/s → 1.0/1.5/3.0 mm. Repeatability ±0.1 mm | Yaskawa flyer B-03-2020 (primary) |
| FANUC M-3iA/6S | 6 kg (8) | Ø1350 × 500 | no 25/305/25 figure found | FANUC product pages (secondary) |
| FANUC M-1iA/1H | 1 kg | Ø280 | "25-200-25 mm pick and place motion time 0.3 s" | FANUC catalog, via search snippet (unverified) |
| Kawasaki YF003N | 3 kg | Ø1300 × 500 | Spec sheet defines the 25/305 mm test motion. We could not retrieve the value (download timed out) | Kawasaki spec (not retrieved) |
| Codian D4-800/1100/1300-HD | ≤ 3 kg | Ø800/1100/1300 | 180/150/120 picks/min (one catalog says 150 for the 1300) | DirectIndustry catalog (secondary) |

**Takeaway.** The 25/305/25 "Adept cycle" is the industry-standard benchmark. The 1 kg class does it in 0.30 s at light load across vendors. Yaskawa's faster numbers exclude the pick dwell, while ABB's include 2 × 35 ms of air activation. Compare like with like.

## 3. Motion generation and line operation in practice

### 3.1 How industrial controllers time-parameterize a pick move

- **ABB QuickMove/TrueMove** [cited: Björkman et al., ABB Robotics, IFAC 2008, primary]:
  - The path is generated by on-line optimization of speed and acceleration: "min T subject to customer constraints (e.g. speed, acc) and robot constraints (e.g. gearbox torque)", including maximum motor torque, motor speed and torque or force in critical structural parts.
  - "The user-specified path must be followed exactly and the specified speed must only be reduced if the robot movements are limited mechanically or electrically."
  - The controller is a two-degrees-of-freedom, model-based structure: feedforward from the rigid and elastic dynamic models, plus feedback.

  This is **time-optimal path parameterization (TOPP)** under dynamic constraints. The established methods are Bobrow et al. 1985 and Shin & McKay 1985, and TOPP-RA (Pham & Pham 2018) as a modern solver. Gnad et al. 2024 use exactly this for the IRB 360, adding strut-force limits and a jerk bound [cited].
- **Path shape.** The PickMaster motion "is controlled by only two positions, usually relative to a tracking frame. The rest of the path is defined by relative offsets and other parameters" [cited: ABB PickMaster product spec 3HAC 5842-1]. These offsets are the approach and depart heights (25 mm in Cycle 1). Corners are blended with zones; the zone radius for the benchmark is not published. Academic delta work uses blended lines, Lamé curves, B-splines or Pythagorean-hodograph curves for the same U path [cited: e.g. Gnad 2024 B-spline U-turn; PH-curve papers, IEEE Access 2018 and Robotica 2023].
- **Jerk.**
  - The rest-to-rest profile that is time-optimal under velocity, acceleration and jerk limits is the 7-segment "double-S" profile [cited: Biagiotti & Melchiorri 2008]. Ruckig computes it on-line for multiple DOF [cited: Berscheid & Kröger, RSS 2021]. It is already a project dependency.
  - **Implied jerk [inferred].** Assume the two 35 ms air activations are dwells. A 0.300 s Cycle 1 then leaves (0.300 − 0.070)/2 = **0.115 s per one-way move** of about 0.33 m of path. A 1-D jerk-limited profile with v ≤ 10 m/s and a ≤ 150 m/s² needs:

    | Jerk | Time for 0.33 m |
    |---|---|
    | 5 000 m/s³ | 128 ms |
    | **10 000 m/s³** | **110 ms** |
    | 100 000 m/s³ | 95 ms |

    So the benchmark implies jerk on the order of 10⁴ m/s³. On a 305 mm stroke the peak speed is only about 6–7 m/s, so the 10 m/s cap rarely binds; acceleration and jerk do.

### 3.2 What a "speed override" scales on a real controller

On ABB (RAPID) [cited: forum quotes of the RAPID reference and a third-party RAPID reference; the official RAPID manual was not retrieved]:
- **`VelSet Override, Max`** scales programmed velocities (%) and caps TCP speed.
- **`AccSet Acc, Ramp`** sets acceleration as a percentage of the normal (model-limited) value (`Acc`, values below 20 are raised to 20). `Ramp` is the percentage rate at which acceleration builds up, i.e. **jerk**, with values below 10 raised to 10. The default `AccSet 100, 100` is restored on RAPID reset.
- **`PathAccLim TRUE \AccMax := a, TRUE \DecelMax := d`** caps the TCP acceleration and deceleration in m/s².

So an integrator's "speed" knob on a picker is in practice an **acceleration and jerk cap**. QuickMove's dynamic limits stay in force on top of it.

**Our action definition [inferred].** α ∈ [0.3, 1] sets:
- the Cartesian acceleration cap to α·a_max (PathAccLim-like);
- the jerk cap to α·j_max (`Ramp` = `Acc`);
- the velocity cap unchanged at 10 m/s.

Torque limits stay active, as in QuickMove. Log α, plus the realized peak acceleration and jerk per pick.

### 3.3 Vision-guided conveyor tracking (ABB PickMaster)

Sources: PickMaster product spec 3HAC 5842-1 (IRB 340 era) [cited]; ABB conveyor tracking datasheet PR10045EN [cited]; IRB 360 datasheet [cited].

1. An **encoder** on the conveyor drive gives belt position. A **trigger**, either every N encoder pulses or a sync switch, fires the camera.
2. At the trigger instant the encoder value is latched. Item positions from vision are stored **relative to that trigger position**, in a frame calibrated to the robot base ("conveyor to robot", "vision pixels to mm with parallax compensation").
3. PickMaster pairs pick targets with place targets and queues them on the controller (up to 254 objects).
4. A **start window** and a **maximum tracking distance** define where along the belt the robot may pick. Items are picked "in direction of increasing x-value, i.e. the items closest to the robot first".
5. The robot moves in the **moving work-object frame**: its TCP path is coordinated with the belt. It touches the product with zero relative velocity, picks during the 35 ms air activation while tracking, then lifts.
6. **Accuracy.**
   - IRB 340 era: ±2 mm at full robot speed with 150 mm/s belts; encoder sampling 20 ms by default; recommended belt ≤ 150 mm/s, maximum 350 mm/s.
   - Current IRB 360 figures: see the tracking table in section 2.1.
   - Generic conveyor tracking: ±2 mm path deviation up to 350 mm/s, and a coordinated fine point within ±0.7 mm.

### 3.4 Belt speeds and line layouts

- **Belt speeds:** 0.15–0.35 m/s for the IRB 340 generation; up to 1.4 m/s tracked by the IRB 360 (ABB) and 2.0 m/s for Yaskawa, with accuracy falling as speed rises (tables above).
- **Layouts** [cited: PickMaster spec §2.2; Packworld; Syntegon]:
  - **Parallel or perpendicular** infeed and outfeed belts.
  - A **master** belt runs independently of the cell. A **slave** infeed can be stopped and started by the cell when the cell has a surplus. Products and boxes can flow in the same or opposite directions.
  - **Several robots in series** share one random flow. Each picks a share and passes the rest downstream. For example, a cookie line has ten deltas along an infeed belt, each with its own camera, feeding one flow wrapper at 90 trays/min.
  - Placement goes into **trays or thermoformed packs** on a second belt, into **indexing** pockets (the IRB 360 supports indexing conveyor control at 3.5 g), or into the **flow-wrapper infeed chain** (flights or lugs at a fixed pitch). Syntegon's RPP cells put 1–3 deltas per cell onto PackSeries/Sigpack wrapper infeed chains.
  - ABB claims "up to 500 products per minute" for flow-wrapped products from an indexing belt with the 8 kg version (multi-pick) [cited: ABB 8 kg press release].
- **Simulation layout we propose [inferred]:**
  - a random-flow infeed belt at 0.2–0.5 m/s across the robot;
  - a counter- or co-current tray belt at constant speed;
  - one robot;
  - a camera trigger simulated as a belt-distance trigger with position noise matching the tracking-repeatability table.

## 4. Modelling it in MuJoCo

### 4.1 Correction: what the official `slider_crank` example shows

`model/slider_crank/slider_crank.xml` in google-deepmind/mujoco (read via `gh api`) has three open-chain bodies and three `position` actuators with `cranksite`, `slidersite` and `cranklength`. It demonstrates the **slider-crank actuator transmission**, including a "broken crank". It contains **no equality constraints**, so it is not a closed-chain modelling pattern. The closed-chain patterns are:

- **Docs: Computation → Equality.** `connect` "connects two bodies at a point, effectively creating a ball joint outside the kinematic tree". Two connects on the same body pair make a hinge, and three non-collinear ones make a weld. The docs advise against using equality constraints for joints that *can* be in the tree ("slower and less accurate"). A loop closure cannot be in the tree, so this is the intended use.
- **Docs: XML reference `equality/connect`.** It can be specified with `body1` + `anchor` (assumed satisfied at `qpos0`), or with `site1` + `site2` (the sites "snap together"; site positions can be changed at runtime). The official test model `test/engine/testdata/equality_site.xml` shows the site form.
- **Menagerie closed chains:**
  - `robotiq_2f85/2f85.xml` closes each four-bar finger linkage with `<connect anchor="0 0 0" body1="right_follower" body2="right_coupler" solref="0.005 1" solimp="0.95 0.99 0.001"/>`.
  - `agility_cassie/cassie.xml` closes the plantar and Achilles rod loops with `connect` at `timestep="0.0005"` and a default of `<equality solref="0.005 1"/>`.
- **Docs: tuning.** For loop closures, use `solref[0]` = 0.004–0.005 at the default 2 ms step, keeping `solref[0] ≥ 2·timestep`. With `refsafe` on, standard integrators clamp `solref[0]` to at least 2·timestep. Recommended global settings are `integrator="implicitfast"`, `cone="elliptic"` and `impratio="10"`.
- **Docs: `equality/tendon`.** "If [tendon2] is omitted, the first tendon is fixed to a constant", namely its length at `qpos0` plus a₀. With a spatial tendon between two sites, this is a **massless rigid rod**: exactly the Codourey lumped forearm.

### 4.2 Two MuJoCo models

**Model A: full closed chain (the reference).**
- 3 upper-arm bodies on hinges, at the motor axes. The armature is the reflected rotor inertia i_G²·I_rotor. Friction goes in `frictionloss` and `damping`.
- 6 strut bodies, children of the arms at the elbow ball centres (±w/2 along the hinge axis).
  - Each strut has **two hinges**, a universal joint: axis 1 is parallel to the motor axis, axis 2 is perpendicular to it and to the strut.
  - A ball joint would also work but adds a spin DOF about the strut axis with near-zero inertia (Θ_xx = 1.5×10⁻⁵ kg·m², Gnad Table 3).
- The platform is a free body. 6 `connect` constraints (`site1` at strut end, `site2` at platform joint) close the loops.
- Mobility: 6 + 3 + 12 − 18 = 3 DOF, and the constraints are not redundant.
- The parallelograms hold the platform orientation. We measured a 0.1 mrad tilt at 150 m/s².
- Axis 4, the telescopic shaft, is optional. For a suction pick with a symmetric cup, a 90° rotation adds nothing to grip loads except a yaw-inertia torque. Add it only if rotation becomes part of the action. It is then a hinge on the platform driven by a motor with the axis-4 limits (2880 °/s, 1200 rad/s²).

**Model B: lumped (Codourey-type), for speed and as the feedforward model.**
- 3 hinged arms, each carrying a point mass r·2m_s at the elbow.
- The platform on 3 slide joints, with mass m_p + 6(1 − r)m_s.
- 3 spatial tendons (elbow site → platform joint site at r_p), each held by `<equality><tendon tendon1="f0"/></equality>`.
- nv = 6, neq = 3.

### 4.3 Actuators and limits

- **Motors.** Use `<motor joint="q0" gear="1" ctrlrange="-τ_max τ_max" ctrllimited="true"/>` with the torque at the gearbox output. Alternatively use `gear = i_G` with motor-side torque, if the gear ratio becomes known.
- **Joint range.** Use `range` (−46°…+100° from the ROS driver) as a soft limit, a safety net only. Motion planning must stay inside it.
- **Optional compliance.** The rigid model gives zero settling time; ABB measures Pst = 30–50 ms to within 0.2 mm (ISO 9283). If residual vibration matters for drops, insert a torsional spring-damper between the motor rotor and the arm, i.e. a second hinge with `stiffness` and `damping` and rotor inertia on the motor side. Tune it to Pst of about 0.03 s and a first mode above 17 Hz, the frame requirement. Not needed in a first version.

### 4.4 Controller: computed torque in a closed chain

Each control step:
1. Get the reference platform state (p, ṗ, p̈)(t) from the motion generator.
2. Compute θ, θ̇, θ̈ with IK and the Jacobian (section 1.4). In Model A, also compute the strut angles, or use Model B as the feedforward model.
3. **Feedforward (projected inverse dynamics).** Run `mj_inverse` with constraints disabled (`mjDSBL_CONSTRAINT`). That gives `qfrc = M(q) q̈ + c(q, q̇) − passive` for the full tree coordinates of the feedforward model. Then solve `[Sᵀ  J_eqᵀ] [τ; λ] = qfrc` by least squares, where:
   - S selects the 3 motor DOFs;
   - J_eq is the equality-constraint rows of `efc_J` from `mj_forward` at the same state;
   - τ is the motor torque;
   - λ is the loop-closure force, which gives strut forces for free.

   This is the standard "constrained inverse dynamics by projection" (e.g. Aghili 2005). It uses MuJoCo's own mass matrix, so the feedforward and the plant agree.
4. **Feedback.** τ = τ_ff + K_p(θ_ref − θ) + K_d(θ̇_ref − θ̇), with saturation from the actuator `ctrlrange`.
5. **Discrete-time detail [measured here].** With MuJoCo's semi-implicit Euler, `qvel` after a step is the mean velocity over that step. Evaluate:
   - the velocity reference at t − dt/2;
   - the feedforward acceleration at t + dt/2.

   Without this, a spurious tracking error proportional to acceleration appears: 0.5 mm at 150 m/s² and dt = 0.5 ms, falling to 0.04 mm once corrected.

### 4.5 Feasibility prototype [measured here]

Throw-away scripts:
- `/tmp/claude-1000/-home-jaseel-priorlab-berlin-hack/1f7a662c-9e44-44c4-ac75-1859400e643a/scratchpad/proto/delta_proto.py`
- `/tmp/claude-1000/-home-jaseel-priorlab-berlin-hack/1f7a662c-9e44-44c4-ac75-1859400e643a/scratchpad/proto/kin.py`

The scratchpad may be cleared; sections 4.2–4.4 describe everything needed to rebuild them.

Setup:
- TU Gdańsk geometry.
- **Placeholder** masses:
  - arm 0.9 kg with CoM at 0.15 m;
  - armature 0.05 kg·m²;
  - strut 0.13 kg;
  - platform 0.6 kg;
  - payload 1.0 kg at 50 mm below the platform.
- A 300 mm x-move with a Ruckig profile (v 10 m/s, a 150 m/s², j 10⁴ m/s³) taking 105.7 ms.
- Feedforward from Model B; PD gains K_p = 2·10⁴ N·m/rad, K_d = 300 N·m·s/rad.

| Configuration | Max TCP error | Max loop violation | Feedback torque (share of 136 N·m peak) | Platform tilt |
|---|---|---|---|---|
| Model A, dt 0.5 ms, eq `solref 0.001 1`, `solimp 0.99 0.999 0.001`, r = ½ | **0.044 mm** | 22 µm | 3.8 N·m (2.8 %) | 0.107 mrad |
| same, r = 2/3 | 0.114 mm | 22 µm | 6.8 N·m (5.0 %) | 0.107 mrad |
| Model A, dt 1 ms, `solref 0.002 1`, `solimp 0.99 0.999` | 0.095 mm | 88 µm | 4.7 N·m | — |
| Model A, dt 0.5 ms, `solref 0.002 1`, default solimp (0.9 0.95) | 0.43 mm | **241 µm** | 5.1 N·m | — |
| Model B self-consistent, dt 0.5 ms, stiff settings | 0.036 mm | 19 µm | 2.4 N·m | n/a |

- **Raw `mj_step` speed** (single thread, this machine): Model A (nv 21, 6 connects) about **26 000 steps/s**, that is 13× real time at 0.5 ms; Model B (nv 6) about 214 000 steps/s.
- **Conclusions:**
  - The closed chain is stable and accurate at 150 m/s² with dt = 0.5 ms, the same step as the suction model.
  - Constraint violation scales roughly with solref², so stiffen loop closures to `solref="0.001 1"` (= 2·dt) and `solimp="0.99 0.999 0.001"`.
  - The lumped model is a good feedforward model, within 3–5 % of peak torque.
- The torque magnitudes come from placeholder inertias and **must not be quoted** as IRB 360 values.

### 4.6 When the effector-only reduction is valid

- **What the grip sees.** The cup is rigidly mounted on the platform. The product therefore sees only the platform's motion: translation, plus yaw if axis 4 is modelled. The parallelograms hold roll and pitch at about 0.1 mrad.
- **Equivalence condition.** An effector that tracks the same platform trajectory (p, ṗ, p̈) gives the same contact loads, up to the effect of the product's reaction force on the robot. That effect is negligible: a 125 g product against 1–2 kg of platform, gripper and reflected arm inertia.
- **The reduction is valid if:**
  - **(a)** the full model's platform acceleration matches the reference within 2 % RMS and 5 % peak of the peak acceleration, over the cycles and payloads we use (test V8);
  - **(b)** the reference trajectories fed to the reduced model were generated **with** the torque-limited time scaling. That is the only place the arm dynamics change the motion: heavier tools or products reach lower accelerations.
- **What the reduction cannot give:**
  - torque-limit-driven slowdowns, unless the trajectory generator includes them;
  - strut forces;
  - structural vibration, which the rigid closed chain does not give either.
- **Validation:** besides V8, run a paired drop-rate comparison on identical seeds. Use ≥ 1000 picks per product × speed cell and a two-proportion test. Drop rates must agree within the confidence interval.

### 4.7 Conveyors and trays with `surfacevel`

Semantics [cited: XML reference `geom/surfacevel`; changelog 3.11.0 and 4.0.0]:
- **The field.** `surfacevel` is `real(6)`, a velocity field σ(x) = v + ω × (x − p) in the geom frame: a constant part v and a rotation ω about the geom origin p.
- **How contacts see it.** It is projected onto each contact's tangent plane, with no normal component. With `condim ≥ 4`, ω also drives torsional friction.
- **What it models.** "Conveyor belts, treadmills and turntables can be constructed with no degrees of freedom. Friction drives touching bodies along the motion of the surface."
- **Whole geom.** It applies to the **entire** geom surface: a box with constant `surfacevel` moves all six faces.
- **Runtime changes.** It "can be modified at runtime" via `model.geom_surfacevel`, which allows belt start/stop, speed ripple and indexing.
- **Solver details.** Contact rows of `efc_vel` report velocity relative to the moving surface. In 4.0.0 the per-contact relative surface velocity was added as `mjContact.surfacevel`.

Official examples:
- **`model/surfacevel/treadmill.xml`:**
  - deck `box size=".76 .255 .05"` with `surfacevel="1.4 0 0 0 0 0"`;
  - end rollers as cylinders with `surfacevel="0 0 0 0 0 -28"`, matched by ω = v/r = 1.4/0.05;
  - "no degrees of freedom"; a passive humanoid is swept off.
- **`model/surfacevel/carousel.xml`:**
  - a supertorus ring with `surfacevel="0 0 0 0 0 .4"`, `friction=".1" priority="1"`;
  - a three-belt cascade defined once as a `frame` and reused with `attach` (flat, perpendicular, inclined 24°);
  - belts at `surfacevel=".7 0 0 ..."`; rollers at ω = −0.7/0.06 = −11.667 rad/s.

Our check [measured here], MuJoCo 4.0.1:
- setup: a 12 g, 60 × 40 × 10 mm product dropped onto a `surfacevel="0.5 0 0 0 0 0"` box with μ = 0.6, elliptic cone, impratio 10, dt 0.5 ms;
- it reached 0.326 m/s at 50 ms and 0.496 m/s at 100 ms;
- it then stayed at **0.5000 m/s**, travelling 0.976 m in 2 s.
- This is consistent with friction-limited acceleration μg ≈ 5.9 m/s².

**Trays.** Trays are free bodies resting on a second `surfacevel` belt, with pockets made of box geoms. If tray positions must be exactly periodic (an indexing or flow-wrapper chain), drive them kinematically with mocap bodies or a slide joint with a position actuator instead. Belt friction then only matters for the products.

## 5. Proposed design

### 5.1 Parameter table (IRB 360-1/1130 class)

| Parameter | Value | Source / status |
|---|---|---|
| Base radius r_b (centre → motor axis) | 0.200 m | TU Gdańsk ROS driver [3rd party]; ABB figure suggests about 0.185 [inferred]. Open question |
| Motor axis depth below base frame | 0.275 m | ABB PS §1.6.1 dimension F [cited]; same in ROS URDF |
| Upper arm L | 0.350 m | ROS driver [3rd party]; ABB swing radius R = 0.366 [cited] |
| Forearm (strut) l | 0.800 m | ROS driver IK constant; strut CoM at 0.40 m in CAD-export URDF [3rd party] |
| Platform joint radius r_p | 0.045 m | ROS driver [3rd party] |
| Parallelogram width w | 0.100 m | ROS URDF elbow ball offsets [3rd party] |
| TCP offset below platform joints | 0.050 m | fits ABB envelope with driver joint limits [inferred, §2.3] |
| Motor joint range | −46.1° … +100.0° | ROS URDF [3rd party]; consistent with ABB envelope [inferred] |
| Working envelope (acceptance) | Ø1130 mm, TCP z 865–1115 mm below base frame (cone to 1165 at R 483.5) | ABB PS [cited] |
| Max TCP speed / acceleration | 10 m/s / 150 m/s² | ABB PS [cited] |
| Jerk cap j_max | about 1·10⁴ m/s³ initial | implied by the 0.30 s cycle [inferred]; refined in calibration |
| Strut mass m_s | 0.13 kg (each of 6) | scaled from Gnad measured 0.150 kg / 1106 mm (identified 0.179) to 800 mm; CAD-URDF 0.1365 kg [cited + inferred] |
| Platform (delta plate) mass | 0.6 kg (range 0.4–1.2) | 6/1600 measured 1.12 kg, identified 1.20 [cited]; 1130 plate smaller [inferred]. Calibrate |
| Upper arm first moment m·c_x | about 0.10 kg·m (range 0.08–0.2) | 6/1600: 0.176 kg·m (500 mm arm) [cited]; scaled by length [inferred]; CAD-URDF 1.40 kg at 0.144 m = 0.20 [3rd party] |
| Upper arm Θ_yy about motor axis, incl. drive | about 0.08 kg·m² (range 0.05–0.26) | 6/1600: 0.258 kg·m² [cited]; scaled ∝ L² [inferred]. Calibrate |
| Drive friction (gearbox output) | Coulomb 4.7 N·m, viscous 1.6 N·m·s/rad | Gnad 2024 RA-L/RAS (6/1600) [cited] |
| Motor torque limit τ_max (gearbox output) | **unknown, calibrate** (§5.5) | not published |
| Axis 4 (optional) | 2880 °/s, 1200 rad/s², lumped inertia 2.9·10⁻³ kg·m² | ABB PS; Gnad RA-L [cited] |
| Gripper + cup | from the suction model | `sim/suction.py` |
| Pick/place dwell (air activation) | 35 ms each | ABB PS §1.7.1 [cited] |
| Timestep | 0.5 ms, `implicitfast`, elliptic cone, impratio 10 | MuJoCo docs + suction model [cited, measured here] |
| Loop closure softness | `solref="0.001 1" solimp="0.99 0.999 0.001"` | measured here |
| Belt speed | 0.2–0.5 m/s infeed (tracking table up to 1.4 m/s) | ABB datasheet, PickMaster spec [cited] |

**Alternative variant.** If the team prefers measured dynamics over measured geometry, model the **IRB 360-6/1600**:
- Gnad 2024 measured its masses and inertias and the 1106 mm strut;
- ABB gives its envelope (R 600, z 1107.5–1412.5) and its cycle times (0.43/0.48/0.60 s at 1/3/6 kg);
- its base and platform radii are unknown, but the same workspace fit can estimate them;
- it is a 100 m/s² robot, not a 150 m/s² one.

### 5.2 MJCF structure (Model A)

```xml
<mujoco model="irb360_class">
  <option timestep="0.0005" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <default>
    <equality solref="0.001 1" solimp="0.99 0.999 0.001"/>
    <default class="motor_joint"><joint type="hinge" armature="I_ROTOR_REFLECTED" damping="1.6" frictionloss="4.7" range="-46.1 100" limited="true"/></default>
    <default class="strut_joint"><joint type="hinge" armature="1e-5"/></default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 Z_BASE"> <!-- base frame; inverted mounting above the belt -->
      <!-- arm i at angle phi_i, motor axis at (r_b cos phi, r_b sin phi, -0.275) -->
      <body name="arm0" pos="0.200 0 -0.275">
        <joint name="q0" class="motor_joint" axis="0 1 0"/>
        <inertial pos="C_ARM 0 0" mass="M_ARM" diaginertia="..."/>
        <body name="s00" pos="L·cosθ0  -0.05  -L·sinθ0">        <!-- elbow ball centre, -w/2 -->
          <joint name="s00a" class="strut_joint" axis="0 1 0"/>     <!-- parallel to motor axis -->
          <joint name="s00b" class="strut_joint" axis="n00"/>       <!-- ⟂ motor axis and strut -->
          <geom type="capsule" size="0.008" fromto="0 0 0  d00" mass="0.13"/>
          <site name="se00" pos="d00"/>                             <!-- strut lower end -->
        </body>
        <body name="s01" ...> ... </body>                           <!-- +w/2 -->
      </body>
      <!-- arm1, arm2 identical, rotated 120°, 240° (use <replicate> or a <frame> + attach) -->
    </body>
    <body name="plat" pos="P0">
      <freejoint/>
      <inertial mass="M_PLAT" .../>
      <site name="pj00" pos="r_p cos phi0 - w/2 tangent ..."/>   <!-- 6 platform ball centres -->
      ...
      <body name="cup" pos="0 0 -0.050"> <!-- suction cup + lip ring from sim/suction.py --> </body>
    </body>
    <geom name="infeed" type="box" size="..." surfacevel="0.3 0 0 0 0 0" friction="..."/>
    <geom name="outfeed" type="box" size="..." surfacevel="-0.2 0 0 0 0 0"/>
  </worldbody>
  <equality>
    <connect site1="se00" site2="pj00"/> <connect site1="se01" site2="pj01"/>
    <!-- ... 6 in total -->
  </equality>
  <actuator>
    <motor name="m0" joint="q0" ctrllimited="true" ctrlrange="-TAU_MAX TAU_MAX"/>
    <!-- m1, m2 -->
  </actuator>
</mujoco>
```

- Build it with `mujoco.MjSpec` or a string template as in `sim/suction.py`.
- Compute the initial configuration with the IK so that `qpos0` is assembled: all 6 connects satisfied, no snapping at t = 0.
- The cup and lip ring from the suction model become children of the platform.

### 5.3 Motion generator (per pick)

1. **Targets.** Pick and place targets come in the belt frames (tracked through `geom_surfacevel` and the elapsed time, plus vision noise).
2. **Path.** The geometric path is: approach (vertical 25 mm) → transfer arc → descend 25 mm, with corners blended at radius ≤ 25 mm. The benchmark's zone size is not published; treat it as an assumption and check sensitivity.
3. **Belt tracking.** For tracked picks, the path is defined in the moving belt frame and the belt velocity is added. The relative TCP–product velocity at touch must be zero.
4. **Time scaling.** Find the minimum-time s(t) along the path subject to:
   - |ṗ| ≤ 10 m/s, |p̈| ≤ α·150 m/s², |p⃛| ≤ α·j_max;
   - |τ| ≤ 0.9·τ_max, via the Model B dynamics.

   Implementation options:
   - **TOPP-RA** (`toppra`) with torque constraints, then jerk smoothing;
   - simpler: a 1-D Ruckig double-S profile on path length s, with caps lowered by bisection until the torque check along the path passes.

   Both match "QuickMove + AccSet" in spirit. Ruckig is already in the repo.
5. **Dwell.** Hold 35 ms at pick (vacuum build-up, while tracking the belt) and 35 ms at place.

### 5.4 Controller

- Projected-inverse-dynamics feedforward (Model B, or Model A itself) plus joint PD.
- Use the dt/2 timing correction (section 4.4).
- Saturate torque with the actuator `ctrlrange`.
- Log per step: motor torques, the 3 equality forces (strut forces), and TCP acceleration.

### 5.5 Calibration (fits the two unknowns that matter)

1. **Fix the geometry** (table 5.1). Set inertias to the scaled priors.
2. **Fit j_max** so the simulated **25/305/25 cycle at 0.1 kg** equals **0.300 s**. The cycle includes 90° axis-4 rotation (not limiting: 72 ms minimum at 1200 rad/s²) and 2 × 35 ms dwell. At this load the 150 m/s² cap should bind, not the torque.
3. **Fit τ_max** so the same cycle at **1.0 kg** equals **0.353 s** (170 ppm). If no τ_max reproduces both with the prior inertias, fit one inertia scale factor as well and say so.
4. **Held-out checks** (not fitted): 90/400/90 at 0.1 kg → 0.444 s (135 ppm) and at 1 kg → 0.500 s (120 ppm). Peak base reaction forces must stay below ABB's operating maxima.

### 5.6 Validation tests

| # | Test | Reference value | Tolerance |
|---|---|---|---|
| V1 | IK ↔ FK round trip; Jacobian vs finite differences; assembled MuJoCo site positions vs analytic FK over the envelope | exact | < 1 µm / < 1e-6 relative |
| V2 | Workspace: every point of the ABB 1130 envelope reachable within the joint range, no singularity (cond(J) bounded) | ABB PS §1.6.1 | 100 % of a 468-point grid |
| V3 | Closed-chain integrity on Cycle 1 at full speed | (ours) | loop violation < 25 µm; platform tilt < 0.5 mrad; TCP error < 0.1 mm |
| V4 | Lumped (Model B) vs full (Model A) feedforward; choose r | (ours) | feedback torque < 5 % of peak |
| V5 | 25/305/25 cycle time, 0.1 kg / 1 kg | 0.300 s / 0.353 s (ABB PS §1.7.1) | fitted (calibration) |
| V6 | **Held out:** 90/400/90 cycle time, 0.1 kg / 1 kg | 0.444 s / 0.500 s | ±5 % (ABB rounds ppm to 5, about ±2 %, and calls figures approximate) |
| V7 | Peak base reaction during Cycle 1 and 2 | Fx ≤ 330 N, Fy ≤ 260 N, dynamic Fz ≤ 170 N, Mz ≤ 100 N·m (ABB PS §1.3.3) | must not exceed |
| V8 | Effector-only reduction: platform acceleration (full model) vs reference on Cycles 1–2 and the line's pick paths | reference trajectory | RMS < 2 %, peak < 5 % of peak acceleration; plus paired drop-rate test (§4.6) |
| V9 | Conveyor: product speed on belt after settling; displacement over 1 s | belt speed 0.2–1.4 m/s | < 0.5 % speed error; < 1 mm per s |
| V10 | Tracked pick: TCP–product relative velocity at touch; placement error with vision/encoder noise | ABB tracking table (1.0 mm at 200 mm/s, 1.5 mm at 350–750 mm/s) | relative velocity < 5 mm/s; error distribution comparable |
| V11 | (optional, if compliance is added) settling after Cycle 1 stop | Pst 0.03–0.05 s to 0.2 mm (ABB PS §1.6.2) | ±50 % |

## 6. Open questions and risks

1. **Exact 1130 link lengths.** The TU Gdańsk driver values fit ABB's envelope, but differ from our reading of ABB's figure:
   - R = 366 against L = 350;
   - r_b ≈ 185 against 200.

   Ways to settle it:
   - (a) ask the TU Gdańsk authors or the delta-hil author which variant and source they used;
   - (b) measure on ABB's RobotStudio library model (ABB says its CAD has no volumetric data, but the geometry is there);
   - (c) fit r_b, L, l, r_p jointly to all ABB envelope dimensions (A–F, R) for both the 1130 and 1600 variants.
2. **Torque limits and inertias for the 1130.** Unknown. They are calibrated from the cycle-time table (section 5.5). If the 2-parameter fit fails the held-out V6, the motion model (path shape, zone radius, dwell overlap) is the first suspect, not the dynamics.
3. **Whether the 35 ms air activation is a dwell or overlaps the motion.** ABB says only that it "takes place during the cycle time". This changes the implied jerk. Treat both readings in calibration and report which one passes V6.
4. **Corner blending radius of the benchmark cycle.** Not published. Run a sensitivity check (0–25 mm).
5. **Lumping fraction r.** The Codourey original was not read; secondary sources conflict. The test (V4) decides.
6. **The official RAPID manual text for `AccSet`/`PathAccLim`** was only seen through forum quotes and a third-party reference.
7. **MJX.** If phase 3 uses MJX, check that `connect` equality and `surfacevel` are supported there before relying on the closed chain. Otherwise use Model B, or the reduced effector validated by V8, on MJX.

## 7. Sources

**ABB (primary)**
- IRB 360 Product specification 3HAC029963-001 rev. Z (2021), working range, ISO 9283 performance, acceleration, cycle times, mounting loads (copy hosted by ULiège): http://matheo.uliege.be/bitstream/2268.2/19593/46/3HAC029963%20PS%20IRB%20360%20on%20IRC5-en.pdf
- IRB 360 datasheet ROB0082EN (2022): https://library.e.abb.com/public/f03878770fef4c2c93757953373aed9b/IRB%20360_datasheet_20220829.pdf
- IRB 360 product guide (older revisions): https://library.e.abb.com/public/1c971fb90d8c34f8c1257c2100468677/ROB0082EN_F_HR.pdf · https://library.e.abb.com/public/c3eafb7f9dd274fac1257b1300579ef8/ROB0082EN_B%20final.pdf
- IRB 360 upgrade kit 6/1600 and 8/1130, ROB0322 (2015): https://library.e.abb.com/public/74edefaef36944b2b3075a2f850f71b3/irb360_rob0322.pdf
- 8 kg FlexPicker press release (2013): https://library.e.abb.com/public/76f011cd7ae30711c1257b7a002c3a01/PressRelease_ABBFlexPicker8kg.pdf
- PickMaster product specification 3HAC 5842-1 (IRB 340 era): https://library.e.abb.com/public/36a9e50804973b42c125766d003e8625/PPAspec.pdf
- Conveyor Tracking data sheet PR10045EN: https://library.e.abb.com/public/51c0fc4c87a036f6c1257b1300579b96/Data%20sheet%20Conveyor_tracking.pdf
- Björkman, Brogårdh, Hanssen, Lindström, Moberg, Norrlöf, "A New Concept for Motion Control of Industrial Robots", IFAC World Congress 2008 (QuickMove/TrueMove): https://skoge.folk.ntnu.no/prost/proceedings/ifac2008/data/papers/2519.pdf
- RAPID `AccSet` / `VelSet` (forum quotes of the manual): https://tech-community.robotics.abb.com/t/acceleration-accset/3346 · https://tech-community.robotics.abb.com/t/jerk-value/11999 ; `PathAccLim` (third-party reference): https://batpartners.github.io/en/instruction/Instruction-PathAccLim

**Delta kinematics and dynamics**
- R. Clavel, "DELTA, a fast robot with parallel geometry", 18th ISIR, 1988; R. Clavel, PhD thesis, EPFL, 1991; US Patent 4,976,582 (1990).
- R. L. Williams II, "The Delta Parallel Robot: Kinematics Solutions", Ohio University, 2016: https://people.ohio.edu/williams/html/PDF/DeltaKin.pdf
- A. Codourey, "Dynamic modeling of parallel robots for computed-torque control implementation", IJRR 17(12):1325–1336, 1998, doi:10.1177/027836499801701205 (abstract: https://www.osti.gov/biblio/302340). Also Codourey, IROS 1996 (mass matrix for axis-decoupling control).
- D. Gnad, H. Gattringer, A. Müller et al., "Dedicated Dynamic Parameter Identification for Delta-Like Robots", IEEE RA-L 9(5):4393–4400, 2024, doi:10.1109/LRA.2024.3380924 (open access: https://epub.jku.at/obvulioa/content/titleinfo/10236547/full.pdf)
- D. Gnad et al., "Identification of Physically Consistent Dynamics Parameter of the ABB IRB 360-6/1600 Delta Robot and its Use for Time-Optimal Motion Planning under Consideration of Constraint Forces", Robotics and Autonomous Systems 182:104823, 2024, doi:10.1016/j.robot.2024.104823 (CC BY: https://epub.jku.at/obvulioa/content/titleinfo/13412718/full.pdf)
- F. Aghili, "A unified approach for inverse and direct dynamics of constrained multibody systems based on linear projection operator", IEEE T-RO 21(5), 2005.
- J. E. Bobrow, S. Dubowsky, J. S. Gibson, "Time-optimal control of robotic manipulators along specified paths", IJRR 1985; K. G. Shin, N. D. McKay, IEEE TAC 1985; H. Pham, Q.-C. Pham, "A new approach to time-optimal path parameterization based on reachability analysis" (TOPP-RA), IEEE T-RO 2018.
- L. Biagiotti, C. Melchiorri, *Trajectory Planning for Automatic Machines and Robots*, Springer 2008 (double-S profile).
- L. Berscheid, T. Kröger, "Jerk-limited real-time trajectory generation with arbitrary target states" (Ruckig), RSS 2021.

**Third-party IRB 360 models**
- TU Gdańsk MecHaRo-Lab ROS 2 / MoveIt driver: https://github.com/micmzr/MecHaRo-Lab_IRB360 (`ws_ros2_abb/src/irb360/urdf/irb360.urdf`, `ws_ros2_abb/src/moveit_irb360_kinematics/.../irb360_kinematics_plugin.cpp`)
- https://github.com/harshalaptraise/delta-hil (`src/deltahil/plant/irb360_pose.py`)

**Other vendors**
- Omron Quattro 650H/HS datasheet: https://assets.omron.com/m/7010d1b816f4776f/original/Quattro-650H-650HS-Datasheet.pdf
- Yaskawa MPP3H flyer: https://yaskawa.co.il/wp-content/uploads/2020/08/Flyer_Robot_MPP3H_E_03.2020.pdf
- FANUC M-3 / M-1 series: https://www.fanucamerica.com/products/robot/m-3ia-6a · https://www.fanuc.co.jp/en/product/catalog/pdf/robot/RM-1iA(E)-06c.pdf
- Kawasaki YF003N spec (not retrieved): https://robotics.kawasaki.com/userAssets1/productPDF/YF003N-E.pdf
- Codian D4 HD catalog: https://pdf.directindustry.com/pdf/codian-robotics/d4-800-hd-d4-1100-hd-d4-1300-hd/126561-567291.html

**Lines and layouts**
- Packworld, "Robotic tray loaders speed cookie loading": https://www.packworld.com/leaders-new/machinery/conveying-accumulation/article/13325677/robotic-tray-loaders-speed-cookie-loading
- Packaging Strategies, "Expanding your horizontal packaging horizons" (delta into flow-wrapper infeed): https://www.packagingstrategies.com/articles/86576-expanding-your-horizontal-packaging-horizons
- Syntegon RPP datasheet: https://www.snackandbakery.com/ext/resources/images/directories/buyer_s_guide/2023/s/3608384/Syntegon_ss3.pdf

**MuJoCo (official)**
- `model/slider_crank/slider_crank.xml`, `model/surfacevel/treadmill.xml`, `model/surfacevel/carousel.xml`, `test/engine/testdata/equality_site.xml`: https://github.com/google-deepmind/mujoco/tree/main/model
- Docs: Computation → Equality constraints, Inverse dynamics (`doc/computation/index.rst`); XML reference `equality/connect`, `equality/tendon`, `geom/surfacevel` (`doc/XMLreference.rst`); Modeling → tuning table and solver parameters (`doc/modeling.rst`); changelog 3.11.0 / 4.0.0 (`doc/changelog.rst`): https://mujoco.readthedocs.io
- MuJoCo Menagerie closed chains: https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotiq_2f85/2f85.xml · https://github.com/google-deepmind/mujoco_menagerie/blob/main/agility_cassie/cassie.xml
