# Model spec for the delta suction pick cell (for review before coding)

Combines the three phase-0 research documents. Each section names its source document, where the equations, parameter tables and citations are:
- `docs/research/model-delta.md`: robot, motion, conveyors
- `docs/research/model-suction.md`: cup mechanics, pneumatics, sensor
- `docs/research/model-products.md`: catalogue, drift, line parameters

Tags used in those documents: **S** sourced, **D/C** derived or calculated from sourced values, **E/I** estimate or inference.

## 1. Robot: IRB 360-1/1130-class delta (model-delta.md §0, §5)

- **Class:**
  - 1 kg payload, Ø1130 mm working range;
  - 10 m/s, 150 m/s²;
  - 25/305/25 cycle: 0.300 s at 0.1 kg, 0.353 s at 1 kg (ABB product specification).
- **Geometry:** base radius 200, motors 275 below the base frame (ABB drawing), upper arm 350, strut 800, platform radius 45, parallelogram width 100, TCP about 50 below the platform joints (mm).
  - The non-ABB values come from a third-party ROS driver for a real IRB 360, and reproduce ABB's envelope.
  - **Decision needed:** ABB's drawing suggests an upper arm of about 366 and a base radius of about 185. Proposal: use the ROS-driver set and run the ABB-drawing set as a sensitivity case.
- **MuJoCo model:**
  - closed chain: 3 hinged upper arms, 6 struts on 2-hinge universal joints, a free platform, 6 `equality/connect` constraints;
  - stiff loop settings (`solref 0.001 1`, `solimp 0.99 0.999 0.001`), dt = 0.5 ms;
  - **measured in the prototype:** 0.044 mm tracking, 22 µm loop error at 150 m/s².
- **Dynamics prior:** Gnad et al. 2024 (IRB 360-6/1600, the only measured set), scaled down.
- **Calibration:**
  - fit the jerk limit to 0.300 s and the motor torque limit to 0.353 s (25/305/25);
  - hold out 90/400/90 (0.444 / 0.500 s, ±5%).
- **Controller:** computed torque (`mj_inverse` feedforward projected onto the 3 motors through the constraint Jacobian) plus joint PD, with torque limits.
- **Motion:** time-optimal along a fixed pick path under Cartesian speed, acceleration and jerk caps and motor-torque limits, as ABB documents for QuickMove.
  - **The speed override scales acceleration and jerk** (ABB `AccSet`, `PathAccLim`). This is our action.
- **Effector-only reduction** (today's `sim/suction.py`): allowed for bulk log generation only if test V8 passes (platform acceleration within 2% RMS and 5% peak of the full model, plus a paired drop-rate test). Otherwise the closed chain generates the logs.

## 2. Suction cup and vacuum (model-suction.md §0, §2–4)

- **Mechanics: Dex-Net 3.0 compliant ring** (Mahler et al., ICRA 2018):
  - each lip segment is a body on a spring-loaded slide joint along the cup axis, which cannot extend past its free length, so an unloaded segment lifts off and opens a gap;
  - optional tendon springs between segments (Dex-Net perimeter and flexion springs);
  - optional spring-mounted cup base (bellows).
- **Holding force:** F = Δp · A_eff, with A_eff the cup's **inner** effective area (Schmalz data matches the inner diameter within 0–6%).
  - **Change from today:** the force is applied with `mj_applyFT` at the cup centre to the product, with the reaction on the cup base, instead of the `adhesion` actuator. The lip springs therefore carry the vacuum preload (Dex-Net f_N = f_z + V).
- **Pneumatics:** isothermal chamber equation `dp/dt = (RT/V)(ṁ_leak − ṁ_ejector) − p·V̇/V`:
  - ejector curve from the Piab datasheet table;
  - leak flow summed over the lip-segment gaps measured in MuJoCo: viscous slit flow (∝ gap³) for narrow gaps, ISO 6358 orifice/choked flow for wide gaps;
  - plus a roughness gap and a porosity term per product.
  - **Already checked offline:** reproduces Piab's published evacuation times within 3% and 9%.
- **What comes out of physics, with no assumed curves:**
  - `seal_kpa` at touch: where ejector flow equals leak flow, as in SMC's adsorption-test method;
  - `vac_min_kpa` during the move: lip tilt and stretch → gap → leak → Δp falls → F falls (the near-miss signal);
  - drop.
- **The 0.45 peel factor is explained:** Dex-Net's ring starts lifting off at a moment of r·f_N/2. Measured −1% and +4%; the rigid-ring limit is r·f_N.
- **Decision needed:** ejector at the cup (one pressure node) or at the robot base with a hose (two nodes). Proposal: two nodes with the hose volume, because delta end effectors are usually fed from a valve or ejector off the platform; use one node only if research on the target end effector shows otherwise.

## 3. Products (model-products.md §2, §5)

Seven products, easy to hard, with priors to replace `sim/delta_toy.py`:

| Product | Mass | Notes |
|---|---|---|
| Pharma blister card | 5 g | Control product |
| Moulded praline | 10.5 g | Vendor cup limit −150 mbar; temperature and dew point affect μ |
| Digestive biscuit | 15 g | Porous; breaks at 12.5 N |
| Flow-wrapped bar | 45 g | Wrinkle leaks; Piab case: worn cup → vacuum loss → drop |
| Chilled cheese wedge | 200 g | Wet: μ 0.2–0.3; CoM 10–17 mm off-centre |
| Pet-food pouch | 85 g | Flexible; moving CoM |
| 10R vial | 18 g | Peel stress test only |

**Drift factors** (the scenarios TabPFN must adapt to):
- **suction-cup wear** (vendor service life about 500k–750k cycles);
- **condensation** on chilled products (μ drops);
- **crumbs and filter loading** (bakery);
- **batch changeovers** (mass and porosity shift).

**Dropped:** "humidity makes biscuits leak during a shift". Moisture uptake takes hours to days (food science), so this must go from the explainer page.

## 4. Conveyors and trays (model-delta.md §4.7)

- Belts are static boxes with MuJoCo's official `geom/surfacevel` (since 3.11). Checked: a 12 g product reaches 0.5 m/s in about 0.1 s.
- Conveyor tracking per ABB PickMaster: TCP velocity matched at touch. Placement error compared with ABB's tracking table (1.0 mm at 200 mm/s, 1.5 mm at 350–750 mm/s).

## 5. What stays unknown, and how we handle it

No published value exists for:
- lip and bellows stiffness and damping (bounded ≤ 1.24 N/mm for SPB1 20);
- lip land width and discharge coefficient;
- food surface permeability and roughness;
- cup-to-food friction;
- the delta motor torque limit;
- real drop rates.

Each becomes a **calibration parameter with a sourced range**. We sweep it and report how sensitive the results are (suction test V15). Where tests must be invariant (peel onset V8 does not depend on stiffness), we check that. Nothing gets a single assumed value presented as fact.

## 6. Validation gates (coding does not move past a gate until it passes)

1. **Pneumatics:** suction tests V1–V4 (evacuation times, time constants, steady seal, static force).
2. **Cup mechanics:** suction V5–V9b and V13–V14 (lift, shear with a 1 mm slip criterion, torsion, peel onset and ultimate, convergence, regression).
3. **Seal and warning:** suction V10–V12 (convex surfaces, seal vs roughness, **dip before drop**), plus the V15 sensitivity sweep.
4. **Delta:** tests V1–V7 (kinematics, workspace, loop integrity, cycle-time fit and held-out check, base forces).
5. **Reduction and line:** delta V8–V10 (effector-only validity, conveyor, tracked pick).

## 7. Build order

1. Pneumatics module (pure Python, unit-tested) → gate 1.
2. Compliant cup in MuJoCo with force through `mj_applyFT` → gates 2 and 3.
3. Delta closed chain, controller, motion planner, calibration → gate 4. Starts from `research/proto_delta/`.
4. Line: belts, trays, product catalogue, tracking → gate 5.
5. Pick-log generator with the real columns, then TabPFN in the loop (PLAN.md phases 3–5).
