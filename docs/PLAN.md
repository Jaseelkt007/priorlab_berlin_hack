# Plan (written Sat 2026-11-21, ~20:30; submission deadline Sun 16:30)

**Rule:** the timeline is a rough guide only. We do not cut scope or replace proper methods with assumptions because of the deadline. Where modelling is not settled, we research the established method (literature, official repos), implement it, and validate it against theory or published data.

**Pitch:** delta robots on packaging lines pick with suction at up to 15 g. Speed against drops is tuned by hand, per product, conservatively. We predict the drop risk of every speed per pick with TabPFN, from the line's own log and the vacuum reading at touch, with no training.

Background: `docs/research/application-delta-picking.md`, the explainer `docs/explainer/delta-pick-line.html`, `docs/research/tabpfn-latency.md`, `docs/research/suction-validation.md`.

## Phases

| # | What | Method (proper, sourced) | Done when |
|---|---|---|---|
| 0 | **Research the models we still lack** | (a) **Suction cup mechanics:** compliant seal and wrench-resistance models, e.g. Dex-Net 3.0's quasi-static spring model of the cup (Mahler et al., ICRA 2018), bellows and lip compliance. (b) **Vacuum pneumatics:** lumped-parameter pressure dynamics (ejector flow against leak flow through the lip gap, cup volume), so that seal level and the dip during the move come out of physics. (c) **Delta robot:** published geometry and dynamics (Clavel delta kinematics; IRB-360-class dimensions, limits and payloads), closed-chain modelling in MuJoCo (`equality/connect` constraints as in the Menagerie Robotiq 2F-85 and Cassie models; note: the official `slider_crank` example shows an actuator transmission, not closed chains). (d) **Conveyor and tray modelling** with MuJoCo's official `surfacevel`. (e) **Product physical data:** masses, dimensions, surface friction and porosity of the catalogue products from sources | a written model spec with citations in `docs/research/`, reviewed before coding |
| 1 | **Suction model, done properly** | Implement the researched cup compliance and vacuum pressure dynamics on top of the validated adhesion and friction contact. Validate lift and shear (already within 4%), peel, and the **pressure response** (seal at touch for porous vs. smooth surfaces, dip under load before failure) against the models and published data | every check matches its reference within stated tolerance, documented |
| 2 | **Delta cell** | Closed-chain delta with real geometry and motor/torque limits (or a justified, validated reduction if the research shows the arm dynamics do not affect the grip); jerk-limited cycle paths (typical 25/305/25 mm); `surfacevel` belts; trays; products with hidden friction, CoM, porosity, softness | standard cycles run at published speeds; tracking and timing match the robot class |
| 3 | **Pick-log generation** | Picks across speeds, products, humidity, cup wear and changeovers, in parallel (CPU workers, MJX if useful). Log with the real columns | datasets for each scenario, with drop rates spanning the speed range; sample committed, full data gitignored |
| 4 | **TabPFN in the loop** | Drop classifier and vacuum-dip regressor, cached context per product; per-pick query (CUDA graph, about 7 ms) and speed map; closed-loop runs | both decision modes work inside the cycle-time budget |
| 5 | **Baselines and evaluation** | Fixed recipes; safe recipe; **GOMP-ST reimplemented** (learned acceleration limits from pressure data); MLP and gradient boosting on the same rows; RL with randomized physics. Metrics: good packs/min, drop rate, picks to recover after a changeover, calibration (reliability, ECE, Brier), latency. Many seeds, confidence intervals, held-out products | results table with intervals; every pitch number traceable to a run |
| 6 | **Visuals and submission** | MuJoCo-rendered clips; explainer page updated with measured numbers; 3-minute video, slides, README with reproduce steps; repo made public after a secrets check | submitted before 16:30 |

## Parallel work

- **Main Claude session:** phases 0 → 5.
- **Second Claude session:** JAX port of TabPFN-3.5-Fast (`docs/handoff/jax-port.md`, branch `jax-port`).
- **Team:**
  - video script and slides from the explainer page;
  - review of results and calibration;
  - README and repo hygiene;
  - **ask the Prior Labs mentors how the track is judged.**

## Rules

- Every number in the pitch comes from a run; illustrative numbers are labelled.
- Shortcuts we have taken are in `DEBT.md`. They are work to replace, not accepted design.
- No weights, tokens or coupon codes in git.
