# Delta robot validation (gate 4), `sim/delta.py`, `tests/test_delta.py`

## Part 1: model, kinematics, control (2026-11-22), 5 of 5 passed

| Test (model-delta.md §5.6) | Result | Reference |
|---|---|---|
| V1 IK satisfies the strut constraint; Jacobian vs finite differences | 0 nm; 2.5e-9 relative | exact |
| V1 assembled MuJoCo chain vs analytic geometry | 0.1 µm loop error; TCP exact | < 1 µm |
| V2 ABB Ø1130 envelope (468 points) within joint limits | θ −45.1…96.8° within −46.1…100°; Jacobian condition ≤ 5.1 | ABB product spec + ROS joint limits |
| V3 300 mm move, 150 m/s², 0.1 kg | TCP 0.143 mm; loop 24.6 µm; tilt 0.105 mrad | loop < 25 µm, tilt < 0.5 mrad, TCP below **ABB path accuracy AT 0.51 mm** |
| V4 feedforward quality (PD share of peak torque) | **exact closed-chain FF 3.8 %** (TCP 0.034 mm); lumped ⅔/⅓ 4.1 %; lumped ½/½ 8.0 % | < 5 % |

## Findings and decisions

1. **The upper-arm length is set to ABB's R = 366 mm**, replacing the ROS driver's 350 mm. With 350 mm, no TCP offset satisfies both ABB's published envelope and the joint limits: 30 mm breaks the upper limit, 45 mm and more breaks the lower one. With 366 mm, the envelope fits for TCP offsets of 30–50 mm and both base radii (200 or 185 mm). The primary source is the consistent one. 350 mm remains the sensitivity case.
2. **The V3 reference was changed from the prototype's own number (0.1 mm) to ABB's published path accuracy (0.51 mm).** That is a physical reference, not an internal one.
3. **The prototype's lumped model silently dropped the elbow masses.** MuJoCo ignores geom masses on a body that has an explicit `<inertial>`. Its good fit (2.8 %) came from a model missing half the strut mass. We reproduced its exact numbers with that effective model (3.79 vs 3.76 N·m), then fixed it.
4. **The lumped (Codourey) model is about 10 % off at light payloads, whatever the strut split** (½/½, ⅔/⅓, ⅓/⅓ by kinetic energy). The struts (0.78 kg) are a large part of the moving mass at 0.1 kg payload. We therefore added the **exact closed-chain feedforward** (`FeedforwardA`). It solves the strut angles by Gauss–Newton on MuJoCo's loop residual, the strut rates from the constraint Jacobian, accelerations by central differences, then runs projected `mj_inverse`. It is precomputed once per path.
5. **Friction.** The plant uses the identified friction law (Gnad et al.): Coulomb F_c·tanh(θ̇/v_s) as an applied torque, plus viscous joint damping. The controller compensates the same law. MuJoCo's `frictionloss` (a soft dry-friction constraint) is not used: compensating it with a smooth law left about 7 % of peak torque to feedback, a model mismatch rather than physics.
6. **Two traps in our own tests:**
   - MJCF hinge ranges are in degrees unless `<compiler angle="radian"/>` is set; we wrote radians, which pinned the arms.
   - After `mj_step`, `site_xpos` describes the pre-integration state, so tracking error must be measured against ref(t), not ref(t + dt). The wrong comparison measured v·dt ≈ 2.75 mm of fake error.

Torques use the prior inertias (Gnad et al. scaled) and are not yet calibrated. Next: motion planner, cycle-time calibration (V5), held-out check (V6), base loads (V7).
