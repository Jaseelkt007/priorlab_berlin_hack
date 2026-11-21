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
