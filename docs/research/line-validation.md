# Line validation (gate 5): `sim/line.py`, `sim/products.py`, `tests/test_line.py`

The pick line: the compliant suction cup (gates 1–3) on the closed-chain IRB 360-class delta (gate 4), picking from a moving belt with ABB PickMaster-style belt tracking, then placing on a pad. One simulated pick is one full cycle on the full closed chain at dt = 0.2 ms:

1. empty approach at full speed, tracking the belt;
2. touch and a 35 ms vacuum dwell (ABB's air-activation time); the vacuum sensor's `seal_kpa` and `evac_ms` are read here, **before** the speed decision;
3. loaded transfer at the speed setting α (ABB AccSet scales the acceleration and jerk caps);
4. outcome at arrival: `dropped`, `place_offset_mm`.

## Tests

`tests/test_line.py`: 8 of 8 passed (with `tests/test_pneumatics.py`, `test_cup.py` and `test_seal.py`: 25 of 25).

| Test | Result | Reference |
|---|---|---|
| V9 belt drives the product (0.2 / 0.5 / 1.4 m/s) | speed error 0.000 % at all three; drift 0.000 mm/s | < 0.5 %, < 1 mm/s; MuJoCo `geom/surfacevel` |
| V10 tracked pick: TCP–product relative velocity at touch | 0.43 mm/s (0.2 m/s belt), 0.67 mm/s (0.5 m/s) | < 5 mm/s (PickMaster matches the TCP to the belt) |
| V10 placement error with vision error (20 picks) | 1.30 mm at 0.2 m/s, 1.96 mm at 0.5 m/s | σ·√(π/2) = 1.25 / 1.88 mm from ABB's tracking table (σ 1.0 / 1.5 mm), ±35 % |
| Drop boundary vs Schmalz sizing (50 g, μ 0.6) | first drop at α 0.5 (74 m/s²): S* = 1.94 | Schmalz F = m(g + a/μ)·S, S ≥ 2 recommended; S* in [1.5, 2.5] |

V9 at 1.4 m/s first failed by +1.7 %. The test had measured a product that ran off the end of the 1.2 m belt and was falling. The window now keeps it on the belt (≤ 0.5 m of travel), and the test checks that it is.

## Findings

1. **The cup-only reduction gives no speed benefit**, so every logged pick runs on the full closed chain. The full delta plus cup takes 62 µs per step plus pneumatics, against 143 µs for the slide-effector `CupCell`. The reduction-validity test V8 is therefore moot.
2. **The drop mechanism, traced.** Under sustained lateral acceleration the compliant cup lets the product tilt: 6.6° at 73 m/s², against 6.8° analytic. At the acceleration reversal, the ~20 Hz tilt mode overshoots to −14°, one lip lifts and the vacuum collapses within 7 ms. The drop boundary matches **Schmalz's horizontal sizing rule F ≥ m(g + a/μ)·S with S ≈ 2** (test below), an external check.
3. **Cup wear is a parallel leak path, not a wider roughness gap.** Adding the wear gap to the roughness gap made the cubic slit law turn the 15 % leak-down rule into 30–40 % on film products. The seal fell by 2.1 kPa per 100k cycles on the cheese, and drops appeared at the slowest speed. Wear now opens a viscous leak K_w·(p_a² − p_c²) in parallel with the product's roughness and porosity, as abraded or cracked lip channels do. It grows linearly with cycles and is calibrated per product to 15 % leak-down at 750k cycles (13–17 % across each film product's seal class).
4. **Cup volumes come from vendor tables**, replacing a Ds³ scaling of the SPB1 20. That scaling put 31 cm³ in a Ø45 flat cup; Schmalz SPF 40 holds 5.09 cm³. Flat cups use the Schmalz SPF series (0.8–5.09 cm³), bellows cups the SPB1 table, and the praline uses the SPG bell (8.1 cm³).
5. **The praline cup is the real Schmalz SPG bell.** A ring sitting on the praline's Ø22 top leaked whenever the 1–1.5 mm vision error pushed segments over the edge. The share of low seals rose from 0 % below 0.5 mm of vision error to 100 % above 3 mm, and drops became speed-independent. The bell is now modelled from its datasheet drawing:
   - a conical wall at 28° from the axis, from the Ø32.1 mouth to the Ø15.7 neck;
   - the seal line on the praline's rim;
   - the vacuum acting on the whole top inside the seal line, through the top's centre;
   - only the volume above the seal circle evacuated (8.1 cm³ minus the cone below the seal line).

   The wall's stiffness is derived from 45 Shore A silicone (Gent: E = 2.0 MPa) and a ~1 mm wall bending between the neck and the seal circle. That derived wall captures the praline as a bell should (capture map below). The ED-65 bellows stiffness used before pinned an off-centre praline to the belt: the far wall meets the rim 1.9·e early and is over-compressed.
6. **Cheese uses a catalogue cup with edge margin**: Schmalz SPF 40 (Ds 41.4 mm, 49.5 N at −0.6 bar, so d_eff 32.4 mm). Its lip stays inside the 50 mm face with a 3.5 mm margin, more than 2σ of the vision error.

### Bell capture map (praline, belt 0.3 m/s, α 0.3; offset e imposed along x)

| lip stiffness | μ 0.3, e 2 / 2.5 / 3.5 / 4.5 mm | μ 0.7, e 2 / 2.5 / 3.5 / 4.5 mm |
|---|---|---|
| SPB1-scaled ring, 4080 N/m (old) | held / held / held / **drop** | **drop** at all four |
| thin silicone wall, 27 N/m (derived) + force at the top's centre | held, seal 14.8 kPa (e ≤ 1 at full speed: place error 0.7 mm, tilt ≤ 11°) | e 3.5: held, seal 11.2 kPa; at 4.5 and μ 0.5: held, seal 5.3 kPa |

A large offset now gives a **lower seal at touch**, which a real vacuum sensor reads before the move. Remaining misses (seal ≈ 1 kPa) need an offset above about 4 mm together with high friction, and they are visible at touch.

## Drop rate vs speed setting (20 picks per cell, random line states)

`scripts/sweep_picks.py 20 data/sweep_v2.csv`: 1,080 picks, 6 line products × 9 settings × 20 picks, random line states (cup 0–900k cycles, 19–28 °C, 35–75 % RH, batch shifts), belt 0.2–0.5 m/s, 1,164 s on 11 laptop workers. It was generated just before the planner was split into an approach and a transfer fixed point (same tolerance): re-running it reproduces every move time exactly and every reference within 4 µm.

| product | seal_kpa (median) | α 0.2 | α 0.3 | α 0.4 | α 0.5 | α 0.6 | α 0.7 | α 0.8 | α 0.9 | α 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| blister | 58.7 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| praline | 14.8 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 5% | 0% |
| digestive | 28.4 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 5% | 0% |
| bar | 48.6 | 5% | 0% | 0% | 5% | 20% | 95% | 100% | 100% | 100% |
| pouch85 | 48.5 | 0% | 0% | 5% | 10% | 45% | 40% | 80% | 100% | 100% |
| cheese_wedge | 56.8 | 0% | 50% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

Median placement error of held picks (mm):

| product | α 0.2 | α 0.3 | α 0.4 | α 0.5 | α 0.6 | α 0.7 | α 0.8 | α 0.9 | α 1 |
|---|---|---|---|---|---|---|---|---|---|
| blister | 1.5 | 1.7 | 1.8 | 1.3 | 1.4 | 1.3 | 1.6 | 1.3 | 1.3 |
| praline | 1.1 | 1.2 | 1.3 | 0.9 | 1.2 | 1.2 | 1.2 | 1.1 | 1.0 |
| digestive | 1.4 | 2.8 | 1.4 | 1.4 | 1.4 | 1.4 | 1.6 | 2.4 | 3.3 |
| bar | 1.6 | 1.7 | 1.8 | 1.3 | 1.9 | 8.1 | – | – | – |
| pouch85 | 3.5 | 13.1 | 13.2 | 16.3 | 14.2 | 14.2 | 12.9 | – | – |
| cheese_wedge | 1.7 | 2.5 | – | – | – | – | – | – | – |

- **Easy products** (blister, praline, digestive) hold at every setting. The few drops are a capture miss (praline: 4.3 mm vision error at μ 0.64, seal 1.1 kPa at touch), an edge miss (bar at α 0.2: 4.65 mm vision error, a 3σ event, seal 13.8 kPa) and one worn-cup drop (digestive: 824k cycles at 129 m/s²). Both misses show as a low `seal_kpa` before the move.
- **The bar, pouch and cheese have their speed limit inside the range:** bar 0.6–0.7, pouch graded from 0.4 to 0.9 (contents slosh), cheese 0.3 (200 g with an off-centre CoM). This is where per-pick speed choice pays.
- The pouch's held placement error is 13–16 mm above α 0.3. The sloshing contents shift the bag on the cup.

## Known gaps

See `DEBT.md`. In particular: lumped bell wall (no shell model); scaled lip stiffness for flat and bellows cups; round flat cup for the bar; box for the wedge; the condensation probability and the film's wet friction are estimates.
