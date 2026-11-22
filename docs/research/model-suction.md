# Suction cup model: mechanics, vacuum pneumatics, MuJoCo design

Phase-0 research, 2026-11-21. Scope: model a single suction cup on a delta-robot pick line well enough that **the vacuum reading (seal at touch, dip during the move) and the drop come out of physics**, and validate it.

How to read this document:
- **[S]** = established, taken from the cited source (number in brackets refers to the source list at the end).
- **[C]** = our own calculation from cited data (method given, reproducible).
- **[I]** = engineering inference or design choice by us. Not from a source. Must be validated or calibrated.
- "Not found" means we searched and found no primary source. The doc says how to calibrate instead.

---

## 0. Summary of the recommendation

1. **Mechanics: the Dex-Net 3.0 compliant ring model, built from real springs.** Dex-Net 3.0 models the cup as a ring of axial springs: lip pressure `p(θ) = k·δz(θ)` [1]. In MuJoCo, each lip segment becomes a small body on a spring-loaded slide joint along the cup axis. The lip can only extend back to its free length, so a segment that unloads lifts off and a gap opens. Products tilt the ring; segments unload; gaps open. Optional tendon springs between neighbouring segments stand in for Dex-Net's perimeter and flexion springs. A spring-mounted cup base on top gives the bellows its axial and tilt compliance.
2. **Pneumatics: a lumped chamber ODE.** The ODE takes the cup and hose volume, a tabulated ejector curve from a vendor datasheet, and leak flows. Leak flows go through:
   - each lip-segment gap (viscous slit flow for narrow gaps [9], ISO 6358 orifice or choked flow for wide gaps [12]);
   - an equivalent roughness gap per product;
   - a porous-product term.

   The pressure sets the holding force `F = Δp·A_eff`. The force is applied physically at the cup centre to the product, and the reaction goes to the cup base. The lip springs therefore carry the vacuum preload, as in Dex-Net (`f_N = f_z + V`).
3. **What comes out of physics:**
   - **Seal at touch:** the root of `Q_ejector(Δp) = Q_leak(Δp)`. This is the SMC "leakage from adsorption test" relation [14].
   - **Dip before the drop:** the lip lift-off gap from the simulated tilt and stretch feeds the leak, which lowers Δp, which lowers F. There is no assumed leak curve.
4. **Key finding from re-analysing our existing spike [C]:** the "peel factor ≈ 0.45" is not arbitrary.
   - It matches the **Dex-Net linear-ring lift-off onset**, `M_onset = r·f_N/2`, with `f_N = F − m·g`. Error is −1% and +4%. See §2.4.
   - The rigid-ring ultimate limit is `M = r·f_N` (Pham & Pham [5], Lee et al. [6]). The real failure lies between these two, and the pressure dynamics decide where.
5. **Validated offline [C]:** the isothermal chamber ODE, fed with Piab's published flow-vs-vacuum table, reproduces Piab's published evacuation-time table within 3% (Si08-3) and 9% (Pi12-2). See §3.2. The pneumatic core is therefore not ad hoc.
6. **Main unknowns:**
   - lip and bellows stiffness and damping (no vendor publishes them);
   - lip land width;
   - permeability and roughness of food surfaces.

   §7 gives a calibration route for each, plus sensitivity ranges.

---

## 1. Starting point and the gap

The current model is `sim/suction.py` and `docs/research/suction-validation.md`:
- 8 rigid lip spheres;
- the MuJoCo `adhesion` actuator with gain = F;
- explicit friction pairs.

Lift and shear match within 4%. Peel fails at ≈0.45× of `F·r/(m·h)`. There is no warning before failure, because the contacts are rigid and there is no pressure state.

How MuJoCo's adhesion actuator behaves, from the docs [20]:
- It "injects forces at contacts in the normal direction". The force is "divided equally between multiple contacts".
- Without the `gap` attribute, it "requires active contacts and cannot apply a force at a distance".
- With `gap`, contacts are detected within `margin+gap` but generate no contact force. The adhesion force can act across that layer.

The official example `model/adhesion/active_adhesion.xml` uses `gap=".01"` on the adhesive geoms. Its comment reads: "a layer of inactive contacts at a 1cm distance from the surface. The contacts can be used by the adhesion actuator, but no contact forces are generated in the gap zone" [21].

**[I]** Our spike sets no `gap` on the lip. Once one side's lip contacts separate, the total F is re-divided among the remaining contacts on the loaded side. The resultant then moves toward the pivot and the hold collapses. This is consistent with the observed sudden release at lift-off onset (§2.4).

---

## 2. Suction cup mechanics

### 2.1 Holding force: Δp × effective area

- **[S]** Theoretical holding force is pressure difference × effective area: `F = Δp·A` [13][15].
  - SMC writes it as `W = P × S × 0.1` (W in N, P in kPa, S in cm²), divided by a safety factor t [14].
  - Pham & Pham model the suction force as `f0 = [0, 0, P·A]` at the cup centroid [5].
- **[S]** Schmalz SPB1 bellows cups (1.5 folds, Elastodur ED-65) [16]:

  | Size | Suction force at −0.6 bar (N) | Pull-off force (N) | Lateral force (N) | Volume (cm³) | Stroke Z (mm) | Min. convex workpiece radius (mm) | d2 (mm) | Ds (mm) |
  |---|---|---|---|---|---|---|---|---|
  | SPB1 10 | 1.5 | 4.0 | 2.0 | 0.5 | 4 | 5 | 5.6 | 9.8 |
  | SPB1 15 | 4.3 | 9.0 | 5.0 | 1.5 | 6 | 8 | 9.5 | 15.4 |
  | SPB1 20 | 9.9 | 16.2 | 9.8 | 3.2 | 8 | 15 | 13.8 | 21.4 |
  | SPB1 25 | 14.9 | 23.4 | 13.0 | 5.3 | 11 | 20 | 16.9 | 26.4 |
  | SPB1 30 | 20.7 | 30.6 | 15.5 | 8.0 | 13 | 25 | 20.0 | 31.4 |
  | SPB1 40 | 35.9 | 46.8 | 24.8 | 18.4 | 16 | 40 | 26.1 | 41.4 |

  Schmalz notes: "suction forces are theoretical values at −0.6 bar and a dry, smooth and even workpiece surface … without safety factors".
- **[C]** Inverting the table for effective area: `d_eff = sqrt(4F/(π·60 kPa))` gives 5.64, 9.55, 14.5, 17.8, 21.0 and 27.6 mm. That is **1.00–1.06 × d2**. So Schmalz's "suction force" uses the inner diameter d2, not the nominal size. **Use `A_eff = F_suction(−0.6 bar)/60 kPa` from the datasheet, not π·(nominal/2)².**
- **[C]** The pull-off force is larger than the suction force: 1.2–2.7×. Its equivalent diameter lies between d2 and Ds (for example SPB1 20: 18.5 mm). Lee et al. measured the same kind of gap: SPB1 60 at 600 mbar gives 78 N suction force and 100.9 N pull-off. They attribute the excess to "adhesive forces … as well as the deformation of the suction cup" [6]. **[I]** When pulled, the lip spreads and the sealed area grows toward Ds. §4.2 offers an optional `A_eff(lip extension)`.
- **[S]** Effective area also shrinks with vacuum for bellows cups. Piab B25XP (PU 30/60 dual hardness, 26 mm OD) [17] gives vertical/parallel lifting forces of:
  - 9.7/8.0 N at 20 kPa;
  - 19.0/12.0 N at 60 kPa;
  - 22.0/15.0 N at 90 kPa.

  Piab BX25P gives 8.0/5.0, 13.0/10.0 and 17.0/12.0 N at the same levels. Piab F-BX25 (food-contact silicone, Shore 40) gives 3.9/2.9, 6.9/4.6 and 8.0/7.3 N at 10, 20 and 40 kPa. Lbf was converted to N.
- **[C]** For B25XP, `d_eff = 24.8 → 20.1 → 17.6 mm` from 20 to 90 kPa. **Use a tabulated `A_eff(Δp)` when the datasheet gives one.** Otherwise use a constant A_eff from the single Schmalz −0.6 bar point, and record this in DEBT.md.

### 2.2 Vendor sizing formulas, safety factors, friction

- **[S] Schmalz load cases** [13]:
  - I (cup horizontal, force vertical): `F_TH = m·(g + a)·S`
  - II (cup horizontal, force horizontal): `F_TH = m·(g + a/μ)·S`
  - III (cup vertical, force vertical): `F_TH = (m/μ)·(g + a)·S`
  - Example acceleration: 5 m/s². Schmalz advises considering emergency stops.
  - Safety factor: "minimum value of 1.5 for smooth and dense workpieces"; "2.0 or greater … for critical, heterogeneous, porous, rough or oiled workpieces".
- **[S] SMC** [14][18]: `W = P × S × 0.1 × 1/t`, with safety factor **t ≥ 4 for horizontal lifting** and **t ≥ 8 for vertical pad contact** (dynamic values).
- **[S] Robotiq** [19]: at least 1.5 for smooth objects, 2.0 for porous or oily objects, up to 4.0 with the cup face vertical.
- **[S] Friction μ:**
  - Schmalz reference values [13]: 0.5 for "wood, metal, glass, stone, etc."; 0.6 for rough surfaces; 0.2–0.3 for wet surfaces; 0.1–0.3 for oiled surfaces. Schmalz states μ must be determined by tests.
  - Kern's KIT dissertation [22] measured much higher values for active vacuum grippers:
    - dry Plexiglas μ = 1.84 (vendor value 0.5);
    - bush-hammered granite 1.14;
    - 120-grit sandpaper with lubricated lip 1.42 (vendor 1.1).

    It concludes that generic vendor μ values are unsuitable for design. **[S]** Kern also observed that on 120-grit paper "the sealing lip gradually buckled, leading to increasing leakage and a continuous decrease of the vacuum" under shear. This is direct evidence of shear-induced leakage before failure.
  - Pham & Pham identified μ = 0.3 with a force/torque sensor [5].
  - Dex-Net 3.0 samples μ ~ N(0.5, 0.1) truncated to [0, 1] [1].
- **[C]** Lateral/vertical capacity ratios from vendor data (test surface not stated; treat as an effective μ):
  - Schmalz SPB1: 1.33 (10 mm), 1.16, 0.99, 0.87, 0.75, 0.69 (40 mm).
  - Piab BX25P: 0.62–0.77. B25XP: 0.63–0.83. F-BX25 silicone: 0.67–0.91.
- **Not found:** measured μ for NBR, silicone or PU cups on biscuit, flow-wrap film (OPP/PE), aluminium foil or chocolate. **Calibration:** sample μ per product from a band, for example 0.3–1.2, covering the Schmalz to Kern range. Record it as a hidden product property.

### 2.3 Rigid-contact wrench models: the upper bound

- **[S] Pham & Pham 2019, "Critically fast pick-and-place with suction cups"** [5]. The cup–object contact is m point contacts on the lip, each with a linearised Coulomb cone, plus a suction force `[0, 0, P·A]` at the centroid. A wrench is feasible if contact forces exist. This is a weak contact-stability condition, the same structure as our current MuJoCo model.
  - Their rig: 12.5 mm cup radius, about 30 kPa vacuum, m = 6 points, μ = 0.3, a 0.551 kg notebook.
  - 20/20 transports succeeded with the constraint; 4/20 without it.
  - At 120–140% speed they saw about 1 cm and 10° of slip.
- **[S] Lee, Sun, Bylard, Sentis 2024** (multi-cup) [6]. Per-cup suction-loss condition with suction force ψ and pad radius r_pad: `f_z + ψ ≥ 0` and `±m_x ± m_y ≤ r_pad·(f_z + ψ)`. This is a diamond-shaped centre-of-pressure condition. Slip is constrained with friction. The load is distributed over cups by minimising spring potential energy (a QP), with a stiffer "bottomed-out" phase.
- **[S] GOMP-ST's analytic baseline (GOMP-FIT)** [4] assumes rigid cups as point contacts, Coulomb friction and no leaks. For their 4-cup, 30 mm rig, "sliding will always occur before moment imbalance".
- Bahr et al. 1996 added moment resistance about the suction centre for a wall-climbing robot. Mantriota 2007/2011 modelled contact forces and torsional friction of flat cups and compared edge contact with full contact [1][23][24]. Full texts are paywalled; known here through Dex-Net's related-work section and abstracts.

**Rigid-ring moment limit [C]:** the centre of pressure must stay inside the ring, so `|M| ≤ r·f_N`, where `f_N = F_vac + f_z`. For a hanging product, `f_N = F − m·(g + a_z)`.

### 2.4 Dex-Net 3.0 compliant model (Mahler et al., ICRA 2018 [1]): the equations

**Assumptions [S]:** quasi-static physics with Coulomb friction; rigid, **non-porous** objects; "a single disc-shaped suction cup made of linear-elastic material". Hardware: ABB YuMi, a "15 mm diameter silicone single-bellow suction cup" and a VM5-NC VacMotion vacuum generator, payload about 0.9 kg.

**(a) Seal formation: a quasi-static spring model C(n, r, h) [S].**
- The undeformed structural springs form a right pyramid of height h. Its base is a regular n-gon of circumradius r (n = 8 in their figure).
- Vertices are `{v1..vn, a}`. There are three spring sets:
  - **perimeter** springs v_i–v_{i+1};
  - **cone** springs v_i–a;
  - **flexion** springs v_i–v_{i+2}, which resist bending.
- Contact state:
  - Perimeter vertices are projected along the approach direction v onto the mesh. "The perimeter springs of C must not deviate from the original undeformed regular n-gon when projected onto a plane orthogonal to v."
  - The apex a stays on the approach line, at average axial distance h from the base vertices: `a* = p − t*·v`, with `t* = min( (1/n)·Σ_i (v_i − p)ᵀv − h, 0 )`.
- A seal is feasible if:
  - the cone faces do not collide with the mesh;
  - there are no holes inside the contact ring;
  - **each** spring's energy stays below a threshold E.
- In practice they use "a per-spring threshold of a 10% change in length" (from Provot's cloth model). They threshold per spring, not in total, "because air gaps are usually caused by local geometry".

**(b) Suction ring contact model [S].** The wrench basis is:
- actuated normal force f_z;
- vacuum force V (a constant pull);
- friction (f_x, f_y);
- torsional friction τ_z;
- elastic restoring torques τ_x, τ_y.

The derivation (Appendix III) assumes the lip is "a ring of infinitesimal springs per unit length": `p(θ) = k·δz(θ)`. The lip displacement is linear: `δz(θ) = δ0 + a·r·cosθ + b·r·sinθ`. So:
```
p(θ) = p0 + px·cosθ + py·sinθ
f_N  = ∫ p dθ = 2π·p0 = f_z + V
|f_t| ≤ µ·f_N ,   |τ_z| ≤ r·µ·f_N             (pure translation / pure rotation bounds)
ellipsoid:  |f_t|²/(µ f_N)² + τ_z²/(r µ f_N)² ≤ 1
τ_x = π·r·p_y ,  τ_y = π·r·p_x ,  ||τ_e|| = π·r·sqrt(px²+py²) ≤ π·r·κ
linearised set used in the QP:
  √3|f_x| ≤ µ f_N,  √3|f_y| ≤ µ f_N,  √3|τ_z| ≤ r µ f_N
  √2|τ_x| ≤ π r κ,  √2|τ_y| ≤ π r κ,   f_z ≥ −V
```
Here κ is "a material-dependent constant modeling the maximum stress for which the suction cup has linear-elastic behavior".

**Parameters [S]:** "In this paper we set V = 250 N and κ = 0.005." The parameters (n, r, h, E, V, µ, κ, ε) "were set by maximizing average precision … using grid search" against YuMi grasps.

**[I] Caution:** V = 250 N and κ = 0.005 are classifier-fitted constants, not physical values. Physically, Δp·A for a 15 mm cup is about 10 N. Use Dex-Net's **structure**, not these numbers.

**Consequence for peel [C], new and important.** Dex-Net's ring has no tension: the lip cannot pull, so `p(θ) ≥ 0`. The first lip lift-off occurs when `|p1| = p0`:
```
M_onset = π·r·p0 = r·f_N / 2          (Dex-Net linear ring, onset of lip lift-off / leak)
M_rigid = r·f_N                       (rigid ring, CoP at the rim: Pham&Pham, Lee et al.)
```
For n discrete equal springs at radius r, `Σcos²θ_i = n/2` gives the same `r·f_N/2`.

Check against our spike [2] (`scripts/spike_suction.py`, peel test: F = 8 N, r = 10 mm, m = 0.1 kg, CoM h below the lip, horizontal acceleration). The normal force is `f_N = F − m·g = 7.02 N`.

| h | Measured a* | Onset `r·f_N/(2·m·h)` | Diff. | Rigid `r·f_N/(m·h)` | Old reference `F·r/(m·h)` |
|---|---|---|---|---|---|
| 2 cm | 17.4 | **17.55** | −1% | 35.1 | 40.0 |
| 4 cm | 9.1 | **8.77** | +4% | 17.5 | 20.0 |

**[I]** Interpretation:
- MuJoCo's soft contacts on a rigid product behave like the Dex-Net linear spring ring.
- With no `gap`, the adhesion actuator re-divides F as soon as the first lip contact is lost, and the hold collapses at onset.
- A real cup leaks at that onset and fails somewhere between `M_onset` and `M_rigid`, decided by the leak and the pressure dynamics. That is what §3–§4 add.
- **Side note [C]:** the spike's shear reference `µF/m` also omits gravity. With `f_N = F − m·g`, the reference becomes 27.1 / 54.1 m/s², and the measured 29.2 / 57.9 m/s² is 7–8% high. The test keeps a product as "held" if it slips less than 1 cm, so slip is tolerated. Re-run with a slip criterion of 1 mm (test V6).

### 2.5 Other compliant and deformation models

- **[S] Hudoklin et al., IEEE RA-L 2022** [7]: "computationally efficient modeling of vacuum suction grippers". Cup deformation is split into "compression force" (axial) and "bending force … modeled by a beam deflection with fixed ends". The model gives seal-formation conditions and load capacity on flat and curved surfaces, and runs "in real-time" in CoppeliaSim. Full text paywalled.
- **[S] Gabriel et al. 2020** [8]: an "experiment-based modeling method that considers the dynamic deformation behavior of vacuum grippers". "Gripper deformation is reversible up to a certain limit". They propose deliberately allowing deformation within this stability range.
- **[S] Stegmaier et al. 2023** [10]: a behaviour model for digital twins of suction cups that simulates "the vacuum curve … mainly based on the force-displacement characteristic". The cup is a variable-volume chamber, where most models use constant-volume chambers.
- **[S] GOMP-ST** (Avigal et al., WAFR 2022) [4]:
  - Slow-motion video shows "the suction cups deform but still maintain a seal" before breaking. "Failures are not an instantaneous response". The learned constraint uses a state history at least as long as the deformation-to-failure time, "> 0.1 seconds".
  - Failure is labelled from the pressure trace: "scans the recorded pressure data to find the time at which suction pressure is lost".
  - Hardware: 4 × 30 mm elastodur flat cups, VacMotion MSV 27 generator, Adafruit MPRLS pressure sensor "close to the suction cups". The sensor runs at **167 Hz** and the UR5 joint data at 125 Hz.
- **[S] Huh et al., IROS 2021** (multi-chamber smart cup) [11]:
  - "The Pvac measure in each chamber does not vary while the vacuum seal remains, but lowers at the onset of the leak."
  - With an 85 kPa source, the measured vacuum rises with surface smoothness, "reaching 74 kPa for the smoothest 600 grit sandpaper".
  - Sensors: MPRLS, 24-bit, "RMS noise of 5.0 Pa", 166.7 Hz. Valve: SMC VQ110, on/off 3.5/2 ms.
  - CFD shows about 0.4 kPa differences between chambers during leaks.
  - **[I]** For us: the pressure signal before failure comes mainly from **leak onset**, not from elastic deformation alone. This supports leak-driven dips.

---

## 3. Vacuum pneumatics

### 3.1 Chamber pressure ODE

**[S]** The standard lumped pneumatic chamber model (Richer & Hurmuzlu 2000 [25]) is energy and mass balance on a control volume with polytropic gas. Written for a vacuum chamber of absolute pressure p and volume V:
```
dp/dt = (n·R·T / V) · (ṁ_in − ṁ_out)  −  n · p · (dV/dt) / V
  ṁ_in  = Σ leak flows from atmosphere into the cup (lip gaps, roughness, porous product)
  ṁ_out = ejector suction mass flow = ρ0 · Q_ej(Δp),   Δp = p_atm − p
```
- **[C]** Use **n = 1 (isothermal)**. The check in §3.2 matches the vendor evacuation table with n = 1 and is 28–35% too fast with n = 1.4.
- The `dV/dt` term is the variable volume of the bellows cup (Stegmaier [10]): `V = V_cup0 − A_b·z_comp + V_hose`, where z_comp is the bellows compression.
- **[S] Constants:**
  - air R = 287 J/(kg·K), γ = 1.4;
  - ANR reference density ρ0 = 1.185 kg/m³ (ISO 6358/8778 reference atmosphere: 20 °C, 100 kPa, 65% RH);
  - the Piab datasheets test at 101.3 kPa ambient, 20 ± 3 °C [26].

### 3.2 Ejector (venturi) characteristic and evacuation time

**[S]** Vendors publish suction flow vs vacuum as tables. Use them directly, with monotone interpolation. Do not assume a curve.

| Piab COAX® cartridge (feed) | Q (Nl/s) at 0 / 10 / 20 / 30 / 40 / 50 / 60 / 70 / 80 −kPa | Max vacuum | Air consumption | Evacuation time (s/l) to 10 / 20 / … −kPa |
|---|---|---|---|---|
| MINI Si08-3 (0.60 MPa) [26] | 1.34 / 0.73 / 0.55 / 0.35 / 0.23 / 0.17 / 0.13 / 0.08 / – | 75 kPa | 0.44 Nl/s | 0.10 / 0.25 / 0.48 / 0.80 / 1.30 / 2.00 / 2.90 |
| MINI Pi12-2 (0.34 MPa flow; 0.314 MPa evac.) [26] | 0.68 / 0.60 / 0.44 / 0.27 / 0.19 / 0.14 / 0.10 / 0.06 / 0.03 | 90 kPa | 0.44 Nl/s | 0.17 / 0.32 / 0.58 / 1.1 / 1.8 / 2.7 / 4.0 / 6.4 |
| MICRO Si02-2 (0.6 MPa) [27] | 0.083 / 0.062 / 0.035 / 0.024 / 0.021 / 0.018 / 0.012 / 0.0059 scfm at 0 / 2 / 5 / 8 / 11 / 14 / 17 / 20 inHg (≈ 0–68 kPa; ×28.3 → l/min) | 10.88 psi ≈ 75 kPa | 0.25 scfm | (s/cf table in sheet) |

The Si02-2 sheet says it is "suitable to integrate close to the suction point in high speed pick-and-place applications of small objects", with flow and evacuation accuracy ±10% [27].

**Vendor evacuation formulas [S]:**
- **Schmalz** [28]: `t = V_G · ln(p_a/p_e) · 1.3 / V̇`. Here V̇ is the generator suction rate, p_a = 1013 mbar absolute, p_e is the final absolute pressure, and 1.3 is an empirical factor. This is the constant-volumetric-flow (pump) form.
- **SMC** [14]: `Q = V·60/T1 + Q_L`, `T2 = 3·T1` (63% and 95% of stable vacuum), `Q_max = (2…3)·Q`.
  - Worked example: 4 mm pad, 1 g chip, 1 m of 2 mm tube (V = 0.0031 l), ZX105 (Q_max 5 l/min).
  - Average flow Q1 = Q_max/2…/3. `T = V·60/Q = 0.109 s`.
- **Stegmaier et al.** (Schmalz co-author) [29]: `t = (V/S)·ln(p0/pv)`. They show that hose flow resistance matters: a 0.4 l reservoir vs a 31.8 m × 4 mm hose of equal volume took 0.36 s vs 2.32 s to reach 700 mbar, a 6.4× difference.

**Check [C] of the ODE against the vendor (`dΔp/dt = p_atm·Q_n(Δp)/V`, isothermal, Piab table interpolated):**

| Si08-3 target | 10 | 20 | 30 | 40 | 50 | 60 | 70 kPa |
|---|---|---|---|---|---|---|---|
| ODE (s/l) | 0.10 | 0.25 | 0.48 | 0.82 | 1.32 | 1.98 | 2.94 |
| Piab table | 0.10 | 0.25 | 0.48 | 0.80 | 1.30 | 2.00 | 2.90 |

All within ±3%. Pi12-2: within −9% to +8%, partly because its flow and evacuation tables use different feed pressures. With n = 1.4 the ODE is 23–35% too fast.

**Conclusion:** the isothermal lumped ODE with the tabulated ejector curve is the right model. Example [C]: V = 5 cm³ (SPB1 20 plus a short hose), Si08-3 → **≈ 10 ms to −60 kPa**.

### 3.3 Leak flows

**(a) Orifice or nozzle flow (ISO 6358) [S] [12].**

ISO 6358-3:2014 §5.2, with upstream p1 = p_atm and downstream p2 = p_cup:
```
subsonic  (b < p2/p1 ≤ 1−Δpc/p1):  q_m = C·ρ0·p1·sqrt(T0/T1) · [1 − ((p2/p1 − b)/(1 − b))²]^m
choked    (p2/p1 ≤ b):             q_m = C·ρ0·p1·sqrt(T0/T1)
```
- C is the sonic conductance, b the critical back-pressure ratio, m the subsonic index ("distributed around 0.5"), Δpc the cracking pressure (0 for a gap).
- For an ideal converging nozzle (ISO 6358-3 eq. 20), C corresponds to `A·sqrt(γ/(R·T0))·(2/(γ+1))^((γ+1)/(2(γ−1)))/ρ0`, and b = 0.528.
- **[C]** That gives **C [dm³/(s·bar)] ≈ 0.20 × A [mm²]**.
- **[C]** The flow is **choked for cup vacuum deeper than about 47 kPa** (p2/p1 < 0.528). In that regime the leak flow does not depend on the cup pressure.
- **[S]** SMC expresses workpiece leakage as `Q_L = 55.5 × C_L` [l/min ANR], with C_L the conductance of the workpiece–pad gap and openings [14]. **[C]** The ISO choked formula gives 60.8×C at 101.3 kPa, so SMC's factor is about 9% lower. Use the ISO formula.

**(b) Narrow gaps: viscous slit flow [S] [9].**

Tiwari & Persson give the compressible laminar flow through a narrow channel of length L_y (along the lip), width L_x (radial lip land) and gap u_c (eq. 13), with a rarefaction (Knudsen) correction (eq. 19):
```
ṁ_visc = (L_y/L_x) · (p_a² − p_c²) · u_c³ / (24·η·R·T)            · [1 + ξ·⟨λ⟩/u_c],  ξ = 32/π
```
η = 1.81e-5 Pa·s (air). The mean free path λ is about 0.1 µm at 1 atm, so the correction matters only below a few µm.

**[I] Combining (a) and (b).** Per lip segment, use `ṁ_i = min(ṁ_visc(g_i), ṁ_ISO(A_i = C_d·L_i·g_i))`.
- **[C]** The crossover is at g ≈ 25 µm for lip land L_x = 1 mm and C_d = 0.7.
- Below the crossover, leak grows as **g³** (gentle onset). Above it, leak grows as **g** (orifice-limited).
- The min() is our simplification. A series-resistance blend is an alternative. Test V3 checks that the result is insensitive to the choice.

**(c) Porous product [S/I].**
- For Darcy flow through a porous layer of area A_p, thickness L_p and permeability k, the isothermal compressible form is `ṁ_por = k·A_p·(p_a² − p_c²)/(2·η·L_p·R·T)` (standard compressible Darcy; same p² structure as (b)).
- Paper and board have standard air-permeance tests (ISO 5636 Bendtsen/Gurley).
- Gabriel et al. 2021 predict leakage from surface roughness, cup diameter and cup count [3] (abstract only).
- **Not found:** permeability of biscuits, crackers or wafers. **[I]** Lump the porous product into one coefficient: `ṁ_por = K_p·(p_a² − p_c²)`, calibrated per product (§3.4).

**(d) Roughness, curvature and wear.** **[I]** Model these as an **equivalent roughness gap g0** added to every lip segment: `g_i,eff = g0 + max(0, dist_i)`.
- This follows Tiwari & Persson's "critical junction" separation u_c [9].
- Huh et al. show the seal level falling monotonically with roughness [11].
- Cup wear becomes a slow increase of g0.
- **[I] Optional extension:** make g0 decrease with lip contact pressure (Persson contact mechanics). Not needed for a first version.

### 3.4 Seal level at touch: what the sensor reads before the move

**[S] SMC "Leakage volume from adsorption test"** [14]: "pick up the workpiece with the ejector … read vacuum pressure P1, obtain the suction flow rate from the flow rate characteristics graph … and render this amount as the leakage of the workpiece". Their worked example: ZH07, 0.45 MPa supply, reading −53 kPa → leakage ≈ 5 l/min (ANR).

The steady seal level is the root of:
```
Q_ej(Δp*) = Q_leak(Δp*) = [Σ_i ṁ_i(g0 + gap_i) + ṁ_por] / ρ0
```

**Calibration rule:** this equation is invertible. A target seal level per product (from a real log, a vendor test or a literature value) plus a known ejector curve fixes the product's g0 or K_p. Without a real log, sample g0 and K_p from ranges chosen to span the seal levels in §5.

Illustration [C] (assumed: lip radius 10.7 mm as SPB1 20, lip land 1 mm, C_d = 0.7, 32 segments):

| Leak condition | Si08-3 (mid-size ejector) | Si02-2 (micro ejector) |
|---|---|---|
| sealed | 75.0 kPa | 73.7 kPa |
| uniform gap 5 µm | 74.9 | 69.8 |
| uniform gap 10 µm | 74.1 | 38.3 |
| uniform gap 20 µm | 64.0 | 10.4 |
| uniform gap 40 µm | 29.9 | 1.8 |
| one-sided lift-off, max gap 25 µm | 72.0 | 17.2 |
| one-sided lift-off, max gap 50 µm | 57.5 | 4.1 |
| one-sided lift-off, max gap 100 µm | 35.5 | 0.6 |

**[I]** Two readings from this table:
- Leaks at the scale of µm to tens of µm decide the seal level. That is the scale of surface roughness and of the lip lift-off in a tilt.
- The ejector size sets how sensitive the seal is to leaks. This is why vendors sell high-flow grippers for biscuits: Schmalz SCG, 270–650 l/min, 20–50 mm plates, "high leakage compensation" [30].

### 3.5 Load-induced dip: the mechanism we need

**[I] Causal chain, all simulated:**
1. Inertial wrench.
2. The lip pressure becomes non-uniform (Dex-Net ring, §2.4), or uniformly unloaded in a lift.
3. Segments unload to zero, then separate by g_i (simulated geometry).
4. Leak `Σ ṁ_i(g_i)` rises: gently (g³) at first, then fast.
5. p rises and Δp falls with the chamber time constant (about 5–15 ms for 5 cm³ [C]).
6. F = Δp·A_eff falls, so f_N falls and both the moment and friction capacity fall.
7. Either the product settles at a lower Δp (a near-miss: the dip is visible to the sensor), or the gaps run away (a drop).

Shear follows the same route through lip buckling and gaps (Kern [22]). In our model, a shear load near the friction limit also tilts the product, because friction acts at the lip plane and the CoM is offset.

**[S] Evidence:**
- Huh: the pressure is flat while sealed and drops at leak onset [11].
- GOMP-ST: deformation lasts more than 0.1 s before the break [4].
- Kern: buckling leads to continuous vacuum loss [22].

### 3.6 Hose and line: when one node is not enough

**[S]** Hose resistance can dominate (Stegmaier: 6.4×) [29]. ISO 6358-3 gives pipe and hose C and b from length, diameter and the Darcy friction factor [12].

**[I]** If the ejector sits on the travelling plate or at the cup (Piab's recommendation for high-speed small-object picking [27]), use **one node**: cup plus a short hose.

If the ejector is remote (for example at the robot base, through the IRB 360's integrated vacuum line, "max vacuum 0.75 bar" [31]), use **two nodes**: V_cup and V_line, connected by a hose conductance (C_h, b_h) from ISO 6358-3. The sensor sits on the cup node, as in GOMP-ST [4].

### 3.7 Vacuum sensor model

**[S]** Sensor references:
- **SMC ZSE20:** switch response "1.5 ms or less" (selectable up to 5 s); IO-Link minimum cycle 2.3 ms [32].
- **Adafruit MPRLS** as used in research: 166.7–167 Hz, 5 Pa RMS noise [4][11].

**[I]** Sensor model: first-order lag τ_s, sampling at f_s, additive noise σ, quantisation. Defaults: τ_s = 1.5 ms, f_s = 435 Hz (2.3 ms IO-Link cycle), σ = 5 Pa (MPRLS noise; the industrial sensor's noise is not found). Log columns:
- `seal_kPa`: the mean over a window after the touch settles;
- `vac_min_kPa` and `vac_dip_kPa` during the move;
- the time of the dip onset.

---

## 4. MuJoCo implementation (4.0.1)

Checked in the project venv [C]: `integrator="discrete"`, geom `gap`, geom `adhesion` and `mujoco.mj_applyFT` all exist in 4.0.1.

### 4.1 Body tree

```
eff  (existing: x/y/z slides, force-driven exact tracking)
└─ cup_base                       ← bellows compliance (optional, step 2)
   joints: bz  slide  axis=cup axis, stiffness k_bz, damping c_bz, range [−Z_stroke, 0]
           bx, by hinges at the bellows root, stiffness k_bt, damping c_bt   (tilt compliance)
   ├─ lip_1 … lip_n               (n = 12…16; 8 is enough for the onset test)
   │   joint: lz_i slide along cup axis, springref 0, stiffness k_l, damping c_l,
   │          range [−δ_max, 0]   (can be compressed; cannot extend beyond free length)
   │   geom:  sphere or short capsule arc at radius r_lip, contype/conaffinity 0,
   │          explicit <pair> with the product: friction µ, condim 3, gap = g_det (e.g. 3 mm)
   └─ (visual cup body)
product (freejoint), as now
```
- **Neighbour coupling (Dex-Net perimeter and flexion springs, optional).** Add fixed tendons `lz_i − lz_{i+1}` (and `lz_i − lz_{i+2}`) with tendon stiffness. This penalises uneven lip deflection. It is what limits sealing on small radii. **[I]** Calibrate it so the minimum convex radius for sealing matches the vendor value (test V10).
- **Why joint springs, not soft contacts, carry the lip compliance [S].** In MuJoCo the contact "stiffness" sets a reference *acceleration*: `a_ref = −b·v − k·r`. The rest penetration is `r = a_u·(1−d)/stiffness` in the direct format [33]. That depends on the body's acceleration and mass, not on a physical N/m. Joint `stiffness` is a real N/m spring. **[I]** Keep the lip–product contacts stiff, and check that contact penetration at nominal load is under 10% of the lip deflection.

### 4.2 Vacuum force: applied where physics puts it

**[I] Recommended:** each step, apply `F_vac = Δp·A_eff` with `mujoco.mj_applyFT`:
- to the **product**, at the cup centre on the sealed face, along the cup axis toward the cup;
- an equal and opposite force to **cup_base**.

Use `d.qfrc_applied`, cleared each step. The load path is then physical:
- product–lip contact compression = F_vac − load (Dex-Net `f_N = f_z + V`);
- the lip springs carry it;
- lift-off happens when a segment's spring reaches its free length.

The resultant stays at the area centroid even when part of the lip has lifted. Pressure is uniform in the cup, which is physically correct. The loss of hold comes from Δp falling, not from re-dividing the force.

**Alternative:** keep the `adhesion` actuator, targeting a "suction face" geom on cup_base, with `gap` so it acts across the layer [20][21]. Drawbacks:
- the force is split equally across the detected contacts, so where it acts depends on the collision points;
- one actuator per body.

Use it only if `mj_applyFT` causes trouble (for example in MJX batching, where `xfrc_applied` is the equivalent).

**Optional `A_eff`:** use `A_eff(Δp)` from Piab-style tables (§2.1). Optionally grow A_eff toward π·Ds²/4 with lip extension, to reproduce the Schmalz pull-off/suction ratio (test V5b).

### 4.3 Per-step algorithm (dt_mj = 0.5 ms)

```
1. after mj_step: for each lip pair i, read d.contact (pair geoms) → dist_i
   (contacts within gap are present with efc_address = −1 when inactive);
   gap_i = max(0, dist_i) if found else g_det (treat as fully open)
2. leak: ṁ_leak = Σ_i ṁ_seg(g0 + gap_i; L_i = 2π r_lip/n, L_x = lip land, C_d)
                + K_p·(p_a² − p²)                    (§3.3)
3. ejector: ṁ_ej = ρ0·Q_ej(Δp)·s(t)   (s = valve state with on/off delay t_v; table interpolation)
4. volume: V = V_cup0 − A_b·max(0, −q_bz) + V_hose ;  dV/dt from q̇_bz
5. integrate dp/dt (§3.1) with N_sub = 10…20 sub-steps (semi-implicit, linearising ṁ in p),
   clamp p ∈ [p_a − Δp_max_ej, p_a]  (two-node: integrate both nodes)
6. F_vac = (p_a − p)·A_eff(Δp) → mj_applyFT on product (+) and cup_base (−)
7. sensor: p_s += dt/τ_s·(p − p_s); sample at f_s, add noise; log
```
**[C]** Sub-stepping is needed because a fully open lip on a 5 cm³ volume can move p by tens of kPa within 0.5 ms (choked leak of several g/s into a few cm³). The ODE is a scalar per cup, so it is cheap.

### 4.4 State variables

| Name | Type | Meaning |
|---|---|---|
| q_lz[1..n], q̇_lz | MuJoCo joint | lip segment compression and velocity |
| q_bz, q_bx, q_by (+ velocities) | MuJoCo joint | bellows axial and tilt deflection |
| product free joint | MuJoCo | product pose |
| p (or p_cup, p_line) | Python scalar(s) | absolute pressure in the cup (and line) |
| s_valve | Python | vacuum valve state and delay |
| p_s | Python | sensor filter state |

### 4.5 Timestep and stability

- **dt = 0.5 ms** (as now). Contact `solref` timeconst ≥ 2·dt, which MuJoCo enforces unless `refsafe` is off [33]. Keep the current 0.002 s.
- **Integrator.** Use `implicitfast`. MuJoCo recommends it for most models; it integrates damping implicitly, while joint springs stay explicit under it and under Euler [33]. **[I]** For an explicit spring on a small lip mass, keep `ω·dt = sqrt(k_l/m_l)·dt ≤ 0.5`, so `m_l ≥ k_l·(dt/0.5)² = k_l·1e-6`. For example, k_l = 300 N/m needs m_l ≥ 0.3 mg. The real cup mass is grams (SPB1 20: 2.2 g [16]), so m_l = m_cup/n is safe.
- MuJoCo 4.0.1's `discrete` integrator folds joint springs into the implicit metric [33]. It is marked as a trial in the docs, so use it only as a cross-check.
- Use `cone="elliptic"`, `impratio` as now, and condim 3 on the lip pairs. Torsion comes from the ring geometry (Dex-Net derives `|τ_z| ≤ r·µ·f_N` from the ring), so condim 4 is not needed (test V7).
- **ODE.** 10–20 sub-steps per MuJoCo step, or the exact exponential update for the linearised step. Halve dt and double the sub-steps as a convergence test (V13).

### 4.6 Seal formation at touch (Dex-Net 3.0 analogue)

**[I]** On approach the lip segments compress against the product. Each segment is constrained to move along the cup axis, as in Dex-Net's projection along v [1]. The seal forms if every segment touches (`dist_i ≤ 0` within the joint range) and the neighbour tendon strain stays below a threshold. Dex-Net uses 10% spring length change [1]. Here the threshold is calibrated against vendor minimum radii (V10).

Uneven or curved products leave some segments with `gap_i > 0`. These enter the leak sum, and the seal level comes out of the ODE (§3.4). So curvature, roughness (g0) and porosity (K_p) all show up in `seal_kPa` through physics.

---

## 5. Parameter table

| Parameter | Symbol | Value / range | Status | Source |
|---|---|---|---|---|
| Ambient pressure | p_a | 101.3 kPa | S | Piab test conditions [26] |
| ANR density | ρ0 | 1.185 kg/m³ | S | ISO 6358 reference atmosphere [12] |
| Gas constant, ratio | R, γ | 287 J/(kg·K), 1.4 | S | standard; Richer & Hurmuzlu [25] |
| Polytropic index | n | 1.0 (isothermal) | C | matches Piab evacuation tables (§3.2) |
| Air viscosity | η | 1.81e-5 Pa·s | S | standard; Tiwari & Persson [9] |
| Ejector curve, mid-size | Q_ej(Δp) | Piab Si08-3 table, 75 kPa max | S | [26] |
| Ejector curve, micro (at cup) | Q_ej(Δp) | Piab Si02-2 table, ≈75 kPa max | S | [27] |
| Ejector, deep vacuum | Q_ej(Δp) | Piab Pi12-2 table, 90 kPa max | S | [26] |
| Valve on/off delay | t_v | 3.5 / 2 ms (SMC VQ110) | S | [11] |
| Delta integrated vacuum | Δp_max | ≤ 75 kPa (IRB 360) | S | [31] |
| Typical operating vacuum | Δp_set | 60 kPa recommended by Piab; 53 kPa (SMC leaking example); 30 kPa (Pham rig); 85 kPa (Huh source) | S | [34][14][5][11] |
| Cup sizes | d | 10–40 mm (SPB1 10–40; Piab 25 mm family; Dex-Net 15 mm; GOMP-ST 30 mm; Pham 25 mm) | S | [16][17][1][4][5] |
| Effective area | A_eff | F_suction(−0.6 bar)/60 kPa ≈ π·d2²/4 (SPB1 20: 1.65e-4 m²) | S/C | [16] |
| A_eff vs vacuum (bellows) | A_eff(Δp) | B25XP: d_eff 24.8 / 20.1 / 17.6 mm at 20/60/90 kPa | C from S | [17] |
| Pull-off / suction ratio | – | 1.2–2.7 (SPB1) | S | [16][6] |
| Cup volume | V_cup0 | SPB1: 0.5 / 1.5 / 3.2 / 5.3 / 8.0 / 18.4 cm³ (10–40 mm); B25XP 1.6 cm³; BX25P 2.9 cm³; F-BX25 5.1 cm³ | S | [16][17] |
| Hose volume | V_hose | π/4·d_i²·L; d_i = 4 mm recommended for ≈ 2 m (SPB1 ≤ 30) | S | [16][14] |
| Bellows stroke | Z | SPB1: 4–16 mm; B25XP 5.6 mm; BX25P 8.4 mm; F-BX25 15.5 mm | S | [16][17] |
| Bellows axial stiffness | k_bz | not published; **upper-bound estimate F_suction/Z**, e.g. SPB1 20 ≤ 1.24 N/mm, assuming full contraction at −0.6 bar ("lifting effect" [35]) | I | calibrate (§7) |
| Lip segment stiffness | k_l | not published; sweep so that total ring stiffness n·k_l ∈ [0.3, 3]×k_bz | I | calibrate (§7) |
| Lip and bellows damping | c_l, c_bz | not published; start at critical for the segment mass and sweep 0.3–2× | I | calibrate |
| Elastomer modulus | E | Gent: Shore 30 → 1.1 MPa, 40 → 1.7, 60 → 3.6, 65 → 4.4 MPa (inaccurate below Shore 40) | S | Gent 1958 via [36] |
| Lip radius | r_lip | ≈ Ds/2 (SPB1 20: 10.7 mm) | S/I | [16] |
| Lip land width (radial) | L_x | not published; 0.5–2 mm | I | calibrate from the seal level |
| Discharge coefficient | C_d | 0.6–0.8 | I | sensitivity only (the choked regime scales linearly) |
| Critical pressure ratio | b | 0.528 (ideal nozzle); real components often lower | S | [12] |
| Equivalent roughness gap | g0 | 0–20 µm smooth to rough; wear increases it | I | calibrate per product from seal_kPa (§3.4); trend [11][9] |
| Porous coefficient | K_p | from target seal level | I | SMC adsorption-test method [14] |
| Friction lip/product | µ | 0.3–1.2 (Schmalz 0.5–0.6 dry, 0.1–0.3 oily or wet; Kern 1.1–1.8 measured; vendor lateral ratios 0.6–1.3) | S (range) | [13][22][16][17] |
| Min. convex radius for sealing | R_min | SPB1: 5 / 8 / 15 / 20 / 25 / 40 mm (10–40) | S | [16] |
| Safety factors (for baseline recipes) | S, t | Schmalz ≥ 1.5 smooth, ≥ 2 porous; SMC t ≥ 4 horizontal, ≥ 8 vertical | S | [13][14] |
| Sensor | τ_s, f_s, σ | 1.5 ms; 435 Hz (IO-Link 2.3 ms) or 167 Hz (MPRLS); 5 Pa | S | [32][4][11] |
| Deformation-to-failure time | – | > 0.1 s (GOMP-ST video) | S | [4] |

**Food-specific seal levels: not found.** We found no published numbers for biscuits, flow-wrap film, foil or chocolate. What exists:
- qualitative vendor statements: porous baked goods need high flow and leak compensation [30]; tool design "depends on the porosity and air permeability of the baked goods" (Schubert, via Baking & Biscuit [37]);
- the SMC leaking-workpiece example (−53 kPa) [14];
- Huh's roughness trend [11].

**[I] Proposal until real logs exist:** define product classes by an equivalent g0 or K_p that puts the seal at the touch-down level shown, with the Si08-3 ejector:
- foil and film-like smooth (seal ≈ 70–75 kPa);
- textured film and creased flow-wrap (≈ 55–70 kPa);
- porous biscuit (≈ 20–50 kPa).

Label these levels as assumptions in DEBT.md until replaced by a real vacuum log (§7).

---

## 6. Validation tests

Each test lists the reference, its source and the tolerance. Tests V1–V4 are pneumatics only and can be unit-tested in Python. The rest run in MuJoCo.

| # | Test | Reference (source) | Tolerance |
|---|---|---|---|
| V1 | Evacuation of a sealed volume (no leak), Si08-3 and Pi12-2 curves | Piab evacuation-time tables (s/l) [26]; offline check ±3% / ±9% (§3.2) | ±10% (Piab's stated accuracy [27]) |
| V2 | Time constants with a linear ejector curve | SMC: 63% at T1, 95% at T2 = 3·T1 [14] | ±5% |
| V3 | Steady seal with a known orifice area A | root of Q_ej = ISO 6358 flow with C = 0.2·A [12]; SMC round trip: recover Q_L from P1 and reproduce P1 [14]; min() vs series-blend leak changes the seal by < 5% for gaps ≤ 10 µm | ±2% (numerical); blend sensitivity reported |
| V4 | Static holding force, sealed | F = Δp·A_eff = 9.9 N for SPB1 20 at 60 kPa [16]; with A_eff(Δp): B25XP 9.7 / 19.0 / 22.0 N at 20/60/90 kPa [17] | ±2% / ±5% |
| V5 | Lift to failure (pressure held constant) | a* = F/m − g (current spike: −1…−4%) | ±4% |
| V5b | Pull-off with A_eff(extension) on | Schmalz pull-off force (SPB1 20: 16.2 N) [16] | ±15% (optional) |
| V6 | Shear to failure, flat product, 1 mm slip criterion | a* = µ·(F − m·g)/m (Dex-Net f_N = f_z + V [1]) | ±5% |
| V7 | Torsion to failure | τ_max = r_lip·µ·f_N (Dex-Net ring derivation [1]); √3-box version is the conservative bound | ±5% |
| V8 | Peel onset: first lip gap > 1 µm under a quasi-static moment ramp | M_onset = r_lip·f_N/2 (Dex-Net linear ring [1]); must not change with k_l (two stiffnesses) | ±5% |
| V9 | Peel ultimate with pressure frozen (ODE off) | M = r_lip·f_N (rigid ring / CoP: Pham&Pham [5], Lee et al. [6]) | ±10% |
| V9b | Peel with ODE on | failure moment in [M_onset, M_rigid]; dip onset coincides with M_onset (Huh: pressure flat until leak onset [11]) | qualitative + onset ±5% |
| V10 | Seal formation on convex cylinders | calibrate the neighbour-coupling limit on SPB1 20 (R_min = 15 mm); predict the others (10: 5, 15: 8, 25: 20, 30: 25 mm) [16]. [C] The rim sag at R_min is 4.0–5.8 mm for 10–40 mm sizes, nearly constant, so one calibration should transfer. | predicted R_min within ±25% |
| V11 | Seal level vs roughness | monotone fall of seal_kPa with g0, plateau near the ejector maximum for smooth surfaces (Huh: 74 kPa of 85 at 600 grit [11]) | monotone; smooth seal within 2 kPa of the ejector maximum |
| V12 | Warning before drop | under acceleration ramps (Ruckig-like, as GOMP-ST), the sensor dip starts before product loss; lead time on the order of tens to hundreds of ms (GOMP-ST > 0.1 s deformation-to-failure [4]) | order of magnitude; report the distribution |
| V13 | Numerical convergence | halve dt, double ODE sub-steps: failure accelerations and seal levels | change < 2% |
| V14 | Regression | old lift and shear numbers (suction-validation.md) | within the old ±4% |
| V15 | Sensitivity sweep | k_l, c_l, L_x, C_d, n over their ranges: report how V8/V9b/V12 change; V8 must be invariant | documented |

---

## 7. Open questions and calibration routes

1. **Lip and bellows stiffness and damping: not published.**
   - Routes, in order:
     - (a) a bench test, if any cup is at hand: load–deflection on a kitchen scale, plus a free-decay video for damping;
     - (b) the Schmalz upper bound k ≤ F_suction/Z;
     - (c) Hudoklin et al. [7] or Gabriel et al. [8] measured data (paywalled; worth getting via a library);
     - (d) Gent E from Shore hardness, with lip geometry from the vendor CAD.
   - Until then, sweep (V15) and expose k_l as a hidden per-cup or wear parameter.
   - Note: **V8 (onset) does not depend on k_l**; only the dip slope and timing do.
2. **Food permeability and roughness: not found.** Calibrate g0 and K_p per product from seal levels (§3.4). Best source would be a real vacuum log from the line (Piab, Schmalz and SMC IO-Link sensors already log it; see `docs/research/application-delta-picking.md`). A Piab, Schmalz or Schubert applications engineer could also give typical seal levels for biscuits and flow-wrap.
3. **Lip land width and discharge coefficient.** They set the gap-to-leak scale (§3.4 table). Sweep them, or fit them to a target seal on a known roughness.
4. **Flow-wrap film specifics.** The film can be drawn into the cup and the bag's trapped air can move. This is not modelled. The likely first-order effect is a larger g0 from creases. Mark as an assumption.
5. **Remote ejector or long hose on the delta.** Decide one node vs two nodes from the real gripper layout (§3.6).
6. **Dex-Net constants.** V = 250 N and κ = 0.005 are fitted, not physical [1]. We use Dex-Net's structure only. The elastic-torque limit πrκ is replaced by lift-off plus the leak physics.
7. **Is the 0.45 peel factor really the Dex-Net onset?** Test V8/V9 with `gap` set on the old rigid model first. This is a 10-minute experiment that confirms the §2.4 interpretation.

---

## Sources

1. J. Mahler, M. Matl, X. Liu, A. Li, D. Gealy, K. Goldberg, "Dex-Net 3.0: Computing Robust Robot Vacuum Suction Grasp Targets in Point Clouds using a New Analytic Model and Deep Learning", ICRA 2018. https://arxiv.org/abs/1709.06670 (full text incl. appendices: https://arxiv.org/pdf/1709.06670)
2. Internal: `docs/research/suction-validation.md` and `scripts/spike_suction.py` (our MuJoCo spike: lift, shear, peel results re-analysed in §2.4).
3. F. Gabriel, S. Baars, M. Römer, K. Dröder, "Grasp Point Optimization and Leakage-Compliant Dimensioning of Energy-Efficient Vacuum-Based Gripping Systems", Machines 9(8):149, 2021 (leakage vs surface roughness, cup diameter and count; full text blocked to our fetcher, known from the abstract). https://doi.org/10.3390/machines9080149
4. Y. Avigal, J. Ichnowski, et al., "GOMP-ST: Grasp Optimized Motion Planning for Suction Transport", WAFR 2022. https://arxiv.org/pdf/2203.08359
5. H. Pham, Q.-C. Pham, "Critically fast pick-and-place with suction cups", ICRA 2019. https://arxiv.org/abs/1809.03151
6. J. Lee, R. Sun, A. Bylard, L. Sentis, "Grasp Failure Constraints for Fast and Reliable Pick-and-Place Using Multi-Suction-Cup Grippers", 2024. https://arxiv.org/abs/2408.03498
7. J. Hudoklin, S. Seo, M. Kang, H. Seong, A. T. Luong, H. Moon, "Vacuum Suction Cup Modeling for Evaluation of Sealing and Real-Time Simulation", IEEE RA-L 7(2):3616–3623, 2022. https://doi.org/10.1109/LRA.2022.3145509
8. F. Gabriel, M. Fahning, J. Meiners, F. Dietrich, K. Dröder, "Modeling of vacuum grippers for the design of energy efficient vacuum-based handling processes", Production Engineering 14:545–554, 2020. https://doi.org/10.1007/s11740-020-00990-9
9. A. Tiwari, B. N. J. Persson, "Theory for diffusive and ballistic air leakage and its application to suction cups", arXiv:1906.01459 (see also "Physics of suction cups", Soft Matter 15:9482, 2019, https://doi.org/10.1039/c9sm01679a). https://arxiv.org/abs/1906.01459
10. V. Stegmaier, T. Eberhardt, W. Schaaf, N. Jazdi, M. Weyrich, "A behavior model for Digital Twins of vacuum suction cups", Procedia CIRP 118:958–963, 2023. https://doi.org/10.1016/j.procir.2023.06.165
11. T. M. Huh, K. Sanders, M. Danielczuk, M. Li, Y. Chen, K. Goldberg, H. S. Stuart, "A Multi-Chamber Smart Suction Cup for Adaptive Gripping and Haptic Exploration", IROS 2021. https://arxiv.org/abs/2105.02345
12. ISO 6358-3:2014, "Pneumatic fluid power: Determination of flow-rate characteristics of components using compressible fluids, Part 3" (§5.2 formulae 1–3; eq. 20). Preview: https://cdn.standards.iteh.ai/samples/56616/21f0d545a09447a2962dcc414057a899/ISO-6358-3-2014.pdf
13. Schmalz Vacuum Knowledge, "Theoretical holding force of a suction cup". https://www.schmalz.com/en/vacuum-knowledge/the-vacuum-system-and-its-components/system-design-calculation-example/theoretical-holding-force-of-a-suction-cup
14. SMC, Vacuum equipment Model Selection (leakage volume, response time, ejector selection): https://www.smcworld.com/assets/products/pickup/en-jp/vacuum_device/select/6-4-p0033-vacuum_en.pdf ; worked example (semiconductor chip): https://www.smcworld.com/assets/products/pickup/en-jp/vacuum_device/select/4-P0891_en.pdf ; selection guide: https://www.smcworld.com/products/pickup/en-hk/vacuum_device/model_selection/
15. Schmalz Vacuum Knowledge, "Design of the suction cup". https://www.schmalz.com/en/vacuum-knowledge/the-vacuum-system-and-its-components/vacuum-suction-cups/design-of-the-suction-cup
16. Schmalz, Datasheet "Bellows Suction Cups SPB1 (1.5 Folds)" (family) and article 10.01.06.02452 (SPB1 20 ED-65). https://pimmedia.schmalz.com/MAM_Library/Dokumente/Datenblatt_Produktfamilie/0_/051/05133/Datasheet_Bellows%20Suction%20Cups%20SPB1%20(1.5%20Folds)_en-EN.pdf ; https://media.schmalz.com/MAM_Library/Dokumente/Datenblatt_Artikel/1/100/10010602452/55908f59d506_Datasheet_Article_10.01.06.02452_en-EN.pdf
17. Piab datasheets: B25XP.4K https://www.motionworld.com/assets/Piab-B25XP_4K-Datasheet-US.pdf ; BX25P.4K https://www.motionworld.com/assets/Piab-BX25P_4K-Datasheet-US.pdf ; F-BX25 food-contact silicone https://www.motionworld.com/assets/Piab-FBX25_DS40-Datasheet-US.pdf
18. SMC ZP/ZPT pad catalogues (safety factor t ≥ 4 horizontal / ≥ 8 vertical): https://static.smc.eu/pdf/ZP.pdf
19. Robotiq, "Best practices for cup selection". https://blog.robotiq.com/best-practices-for-cup-selection
20. MuJoCo XML reference: `actuator/adhesion`, `geom/gap`, `geom/adhesion`. https://mujoco.readthedocs.io/en/stable/XMLreference.html (source: https://github.com/google-deepmind/mujoco/blob/main/doc/XMLreference.rst)
21. MuJoCo official example `model/adhesion/active_adhesion.xml`. https://github.com/google-deepmind/mujoco/blob/main/model/adhesion/active_adhesion.xml
22. P. Kern, "Elastomerreibung und Kraftübertragung beim Abscheren von aktiv betriebenen Vakuumgreifern auf rauen Oberflächen", Dissertation, KIT, 2016. https://publikationen.bibliothek.kit.edu/1000065103
23. G. Mantriota, "Theoretical model of the grasp with vacuum gripper", Mech. Mach. Theory 42(1):2–17, 2007. https://doi.org/10.1016/j.mechmachtheory.2006.03.003 ; G. Mantriota, A. Messina, "Theoretical and experimental study of the performance of flat suction cups in the presence of tangential loads", Mech. Mach. Theory 46(5):607–617, 2011. https://doi.org/10.1016/j.mechmachtheory.2011.01.003
24. B. Bahr, Y. Li, M. Najafi, "Design and suction cup analysis of a wall climbing robot", Computers & Electrical Engineering 22(3):193–209, 1996. https://doi.org/10.1016/0045-7906(95)00039-9
25. E. Richer, Y. Hurmuzlu, "A High Performance Pneumatic Force Actuator System: Part I — Nonlinear Mathematical Model", ASME J. Dyn. Sys., Meas., Control 122(3):416–425, 2000. https://doi.org/10.1115/1.1286336
26. Piab VGS™3010 datasheets with COAX® MINI Si08-3 and Pi12-2 flow and evacuation tables: https://shop.tech-con.ro/media/catalog/product/pdf/9901167dt.pdf ; https://shop.tech-con.ro/media/catalog/product/pdf/9900041dt.pdf
27. Piab datasheet COAX® cartridge MICRO Si02-2 (01.13.593). https://www.motionworld.com/assets/Piab-01_13_593-Datasheet-US.pdf
28. Schmalz Vacuum Knowledge, "Calculation of evacuation times"; "Vacuum generator selection". https://www.schmalz.com/en/support/know-how/vacuum-knowledge/the-vacuum-system-and-its-components/system-design-calculation-example/calculation-of-evacuation-times ; https://www.schmalz.com/en/support/know-how/vacuum-knowledge/the-vacuum-system-and-its-components/system-design-calculation-example/vacuum-generator-selection
29. V. Stegmaier, W. Schaaf, N. Jazdi, M. Weyrich, "Efficient Creation of Behavior Models with Variable Modeling Depths Used in Digital Twins", arXiv:2411.07883 (eq. 1.1, hose vs reservoir); and "Simulation Model for Digital Twins of Pneumatic Vacuum Ejectors", Chem. Eng. Technol. 46(1):71–79, 2023, https://doi.org/10.1002/ceat.202200358. https://arxiv.org/abs/2411.07883
30. Schmalz, "Flow gripper SCG for handling dry food products". https://www.schmalz.com/en-us/solutions/media-center/flow-gripper-scg-for-handling-dry-food-products
31. ABB IRB 360 datasheet (integrated vacuum supply "Max. 7 bar / max vacuum 0.75 bar"). https://library.e.abb.com/public/f03878770fef4c2c93757953373aed9b/IRB%20360_datasheet_20220829.pdf
32. SMC ZSE20(F)/ISE20 digital pressure switch (response ≤ 1.5 ms; IO-Link min. cycle 2.3 ms). https://static.smc.eu/pdf/ZSE20(F)_ISE20.pdf ; https://www.smcworld.com/webcatalog/en-jp/series/ZSE20(F)-ISE20-IO-E
33. MuJoCo documentation, Modeling: solver parameters (solref direct format, penetration at rest), integrator choice. https://mujoco.readthedocs.io/en/stable/modeling.html (source: https://github.com/google-deepmind/mujoco/blob/main/doc/modeling.rst)
34. Piab FAQ (recommended vacuum level ≈ 60 kPa for lifting with suction cups). https://piab.com/support/faq/
35. Schmalz glossary, "Bellows suction cups" ("contracts while lifting the workpiece (lifting effect)"). https://www.schmalz.com/en/glossary/bellows-suction-cups
36. A. N. Gent, "On the relation between indentation hardness and Young's modulus", Trans. Inst. Rubber Ind. 34:46–57, 1958, https://doi.org/10.5254/1.3542351 (formula E[MPa] = 0.0981(56 + 7.62336 S)/(0.137505(254 − 2.54 S)), as given at https://en.wikipedia.org/wiki/Shore_durometer; deviates from data below Shore A 40)
37. W. Sedlak (Gerhard Schubert GmbH), "Always a secure hold", Baking & Biscuit. https://bakingbiscuit.com/always-a-secure-hold/
