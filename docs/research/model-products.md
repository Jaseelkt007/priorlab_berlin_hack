# Product catalogue for the delta-pick simulator (sourced)

Researched 2026-11-21 (phase 0e of `docs/PLAN.md`). Replaces the placeholder `PRODUCTS` dict in `sim/delta_toy.py`.

## How to read this document

Every value carries a tag:

| Tag | Meaning |
|---|---|
| **S** | Taken from the cited source. `[Sn]` points to the source list at the end. |
| **S\*** | Taken from the cited source, but read only through a search-engine excerpt, because the page itself blocked our fetch (HTTP 403 or an expired certificate). Re-check before quoting in the pitch. |
| **D** | Derived by calculation from sourced values. The calculation is shown. |
| **E** | Our estimate. The reasoning and a range are given. Treat it as a simulator prior, not a fact. |

**Biggest gap, stated up front.** We found **no published friction coefficient between a suction-cup elastomer and any food surface** (biscuit, chocolate, film, cheese), and **no published air permeability of a biscuit**. Schmalz says outright that generally valid cup–workpiece friction values cannot be given and must be measured on the real part [S1]. Every μ and every permeability below is therefore **E**, anchored on Schmalz's reference values for rigid materials [S1] and on bread and dough permeability data [S36, S37].

---

## 1. Evidence: what delta robots really pick with suction

| Product family | Case (vendor / site) | Robot, gripper, rate | Tag |
|---|---|---|---|
| Biscuits, plain and chocolate-based, sandwich biscuits | Schubert lightline Pickerline at a confectionery producer | 8 T4 (delta-type) robots, **round suction cup per biscuit**, up to 530 products/min, 1,600 mm product belt, 3D scanner checks orientation and height (±0.5 mm), 5 formats, 10 products per tray [S17] | S |
| Biscuits ("ultra-thin, fragile") | Schubert, largest picker line | 48 F4 robots, close to 5,000 biscuits/min (about 104/robot/min, D); chocolate-coated rice wafers: 28 F4, about 2,500/min [S19] | S (rate), D (per robot) |
| Cookie sandwich | Syntegon (Sigpack) Delta at Pepperidge Farm | more than 95 picks/min per robot in a 1,000 mm work area; capping line of 12 robots at more than 1,300 pieces/min [S22] | S\* |
| Pralines, wrapped and unwrapped | Schubert TLM line at Toms (Anthon Berg) | 6 TLM-F4 robots, quick-change **suction and gripper tools**, up to 529 chocolates/min, 44 trays/min of 24 [S18] | S |
| Moulded chocolate squares, Santas, bunnies | Syntegon RPP feed placer, US chocolate maker | delta robots with **vacuum end-of-arm tools** pick from moulds (102–144 products per mould); downstream flow wrapper up to 1,088 products/min [S21] | S |
| Chocolates (general) | Cama Triaflex loading unit | 3 (up to 4) delta robots, 120 cycles/min each, multi-pick gripper, up to 300 products/min [S23] | S |
| Flow-packed sweet bars, snack bags | Piab success story, ABB FlexPicker | worn competitor cup let vacuum fall and products dropped mid-move; Piab DURAFLEX rectangular cup RB20x45P + COAX MINI ejector "dramatically decreased the number of dropped products", +70 % productivity [S7] | S |
| Cheese wedges | Sachsenmilch (Germany), Delta-3 pickers | 4 pickers, 3D inspection of each wedge, **Venturi vacuum**; a wedge that "cannot be picked up due to drawn in secondary air" is sent to a chute; missed wedges pass to the next picker [S24] | S\* |
| Snack-size cheeses | Ilchester Cheese (UK) | 2 ABB IRB 340 FlexPickers with vision load raw cheeses into a wrapper [S25] | S |
| Liquid-filled pet-food pouches | Mackle Petfoods (OMRON) | 2 delta robots, 2 pouches per cycle, 135 pouches/min, ±1 mm placement. **Gripper is 3D-printed, suction not confirmed** [S26] | S |
| Pouches with fluids, solids or powder | Schmalz SPB2f bellows cup | designed "particularly in conjunction with delta robots (case packer and pick-and-place)" [S4] | S |
| Pharma blister cards | Dividella P5 cartoner; FANUC M-1iA delta (Flexlink); Marchesini-type line | delta transfer with **suction arm** (up to 500 cartons/min) [S28]; M-1iA blister retrieval, target 60/min [S27]; 320 blisters/min line with a suction-head robot [S29] | S |
| Injection vials | STIWA inspection loader | robot with suction grippers lifts 105 vials at once, 180 vials/min. **Not a delta** [S30] | S |
| Poultry fillets | JBT RoboPacker; Vendée Concept Delta Calibra | up to 180 pieces/min with a vacuum-gripper option [S60]; delta, up to 3 kg, 110 picks/min, gripper type not stated [S60] | S |

ABB's own guidance [S13]: the delta picks "from less than 50 grams to 3 kg"; the robot may accelerate "up to 12 times the force of gravity", which affects "the stability of the product in the gripper"; vacuum gripping is "generally faster, lower weight, simpler"; motion is 70–85 % of the cycle and pick-and-place 15–30 %; vision must deliver "over 99.9 percent reliable data".

---

## 2. The catalogue (7 products, easy to hard)

Chosen to span the failure mechanisms our simulator must reproduce: **leak** (porosity, film wrinkles, secondary air), **shear and slip** (low μ, wet surfaces), **peel** (tall or off-centre CoM), **compliance** (flexible pouch, film), **damage** (brittle biscuit), and **drift** (humidity, temperature, wear).

Cup diameter D, vacuum p and holding force F refer to one cup. h = CoM distance below the picked face.

### 2.1 Summary table

| # | Product | Difficulty | Mass mean | Mass spread (1σ) | Dims L×W×H (mm) | h (mm) | Surface at the cup | μ cup–product | Cup (type, D) | Working vacuum | Main failure mode |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | Pharma blister card, 10 tablets | easy | 5 g **D** (4–7) | 0.1 g **E** | 95×46×4 **E** | 2 **E** | 20 µm hard-temper Al lidding [S53] **S**, flat, impermeable | 0.5 **E** (0.4–0.6) | flat TPU/silicone, 2×Ø15–20 **E**; TPU outlived silicone on a high-speed blister line [S9] **S** | −600 mbar **E** (dense: −600…−800 [S2] **S**) | shear at high accel.; cup wear (lip) |
| P2 | Moulded praline, unwrapped | easy | 10–11 g **S** [S50] | 0.2 g **E** (CV 1–3 %) | Ø22×20 **S** [S50] | 8 **E** (6–10) | glossy chocolate, non-porous, decorated tops [S5] | 0.5 **E** (0.3–0.7); falls with condensation/soft surface | Schmalz SPG bell, Ø33, silicone 45 or 65 Shore A [S5] **S** | **max −150 mbar** (vendor recommendation) [S5] **S** | surface damage; temperature-driven softening and condensation |
| P3 | Digestive biscuit (plain) | medium | 15 g **D** [S32] | 0.3 g **E** (0.15–0.45) | Ø72×7.2 **S** [S31] | 3.6 **D** | rough, porous baked crust; apparent density 0.51 g/cm³ **D** | 0.6 **E** (0.5–0.7; Schmalz "rough" 0.6 [S1]) | silicone bellows Ø25–30 **E** (Piab B-BL family for bakery [S6]) | −200…−400 mbar **E** (Schmalz range for porous parts [S2] **S**) | leak through pores → low seal; breakage (12.5 ± 1.2 N [S31]) |
| P4 | Flow-wrapped bar (chocolate/wafer) | medium | 45 g **S** (41.5–50 g labels [S63]) | 0.7 g **E** (CV ≈ 1.5 %) | 120×30×15 **E** | 7.5 **D** (+ film slack) | metallised OPP, ~18–30 µm **E** [S62]; fin seal, wrinkles, loose film | 0.45 **E** (0.3–0.6) | rectangular bellows 20×45 (Piab RB20x45P, made for flow-packed bars) [S7] **S** | −400…−600 mbar **E** | leak at wrinkles/seal; film tenting → peel; cup wear → vacuum loss [S7] |
| P5 | Vacuum-packed cheese wedge (chilled) | hard | 200 g **E** (150–250) | 3 g **E** (2–5) | triangle 100×70, 50 thick **E** | 25 **D** | shrink film over cheese, wet when it condenses | 0.25 **E** (0.2–0.3; Schmalz "wet" [S1]) | silicone bellows Ø40–50 **E** | −500…−700 mbar **E** | secondary air at film folds [S24]; low μ when wet; off-centre CoM |
| P6 | Wet pet-food pouch, 85 g | hard | 85 g net **S** [S54] + ~4 g film **E** | 1.5 g **E** | 140×90×15 **E** (retail listing 140×95×10 [S56]) | 7.5 **D** (shifts) | PET/AL/NY/RCPP retort laminate [S55], smooth, flexible | 0.45 **E** (0.3–0.6; 0.2–0.3 with gravy residue) | Schmalz SPB2f 40–50 with bag stabiliser [S4] **S** | −400…−600 mbar **E** | compliance; contents slosh → moving CoM; film drawn into cup |
| P7 | Filled 10R injection vial (stress test) | hard | 18 g **D** (16–22) | 0.3 g **E** | Ø24×45 **S** [S52] | 27 **E** (25–30) | plastic flip-off cap, ~Ø20 **E** (crown Ø20 [S52]) | 0.5 **E** | flat/bellows Ø10–15 **E** | −600…−800 mbar **E** | **peel**: tall, small cup; liquid slosh |

### 2.2 Per-product notes

#### P1 Pharma blister card (easy)
- **Materials (S):** PVC cavity web 250–300 µm, aluminium lidding 20 µm [S53]; "a 20 µm rigid aluminium foil heat-sealable lacquered to PVC" is the basic standard [S53].
- **Mass (D):** a 95×46 mm card has 1.5 g of PVC (43.7 cm² × 0.025 cm × 1.39 g/cm³) and 0.24 g of Al. With 10 tablets of 0.25–0.5 g each (**E**), the card weighs **4–7 g**. Card size is **E**: we found no published standard footprint.
- **Picking:** delta transfer with suction arm in cartoners [S28]; FANUC M-1iA delta for blister retrieval at 60/min [S27].
- **Cup wear (S):** on a high-speed blister line, silicone cups "were failing far earlier than expected due to the speed, cycle rate, and repetitive contact"; Piab F20MF.40 TPU flat cups gave longer life [S9]. This is a good, sourced reason to include **cup wear drift** for this product.
- **Sim:** a rigid flat plate, an impermeable smooth surface, near-perfect seal. It is the **control product**: drops should only appear at extreme accelerations or with worn cups.

#### P2 Moulded praline (easy, but temperature-sensitive)
- **Mass and size (S):** a commercial praline mould gives about 11 g at Ø22 × 20 mm; Felchlin's Amo praline is 10 g [S50]. Other common moulds give 7.5 g; small half-shells 2.2 g [S50].
- **Density (S):** 1.3 g/cm³ at 25 °C and 1.2 at 50 °C for coating chocolates (pycnometer) [S45]; 1.18–1.31 g/cm³ for commercial chocolate candies (helium pycnometry) [S46].
- **Cup (S):** Schmalz SPG is made for "dynamic handling and packing of chocolates in combination with picker robots". Its features [S5]:
  - suction area Ø33 mm, volume 8.1 cm³, silicone 45 or 65 Shore A;
  - round chocolates up to Ø35 mm, oval or square up to a 30 mm diagonal;
  - a "conical shape safely encloses even square pralines";
  - "high nominal flow compensates for leakages due to highly structured chocolates";
  - **"We recommend a maximum operating vacuum of −150 mbar."**
- **Margin (D):** even at −150 mbar, 15 kPa × the praline's top area (about 3.8 cm² for Ø22) gives about **5.7 N** against a weight of 0.11 N. The praline is a *high-margin* product for hold. Its risks are **surface damage and temperature** (see §3.3), not hold force.
- **Alternatives (S):** Piab piSOFTGRIP (soft fingers with vacuum cavities) for "fragile objects such as chocolate and waffle products" [S6].

#### P3 Digestive biscuit (medium: porous, brittle)
- **Size (S):** circular digestive biscuit, **Ø72 mm × 7.2 mm** [S31].
- **Mass (D):** McVitie's Digestives, 360 g pack = 24 biscuits → **15 g** per biscuit [S32]. The chocolate-coated version is labelled **16.7 g** per biscuit [S32].
- **Apparent density (D):** 15 g / (π × 3.6² × 0.72 cm³) = **0.51 g/cm³**. This agrees with sugar cookies at 0.56 and shortbread at 0.66 g/cm³ measured by CT and pycnometry [S35, S\*].
- **Porosity (S):** butter cookies ≈ 0.6 (bulk density by solid displacement, true density by pycnometry) [S33]; baked cookies 0.145–0.373 [S34]. Use **0.3–0.6** for a digestive (**E**).
- **Air permeability (E):** none published for biscuits. The bracket we use:
  - bread crumb 10⁻¹² to 3.6×10⁻¹⁰ m² [S36];
  - pre-gelatinised dough 0.02 darcy at porosity 0.10 to about 23 darcy at 0.60 (1 darcy ≈ 0.99×10⁻¹² m²) [S37].
  - Prior for a biscuit: **k ≈ 10⁻¹³–10⁻¹¹ m²** (log-uniform). The surface crust and docker holes make the leak path uneven across the face, so seal varies with pick position.
- **Strength (S):** centre-loaded, edge-supported digestive biscuits break at **12.5 ± 1.2 N** when sound and **9.6 ± 2.62 N** with natural checks (cracks); failure stress about 420 kPa; deep radial cracks near the centre cut strength by up to 50 % [S31].
- **Cup and vacuum:**
  - Piab's B-BL cups target mass-produced baked goods, including "thin crackers ... designed to avoid breakage" [S6] (**S**).
  - Schmalz runs porous parts at **−200 to −400 mbar** and gets the holding force from more flow and more area [S2] (**S**, stated for cardboard and MDF; transferring it to biscuits is **E**).
  - Schmalz requires safety factor ≥ 2.0 for "porous, rough" parts [S1] (**S**).
- **Variability:** weight dispersion "is primarily due to variability in dough piece weight", linked to dough-sheet thickness [S43] (**S**). A rotary moulder vendor claims weight CV ≤ 1 % [S44] (**S**, vendor claim). σ = 0.3 g (CV 2 %) is **E**: it allows for moisture loss in baking and inter-lane differences.

#### P4 Flow-wrapped bar (medium: film)
- **Mass (S):** single bars on sale in the EU and UK: KitKat 4-finger 41.5 g, Snickers 48–50 g [S63]. We use 45 g.
- **Dimensions (E):** no manufacturer dimension sheet found (retailer listings were contradictory). 120×30×15 mm gives 54 cm³, which at 45 g means 0.83 g/cm³. That is plausible for wafer or nougat with chocolate (solid chocolate is 1.2–1.3 [S45]).
- **Film (E):** metallised OPP with cold seal is the usual chocolate-bar flow wrap [S62]. One converter lists 0.7–1.2 mil (18–30 µm) for a flow-wrap film, but the excerpt is ambiguous [S62].
- **Cup (S):** the Piab DURAFLEX rectangular cup (RB20x40P / RB20x45P; sources differ on the part number):
  - developed for flow-packed bars of sweets;
  - a "relatively firm lip" avoids the lip being sucked in;
  - material combines rubber-like elasticity with PU wear resistance [S7, S8].
- **Failure evidence (S):** at the FlexPicker site in [S7], the old cup "wore out very quickly"; "the vacuum level decreased and the product was dropped during its movement". Wear → leak → vacuum dip → drop is exactly the chain our vacuum-sensor feature must reproduce.
- **Sim (E):**
  - Film adds a compliant layer: the product can rotate slightly in the bag, which lowers the effective peel stiffness.
  - Wrinkles and the fin seal under the lip add leak area. Model this as a per-pick random leak fraction that grows with pick offset from the bar centreline.

#### P5 Vacuum-packed cheese wedge, chilled (hard: heavy, wet, off-centre CoM)
- **Evidence (S\*):** Sachsenmilch picks wedges with Delta-3 pickers and Venturi vacuum. The outer wedges "vary significantly" in size, the middle one does not. Wedges that cannot be picked "due to drawn in secondary air" go to a chute [S24]. Piab on vacuum-packed cheese blocks: "risk that the block of cheese will fall off due to leaking air" [S58].
- **Mass (E):** 200 g (150–250 g) is a typical retail wedge. We did not find a sourced mass for the Sachsenmilch wedges. If sold at fixed weight, the EU TNE for 100–200 g is 4.5 % and for 200–300 g is 9 g [S51] (**S**).
- **Spread:** one cheese portioning line ran giveaway of 4–5 % before an upgrade [S57] (**S**). Packers aim slightly over nominal, so σ ≈ 3 g (2–5 g) is **E**.
- **Density (E):** 1.05–1.15 g/cm³. A study measured the true density of 22 cheeses and fitted linear equations on moisture and fat [S64], but we could not access its values.
- **Geometry (E):** a right-triangle prism 100×70 mm, 50 mm thick (175 cm³ → 193 g at 1.1 g/cm³), picked on a triangular face. The centroid sits at one third of each leg from the right angle. A camera centring on the bounding box misses it by about 10–17 mm (**D**). This is a **physical, sourced-mechanism** source of off-centre picks.
- **Surface:** wet surfaces μ = 0.2–0.3 [S1] (**S**). Condensation on chilled product: see §3.3.

#### P6 Wet pet-food pouch, 85 g (hard: flexible, moving CoM)
- **Formats (S):** 4-side-seal pouches in 50 g and 85 g, stand-up pouches in 100 g [S54]. Retort laminate PET/AL/NY/RCPP for 85 g cat-food pouches [S55]. One retail listing gives 1 × 9.5 × 14 cm [S56] (weak source).
- **Cup (S):** Schmalz SPB2f, made "for dynamical handling of pouches and other types of very flexible packaging ... containing fluids, solids or powder", "stable in extremely quick packaging processes, particularly in conjunction with delta robots" [S4]. It has an integrated bag stabiliser so "even thin and soft films" are not sucked in [S4].
- **Holding forces (S, theoretical, smooth dry surface, no safety factor) [S4]:**

  | Cup | F @ −200 mbar | F @ −400 mbar | F @ −600 mbar | Volume |
  |---|---|---|---|---|
  | SPB2f 30 | 4.3 N | 8.5 N | 12.8 N | 7.2 cm³ |
  | SPB2f 40 | 6.8 N | 13.6 N | 20.4 N | 12.9 cm³ |
  | SPB2f 50 | 13.7 N | 27.3 N | 41.0 N | 26.5 cm³ |

  **Important for the sim (D):** these forces are far below p × π D²/4 of the nominal lip diameter. At −600 mbar a Ø50 lip would give 118 N, but Schmalz lists 41 N, which corresponds to an effective diameter of 29.5 mm. For SPB2f 15/20/25 the effective diameter (4.1 / 6.0 / 8.0 mm) equals the listed inner diameter d2 [S4]. So the **effective area of a bellows cup with a stabiliser is set by its inner opening, not the lip**. Our adhesion gain must use an effective area, not the nominal one.
- **Rate:** 135 pouches/min with 2 deltas, 2 pouches per cycle [S26] (**S**, gripper type not stated).
- **Sim (E):** model as a stiff film shell with a viscous inner mass (or two bodies joined by a soft slide joint). CoM moves 5–15 mm under 10 g lateral acceleration (**E**).

#### P7 Filled 10R injection vial (hard, stress test for peel)
- **Glass (S):** ISO 8362-1 10R: outer Ø24 mm, height 45 mm, mass 9.5 g, crown Ø20 mm. 2R: Ø16, height 35, 4.4 g, crown 13 mm. 6R: Ø22, height 40, 7.9 g [S52].
- **Filled mass (D/E):** 9.5 g glass + 5–10 ml fill + stopper and crimp cap ~2 g (**E**) → **16–22 g**.
- **CoM (E):** the liquid sits low, so h ≈ 25–30 mm below the cap top. The ratio h / cup radius of about 4 makes this the **peel-dominated** product.
- **Caveat:** we found suction picking of vials only on a non-delta loader (105 vials per lift, 180/min [S30]). Keep it as a stress test, not as a headline product.

#### Not included, and why
- **Raw poultry fillets:** delta vacuum picking exists [S60], but we found no sourced surface or mass data in the time available. A good next product: soft, wet, irregular, 150–300 g (**E**).

---

## 3. Variability and drift on a real line

### 3.1 Weight distributions and legal limits (EU e-mark)
Directive 76/211/EEC, Annex I [S51] (**S**), for each pack (not each biscuit inside a pack):
1. "the actual contents shall not be less, on average, than the nominal quantity";
2. only a small proportion of packs may fall short by more than the tolerable negative error (TNE). The usual reading, from the reference test plan, is ≤ 2.5 %; that figure is not quoted in our excerpt;
3. no pack bearing the ℮ may fall short by more than **2 × TNE**.

| Nominal Qn | TNE |
|---|---|
| 5–50 g | 9 % of Qn |
| 50–100 g | 4.5 g |
| 100–200 g | 4.5 % of Qn |
| 200–300 g | 9 g |
| 300–500 g | 3 % of Qn |
| 500–1,000 g | 15 g |
| 1,000–10,000 g | 1.5 % of Qn |

Consequences for the sim (**D**):
- 45 g bar: TNE = 4.05 g; 85 g pouch: TNE = 4.5 g; 200 g wedge: TNE = 9 g.
- Packers run filler targets above nominal to stay within the rules; the way to cut giveaway is to cut the filler's standard deviation [S65].
- One cheese portioning line ran 4–5 % giveaway [S57].
- So product mass on a real line is **slightly above nominal, with σ well inside TNE/2**. Our σ values above respect that.
- Checkweighers upstream log the mean and σ trend [S65]. Checkweigher weight is therefore a realistic per-pick input column.

### 3.2 Humidity (biscuits, baked goods)
- **Fresh biscuit moisture (S):** rotary-moulded biscuits are baked to 2.0 ± 0.2 % moisture (wet basis) [S42].
- **Critical water activity (S):** crackers, popcorn and chips lose crispness above aw 0.35–0.50, depending on product [S38]. Cookies keep crispness below a critical RH and lose texture abruptly above it (sigmoidal curve) [S39]. Commercial biscuits stored at 35 °C gained moisture and aw, with significant texture changes in the first 40 days [S40].
- **Mechanics vs aw (S):**
  - crackers: maximum compression force falls as aw rises from 0.14 to 0.80;
  - crisp bread: fracture stress rises up to aw ≈ 0.26–0.32, then drops sharply [S38].
- **Uptake speed (S/E):** moisture moves through biscuit by Fickian diffusion, with an external transfer resistance at the air/biscuit interface of about 0.018 m/s [S41] (**S**). Our estimate (**E**): a 7 mm biscuit exposed at 50–70 % RH changes measurably only over hours to days, not during the seconds it spends on the pick belt.
- **Line implication (E):**
  - Humidity drift matters through **upstream buffers and cooling conveyors**. Long cooling belts make RH, draughts and room temperature a factor in biscuit checking (cracking) [S66] (**S**).
  - It matters less through the pick itself.
  - Model humidity as a **shift-scale drift** in mass (+0 to +2 %), in breaking strength (± up to ~20 %) and in a higher fraction of checked (cracked) biscuits, which break at 9.6 ± 2.6 N instead of 12.5 ± 1.2 N [S31].
  - The effect of humidity on **vacuum leak through a biscuit is unpublished**. The current `delta_toy.py` assumes humidity cuts seal on porous items. Keep it only as a labelled assumption, with a small effect (≤ 10 % seal change over 45–75 % RH, **E**).
- **Moisture gradients (S):** moisture gradients above 1.5 %/cm produced significant cracking; 1.11 %/cm still gave 2.8 % cracked biscuits [S66].

### 3.3 Temperature (chocolate, chilled cheese)
- **Chocolate (S):**
  - stable form V cocoa butter melts at 33.8 °C [S48];
  - storage 18–20 °C at ≤ 50–55 % RH [S47];
  - cooling tunnels step down to about 15 °C, then warm the surface above the room dew point at the exit to avoid condensation [S49];
  - one patent example wraps at 20 °C room temperature with a 7 °C dew point [S49];
  - condensed moisture dissolves surface sugar, which recrystallises as **sugar bloom** [S49].
- **Chocolate in the sim (E):**
  - above ~25 °C the surface softens: lower μ, risk of cup marks;
  - below the dew point it gets wet: μ → 0.2–0.3 [S1].
  - Use product surface temperature minus room dew point as a hidden drift variable.
- **Dew points (D, Magnus formula):** 20 °C/50 % RH → 9.3 °C; 20 °C/60 % → 12.0 °C; 22 °C/60 % → 13.9 °C; 25 °C/70 % → 19.1 °C.
  - A chilled cheese wedge at 2–8 °C entering a 20–22 °C packing hall is **below the dew point in all these cases**, so its film fogs.
  - We found **no published measurement** of how this changes suction grip on cheese (**gap**).
  - Model it as wet μ (0.2–0.3 [S1]) with probability rising with RH.

### 3.4 Suction-cup wear
- **Vendors:**
  - Schmalz's Elastodur ED-65 gives "three times the service life compared to nitrile rubber (NBR)" [S10] (**S**);
  - Schmalz's maintenance rule is condition-based: "check the suction cup ... for wear, cracks and leaks and replace if necessary" [S12] (**S**);
  - TPU outlasted silicone on a high-speed blister line [S9] (**S**).
- **Interval (S, weak):** a CMMS vendor suggests replacing cups every 500k–750k cycles, or when leak-down exceeds 15 % of baseline [S11]. This is not a manufacturer figure.
  - At 100 picks/min, 2 shifts, 500k cycles is about **5 shift-days** (**D**). Wear is therefore a realistic within-week drift.
- **Mechanism (S):** repeated flexing and rubbing wear the lip from rounded to flat; cycling eventually cracks the skirt by fatigue (patent literature, via [S67]). Worn cup → vacuum falls during the move → drop [S7].
- **Materials (S, Schmalz 2025 material guide [S3]):**

  | Material | Shore A | Long-term temperature | Note |
  |---|---|---|---|
  | NBR | 40–80 ±5 | −10…+70 °C | universal |
  | Silicone (SI / SI-HD) | 40–70 ±5 (HD 65) | −30…+180 °C | FDA 21 CFR 177.2600, direct food contact (also [S4, S5]) |
  | PU | 55 ±5 | −30…+100 °C | |
  | Vulkollan | 72 ±5 | −30…+80 °C | |
  | Elastodur ED | 60–85 ±5 | −25…+80 °C | |

  Piab food cups: silicone approved under FDA 21 CFR 177.2600 and EU 1935/2004, with metal-detectable versions [S6].
- **Sim (E):** wear state w ∈ [0, 1] increases with picks:
  - effective lip leak area rises roughly linearly with w;
  - lip friction falls ~10–20 % at w = 1;
  - leak-down at w = 1 is about 15 % worse than new [S11].

### 3.5 Dust and crumbs
- **S:** vacuum grippers are dust-sensitive. Schmalz lists a dirty filter as a cause of poor grip [S12]. The SPB2f takes an optional insert filter [S4]. Piab dairy cups have "no dust pockets" and "can be equipped with filters" to prevent clogging [S6]. Coval recommends 60 or 100 µm stainless filters [S67].
- **Gap:** no quantified crumb-loading rate or vacuum loss per hour for bakery lines was found.
- **E:** for P3, add a slowly rising flow restriction, filter conductance −0…30 % per shift, reset at cleaning. It shows as longer evacuation time, a feature the vacuum sensor already gives.

### 3.6 Other per-pick randomness
- **Pick offset:** IRB 360 conveyor-tracking repeatability is 1.0 mm at 200 mm/s, 1.5 mm at 350–750 mm/s, 5.0 mm at 800–1,400 mm/s [S14] (**S**). Use σ ≈ repeatability plus vision error (**E**, 1–2 mm).
- **Product orientation and shape:** 3D scanners check height to ±0.5 mm and detect missing cream filling [S17] (**S**).

---

## 4. Line parameters

| Parameter | Value | Tag |
|---|---|---|
| Picks per robot per minute (vendor maxima) | Syntegon RPP: up to 120 per robot, 140 max for a single arm [S20]; Cama Triaflex: 120 cycles/min [S23]; ABB IRB 365: up to 120 picks/min (1 kg, 25/305/25 mm) [S66b]; Bosch Astor: 140 units/min [S64b] | S |
| Picks per robot per minute (installed lines) | Pepperidge Farm > 95 [S22] (S\*); Schubert 530/8 ≈ 66 [S17]; 529/6 ≈ 88 [S18]; ≈ 104 on the 48-robot biscuit line [S19] | S / D |
| Robots per line | 2 (Syntegon basic RPP [S20]; Mackle [S26]), 3–4 (Cama [S23]), 4 (Sachsenmilch [S24]), 6 (Toms [S18]), 8 (Schubert [S17]), 12–14 (Pepperidge [S22]), 28 and 48 (Schubert [S19]) | S |
| Cycle time | IRB 360-1/1130, 25/305/25 mm: 0.30 s at 0.1 kg, 0.36 s at 1 kg; 3 kg version 0.52–0.54 s. **Includes 35 ms gripper activation at pick and at place** [S14, S15] | S |
| Acceleration | ~150 m/s² (1 kg version), ~100 m/s² (3 kg); "up to 12 g" [S13, S15] | S |
| Product belt speed | tracking tested 200–1,400 mm/s; start/stop conveyors 500 mm/s [S14] (S); one user's line at 22 m/min ≈ 0.37 m/s [S16] (S, forum); Cama belt 110 m/min [S23] (S\*, could not verify) | S / S\* |
| Belt width | 1,600 mm (Schubert biscuit line) [S17] | S |
| Product pitch | random flow, located by vision (Greymans, FANUC [S65b]; Bosch Astor [S64b]); "spreading rakes" separate chocolates to the minimum spacing the scanner needs [S18]. Pitch per lane = belt speed / products per lane per minute (D). Example: 0.37 m/s and 100 products/min per lane → 0.22 m pitch | S / D |
| Missed or failed picks | pass to the next robot downstream; at the end, unpicked product is discharged by chute or recirculated [S24]; defective products are not picked [S18]; incomplete trays go to a buffer belt for manual refill [S18] | S |
| Vision reliability | "over 99.9 percent reliable data" required [S13] | S |
| Placement accuracy | ±1 mm (Mackle) [S26] | S |
| Vacuum levels | dense: −600…−800 mbar; porous: −200…−400 mbar; catalogue forces at −600 mbar; −600 → −900 mbar gives 1.5× force but 3× evacuation time and energy [S2]; chocolates ≤ −150 mbar [S5] | S |
| Sizing rules | load cases: vertical FTH = m(g+a)S; horizontal FTH = m(g + a/μ)S; S ≥ 1.5 smooth/dense, ≥ 2.0 porous/rough/oily, ≥ 2.5 when swivelling [S1] | S |
| μ reference values | oily 0.1–0.3; wet 0.2–0.3; wood/metal/glass/stone 0.5; rough 0.6 [S1]. Rubber on oily metal ≈ 0.2, raised to 0.4–0.5 by friction cups [S61] | S |

---

## 5. Suggested simulator priors (replacing `delta_toy.PRODUCTS`)

All values come from §2. The distributions are **E** unless the tag says otherwise. Masses in g, lengths in mm, vacuum in kPa below ambient.

| key | mass ~ N(μ, σ) | dims | h | cup D (eff.) | p_set | μ ~ U | leak / permeability prior | special |
|---|---|---|---|---|---|---|---|---|
| `blister` | N(5, 0.1) | 95×46×4 | 2 | 2× Ø16 (eff ≈ lip) | 60 | U(0.4, 0.6) | near zero | wear-sensitive (TPU vs silicone) |
| `praline` | N(10.5, 0.2) | Ø22×20 | 8 | Ø33 bell (eff ≈ product top Ø22) | **15** (S) | U(0.3, 0.7) | low, decoration adds leak | temperature/dew point → μ |
| `digestive` | N(15, 0.3) | Ø72×7.2 | 3.6 | Ø28 bellows | 30 | U(0.5, 0.7) | k ∈ 10⁻¹³–10⁻¹¹ m² log-uniform | breakage at 12.5 ± 1.2 N (checked: 9.6 ± 2.6 N); dust |
| `bar` | N(45, 0.7) | 120×30×15 | 7.5 | 20×45 rect | 50 | U(0.3, 0.6) | wrinkle leak ∝ offset from centreline | film compliance |
| `cheese_wedge` | N(200, 3) | tri 100×70, t = 50 | 25 | Ø45 bellows | 60 | U(0.2, 0.3) wet / U(0.4, 0.6) dry | fold leak (secondary air) | centroid offset 10–17 mm |
| `pouch85` | N(89, 1.5) | 140×90×15 | 7.5 ± slosh | SPB2f 50 (eff Ø29.5, S) | 50 | U(0.3, 0.6) | low | moving CoM, film into cup |
| `vial10r` | N(18, 0.3) | Ø24×45 | 27 | Ø12 | 70 | U(0.4, 0.6) | near zero | peel-dominated |

Drift variables, all hidden from the camera but partly visible to the vacuum sensor:
- room RH and temperature against product temperature (dew point);
- cup wear w (picks since change);
- filter loading (bakery);
- fraction of cracked biscuits;
- product batch shifts in mass and porosity (changeover).

---

## 6. Gaps (could not source)

1. Friction coefficient between cup elastomers and food or film surfaces: none published. Measure, or keep wide priors.
2. Air permeability of biscuits and crackers: none published. Bread and dough values are used as bounds.
3. Effect of humidity on suction seal for porous food: none published.
4. Effect of condensation on vacuum grip on chilled packs: none published.
5. Quantified cup wear curves (leak vs cycles) from a manufacturer: none. Only the 3× ED/NBR claim and a CMMS vendor rule of thumb.
6. Exact dimensions of branded bars, pouches and wedges: only retail listings, which were contradictory. These dimensions are marked **E**.
7. Real drop rates on production lines: not published (also noted in `application-delta-picking.md`).
8. The Pepperidge Farm, Sachsenmilch and Cama belt-speed numbers were read only through search excerpts (**S\***).

---

## Sources

Read directly unless marked (excerpt).

- [S1] Schmalz, *Theoretical holding force of a suction cup* (load cases, safety factors, μ reference values; "not possible to specify generally valid values"). https://www.schmalz.com/en/vacuum-knowledge/the-vacuum-system-and-its-components/system-design-calculation-example/theoretical-holding-force-of-a-suction-cup
- [S2] Schmalz, *Energy required for vacuum generation* (porous −200…−400 mbar, airtight −600…−800 mbar, catalogue at −600 mbar, −600→−900 mbar trade-off) (excerpt). https://www.schmalz.com/en/vacuum-knowledge/basic-knowledge/energy-required-for-vacuum-generation/
- [S3] Schmalz, *Selection guidelines: material of the suction cup* (2025). https://media.schmalz.com/MAM_Library/Dokumente/Auswahlhilfen/VA/272fbdc64cc1_Document_Selection_Guidelines_Material_Suction_Cup_EN_2025.pdf
- [S4] Schmalz, *Bellows Suction Cups SPB2f (2.5 folds)* datasheet. https://pimmedia.schmalz.com/MAM_Library/Dokumente/Datenblatt_Produktfamilie/0_/050/05031/Datasheet_Bellows%20Suction%20Cups%20SPB2f%20(2.5%20Folds)_en-EN.pdf
- [S5] Schmalz, *Chocolate Suction Cups SPG* datasheet. https://pimmedia.schmalz.com/MAM_Library/Dokumente/Datenblatt_Produktfamilie/0_/043/04302/Datasheet_Chocolate%20Suction%20Cups%20SPG_en-EN.pdf
- [S6] Piab, *Food automation* (B-BL bakery cups, piSOFTGRIP, silicone FDA/EU approval, dairy cups with filters). https://www.piab.com/industries/food--beverage/food-automation
- [S7] Piab success story, *70 % improvement of the robot's production through less dropping of products* (FlexPicker, RB20x45P). https://piab.com/success-stories/70-improvement-of-the-robots-production ; Piab/A3 product note *Raising the bar with rectangular suction cup* (excerpt). https://www.automate.org/products/piab-usa-inc/raising-the-bar-with-rectangular-suction-cup
- [S8] Piab piGRIP / BGX notes (50 Shore silicone lip, bag insert) (excerpt). https://www.packworld.com/machinery/controls-automation/product/22210730/piab-usa-inc-bag-handling-suction-cup
- [S9] TSI Solutions, *Vacuum cup material selection* (silicone → Piab F20MF.40 TPU on a high-speed blister line) (excerpt). https://www.tsisolutions.us/blog/588/vacuum-cup-material-selection
- [S10] Schmalz, *Safe automatic packaging of pharmaceuticals and food* (ED-65 three times the service life of NBR; FDA / EC 1935/2004) (excerpt). https://www.schmalz.com/en/career-company/latest/news/safe-automatic-packaging-of-pharmaceuticals-and-food
- [S11] Oxmaint (CMMS vendor), *Robotic pick-place maintenance* (500k–750k cycles, 15 % leak-down rule; not a manufacturer figure) (excerpt). https://oxmaint.com/industries/fmcg/robotic-pick-place-maintenance-fmcg-gripper-cmms
- [S12] Schmalz SBPG operating manual, troubleshooting and maintenance (excerpt). https://www.manualslib.com/manual/2140790/Schmalz-Sbpg.html?page=25
- [S13] R. Tallian (ABB), *Creating an optimal high-speed robotic picking solution* (white paper). https://library.e.abb.com/public/d02772b2e8cc14fcc12575bb005c2c06/ABB%20white%20paper%20Rick%20Tallian%20High%20speed%20picking.pdf
- [S14] ABB, IRB 360 datasheet (conveyor tracking table, cycle times) (excerpt). https://library.e.abb.com/public/f03878770fef4c2c93757953373aed9b/IRB%20360_datasheet_20220829.pdf
- [S15] ABB, IRB 360 product specification ROB0082EN_B (2009; 35 ms gripper activation, 100 m/s² for the 3 kg version) (excerpt). https://library.e.abb.com/public/c3eafb7f9dd274fac1257b1300579ef8/ROB0082EN_B%20final.pdf
- [S16] ABB Robotics tech community, *Wrong pick positions conveyor tracking* (22 m/min line) (excerpt). https://tech-community.robotics.abb.com/t/wrong-pick-positions-conv-tracking-pickmaster-3/5039
- [S17] Schubert, case study: pick-and-place machine at a confectionery producer. https://www.schubert.group/en/case-studies/pick-and-place-machine-confectionery-producer
- [S18] Schubert, case study: chocolate packing line at Toms. https://www.schubert.group/en/case-studies/chocolate-packing-line-toms/
- [S19] Schubert, *25 years of pick & place machines*. https://www.schubert.group/en/magazin/25-years-of-pick-place-machines-from-schubert-with-unmatched-versatility-to-take-on-any-packaging-task/
- [S20] Syntegon, robotic packaging (RPP: up to 120 picks per robot per minute; 140 max) (excerpt). https://www.syntegon.com/solutions/food/robotic-packaging/
- [S21] Syntegon, *Flexible robotic system solution for chocolate squares and seasonal shapes*. https://www.syntegon.com/news/flexible-robotic-system-solution-for-chocolate-squares-and-seasonal-shapes
- [S22] A3 case study, *Sigpack Delta robot for packaging at Pepperidge Farm* (page blocked; excerpt). https://www.automate.org/case-studies/sigpack-delta-robot-for-packaging-at-pepperidge-farm
- [S23] Control Design, *Candymaker combines, simplifies controls* (Cama Triaflex, 120 cycles/min, 300 products/min). https://www.controldesign.com/articles/2012/plc-candymaker-combines-simplifies-controls ; belt speed 110 m/min only in the A3 excerpt: https://www.automate.org/case-studies/chocolate-bars-uniquely-packaged
- [S24] International Dairy Magazine 11–12/2021, Sachsenmilch wedge packing (page cert expired; excerpt). https://dairy-international.com/wp-content/uploads/epaper/1000024864/IDM_11_12_2021/page_10.html
- [S25] ABB, Ilchester Cheese FlexPicker reference (excerpt). https://library.e.abb.com/public/7d48f3e94529ff31c12574560050898d/EN_30-31_ilchester_korr2_abb_pack_108.pdf
- [S26] OMRON, Mackle Petfoods case study (excerpt). https://robotics.omron.com/case-studies/automated-packaging-solution-mackle-petfoods/
- [S27] FANUC, Flexlink / Lynker blister case (excerpt). https://www.fanuc.eu/it/en/customer-cases/lynker
- [S28] Dividella (Körber) P5 cartoner (excerpt). https://www.medicalexpo.com/prod/dividella-ag-a-korber-group-company/product-114016-1012247.html
- [S29] Packaging Strategies, *New robotized blister packaging line debuts* (excerpt). https://www.packagingstrategies.com/articles/87220-new-robotized-blister-packaging-line-debuts
- [S30] STIWA, vial inspection loader datasheet (excerpt). https://www.stiwa.com/fileadmin/user_upload/content/automation/automation/Referenzen/Life_Science/Datenblatt_Vial_v01_EN_US_web.pdf
- [S31] Garcia-Armenta, Gutierrez, Anand, Cronin, *Analysis of the failure of cracked biscuits*, J. Food Eng. 196 (2017) 52–64 (abstract and index excerpt). https://cora.ucc.ie/handle/10468/3318
- [S32] Sainsbury's product pages: McVitie's Digestives 360 g (24 biscuits); Milk Chocolate Digestives (16.7 g per biscuit) (excerpt). https://www.sainsburys.co.uk/groceries/product/mcvitie-s-digestives-the-original-biscuits-360g ; https://www.sainsburys.co.uk/groceries/product/mcvitie-s-100g-milk-chocolate-digestive-biscuits
- [S33] Hicsasmaz & Clayton (1992), *Food Structure* 11(2) (bread 0.9, butter cookie 0.6 porosity) (excerpt). https://digitalcommons.usu.edu/foodmicrostructure/vol11/iss2/4
- [S34] Univ. of Illinois, microstructural characterisation of a wheat-based food (cookie porosity 0.145–0.373) (excerpt). https://experts.illinois.edu/en/publications/microstructural-characterization-of-a-wheat-based-food-material-u/
- [S35] PMC4384159, Table 5 (apparent density: sugar cookie ≈ 0.56, shortbread ≈ 0.66 g/cm³) (excerpt; page blocked). https://pmc.ncbi.nlm.nih.gov/articles/PMC4384159/table/T5
- [S36] Chaunier et al., *Food Biophysics* 3 (2008) 344–351, gas permeability of bread crumb (excerpt). https://link.springer.com/doi/10.1007/s11483-008-9073-8
- [S37] Goedeken & Tong (1993), *Permeability measurements of porous food materials* (dough 0.02–23 darcy) (excerpt). https://agris.fao.org/search/en/records/65deb19d6eef00c2cea0f125
- [S38] Katz & Labuza, J. Food Sci. 46 (1981) 403–409, via the review by Jakubczyk et al. (aw 0.35–0.50; cracker and crisp-bread mechanics vs aw) (excerpt). https://agro.icm.edu.pl/agro/element/bwmeta1.element.agro-article-7263b213-84b8-42ca-9721-e13acf9f5dd4/c/58_1_7_45-51.pdf
- [S39] Carter et al. (2015), J. Food Meas. Charact., critical RH for cookie crispness (excerpt). https://link.springer.com/10.1007/s11694-015-9254-3
- [S40] Romani et al. (2016), moisture adsorption of biscuits during storage (excerpt). https://www.pubmed.gov/26575718/
- [S41] *Moisture diffusivity and transfer modelling in dry biscuit* (external resistance ≈ 0.018 m/s) (excerpt). https://agris.fao.org/search/ru/records/65e01d5563b8185d9cb15d78
- [S42] Molina, Vaz, Bouchon, *Foods* 10 (2021) 621 (rotary-moulded biscuits, 2.0 ± 0.2 % moisture). https://pmc.ncbi.nlm.nih.gov/articles/PMC7999813/
- [S43] Cronin & Preis, J. Food Eng. 46 (2000) 217–225, doi:10.1016/S0260-8774(00)00053-4. https://research.ucc.ie/en/publications/statistical-analysis-of-biscuit-physical-properties-as-affected-b/
- [S44] Bühler RotaMold Smart brochure (weight CV ≤ 1 %, vendor claim) (excerpt). https://dam.buhlergroup.com/asset/1664481/PowerPoint%20Presentation
- [S45] Luccas et al. (1997), specific heat and true density of coating chocolate (IIR) (excerpt). https://iifiir.org/en/fridoc/experimental-determination-of-the-specific-heat-and-true-density-of-113130
- [S46] Micromeritics application note, density of commercial items (excerpt). https://downloads.micromeritics.com/Appnotes/Higher-Accuracy-Plus-Faster-Results-when-Measuring-the-Density-of-Commercial-Items.pdf
- [S47] Sécurité alimentaire Luxembourg, *Blanchiment du chocolat* (18–20 °C, ≤ 50–55 % RH) (excerpt). https://securite-alimentaire.public.lu/dam-assets/fr/publications/conseils/blanchiment-chocolat.pdf
- [S48] ChemistryViews, cocoa-butter polymorphs (form V 33.8 °C) (excerpt). https://chemistryviews.org/?p=40521
- [S49] US 6391356, US 6419970, US 6406733 (tunnel exit above dew point; wrap at 20 °C / 7 °C dew point); IFT Food Technology Dec 2004 (sugar bloom) (excerpt). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6391356 ; https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6419970 ; https://www.ift.org/news-and-publications/food-technology-magazine/issues/2004/december/columns/processing
- [S50] Felchlin Amo praline sheet (10 g); polycarbonate praline mould 22 mm × 20 mm ≈ 11 g; Chocolate World curve mould ≈ 7.5 g (excerpt). https://www.felchlin.com/uploads/PR10093_Amo_Champagne_Caramel_Praline_EN_IN.pdf ; https://www.cfw.co.uk/Chocolate-World-Mould-Praline-Curve-Frank-Haasnoot
- [S51] Council Directive 76/211/EEC, Annex I (as amended by 78/891/EEC). https://legislation.gov.uk/eudr/1976/211/annexes/data.htm?view=plain ; https://eur-lex.europa.eu/eli/dir/1976/211/oj
- [S52] SCHOTT Pharma, *Core Vials in StandardLine quality* (ISO 8362-1 dimension and mass table). https://media.schott.com/api/public/content/0815cb6af9774080a9ff85e150f8a724
- [S53] Adelphe, *Guide blister* (PVC 250–300 µm, Al 20 µm); lidding foil presentation (20 µm standard) (excerpt). https://www.adelphe.fr/wp-content/uploads/2026/03/guide-blister_en.pdf
- [S54] Huhtamaki, wet pet-food packaging (50/85 g 4-side-seal, 100 g stand-up) (excerpt). https://www.huhtamaki.com/en/flexibles-pet-food/flexibles-wet-pet-food/
- [S55] M-First Pack, 85 g retort pouch (PET/AL/NY/RCPP, 121–135 °C) (excerpt). https://www.mfirstpack.com/85g-pet-wet-food-retort-pouch-product
- [S56] Desertcart listing, Schesir 85 g pouch, 1 × 9.5 × 14 cm (weak, retail) (excerpt). https://www.desertcart.ma/products/257842129
- [S57] Rockwell Automation, Marchant Schmidt cheese portioning (4–5 % giveaway) (excerpt). https://www.rockwellautomation.com/en-us/company/news/case-studies/solutions-in-action--marchant-schmidt--inc--.html
- [S58] Piab, TGW cheese-block handling (excerpt). https://www.piab.com/TGW
- [S59] Food Engineering, Arla cheese rounds with Piab vacuum cruise control. https://www.foodengineeringmag.com/articles/86841-cruise-control-option-eases-robotic-conveying
- [S60] JBT Marel RoboPacker (excerpt) https://jbtmarel.com/en/news/robopacker/ ; Vendée Concept Delta Calibra (excerpt) https://wattagnet.com/WATTPoultry-Annual-Buyers-Guide/poultry-processing/product/15483347/vendee-concept-delta-calibra-robot
- [S61] Automotive Manufacturing Solutions, friction cups on oily sheet (μ ≈ 0.2 → 0.4–0.5) (excerpt). https://automotivemanufacturingsolutions.com/speeding-up-during-global-slowdown/6694.article
- [S62] Napco, chocolate bar wrap (excerpt) https://napconational.com/food/confectionery/chocolate-bar-wrap ; TC Transcontinental confectionery films (0.7–1.20 mil, ambiguous) (excerpt) https://tctranscontinental.com/en-ca/node/814
- [S63] Sainsbury's, KitKat 4 finger 41.5 g; Tesco, Snickers single 48 g (excerpt). https://www.sainsburys.co.uk/groceries/product/kitkat-4-finger-milk-chocolate-bar-41-5g ; https://www.tesco.com/shop/en-GB/products/281025126
- [S64] Iezzi, Locci, Mucchetti, *Cheese true density prediction by linear equations*, J. Food Process Eng. 36 (2013) 462–469 (values not accessed). https://zendy.io/title/10.1111/jfpe.12008
- [S64b] Matrox Imaging, delta robot sorts biscuits (Bosch Astor, 140/min) (excerpt). https://imaging.matrox.com/en/media/case-studies/matrox-imaging-library-helps-delta-robot-sort-biscuits
- [S65] Mettler-Toledo, *Checkweighing guidance* (giveaway vs σ, trend monitoring) (excerpt). https://www.mt.com/gb/en/home/library/know-how/product-inspection/PI-Guides/checkweighing-guide/checkweighing-wiki.html
- [S65b] FANUC, Greymans waffles case (random product on belt, visual line tracking) (excerpt). https://www.Fanuc.eu/~/media/corporate/customer-cases/greymans/pdf/ro%20greymans%20nl-en-lr.pdf?la=en
- [S66] Biscuit People Conference 2019, *Checking in crackers* (cooling-belt RH and draughts); microwave post-baking study on moisture gradients and cracking (excerpt). https://archive.biscuitpeople.com/wp-content/themes/understrap-child/img/bpc2019/presentations/bpc2019-day1-4-annelyse-bartolo-checking-in-crackers.pdf ; https://agris.fao.org/search/en/records/65de2a3eb766d82b18fc7158
- [S66b] ABB, IRB 365 (up to 120 picks/min, 1 kg, 25/305/25 mm) (excerpt). https://www.abb.com/global/en/areas/robotics/products/robots/delta-robots/irb-365
- [S67] Wear-mechanism patents and filter guidance, via search excerpts: Robotiq on dust sensitivity https://blog.robotiq.com/how-robots-with-vacuum-grippers-improve-food-safety ; Coval washdown documentation (60/100 µm filters) https://issuu.com/coval/docs/washdown_doc_coval_2023_en_issu
