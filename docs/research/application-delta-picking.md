# Application: high-speed delta-robot picking with suction

Researched 2026-11-21. Vendor figures are marked as vendor claims, and illustrative calculations are marked as such. Market-report numbers are summaries of paid reports, so their methods were not checked.

## 1. The problem in plain words

On a food or pharma packaging line, a conveyor carries loose products: biscuits, chocolates, cheese portions, sachets, blister packs, flow-wrapped bars. A delta robot hangs above the belt. For each product it:
1. puts a suction cup on it,
2. lifts it,
3. flies it 0.3–0.5 m sideways,
4. drops it into a tray or carton.

It does this in about 0.4 s, with accelerations of 100–150 m/s² (10–15 g).

The suction cup holds the product with a limited force: vacuum × cup area. When the robot accelerates, the product pushes against that grip with mass × acceleration. A tall or off-centre product also gets a tipping moment that **peels** the cup off. When the grip loses, the product:
- **drops** onto the belt or the machine (scrap, possibly a stop to clean up),
- **lands out of position** in the tray (reject or downstream jam),
- or gets **damaged** (cracked biscuit, dented pack).

The robot's speed setting decides the trade-off:

```
faster  -> more picks per minute -> more drops and misplacements
slower  -> fewer failures        -> lost output (often the line's bottleneck)
```

The right setting depends on the product. Some products vary in ways the camera cannot see: weight, porous or oily surfaces that leak vacuum, an off-centre filling, moisture changes during a shift.

## 2. Evidence that this problem exists

- **ABB (largest delta-robot maker), white paper on high-speed picking:** high-speed motion creates forces that impact the stability of the product in the gripper, as well as the potential for product damage.
- **Robot specs show how extreme the motion is.** ABB IRB 360 FlexPicker:
  - up to 10 m/s and about 150 m/s² (1 kg version) or about 100 m/s² (3 kg version);
  - typical cycles 0.36–0.44 s;
  - the 8 kg model does up to 100 cycles/min.
  - Syntegon (vendor claim): up to 120 picks per robot per minute.
- **Research names the exact failure.** UC Berkeley's GOMP-ST (WAFR 2022): "high-speed motions can cause failure of suction cups". They had to learn failure limits from physical trials (see §3).
- **Failed picks stop lines.** A Lund University thesis on a delta robot stacking packages: errors otherwise require stopping the line for manual correction.
- **Small failure rates are expensive at volume.** Locus Robotics (a vendor), on picking: even fractions of a percent of failed picks scale to significant cost.
- **Line economics:**
  - High-speed food packaging lines typically run at **55–68% OEE**, with minor stops and changeovers as the main losses (Oxmaint; vendor content with example figures).
  - Worked example (Shoplogix, illustrative): 10 micro-stops/hour × 10 s each is about 2.8% of time, or about 3,300 packs lost per 8-hour shift on a 250 packs/min line.
  - Changeovers that produce extra start-up scrap can leave OEE flat or worse.

## 3. How it is handled today

| Approach | What it does | What it leaves open |
|---|---|---|
| **Fixed speed/acceleration per product recipe** (set by the integrator at commissioning) | conservative settings chosen once by experience | leaves speed unused on easy items, still fails on unusual ones, and does not track drift during a shift |
| **Vacuum switch / sensor per pick** (standard on vacuum grippers) | confirms "item held" or "item lost" | **reacts after** the failure; does not change how the robot moves |
| **Smart vacuum condition monitoring** (Piab IO-Link ejectors, Schmalz smart vacuum, Festo AX with AI anomaly detection, SMC IO-Link sensors) | detects leaks, wear and evacuation-time drift for **maintenance** | aimed at the equipment, not at choosing each pick's motion. But it means **the data stream already exists on modern grippers** |
| **ABB PickMaster Twin** (digital twin) | offline simulation of the line; ABB claims changeover from hours to minutes and 15% faster picking | a model of layout and flow; it does not learn each product's grip physics from live picks |
| **Research: GOMP-ST** (Berkeley, 2022) | repeatedly moves a real robot with a **sample object**, measures vacuum pressure, learns failure limits and plugs them into a time-optimal planner as acceleration limits. Result: **16–58% shorter transport, near 100% success** (UR5, 1.3–1.7 kg objects, 420 experiments) | learning needs dedicated physical trials with a sample object; no calibrated uncertainty; no continual update during production |

GOMP-ST proves the value: learned, item-aware acceleration limits make suction transport **16–58% faster without failures**. What is missing for production is doing that **per product, from normal production picks, without dedicated trials or retraining, and staying current as products drift**.

## 4. How our solution fits

**What we build:** a world model that, for the next pick, predicts the probability of a drop and the placement error, for each candidate speed setting. It learns in context from the line's own recent pick log, using TabPFN.

| Input (all already on the line) | Source |
|---|---|
| product type / recipe, length, width, height | production order, camera |
| weight | **checkweigher** upstream (standard on food lines) |
| `seal_kPa`: vacuum reached when the cup touches | **vacuum sensor** in the gripper: a leaky or uneven surface shows here *before* the robot accelerates |
| `pick_offset_mm`: cup vs product centre | camera |
| belt speed | conveyor encoder |
| **actions:** acceleration override, vacuum setpoint, release height | robot program |

| Output (predicted, then measured) | How measured |
|---|---|
| P(drop) | vacuum lost mid-move / item missing in tray |
| placement offset and rotation | camera over the tray |

**What happens on each pick:**
1. The cup touches; the vacuum sensor reads the seal (about 10 ms, no extra cycle time).
2. TabPFN scores 10–20 candidate speed settings in one batched call.
3. The robot takes the fastest one under the risk limit.
4. The result is logged as a row.

**What happens on a product changeover:** the context is switched to that product's rows (or starts from similar products with wide uncertainty, so the planner is cautious). It sharpens within minutes, because a delta makes 100+ picks per minute.

**Fit with the related work:**
- **vs GOMP-ST:** the same goal (item-aware acceleration limits), but learned **in context from production data**, with **no dedicated trials, no training**, a calibrated spread, and continuous updating.
- **vs smart vacuum monitoring:** uses the same sensor stream, but **to decide how to move**, not only to raise a maintenance alert.
- **vs fixed recipes:** recovers the speed that conservative recipes waste, and slows down only for the picks that need it.
- **vs PickMaster-style twins:** complementary. The twin handles layout and flow; we handle per-pick grip physics.

**Who would buy it:**
- robot OEMs' picking software (ABB PickMaster, FANUC, Yaskawa),
- packaging-machine builders (e.g. Syntegon),
- vacuum-gripper makers that already sell smart sensors (Piab, Schmalz, Festo).

It would sell as a software add-on, because the sensors and data already exist.

## 5. How big the business is

**Delta robot market** (report summaries; definitions differ):

| Source | Estimate |
|---|---|
| Global Industry Analysts | ~$2.2 B (2024) → $4.2 B (2030), 11.4% CAGR |
| MarketIntelo | ~$2.1 B (2025) → $3.6 B (2030); pick-and-place is the largest application (32.5%) |
| Technavio | +$2.1 B growth 2024–2029, 16.8% CAGR, driven by high-speed packaging pick-and-place |
| Next Move | $4.14 B (2022) → $7.51 B (2030); 250k units (2023) → 641k units (2030) |

**Food robotics** (GM Insights / Market Glass summary): $3.07 B (2025) → $13.16 B (2035); delta robots 24.6% share.

**Illustrative value per line** (assumptions, not data). A 250 packs/min line, robot cell is the bottleneck, 2 shifts × 8 h × 250 days:
- **+5% throughput** at the same reject rate gives about 6.0 M extra packs per year.
- At €0.02–0.05 contribution margin per pack, that is about **€120k–300k per year per line**.
- Fewer drops and stops add to this.

GOMP-ST's 16–58% transport-time reduction suggests 5% overall is a cautious figure, but **we have to measure it in our own simulation**.

**Illustrative software opportunity:** if even 50,000 installed deltas used a €1–2k per robot per year add-on, that is €50–100 M per year. This is a back-of-envelope figure.

## 6. What we must be honest about

- Our results will come from **simulation** (MuJoCo, with suction modelled by the built-in adhesion actuator plus friction). We need to validate the suction model against physics the way we validated the Franka grip.
- The value per line depends on the robot being the bottleneck and on real drop rates, which we could not find published.
- GOMP-ST shows the speed-up is real on hardware. Our contribution is getting there **without dedicated training**, **with calibrated risk**, **adapting during production**.

## Sources

- ABB: [IRB 360 FlexPicker](https://new.abb.com/products/robotics/industrial-robots/irb-360) · [IRB 360 datasheet](https://library.e.abb.com/public/f03878770fef4c2c93757953373aed9b/IRB%20360_datasheet_20220829.pdf) · [IRB 360 product guide](https://library.e.abb.com/public/1c971fb90d8c34f8c1257c2100468677/ROB0082EN_F_HR.pdf) · [8 kg FlexPicker press release](https://library.e.abb.com/public/76f011cd7ae30711c1257b7a002c3a01/PressRelease_ABBFlexPicker8kg.pdf) · [White paper: high-speed picking](https://library.e.abb.com/public/d02772b2e8cc14fcc12575bb005c2c06/ABB%20white%20paper%20Rick%20Tallian%20High%20speed%20picking.pdf) · [Industrial delta robots](https://campaign-ra.abb.com/delta-robots-picking-packaging) · [PickMaster Twin](https://www.abb.com/global/en/areas/robotics/products/software/pickmaster-twin) · [PickMaster Twin launch](https://new.abb.com/news/detail/32511/abb-introduces-pickmasterr-twin-with-digital-twin-technology-to-reduce-commissioning-times)
- Research: [GOMP-ST (arXiv 2203.08359)](https://arxiv.org/pdf/2203.08359) · [WAFR 2022 version](https://autolab.berkeley.edu/assets/publications/media/WAFR_2022_GOMP_SucIT.pdf) · [Lund thesis on delta robot error handling](https://www2.iea.lth.se/publications/MS-Theses/Full%20document/5434_full_document.pdf)
- Vacuum monitoring: [Piab predictive maintenance](https://piab.com/news/digitizing-vacuum-generation-enables-predictive-maintenance-of-robotic-handling-systems) · [Schmalz / vacuum for the digital factory](https://www.industrial-production.de/handling-technology/vacuum-technology-for-the-digital-factory.htm) · [Festo AX vacuum](https://www.festo.com/us/en/p/festo-ax-asset-type-vacuum-id_GSDA_VAC) · [SMC vacuum sensors](https://www.smc.eu/en-gb/products/vacuum-switches~22446~nav)
- Line economics: [Oxmaint OEE guide](https://oxmaint.com/industries/food-manufacturing/oee-optimization-food-production-lines-practical-guide) · [Shoplogix OEE for packaging](https://shoplogix.com/oee-for-packaging-industry/) · [iFactory OEE roadmap](https://ifactoryapp.com/article/food-packaging-line-oee-improvement) · [Packaging World top OEE losses](https://www.packworld.com/home/news/13343489/top-10-sources-of-oee-losses) · [Locus Robotics on grasping](https://locusrobotics.com/blog/robotic-grasping-warehouse-automation) · [Syntegon robotic packaging](https://www.syntegon.com/solutions/food/robotic-packaging/)
- Market: [Technavio delta robots](https://www.technavio.com/report/delta-robots-market-industry-analysis) · [Global Industry Analysts](https://www.marketresearch.com/Global-Industry-Analysts-v1039/Delta-Robots-42758865) · [MarketIntelo](https://marketintelo.com/report/delta-robot-market) · [Next Move Strategy Consulting](https://www.marketresearch.com/Next-Move-Strategy-Consulting-v4226/Delta-Robots-Axis-Type-Double-37119574) · [GM Insights food robotics](https://www.gminsights.com/industry-analysis/food-robotics-market)
