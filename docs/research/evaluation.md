# Evaluation: TabPFN choosing the speed of every pick

Question: on a simulated IRB 360-class suction pick line (gates 1–5), does choosing each pick's speed setting with TabPFN, using the line's own log as in-context data, give more good picks per minute at the same or lower drop rate than today's fixed per-product recipe? Does it keep doing so when the line drifts or a new product arrives?

## Data

| Set | How | Rows |
|---|---|---|
| Commissioning log (in-context data) | `scripts/sweep_picks.py`: 6 line products × 9 speed settings × 50 picks under random line states. Cup 0–900k cycles; room 19–28 °C and 35–75 % RH; batch mass and seal shifts; belt 0.2–0.5 m/s. Split between laptop (s 0–24) and server (s 25–49); seeds are disjoint and the results deterministic per seed. | 2,700 |
| Evaluation bank | `scripts/make_bank.py`, streams in `sim/scenarios.py`. Each instance (item, line state, belt speed, vision error) is simulated at **all 9 settings**: exact counterfactuals. | 940 instances, 8,460 rows |

Scenarios (one stream per product):
- **wear:** the cup ages from 300k to 1.05M cycles, for all 6 products, 60 picks each;
- **humid:** room RH rises 40 → 80 % at 23 °C, so chilled products condense (cheese, praline), 90 picks each;
- **new:** a product absent from the log (pouch, cheese, bar), random line states, 80 picks each;
- **batch:** a new film batch after 20 picks (seal class −12 kPa, mass +3 %), for the bar and the pouch, 80 picks each.

## Decision and policies (`policy/policies.py`)

Each pick is decided after the 35 ms vacuum dwell, before the loaded move. The policy sees only what a real line knows at that moment:
- checkweigher mass;
- 3-D vision size;
- product and room temperature, room humidity;
- the cup cycle counter, cup size and vacuum set point;
- belt speed;
- the vacuum sensor's `seal_kpa` and `evac_ms`.

It chooses α ∈ {0.2, …, 1.0}.

- **Fixed recipe (today's practice):** one α per product, the fastest whose commissioning drop rate, and that of every slower setting, is at most the target. A variant runs one step slower as a safety margin. A product without a recipe runs at the slowest setting.
- **TabPFN-3.5:** P(drop | features, α) for the 9 settings of this pick, with the log as context, no training. The policy takes the fastest α with P ≤ target, or the slowest if none qualifies. The model is refit every 5 picks; a live line refreshes its speed map about once per second, which is every 2–3 picks.
- **Logistic regression** and **gradient-boosted trees** (scikit-learn): the same decision rule on the same log, retrained every 5 picks.
- **Oracle:** the fastest α that does not drop this very item. It knows the hidden friction, wetness and vision error; it is an upper bound.

**Replay with bandit feedback.** Every policy runs over the same stream and gets back only the outcome at the α it chose, which it appends to its log. That is exactly what a line would learn. The bank makes this exact: each policy's choice is looked up, not re-simulated.

**Metrics.**
- Good picks per minute: 60 × held picks / Σ cycle time. Cycle time is the simulated approach, both 35 ms dwells and the loaded move at the chosen α.
- Drop rate.
- 95 % bootstrap intervals over picks (2,000 resamples).
- Learning curves also report log-loss, Brier score, ROC AUC and expected calibration error on the bank.

## Results

### 1. How many logged picks does the drop model need? (`scripts/learning_curve.py`)

Context: random subsets of the 2,700-row commissioning log, 3 repeats per size. Test: all 8,460 bank rows, every speed of every item; these are unseen items, line states and drift. Means over repeats (`results/eval/learning_curve.csv`, figure `docs/figures/learning_curve.png`):

| context rows | log-loss: TabPFN / logistic / trees | calibration error (ECE) | AUC |
|---|---|---|---|
| 100 | **0.219** / 0.253 / 0.310 | 0.061 / 0.048 / 0.064 | **0.974** / 0.964 / 0.955 |
| 250 | **0.159** / 0.201 / 0.222 | **0.025** / 0.035 / 0.040 | **0.985** / 0.978 / 0.978 |
| 1,000 | **0.127** / 0.167 / 0.209 | **0.012** / 0.026 / 0.045 | **0.989** / 0.983 / 0.985 |
| 2,700 | **0.124** / 0.157 / 0.176 | **0.011** / 0.018 / 0.040 | **0.990** / 0.985 / 0.987 |

- **TabPFN with 250 logged picks predicts drops as well as logistic regression with 2,700** (log-loss 0.159 vs 0.157), and better than boosted trees with 2,700 (0.176).
- **It flattens at about 1,000–2,000 rows**, which sets the context budget. At this size the per-pick query is fast, see section 5.
- **Calibration:** at 2,700 rows the predicted probabilities are within about 1 percentage point of the observed rates on average (ECE 0.011). That is what makes "fastest speed under X % risk" mean what it says.

**Decisions from a static model** (no feedback; fastest setting with predicted risk ≤ target, per bank item):

| target | TabPFN good/min · drops | logistic | boosted trees |
|---|---|---|---|
| 5 % (2,700 rows) | **152.6 · 1.2 %** | 152.2 · 1.3 % | 145.3 · 9.5 % |
| 10 % (2,700 rows) | **153.4 · 1.3 %** | 149.8 · 4.3 % | 140.0 · 13.7 % |
| 5 % (250 rows) | **150.2** · 2.9 % | 147.6 · 0.9 % | 146.3 · 7.3 % |

On logged products a well-specified logistic regression is close to TabPFN at a strict target. The boosted trees are overconfident: their drop rate runs at twice the target. TabPFN's advantage grows as the target loosens, because its probabilities stay calibrated.

### 2. Where will the product land? (`scripts/placement.py`)

TabPFN-3.5 regressor, quantiles of its predictive distribution, fitted on the 1,919 held picks of the commissioning log and tested on the 4,926 held picks of the bank (`results/eval/placement.csv`):

| model | median error (mm) | pinball loss (q 5/50/95) | 90 % interval: coverage · width |
|---|---|---|---|
| **TabPFN-3.5** | **1.18** | **0.296** | 85 % · 4.5 mm |
| boosted trees (quantile loss) | 1.28 | 0.341 | 77 % · 4.5 mm |
| linear quantile regression | 1.71 | 0.416 | 89 % · 6.6 mm |

For the same interval width as the trees, TabPFN's intervals cover 85 % against the trees' 77 %; the linear model needs intervals 44 % wider to cover 89 %. On the pouch, whose sloshing contents shift it on the cup, the median error is 3.5 mm and the 90 % interval widens to 14 mm and still covers 92 %. The model knows when it is unsure.

### 3. Closed loop: choosing every pick's speed (`scripts/run_eval.py`, `scripts/report.py`)

**Decision rule used for the comparison.** Every policy maximizes the same expected value per second: (1 − (1 + c)·P(drop)) / cycle time, where a drop costs c picks' worth (scrap, clean-up, possibly a stop).
- **The fixed recipe** applies this rule per product, to the commissioning log's drop rate and cycle time at each setting.
- **The learned policies** apply it per pick, to their predicted P(drop) and the cycle time the planner knows before the move.

The only difference is therefore per-pick prediction against a per-product setting. Sweeping c from 1 to 100 traces each policy's trade-off.

Learned policies also use **safe-set expansion**: never more than one setting above the fastest setting already seen to hold for this product. This is the idea behind SafeOpt, and how engineers ramp up a new product. Without it, most of TabPFN's drops on new products came from single jumps to settings it had never seen; with the risk-target rule, on the new cheese, all 4 (5 % target) and all 5 (10 % target) drops came from such jumps. Safe-set expansion only binds for products the log lacks.

Net value per minute = 60 × (held − c × dropped) / Σ cycle time, all 940 bank items pooled. Uplifts are paired bootstrap intervals over the same items (figure `docs/figures/value_vs_cost.png`).

**A. Realistic commissioning: 10 test picks per speed and product (540 rows) for every policy** (`results/eval_small/`)

| drop cost c | recipe tuned on the log | recipe 1 step slower | recipe 2 steps slower | logistic | boosted trees | **TabPFN** | TabPFN − best recipe (95 % CI) |
|---|---|---|---|---|---|---|---|
| 1 | 121.0 (10.1 % drops) | 145.3 | 142.6 | 149.1 | 128.1 | **149.0** | **+3.8 (+1.6, +5.7)** |
| 3 | 90.3 | 141.8 | 140.8 | 144.2 | 112.0 | **144.7** | +2.9 (−0.8, +6.6) |
| 10 | −16.9 | 129.6 | 134.3 | 136.9 | 53.3 | **133.8** | −0.5 (−9.0, +7.9) |
| 30 | −323 | 94.8 | 115.9 | 113.5 | −68.8 | **116.4** | +0.5 (−18.5, +19.5) |
| 100 | −1,396 | −27.1 | 51.3 | 64.7 | −332 | **80.8** | +29.4 (−2.1, +76.2) |

"Best recipe" is whichever safety margin turned out best at that cost, chosen in hindsight, which favours the recipe.

- **The recipe tuned naively on 10 tests per speed fails:** 10.1 % of picks drop. At 60 % speed the bar showed 0 drops in 10 tests by chance, while its true drop rate there is 30–60 %.
- **An engineer's safety margin fixes that, but the right margin depends on the cost:** one step is best at low cost, two steps at high cost.
- **TabPFN matches or beats the best margin at every cost without anyone choosing one.** The gain is significant at c = 1, and at c = 100 it beats logistic regression significantly: +16.1 (+0.4, +46.8).
- **Logistic regression** on the same features is close to TabPFN at low and medium costs.
- **Boosted trees** are overconfident and lose badly as soon as drops cost anything.

**B. Abundant commissioning: 50 test picks per speed and product (2,700 rows)** (`results/eval/`)

| drop cost c | recipe tuned on the log | logistic | boosted trees | **TabPFN** | TabPFN − recipe (95 % CI) |
|---|---|---|---|---|---|
| 1 | 146.7 | 148.2 | 135.8 | **148.7** | +2.0 (−0.5, +4.3) |
| 3 | 143.2 | 143.6 | 121.9 | **142.7** | −0.5 (−4.6, +3.4) |
| 10 | 138.8 | 133.1 | 99.2 | **135.2** | −3.6 (−10.7, +1.8) |
| 30 | 126.4 | 114.9 | 42.3 | **122.1** | −4.3 (−14.1, +0.8) |
| 100 | 83.0 | 65.4 | −75.5 | **82.7** | −0.3 (−0.7, −0.0) |

With 450 dedicated test picks per product, a recipe tuned on the true drop rates is as good as per-pick choice; no difference is significant except a 0.3 net/min loss at c = 100. TabPFN beats the other learned models at every cost from 10 upward (logistic is ahead by 0.9 at c = 3).

**Why per-pick choice cannot gain more on this line.** Within the product × speed cells where items both hold and drop, TabPFN ranks items well:

| cell | AUC within the cell |
|---|---|
| bar at 60 % | 0.84 |
| cheese at 30 % | 0.81 |
| pouch at 50–80 % | 0.65–0.72 (hidden slosh frequency) |

But a faster setting pays off only for items whose risk there is a few percent or less, and with a 30–60 % base rate few items qualify. Most per-pick risk on this line comes from what no sensor sees: cup–product friction, the vision error that pushes a lip over an edge, and the pouch's slosh frequency. The oracle, which knows them, reaches 164 good picks/min against about 148.

**C. New products** (absent from the log; large log, safe-step, c = 10)

| product | slow default (no recipe yet) | logistic | boosted trees | **TabPFN** | oracle |
|---|---|---|---|---|---|
| bar | 129.5 · 0 % | 147.5 · 5.0 % | 147.7 · 6.2 % | **146.6 · 0 %** | 169.1 |
| pouch | 129.3 · 0 % | 129.3 · 0 % | 146.6 · 5.0 % | 129.3 · 0 % | 174.1 |
| cheese | 125.1 · 2.5 % | 125.8 · 2.5 % | 118.4 · 8.8 % | 122.2 · 5.0 % | 134.8 |

Cells are good picks/min · drop rate.

- **Bar:** TabPFN reaches the trees' throughput, +13 % over the slow default, with no drops.
- **Pouch:** at c = 10 it stays at the default. At c = 1 it climbs to 146.4 good picks/min (2.5 % drops).
- **Cheese:** the product is at its limit even at the slowest setting, and the attempts to go faster cost drops.

The demo clip (`media/demo_new_bar.mp4`) shows the bar stream with the same cost rule, before safe-set expansion was added. That run climbs from 20 % to 50–60 % over 77 picks with no drop.

**D. Growing vs frozen log, risk-target rule** (`results/eval/summary_frozen.csv`)

- **New products:** appending each pick's outcome is what lets TabPFN leave the slow default. On the bar, 129.5 (frozen) becomes 147–153 good picks/min (growing, across targets). On the pouch, 129.3 becomes up to 140.
- **Drift the log already covers** (wear, humidity, batch): frozen and growing perform the same. The commissioning log spans those conditions, and TabPFN reads the cup counter, humidity, product temperature and seal directly.

**E. Other checks**

- **TabPFN-3.5-Fast** performs like 3.5 in the loop: 150.2 vs 150.7 good picks/min at the 5 % target, 2.0 % vs 1.9 % drops. The fast model is the one the JAX port accelerates.
- **Bounded exploration** (ε = 0.1, one step up when the predicted risk is under twice the target) almost never fires: at most 3 picks per stream across all 13 streams. It gains nothing, and on the new bar it costs drops (3.7 % → 6.2 % at the 5 % target). The commissioning log already covers every setting, and safe-set expansion with the cost rule climbs on its own.

### 4. Latency (`scripts/bench_tabpfn.py`, RTX 4050 laptop GPU, PyTorch, real log)

| model, 1,000-row context | refit | one pick (9 settings) | 1,800-row speed map |
|---|---|---|---|
| TabPFN-3.5-Fast | 235 ms | 23 ms (p95 34) | 61 ms |
| TabPFN-3.5 | 462 ms | 52 ms (p95 91) | 98 ms |

The decision has the 35 ms vacuum dwell to happen in: 3.5-Fast fits it in PyTorch. The JAX port of the 3.5-Fast query path (`docs/research/tabpfn-jax-port.md`) answers in 3.7 ms, measured on the earlier toy log with the same column count; it has not been re-measured on this log.

## Summary

- TabPFN is the best probabilistic model of drops at every log size, calibrated from a few hundred picks. It is also the best model of where a held product lands.
- In closed loop, with the small commissioning log a line realistically has, it matches or beats the best hand-set recipe at every drop cost, with no margin to choose. The recipe tuned naively on that log drops 10 % of picks. It also handles a new product without any tuning, though not every new product gains.
- With an abundant, dedicated test campaign (450 picks per product), a perfectly tuned recipe is as good. On this line most per-pick risk is invisible to the sensors, so there is little left for per-pick choice.

