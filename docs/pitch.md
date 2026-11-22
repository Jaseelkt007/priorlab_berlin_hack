# Pitch kit: slide outline (5 min) and video script (3 min)

Every number here is measured; the source file is given in brackets. Figures are in `docs/figures/`, clips in `media/`.

## Slides (5 minutes, about 8 slides)

1. **Title.** "How fast can this pick be? TabPFN sets the speed of every suction pick." Team names. Prior Labs track: TabPFN for spatial awareness.
2. **The problem.** A delta robot (IRB 360 class) picks food and pharma products with a suction cup at up to 150 m/s². Every product runs at one cautious speed recipe, set by hand at commissioning. Too slow for most picks, still too fast for the bad ones (worn cup, wet cheese, sloshing pouch, a new product).
   Visual: `media/pick_bar.mp4`, the same bar placed at 40 % and flung off at 100 %.
3. **The idea.** The line already logs what a speed decision needs: mass, size, temperature, humidity, cup cycles, vacuum at touch. TabPFN reads that log as in-context data and predicts, for every pick, the drop risk at each of 9 speed settings and where the product will land. Then the robot takes the best setting. No training; each outcome is appended to the log.
   Visual: the decision diagram in the README.
4. **A simulation we can trust.** Five validation gates against vendor data:
   - Piab evacuation tables within 3 %;
   - ABB cycle times, held-out −2.5 %;
   - ABB tracking accuracy;
   - Schmalz sizing rule (S* = 1.94).

   Real cups: Schmalz SPF, SPB1 and SPG bell. Physics we did not put in by hand: pouch slosh makes 60 % riskier than 70–80 % for 4–5 Hz contents.
   Visual: `docs/figures/drop_heatmap.png`.
5. **Results 1: few rows, calibrated.**
   - With 250 logged picks, TabPFN predicts drops as well as logistic regression with 2,700.
   - Calibration error 1.1 % at 2,700 rows, against 1.8 % (logistic) and 4.0 % (boosted trees).
   - Where the product lands: median error 1.18 mm, best of the three models [evaluation.md §1–2].

   Visual: `docs/figures/learning_curve.png`.
6. **Results 2: closed loop, same objective for everyone** (good picks − c × drops per minute) [evaluation.md §3].
   - **Small commissioning log** (10 test picks per speed and product), the realistic case:
     - the recipe tuned on it drops 10 % of picks;
     - TabPFN matches or beats the best hand-set safety margin at every drop cost, with no margin to choose (c = 1: +3.8 net picks/min, 95 % CI +1.6 to +5.7).
   - **New product:** the bar runs at 146.6 good picks/min with 0 drops, against 129.5 for the slow default (+13 %). Logistic regression and boosted trees reach the same speed with 5–6 % drops.
   - **Honest limit:** with 450 dedicated test picks per product, a perfectly tuned recipe is as good, because most per-pick risk here is invisible to sensors.

   Visual: `docs/figures/value_vs_cost.png`.
7. **Demo: a product it has never seen.** On a bar absent from the log, TabPFN starts at 20 %, climbs to 50–60 % as its own picks arrive, and drops nothing.
   Visual: `media/demo_new_bar.mp4`.
8. **Fast enough for the loop, and what's next.**
   - One query for all 9 speeds: 23 ms with TabPFN-3.5-Fast on a laptop GPU, inside the 35 ms vacuum dwell. The JAX port answers in 3.7 ms.
   - Next: a real cell, measured seal-leak curves, the rectangular bar cup.

## Video (3 minutes; a team member on camera)

| time | on screen | said |
|---|---|---|
| 0:00–0:25 | person on camera, then `media/pick_bar.mp4` | "Delta robots pick a hundred products a minute with a suction cup. Run too fast and products fly off; too slow and the line loses output. Today one speed per product is set by hand." |
| 0:25–0:55 | README diagram, heatmap | "The line already logs everything needed to choose better: weight, temperature, cup wear, the vacuum at touch. TabPFN reads that log as context and predicts the drop risk at every speed, for every pick, with no training." |
| 0:55–1:30 | gates table, `media/pick_praline.mp4` | "We built the line in MuJoCo from vendor datasheets and checked every part: vacuum against Piab, the robot against ABB's cycle times, the drop boundary against Schmalz's sizing rule." |
| 1:30–2:30 | `media/demo_new_bar.mp4`, then `value_vs_cost.png` | "Here it meets a product it has never seen. It starts slow, and each pick it makes is appended to its log, so it climbs to the product's limit without a single drop. With the small commissioning log a line really has, it matches or beats the best hand-tuned recipe at every cost of a drop, without anyone tuning it." |
| 2:30–3:00 | learning curve, person on camera | "250 logged picks are enough, the probabilities are calibrated, and a decision takes 23 milliseconds. The data is already on every line." |
