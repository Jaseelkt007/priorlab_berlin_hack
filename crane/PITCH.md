# Pitch kit, crane: slide outline (5 min) and video script (3 min)

Every number here is measured; the source file is given in brackets. Figures are in `crane/figures/` (made by
`make_figures.py`), clips in `crane/results/`. "Lab crane" = real data; "MuJoCo" = simulation. Always say which.
Interactive page: https://claude.ai/artifact/7rjNJLFjPqkE2g74ekxL31 · diagrams: [`EXPLAINED.md`](EXPLAINED.md).

## Slides (6 slides; the deck: https://claude.ai/artifact/4EJUDBdYFHaYS2yYs6DnKQ, speaker notes on every slide)

1. **Title.** "Where will the load be in 1 second? TabPFN tunes an overhead crane's anti-sway for every load: no training, just new rows in a table." Team names. Prior Labs track.
2. **The problem.** The load keeps swinging after every stop; operators wait, swinging loads hit racks or people. 154 of 297 US crane-related work deaths (2011–17) were workers struck by an object or equipment [BLS]. 20+ s of waiting per move when the anti-sway is set for the wrong sling [`results/live_runs.json`]. Visual: `results/still_b.png`.
3. **Why today's fix breaks.** Anti-sway = two half pushes, half a swing apart; the timing comes from a sling length someone types in. Typed 0.5 m, real 2.0 m: 10 cm of swing survives, above the ±5 cm set-down limit. Visual: `figures/antisway.png`.
4. **The idea.** Sensors → one row of 12 numbers every 0.1 s → TabPFN reads the crane's own table (no training) → load position in 1 s with a 90% range, and the best anti-sway setting. 1 s later the truth arrives and the row joins the table.
5. **Does the prediction work?** Prediction error = distance between where a model said the load would be 1 s later and where it really was. Real lab crane, a load it never saw, after 5 s: physics 0.50 cm, standard ML (retrained) 0.37 cm, **TabPFN 0.10 cm, 5× closer** [`results/exp2_real.json`]. Full-size crane, sling longer than typed in: physics 4.7 cm, **TabPFN 0.70 cm, 7× closer** [`results/influence.json`]. Visual: `figures/prediction_simple.png`.
6. **The crane uses it.** Each cycle TabPFN picks the anti-sway setting: wrong at first, right from cycle 5; **47 s per cycle instead of 68**, 7 of 9 placed within ±5 cm [`results/live_runs.json`]; **−35%** over 12 new loads [`results/closed_loop.json`]. Simulation; no collisions, so the gain is time. Next: real crane logs as operator advice. Visual: `figures/live_cycles.png`.

Details for questions (speaker notes, not slides): 130 ms per forecast on a laptop GPU; the anti-sway choice runs while the beam is lifted; timing the set-down alone failed (load still moving at 15 cm/s, `results/setdown_advisor.json`); real data is lab-scale.

## Video (3 minutes; a team member on camera)

| time | on screen | said |
|---|---|---|
| 0:00–0:20 | person on camera, then `results/still_a.png` | "When an overhead crane stops, the load keeps swinging. The operator waits before setting it down, and a swinging load can hit racks or people." |
| 0:20–0:45 | `figures/antisway.png` | "Anti-sway cancels the swing with two half pushes, half a swing apart. But the timing depends on a sling length someone types in. Get it wrong and ten centimetres of swing survive." |
| 0:45–1:10 | page section 0b diagram | "We turn the crane's own sensors into a table: the swing now and a moment ago, the trolley, the rope, the planned move. TabPFN reads it and predicts where the load will be one second later, with a range, and which anti-sway setting gives the shortest safe wait. No training: every move it makes becomes one more row." |
| 1:10–1:40 | `figures/real_crane_period.png`, `figures/real_crane_adaptation.png` | "On a real lab crane the textbook pendulum is wrong in every run. On a payload it has never seen, five seconds of watching cut TabPFN's error three times, without retraining." |
| 1:40–2:30 | live window (or `results/closed_loop.mp4`), then `figures/live_cycles.png` | "Here is a full-size crane in MuJoCo, picking and placing a beam on long slings. Green is TabPFN's prediction, orange the physics model with the typed-in sling. The first cycle uses the wrong setting and waits 24 seconds. Then it learns: from the fifth cycle it keeps the right setting, every placement is within five centimetres, and a cycle takes 47 seconds instead of 68." |
| 2:30–3:00 | `figures/closed_loop.png`, person on camera | "Across twelve new loads the crane is 35% faster than industry anti-sway set for the wrong sling. It runs on a laptop GPU, needs no retraining, and can start as advice to the operator." |
