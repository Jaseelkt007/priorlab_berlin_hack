# Plan to submission (written Sat 2026-11-21, ~20:30; deadline Sun 16:30)

**Pitch:** delta robots on packaging lines pick with suction at up to 15 g. Speed against drops is tuned by hand, per product, conservatively. We predict the drop risk of every speed per pick with TabPFN, from the line's own log and the vacuum reading at touch, with no training.

Background: `docs/research/application-delta-picking.md`, the explainer `docs/explainer/delta-pick-line.html`, latency in `docs/research/tabpfn-latency.md`, suction validation in `docs/research/suction-validation.md`.

## Phases

| # | When (target) | What | Done when | Cut if late |
|---|---|---|---|---|
| 1 | Sat 21:00–23:30 | **Delta cell in MuJoCo.** Effector path with IRB-360-class limits (10 m/s, 150 m/s², jerk-limited); conveyor and trays with MuJoCo's official `surfacevel`; suction per the official `active_adhesion` pattern (`gap`) on our validated lip ring; product catalogue (biscuit, pouch, cheese) with hidden friction and CoM; vacuum sensor from lip forces (seal at touch, minimum vacuum during the move) | one product picked and placed; drops appear at high speed for weak seals; log row written with the real columns | trays: place on a flat belt |
| 2 | Sat 23:30–Sun 02:00 | **Pick-log generator.** Thousands of picks across speeds, products, humidity and cup wear, on parallel CPU cores; files in `data/` (gitignored) plus a small committed sample | ≥ 5k rows for each product; drop rate 5–20% across the speed range | fewer products |
| 3 | Sun 02:00–06:00 | **TabPFN in the loop.** Drop classifier plus vacuum-dip regressor, cached context per product. Per-pick query (CUDA graph, about 7 ms) and the speed map. Closed-loop runs of the guided shift (humidity, changeover, wear) against fixed recipes, safe 50%, a GOMP-ST-style learned limit, and an MLP on the same rows | scoreboard: good packs/min, drop rate, picks to recover after a changeover, calibration (reliability diagram, ECE) | drop the GOMP-ST baseline |
| 4 | Sun 06:00–10:00 | **Results and visuals.** Plots, numbers into the explainer page (replacing illustrative ones), MuJoCo-rendered clips of the same moment under the recipe and TabPFN | every number on the page comes from a run | static plots only |
| 5 | Sun 10:00–14:30 | **Submission material.** 3-minute video (pitch, how it works, demo, a person on camera), slides for the 5-minute stage pitch, README with reproduce steps | uploaded, links work without special access | — |
| 6 | Sun 14:30–16:30 | **Buffer and submit** at https://tally.so/r/GexoxQ; make the repo public after a secrets check | submitted before 16:30 | — |

## Parallel work

- **Main Claude session:** phases 1 → 3, in that order.
- **Second Claude session:** JAX port of TabPFN-3.5-Fast (`docs/handoff/jax-port.md`), branch `jax-port`. Optional: it only goes into the demo if it beats the CUDA graph on speed-map compute while matching probabilities.
- **Team (4 people):**
  - one on video and slides from Sunday morning (script and shot list can start tonight from the explainer page);
  - one reviewing results and calibration;
  - one on the README and repo hygiene (secrets check before going public);
  - one floating / pairing on phase 3.
- **Ask Prior Labs mentors tonight:** how the track is judged. Adjust phase 4 emphasis accordingly.

## Rules we keep

- Every number in the pitch comes from a run; illustrative numbers are labelled as such.
- Every shortcut goes into `DEBT.md`.
- No weights, tokens or coupon codes in git.
