# Results

One file per experiment. "Real" = the Nazarbayev University lab tower crane (CC BY 4.0, fetched by
`data_tools/download_dataset.py`, not in git); "MuJoCo" = our simulated overhead crane. All runs one seed unless noted.

| File | What | Data | Produced by |
|---|---|---|---|
| `exp1_real.json` | forecast 0.5 / 1 s ahead on 3 held-out tests (cable length, beam payload, hoisting): textbook and fitted pendulum, gradient-boosted trees, TabPFN | real | `exp1_real.py` |
| `exp2_real.json` | adaptation curve on the unseen beam (0–20 s of its rows); physics + TabPFN residual model | real | `exp2_real.py` |
| `mujoco_demo.json`, `mujoco_demo.mp4`, `still_a.png`, `still_b.png` | held-out load (400 kg beam, 2.6 m slings): TabPFN vs physics 1 s ahead, video and stills | MuJoCo | `mujoco_demo.py` |
| `mujoco_frame.png` | test render of the crane scene | MuJoCo | `mujoco_crane.py --frame` |
| `playground.json`, `fleet_rows.json` | 81-setting grid with precomputed TabPFN predictions (report 4b); the 3,000 fleet rows it used | MuJoCo | `playground_precompute.py` |
| `clip_rope4.5_sling2.5_load0.5t_speed100.mp4` | example clip for one setting | MuJoCo | `render_clip.py` |
| `influence.json` | effect of each slider (rope, sling, payload, speed) and permutation importance of TabPFN's inputs | real + MuJoCo | `influence.py` |
| `ablation.json` | fewer inputs: forecast (12 → 4) and anti-sway choice (5 → 2) | real + MuJoCo | `ablation.py` |
| `closed_loop.json`, `closed_loop.mp4` | TabPFN chooses speed and anti-sway on 12 new loads vs industry anti-sway and slow-and-safe (`closed_loop_run.json`, the replay, is local only) | MuJoCo | `mujoco_closed_loop.py` |
| `setdown_advisor.json` | timing the set-down instead of killing the swing | MuJoCo | `setdown_advisor.py` |
| `live_fleet_rows.json` | 3,000 rows from 40 back-and-forth jobs: context for the live forecast | MuJoCo | `live_crane.py` (built on first run) |
| `pickplace_decision_fleet.json` | 120 commissioning moves: context for the live anti-sway choice | MuJoCo | `live_pick_place.py` (built on first run) |
| `live_runs.json` | live pick-and-place runs (9, 13 and 35 cycles): setting chosen, cycle time, expected vs actual wait, placement | MuJoCo | `live_pick_place.py --auto-tune` |

Figures in `../figures/` are made from these files by `make_figures.py`; README pictures are in `../docs/img/`.
