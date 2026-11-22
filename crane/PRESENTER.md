# Crane: presenter's cheat sheet (one of our 4 ideas)

Page to show: https://claude.ai/artifact/7rjNJLFjPqkE2g74ekxL31 · Live demo: `crane/live_pick_place.py --auto-tune`

## The idea in one sentence
**A crane's load keeps swinging after every move; anti-sway fixes that only if someone types in the right sling length. TabPFN learns the swing from the crane's own recent moves and picks the anti-sway setting itself: no training, just new rows.**

## 75-second script (your slot in the 4-idea pitch)
| Time | Say | Show |
|---|---|---|
| 0:00–0:15 | "When an overhead crane stops, the load keeps swinging. The operator waits before setting it down, and a swinging load can hit racks or people." | page section 0 or 1 |
| 0:15–0:30 | "Anti-sway cancels the swing with two half-sized pushes, half a swing apart. But the timing depends on the sling length someone types in. Wrong sling, and the swing comes back." | page section 2, panel 3 |
| 0:30–0:50 | "TabPFN reads a table of the crane's own motion: 12 sensor values every 0.1 s. It predicts where the load will be in 1 second, with a range. Before each move, it predicts the wait for every anti-sway setting and picks the best." | page 3c diagram |
| 0:50–1:15 | "On a real lab crane, 5 seconds of watching a new load cut its error 3×. In a full-size MuJoCo crane, the live pick and place learns the right setting in a few cycles: 48 instead of 68 seconds per cycle, with no retraining." | live window or the 6b table |

## The 5 numbers to remember
1. **3×** lower 1-s forecast error after 5 s of an unseen load (real lab crane, 0.30 → 0.10 cm).
2. **0.8 vs 2.2 cm**: TabPFN vs the physics model over 81 MuJoCo settings; **0.7 vs 4.7 cm** with a hidden long sling.
3. **−35%**: 12 new loads in 261 s instead of 403 s (closed loop).
4. **−29%**: live pick and place, 35 cycles, mean 48.2 s vs about 68 s.
5. **0 retraining**: adaptation = adding rows; about 130 ms per prediction on a laptop GPU.

## Live demo (about 30 s)
1. Start it before your slot (it needs about 15 s to load):
   `! cd ~/priorlab_berlin_hack && set -a && . ./.env && set +a && TABPFN_NO_BROWSER=1 MUJOCO_GL=glfw crane_tmp/.venv/bin/python crane/live_pick_place.py --auto-tune`
2. Point at the **dark blue beam** (real), the **green ghost** (TabPFN, 1 s ahead) and the **orange ghost** (physics with the wrong sling).
3. Point at the top-right: "anti-sway … TabPFN chooses" and "last cycle: … s, wait … s".
4. If it's still early: "the first cycles it tries settings; after that it sticks with 1.5 m."
5. Backup if the window misbehaves: play `crane/results/closed_loop.mp4`, or use the page's playground (5b).

## How it fits "one table, many robots"
Same recipe as the other three ideas: **turn the robot's sensors into rows → TabPFN predicts what happens next, with a range → the robot acts on it, and new rows are the adaptation.** For the crane, "what happens next" is where the swinging load will be, and "acting" is choosing the anti-sway setting.

## Likely judge questions
| Question | Answer |
|---|---|
| Is this real data? | Partly. The forecast is validated on a real lab tower crane (Nazarbayev University dataset, CC BY 4.0). The full-size crane and the closed loop are MuJoCo simulation. We say so on every number. |
| Why not just measure the sling? | Nobody measures it reliably today; it changes with every rigging. We infer it from the swing itself (the sway period), never from a typed-in value. |
| Why TabPFN, not a neural net or physics? | No training loop: a new load is a few new rows. It works on small tables (about 120 moves). It gives calibrated ranges we use as a safety rule. Physics wins only when its pendulum length is right; we show that too. |
| What does "tuning" mean? Fine-tuning? | No. TabPFN's weights never change. We tune **one anti-sway number** (the sling length it assumes) per load, chosen from TabPFN's predictions. |
| What are the inputs? | Forecast: 12 (swing now and its history, trolley position/speed/acceleration, rope length, payload, planned move). Anti-sway choice: 3 (sway period, speed, tuning error). The ablation showed the forecast needs all 12, while the choice is as good with 3. |
| Does it improve safety? | Not shown yet: no strategy collided, so our gain is time. The 99% range is already used as a "stay clear of the racks" rule. |
| What failed? | Timing the set-down alone: TabPFN knew the position, but the load was still moving sideways at 15 cm/s. The swing has to be killed, not just timed. Also, early in the live run it sometimes picks the wrong setting before it has its own rows. |
| Is it fast enough? | About 130 ms per forecast and 2–3 s per anti-sway decision on a GTX 1650. The decision happens while the beam is being lifted, so it costs no cycle time. |
| Next steps? | A magnet or grab release demo, then real industrial crane logs in shadow (advisory) mode. Note: TabPFN-3.5 weights are non-commercial. |

## Words to use (and avoid)
- Say **"in-context"** or **"adapts by adding rows"**, not "trains" or "fine-tunes".
- Say **"in simulation"** for MuJoCo numbers and **"on a lab crane"** for real-data numbers.
- Say **"anti-sway setting"** or **"assumed sling length"**, not "tuning the model".
