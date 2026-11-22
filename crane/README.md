# Crane track: where will the load be in 1 second?

Part of our Prior Labs entry ("TabPFN for spatial awareness"). Branch: `crane`.

**Idea.** An overhead crane's load swings after every move. Anti-sway control cancels that swing only if it knows the pendulum, and a new load or a wrong sling entry breaks it. TabPFN reads the crane's own recent moves as a table and predicts the swing 1 s ahead with a 90% range. When a new load arrives, its rows are added to the table; nothing is retrained.

- **Explainer page:** `explainer.html`. It's one HTML file: open it in a browser. It has a live 4-strategy crane simulation, the physics, the table, calibration, the real-data results, the MuJoCo demo, the full problem and training spec, and sources.
- **Pitch and video script:** `PITCH.md`.

## Present the live demo from another laptop

Everything the demo needs is in this folder: the code, the cached tables TabPFN reads (`results/live_fleet_rows.json`, `results/pickplace_decision_fleet.json`) and launchers. Only your own TabPFN key stays out of git.

1. **Get the code:** `git clone https://github.com/Jaseelkt007/priorlab_berlin_hack.git && cd priorlab_berlin_hack && git checkout crane`
2. **Install uv** (once): https://docs.astral.sh/uv/getting-started/installation/
3. **TabPFN key:** log in at https://platform.priorlabs.ai, accept the **TabPFN-3.5** licence, copy your API key, and create `.env` in the repo root containing one line, `TABPFN_TOKEN=yourkey` (no quotes, no `< >`). `.env` is git-ignored.
4. **Start the demo.** The first run creates `crane/.venv`, installs the packages (needs internet, a few minutes) and downloads the TabPFN weights once.
   - Linux / macOS: `crane/run_live.sh`
   - Windows (PowerShell, in the repo folder): `powershell -ExecutionPolicy Bypass -File crane\run_live.ps1`
5. **Pre-flight check** (after the first start, before presenting): `crane/.venv/bin/python crane/check_setup.py` (Windows: `crane\.venv\Scripts\python.exe crane\check_setup.py`). Every line should say OK.

Options: `run_live.sh predict` = live forecast only; extra flags are passed through, e.g. `run_live.sh pick --stopped` (start idle, Enter to begin) or `--manual` (one cycle per Enter).

**Before you go on stage:** start the window 3–5 minutes early and let it run. The first cycles are TabPFN trying settings; from about cycle 5 it keeps the right one, and the cycle table (bottom right) shows the learning. An NVIDIA GPU makes the green forecast ghost smooth. Without one it runs on CPU and the ghost lags; everything else works. On macOS the launcher uses `mjpython`, which MuJoCo's viewer requires.

**In the window:** dark blue = real beam; green = TabPFN's predicted position in 1 s; orange = physics model with the typed-in 0.5 m sling. Top left: mode, state, current cycle timer. Top right: both forecasts, the last 1-s check, the anti-sway setting and what's expected for this move (TabPFN prediction, or measured if already tried on this load). Bottom right: every cycle's time, expected → actual wait, placement, OK/MISS.

## Results

**Real lab crane** (Nazarbayev University tower crane, see Data). Mean error at the payload 1 s ahead, on whole runs held out from the context:

| Held out | Fitted pendulum | Trees | TabPFN | Physics + TabPFN |
|---|---|---|---|---|
| New cable length (0.5 m) | 0.34 cm | 1.30 | 1.52 | **0.19** |
| New payload (beam) | 0.57 | 0.48 | **0.35** | 0.36 |
| Hoisting during travel | 0.82 | 1.20 | 0.81 | **0.78** |

**Adaptation on the unseen beam:** after 5 s of the beam's own rows, TabPFN's 1-s error drops from 0.30 to 0.10 cm (3×). Its 90% range goes from holding 32% of outcomes to 91%. Trees retrained on the same rows reach 0.37 cm; the refitted pendulum 0.50 cm. Physics alone wins at +0.5 s when only the rope length changes.

**MuJoCo, full-size crane, in simulation:** a held-out 400 kg beam on 2.6 m slings, with the anti-sway set for 0.5 m:

| | TabPFN | Physics model |
|---|---|---|
| First move | 3.5 cm | 7.0 cm |
| Later moves | **2.6 cm** (90% range holds 90%) | 10.9 cm |

Video: `results/mujoco_demo.mp4`.

**Closed loop, MuJoCo, in simulation:** before each move, the crane asks TabPFN (one batched call, 7 speeds × 4 anti-sway tunings) for the predicted swing and settle time, and takes the fastest move whose 99th-percentile swing clears the racks. Its real outcome becomes the next row. On 12 new loads on long slings:

| Strategy | 12 moves | Moves / hour | Mean settle wait | Collisions |
|---|---|---|---|---|
| Industry anti-sway (set for 0.5 m sling) | 403 s | 107 | 19.9 s | 0 |
| Slow and safe (40%) | 333 s | 130 | 1.5 s | 0 |
| **TabPFN decides** | **261 s** | **166** | 7.4 s | 0 |

Video: `results/closed_loop.mp4`. TabPFN still misjudged 3 of the 12 moves (long settles). No strategy collided, so the gain shown is time, not safety.

**Playground** (page section 6c): 81 precomputed MuJoCo runs (rope × sling × payload × speed). TabPFN's 1-s error averages 0.8 cm vs 2.2 cm for physics; its 90% range holds 80%. To render any other combination: `render_clip.py --rope R --sling S --payload T --speed F`.

**Caveats:**
- The lab crane is about 10× smaller than an industrial one, with one run per condition and one seed.
- The beam run has a single trolley move at the start.
- The MuJoCo demo was rendered offline (7–8 s TabPFN fits on a GTX 1650).

## Run it

```sh
uv venv --python 3.12 crane/.venv
uv pip install --python crane/.venv/bin/python -r crane/requirements.txt
python3 -I crane/data_tools/download_dataset.py           # lab-crane data, 11 MB, CC BY 4.0
set -a; . ./.env; set +a; export TABPFN_NO_BROWSER=1       # TABPFN_TOKEN; accept the TabPFN-3.5 licence first
crane/.venv/bin/python crane/exp1_real.py                 # 3 held-out tests, 4 predictors (~2 min)
crane/.venv/bin/python crane/exp2_real.py                 # adaptation curve + physics+TabPFN (~5 min)
MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_crane.py --check --frame
MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_demo.py # fleet data, held-out job, TabPFN, video (~5 min)
MUJOCO_GL=egl crane/.venv/bin/python crane/mujoco_closed_loop.py   # closed loop, 3 strategies x 12 loads + video (~10 min)
MUJOCO_GL=egl crane/.venv/bin/python crane/playground_precompute.py # page playground data + fleet rows (~2 min)
MUJOCO_GL=egl crane/.venv/bin/python crane/render_clip.py --rope 5.2 --sling 2.8 --payload 0.7 --speed 0.9
MUJOCO_GL=glfw crane/.venv/bin/python crane/live_crane.py      # LIVE 3D window: TabPFN predicts 1 s ahead in real time
MUJOCO_GL=glfw crane/.venv/bin/python crane/live_pick_place.py # LIVE pick and place with anti-sway tuning keys
MUJOCO_GL=glfw crane/.venv/bin/python crane/live_pick_place.py --manual   # one cycle per Enter; --stopped starts idle
MUJOCO_GL=glfw crane/.venv/bin/python crane/live_pick_place.py --auto-tune # TabPFN chooses the anti-sway setting (key 3)
```

`python3 -I crane/exp1_real.py --no-ml` runs only the physics baselines, with the standard library.

## Files

| File | What |
|---|---|
| `real_data.py` | loads the lab-crane runs into 10 Hz rows (12 features, targets +0.5 s / +1.0 s); pendulum baseline |
| `exp1_real.py`, `exp2_real.py` | real-data experiments; results in `results/*.json` |
| `mujoco_crane.py`, `mujoco_demo.py` | MuJoCo overhead crane (rail, trolley, rope, hook, slings, beam), demo and video |
| `mujoco_closed_loop.py` | closed loop: TabPFN chooses speed and anti-sway tuning per move, vs. industry anti-sway |
| `live_pick_place.py` | **live pick and place**: hoist lowers, grabs the beam (weld), lifts, travels, settles, places it in the other bay, releases, repeats; Enter start/stop, 1 auto / 2 manual; ←/→ change the anti-sway's assumed sling (wrong 0.5 m: 69.5 s cycle; right 2.0 m: 47 s) |
| `live_crane.py` | **live** MuJoCo window: the crane shuttles between bays, TabPFN's 1-s-ahead ghost updates in real time; change rope/sling/payload/speed with keys (Up/Down, PgUp/PgDn, Home/End, Ins/Del) or by typing `rope 5` in the terminal |
| `playground_precompute.py`, `render_clip.py` | page playground (option A, 81 precomputed runs) and any-parameter MP4 clip (option B) |
| `data_tools/` | dataset download, the period check (measured vs. textbook pendulum), an xlsx reader |
| `sim-check/` | Node checks of the explainer page's in-browser simulation (`node crane/sim-check/test.js 30`) |
| `DATASETS.md`, `LESSONS.md` | dataset search and citation; the four Prior Labs lessons with links |

## Data and credit

Lab tower-crane dataset: Alhassan, A., Arif, S. M. U., Talapiden, K., Shehu, M. A., Haruna, A., Do, T. (2025). *Trajectories and Payload Oscillations Data of Tower Crane Systems with varied Payload Masses, Shapes, and Cable Lengths for Machine Learning-Based Control, System Identification, and Industrial Automation*. Mendeley Data, V2. https://doi.org/10.17632/x69hyyv3zw.2, licensed CC BY 4.0. It is not stored in this repo: `data_tools/download_dataset.py` fetches it.
