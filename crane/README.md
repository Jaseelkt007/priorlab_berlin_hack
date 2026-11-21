# Crane track: where will the load be in 1 second?

Part of our Prior Labs entry ("TabPFN for spatial awareness"). Branch: `crane`.

**Idea.** An overhead crane's load swings after every move. Anti-sway control cancels that swing only if it knows the pendulum, and a new load or a wrong sling entry breaks it. TabPFN reads the crane's own recent moves as a table and predicts the swing 1 s ahead with a 90% range. When a new load arrives, its rows are added to the table; nothing is retrained.

- **Explainer page:** `explainer.html`. It's one HTML file: open it in a browser. It has a live 4-strategy crane simulation, the physics, the table, calibration, the real-data results, the MuJoCo demo, the full problem and training spec, and sources.
- **Pitch and video script:** `PITCH.md`.

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
```

`python3 -I crane/exp1_real.py --no-ml` runs only the physics baselines, with the standard library.

## Files

| File | What |
|---|---|
| `real_data.py` | loads the lab-crane runs into 10 Hz rows (12 features, targets +0.5 s / +1.0 s); pendulum baseline |
| `exp1_real.py`, `exp2_real.py` | real-data experiments; results in `results/*.json` |
| `mujoco_crane.py`, `mujoco_demo.py` | MuJoCo overhead crane (rail, trolley, rope, hook, slings, beam), demo and video |
| `data_tools/` | dataset download, the period check (measured vs. textbook pendulum), an xlsx reader |
| `sim-check/` | Node checks of the explainer page's in-browser simulation (`node crane/sim-check/test.js 30`) |
| `DATASETS.md`, `LESSONS.md` | dataset search and citation; the four Prior Labs lessons with links |

## Data and credit

Lab tower-crane dataset: Alhassan, A., Arif, S. M. U., Talapiden, K., Shehu, M. A., Haruna, A., Do, T. (2025). *Trajectories and Payload Oscillations Data of Tower Crane Systems with varied Payload Masses, Shapes, and Cable Lengths for Machine Learning-Based Control, System Identification, and Industrial Automation*. Mendeley Data, V2. https://doi.org/10.17632/x69hyyv3zw.2, licensed CC BY 4.0. It is not stored in this repo: `data_tools/download_dataset.py` fetches it.
