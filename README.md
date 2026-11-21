# priorlab_berlin_hack

HackLab #2 Berlin, Prior Labs track: **TabPFN for spatial awareness**. The brief is in [`docs/HACKATHON.md`](docs/HACKATHON.md).

## Team setup

1. Clone the repo and add your TabPFN key:
   ```sh
   git clone https://github.com/Jaseelkt007/priorlab_berlin_hack.git
   cd priorlab_berlin_hack
   echo "TABPFN_TOKEN=<your key from platform.priorlabs.ai>" > .env   # git ignores .env
   ```
2. Install the shared Claude Code skills once per machine, so everyone gets the same `/potato-start` and `/potato-help` and the version reminder:
   ```sh
   git clone https://github.com/Jaseelkt007/potato-skills ~/potato-skills
   sh ~/potato-skills/install.sh
   ```
   Optional: the pstack skills it mentions, via `npx skills add backnotprop/pstack -g`.
3. Open Claude Code in this folder. The coding rules are already in `CLAUDE.md` (hackathon mode) and load automatically. Running `/potato-start hackathon` again is safe and changes nothing. **Never run it with another mode**: the block is shared, so a different mode changes the rules for the whole team.

## Fly flight data (flybody)

Everything lives inside this repo; `third_party/` and `data/` are gitignored.

```sh
git clone https://github.com/TuragaLab/flybody.git third_party/flybody
git -C third_party/flybody checkout d015e9bfe441bd90ae431bac24c55cb74bdbce26
conda create -y -n flybody -c conda-forge python=3.10 pip
conda run -n flybody pip install -e ./third_party/flybody "tensorflow==2.8.0" "tensorflow-probability==0.16.0" "protobuf==3.20.*" imageio imageio-ffmpeg
```

Download `trained-fly-policies.zip` and `datasets_flight-imitation.zip` **in a browser** from https://doi.org/10.25378/janelia.25309105 (figshare blocks command-line downloads), unzip both into `data/`, then:

```sh
MUJOCO_GL=egl conda run -n flybody python collect/preview_flight.py --traj 0 --out flight_preview.mp4
```

Watch it live in the MuJoCo viewer (orange fly = policy, ghost + cyan dots = the real fly's recorded flight):

```sh
conda run -n flybody python collect/view_flight.py --traj 0
```

Collect the TabPFN data (plan and column layout: [`docs/data/collection-plan.md`](docs/data/collection-plan.md)); output goes to `data/flights/`:

```sh
for p in 1 2 3 4 5 6; do MUJOCO_GL=egl conda run -n flybody python collect/collect_flights.py --pass-id $p; done
```

## TabPFN experiments (uv)

The experiments only read `data/flights/*.npz`, so they use a light uv env (Python 3.12), separate from the flybody conda env:

```sh
pip install uv          # once
uv sync                 # cloud client: needs TABPFN_TOKEN in .env
uv sync --extra local   # GPU machines: adds the local `tabpfn` package (3.5 / 3.5 Fast)
uv run python <script>
```

## Experiments

Data plan and columns: [`docs/data/collection-plan.md`](docs/data/collection-plan.md). All experiments hold out the same 27 flight pairs. A–C call the TabPFN API (priced first, results cached in `results/<exp>/cache`, so reruns are free); D runs TabPFN 3.5 Fast locally on a GPU inside the flybody env.

| | What | Run |
|---|---|---|
| A | Where will the fly be in 1/3/10 wingbeats (quantiles vs baselines) | `uv run python experiments/exp_a_forward.py` |
| B | Adapt to a weak wing / heavier body by adding rows to the context | `uv run python experiments/exp_b_adapt.py` |
| C | Copy the flight policy offline, accuracy vs context size | `uv run python experiments/exp_c_clone.py` |
| D | TabPFN flies the fly (closed loop) | `MUJOCO_GL=egl conda run -n flybody python collect/closed_loop.py --context-steps 1500 --flights 3 --video` |

D needs `torch` (CUDA) and `tabpfn` in the flybody env (`pip install torch --index-url https://download.pytorch.org/whl/cu126 "tabpfn>=10,<11"`) and the TabPFN 3.5 licence accepted at https://ux.priorlabs.ai/accept-license?hf_repo_id=tabpfn_3_5. A 3 GB GPU fits only a 256-step context; use the 6 GB RTX 4050 for 1,000+.

## Cloud GPU (big contexts)

For contexts beyond what a laptop GPU holds (we use a Nebius H100, 80 GB), set up the machine with one script. It builds `.venv-fly` with the same versions as the `flybody` conda env:

```sh
# from your machine: code, data, key and TabPFN weights (data/ and .env are not in git)
rsync -a --exclude data/ --exclude third_party/ --exclude .venv/ --exclude '*.mp4' ./ <host>:priorlab_berlin_hack/
rsync -a data/flights data/policies data/flight-imitation <host>:priorlab_berlin_hack/data/
rsync -a ~/.cache/tabpfn/tabpfn-v3.5-fast-*.safetensors <host>:.cache/tabpfn/
# on the GPU machine
cd priorlab_berlin_hack && bash scripts/setup_gpu_box.sh
MUJOCO_GL=egl .venv-fly/bin/python collect/synthetic_paths.py --controller tabpfn --context-steps 4096 --paths curve --no-video
```
