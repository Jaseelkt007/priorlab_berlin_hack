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
