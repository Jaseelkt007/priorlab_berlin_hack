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
3. Simulation setup:
   ```sh
   ./scripts/fetch_assets.sh                 # official MuJoCo Menagerie Franka model, pinned commit
   uv venv --python 3.12 && uv pip install -r pyproject.toml
   .venv/bin/python scripts/spike_dynamic.py # one grid of real pick-and-place episodes
   ```
4. Open Claude Code in this folder. The coding rules are already in `CLAUDE.md` (hackathon mode) and load automatically. Running `/potato-start hackathon` again is safe and changes nothing. **Never run it with another mode**: the block is shared, so a different mode changes the rules for the whole team.
