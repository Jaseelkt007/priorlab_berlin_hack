#!/usr/bin/env bash
# Set up a fresh Linux GPU machine (e.g. a cloud H100) to run the closed-loop experiments.
# Run inside the repo checkout on that machine:   cd fly && bash scripts/setup_gpu_box.sh
# Creates .venv-fly (Python 3.10: flybody + MuJoCo + TF policy + local TabPFN on CUDA), the same
# versions as the team's `flybody` conda env. Data, .env and TabPFN weights are copied separately
# (see README, "Cloud GPU").
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d third_party/flybody ]; then
  git clone https://github.com/TuragaLab/flybody.git third_party/flybody
fi
git -C third_party/flybody checkout -q d015e9bfe441bd90ae431bac24c55cb74bdbce26

uv venv --python 3.10 .venv-fly
# torch first, from the CUDA 12.6 index (works on drivers from CUDA 12.6 up); otherwise tabpfn pulls the default build.
VIRTUAL_ENV=.venv-fly uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=.venv-fly uv pip install -e ./third_party/flybody \
  "tensorflow==2.8.0" "tensorflow-probability==0.16.0" "protobuf==3.20.3" "numpy==1.26.4" \
  "mujoco==3.17.0" "dm_control==1.0.50" "scipy==1.15.3" "scikit-learn==1.7.2" "h5py" \
  "imageio" "imageio-ffmpeg" "python-dotenv" "tabpfn==10.2.0"

MUJOCO_GL=egl .venv-fly/bin/python - <<'EOF'
import torch, tabpfn, mujoco, tensorflow as tf
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("tabpfn", tabpfn.__version__, "mujoco", mujoco.__version__, "tf", tf.__version__)
EOF
