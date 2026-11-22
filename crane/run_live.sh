#!/usr/bin/env bash
# Live crane demo on Linux or macOS. Usage (from anywhere):
#   crane/run_live.sh            # live pick and place, TabPFN chooses the anti-sway (the demo)
#   crane/run_live.sh predict    # live forecast only (crane shuttles, no picking)
#   crane/run_live.sh pick --stopped --manual     # extra options are passed through
# Needs: uv (https://docs.astral.sh/uv/getting-started/installation/), internet on first run, ./.env with TABPFN_TOKEN.
set -e
cd "$(dirname "$0")/.."
APP="${1:-pick}"; [ $# -gt 0 ] && shift
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
[ -d crane/.venv ] || uv venv --python 3.12 crane/.venv
uv pip install --python crane/.venv/bin/python -q -r crane/requirements.txt
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -n "${TABPFN_TOKEN:-}" ] || { echo "Put TABPFN_TOKEN=<your key> in ./.env (platform.priorlabs.ai, accept the TabPFN-3.5 licence)"; exit 1; }
export TABPFN_NO_BROWSER=1
PY=crane/.venv/bin/python
if [ "$(uname)" = "Darwin" ]; then PY=crane/.venv/bin/mjpython; else export MUJOCO_GL=glfw; fi   # macOS needs mjpython for the viewer
if [ "$APP" = "predict" ]; then exec "$PY" crane/live_crane.py "$@"; else exec "$PY" crane/live_pick_place.py --auto-tune "$@"; fi
