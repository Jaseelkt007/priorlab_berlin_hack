#!/usr/bin/env bash
# Every video in drone/results/videos/ plus drone/results/world.gif and drone/results/compare.gif, as listed in drone/results/README.md.
#
#   drone/make_videos.sh            # from the repo root, in the priorlab env (or a venv with requirements)
#
# Needs drone/data/ with the saved predictions (drone/README.md, "Reproduce every result").
# Renders offscreen: on a headless GPU server run it with
#   MUJOCO_GL=egl drone/make_videos.sh
# About 5 minutes on a GPU (the videos render in parallel); much slower on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
V=drone/results/videos
mkdir -p "$V/1_world" "$V/2_pilotA_flight54" "$V/3_newpilotB_flight78"
video() { $PY drone/pitch_video.py "$@" | tail -2; }

# 1. the world: building, red zone, intruder, the X2 defender hovering
( $PY drone/intruder.py --drone cf2 --view 54 --record "$V/1_world/world_scene.mp4" &&
  $PY -c "import sys; sys.path.insert(0, 'drone'); from pitch_video import to_gif; to_gif('$V/1_world/world_scene.mp4', 'drone/results/world.gif', 0, 14)" ) &

# 2. pilot a (the pilot the examples came from), test flight 54: one model per video
video --pilot a --flight 54 --model imm         --out "$V/2_pilotA_flight54/1_IMM.mp4" &
video --pilot a --flight 54 --model imm_retuned --out "$V/2_pilotA_flight54/2_IMM-retuned.mp4" &
video --pilot a --flight 54 --model tabpfn_a    --out "$V/2_pilotA_flight54/3_TabPFN-A-bigdrone-only.mp4" &
video --pilot a --flight 54 --model tabpfn      --out "$V/2_pilotA_flight54/4_TabPFN-B.mp4" &
wait

# 3. pilot b (no model has seen this pilot), flight 78: every model, the live version, then the best three together
video --pilot b --flight 78 --model imm         --out "$V/3_newpilotB_flight78/1_IMM.mp4" &
video --pilot b --flight 78 --model imm_retuned --out "$V/3_newpilotB_flight78/2_IMM-retuned.mp4" &
video --pilot b --flight 78 --model tabpfn_a    --out "$V/3_newpilotB_flight78/3_TabPFN-A-bigdrone-only.mp4" &
video --pilot b --flight 78 --model tabpfn      --out "$V/3_newpilotB_flight78/4_TabPFN-B.mp4" &
video --pilot b --flight 78 --model combo       --out "$V/3_newpilotB_flight78/5_IMM-plus-TabPFN.mp4" &
video --pilot b --flight 78 --model live        --out "$V/3_newpilotB_flight78/6_LIVE-IMM-plus-TabPFN-every-0.3s.mp4" &
video --pilot b --flight 78 --model imm,tabpfn,combo \
      --out "$V/3_newpilotB_flight78/7_COMPARE-IMM-TabPFN-IMMplusTabPFN.mp4" --gif drone/results/compare.gif &
wait
echo "done: $(find "$V" -name '*.mp4' | wc -l) videos in $V; GIFs: drone/results/world.gif, drone/results/compare.gif"
