#!/bin/sh
# Fetch the official Franka MJCF models from MuJoCo Menagerie (Apache-2.0), pinned to one commit.
set -e
cd "$(dirname "$0")/.."
REPO=https://github.com/google-deepmind/mujoco_menagerie.git
DEST=third_party/mujoco_menagerie
COMMIT=${MENAGERIE_COMMIT:-370c43965323c0944b433b202aac5058b5f907a9}
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout --depth 1 "$REPO" "$DEST"
  git -C "$DEST" sparse-checkout set franka_emika_panda franka_fr3 LICENSE
fi
if [ -n "$COMMIT" ]; then git -C "$DEST" fetch --depth 1 origin "$COMMIT" && git -C "$DEST" checkout -q "$COMMIT"; else git -C "$DEST" checkout -q; fi
echo "menagerie at $(git -C "$DEST" rev-parse --short HEAD)"
ls "$DEST"
