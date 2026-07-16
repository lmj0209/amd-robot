#!/usr/bin/env bash
# Fetch Unitree Go2 + Z1(gripper) MJCF and meshes from MuJoCo Menagerie
# (BSD-3-Clause). Meshes (.obj/.stl) are large and gitignored, so they are
# fetched at build time by this script. Record the printed commit in
# assets/manifest.yaml (menagerie_commit) for reproducibility.
#
# Run from anywhere:  bash scripts/fetch_menagerie.sh
set -euo pipefail

MENAGERIE_REF="${MENAGERIE_REF:-main}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${REPO_ROOT}/assets/menagerie"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">> sparse-cloning mujoco_menagerie @ ${MENAGERIE_REF}"
git clone --depth 1 --filter=blob:none --sparse \
  "https://github.com/google-deepmind/mujoco_menagerie.git" "$TMP/menagerie"
git -C "$TMP/menagerie" sparse-checkout set unitree_go2 unitree_z1
git -C "$TMP/menagerie" sparse-checkout reapply
COMMIT="$(git -C "$TMP/menagerie" rev-parse HEAD)"
echo ">> menagerie commit: ${COMMIT}"

rm -rf "${DST}/unitree_go2" "${DST}/unitree_z1"
mkdir -p "${DST}"
cp -r "$TMP/menagerie/unitree_go2" "${DST}/"
cp -r "$TMP/menagerie/unitree_z1" "${DST}/"

echo ">> fetched files:"
( cd "${DST}" && find unitree_go2 unitree_z1 -type f | sort )
echo ""
echo ">> DONE. Record this in assets/manifest.yaml (menagerie_commit): ${COMMIT}"
