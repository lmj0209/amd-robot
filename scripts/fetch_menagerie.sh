#!/usr/bin/env bash
# Fetch Unitree Go2 + Z1(gripper) MJCF and meshes from MuJoCo Menagerie
# (BSD-3-Clause). The complete pinned source directories are gitignored and
# fetched at build time by this script. The default commit must match
# assets/manifest.yaml (menagerie_commit).
#
# Run from anywhere:  bash scripts/fetch_menagerie.sh
set -euo pipefail

DEFAULT_MENAGERIE_REF="71f066ad0be9cd271f7ed58c030243ef157af9f4"
MENAGERIE_REF="${MENAGERIE_REF:-${DEFAULT_MENAGERIE_REF}}"
MENAGERIE_REPO="${MENAGERIE_REPO:-https://github.com/google-deepmind/mujoco_menagerie.git}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${REPO_ROOT}/assets/menagerie"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Some minimal ROCm images ship a valid system CA bundle without teaching Git
# where to find it. Keep TLS verification enabled and scope the fallback to the
# network command below; never mutate the user's global Git configuration.
GIT_CA_ARGS=()
if [[ -z "$(git config --get http.sslCAInfo || true)" ]]; then
  for ca_file in \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/cert.pem; do
    if [[ -f "$ca_file" ]]; then
      GIT_CA_ARGS=(-c "http.sslCAInfo=$ca_file")
      break
    fi
  done
fi

echo ">> sparse-cloning mujoco_menagerie @ ${MENAGERIE_REF}"
git init --quiet "$TMP/menagerie"
git -C "$TMP/menagerie" remote add origin "$MENAGERIE_REPO"
git -C "$TMP/menagerie" config remote.origin.promisor true
git -C "$TMP/menagerie" config remote.origin.partialclonefilter blob:none
git -C "$TMP/menagerie" sparse-checkout init --cone
git -C "$TMP/menagerie" sparse-checkout set unitree_go2 unitree_z1
git "${GIT_CA_ARGS[@]}" -C "$TMP/menagerie" fetch \
  --depth 1 --filter=blob:none \
  origin "$MENAGERIE_REF"
git -C "$TMP/menagerie" checkout --quiet --detach FETCH_HEAD
COMMIT="$(git -C "$TMP/menagerie" rev-parse HEAD)"
echo ">> menagerie commit: ${COMMIT}"

if [[ "$MENAGERIE_REF" =~ ^[0-9a-f]{40}$ && "$COMMIT" != "$MENAGERIE_REF" ]]; then
  echo ">> ERROR: resolved commit ${COMMIT} does not match ${MENAGERIE_REF}" >&2
  exit 1
fi

rm -rf "${DST}/unitree_go2" "${DST}/unitree_z1"
mkdir -p "${DST}"
cp -r "$TMP/menagerie/unitree_go2" "${DST}/"
cp -r "$TMP/menagerie/unitree_z1" "${DST}/"

echo ">> fetched files:"
( cd "${DST}" && find unitree_go2 unitree_z1 -type f | sort )
echo ""
echo ">> DONE. Record this in assets/manifest.yaml (menagerie_commit): ${COMMIT}"
