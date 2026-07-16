#!/usr/bin/env bash
# Reproducible RGC environment setup for this project.
#
# Base image (RGC):  amd-oneclick-base:rocm7.2.1-py3.12-v20260416
# GPU:               Radeon PRO W7900 (gfx1100 / RDNA3), ~48 GB
# ROCm userspace:    7.2.1 (rocm-core 7.2.1.70201-81~24.04)
#
# Run on a FRESH RGC instance of the base image above, from the repo root:
#     bash scripts/rgc_setup.sh
# Then, before any jax/training command in a new shell:
#     source .venv/bin/activate && export HSA_OVERRIDE_GFX_VERSION=11.0.0
#
# PRINCIPLE: every command that is verified to work on RGC must be added here,
# in order. This file is the single source of truth for the environment; keep
# docker/Dockerfile in sync with it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 1. venv (RGC system Python is PEP 668 externally-managed) ---
python -m venv .venv
.venv/bin/pip install --upgrade pip

# --- 2. gfx1100 / RDNA3 must override the reported GFX version for JAX/HIP ---
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# --- 3. JAX-ROCm matching userspace ROCm 7.2.1 ---
.venv/bin/pip install -U "jax[rocm7-local]"

# --- 4. Verify JAX sees the GPU (Gate G0 core check) ---
.venv/bin/python -c "import jax; print(jax.devices()); print(jax.default_backend())"

# =============================================================================
# After step 4 prints a GPU device, install the rest of the stack.
# Each line below is PENDING verification on RGC; uncomment as it is confirmed
# and move it above this block once verified.
# =============================================================================
# .venv/bin/pip install mujoco brax
# .venv/bin/pip install mujoco-playground        # or install from git per its docs
# .venv/bin/pip install -e . --no-deps           # this project, editable
# .venv/bin/python scripts/smoke_test.py         # Gate G0 fail-closed smoke test
