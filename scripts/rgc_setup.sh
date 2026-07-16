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

# --- 3. JAX-ROCm matching userspace ROCm 7.2.1 (verified: jax 0.10.2) ---
.venv/bin/pip install -U "jax[rocm7-local]"

# --- 4. Verify JAX sees the GPU (Gate G0 core check) ---
.venv/bin/python -c "import jax; print(jax.devices()); print(jax.default_backend())"
# expect: [RocmDevice(id=0)] / gpu

# --- 5. MuJoCo + Brax (verified: mujoco 3.10.0, brax 0.14.2) ---
.venv/bin/pip install mujoco brax

# --- 6. MJX-on-GPU check (Gate G0 sim check) ---
.venv/bin/python - <<'PY'
import jax, jax.numpy as jnp, mujoco
from mujoco import mjx
xml = '<mujoco><option timestep="0.005"/><worldbody><geom type="plane" size="2 2 0.1"/><body pos="0 0 0.3"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body></worldbody></mujoco>'
m = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(m, impl="jax"); d0 = mjx.make_data(m, impl="jax")
d = jax.jit(lambda d: mjx.step(mx, d))(d0); d.qpos.block_until_ready()
print("MJX impl:", mx.impl, "| finite:", bool(jnp.isfinite(d.qpos).all()))
PY

# --- 7. MuJoCo Playground (NOT on PyPI; install from git) ---
.venv/bin/pip install git+https://github.com/google-deepmind/mujoco_playground.git

# --- 8. Playground load check (Gate G0) ---
.venv/bin/python - <<'PY'
import jax, jax.numpy as jnp
from mujoco_playground import registry
env = registry.load("CartpoleBalance", config=registry.get_default_config("CartpoleBalance"), config_overrides={"impl": "jax"})
s = jax.jit(env.reset)(jax.random.PRNGKey(0)); a = jnp.zeros((env.action_size,), jnp.float32)
ns = jax.jit(env.step)(s, a); ns.data.qpos.block_until_ready()
print("playground finite:", bool(jnp.isfinite(ns.data.qpos).all()), "| impl:", getattr(env.mjx_model, "impl", "n/a"))
PY

# --- NOTE: brax + jax 0.10.2 requires a jax.device_put_replicated shim.    ---
# --- It is applied automatically by src/amd_robo/platform/_compat.py, which ---
# --- smoke_test.py and our training code import. No manual action needed.   ---

# =============================================================================
# PENDING verification on RGC. Uncomment as each is confirmed.
# =============================================================================
# .venv/bin/pip install -e . --no-deps           # this project, editable
# .venv/bin/python scripts/smoke_test.py         # Gate G0 fail-closed smoke test (Brax PPO + checkpoint roundtrip)
