# Reproducibility spec that mirrors scripts/rgc_setup.sh.
#
# NOTE: the base image below lives inside RGC. Build it only where that image is
# accessible (e.g. an RGC instance that has docker, or a CI runner with the
# image). For day-to-day dev on RGC, prefer running scripts/rgc_setup.sh on the
# public base image directly -- no build needed. Keep this file in sync with
# scripts/rgc_setup.sh.
#
# Base image (RGC): amd-oneclick-base:rocm7.2.1-py3.12-v20260416
# GPU:              Radeon PRO W7900 (gfx1100 / RDNA3)
FROM amd-oneclick-base:rocm7.2.1-py3.12-v20260416

# gfx1100 / RDNA3 override (needed whenever JAX/HIP runs) + ROCm libs on path
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace/amd-robo

# System Python on this base is externally-managed -> use a venv
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -U "jax[rocm7-local]"

# TODO(Phase 0): append the verified installs here in the same order as
# scripts/rgc_setup.sh -- mujoco, brax, mujoco-playground, then
# `pip install -e . --no-deps`. Do not add a line until it is verified on RGC.

ENV PATH=/opt/venv/bin:${PATH}

COPY . .

# Default: fail-closed Gate G0 smoke test
CMD ["python", "scripts/smoke_test.py"]
