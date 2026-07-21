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
ENV LLVM_PATH=/opt/rocm/llvm
ENV HIP_DEVICE_LIB_PATH=/opt/rocm-7.2.1/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
ENV XLA_FLAGS="--xla_gpu_enable_command_buffer="
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace/amd-robo

# System Python on this base is externally-managed -> use a venv
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -U "jax[rocm7-local]==0.10.2" \
 && /opt/venv/bin/pip install "mujoco==3.10.0" "brax==0.14.2"

COPY patches/brax-0.14.2-training-state.patch /tmp/
RUN BRAX_SITE_PACKAGES="$( \
      /opt/venv/bin/python -c \
      'import pathlib, brax; print(pathlib.Path(brax.__file__).resolve().parents[1])' \
    )" \
 && patch --directory "$BRAX_SITE_PACKAGES" --strip 0 \
      < /tmp/brax-0.14.2-training-state.patch \
 && /opt/venv/bin/pip install \
      git+https://github.com/google-deepmind/mujoco_playground.git@43d180a226da3aae091d918b63c06c3a343519ad

ENV PATH=/opt/venv/bin:${PATH}

COPY . .
RUN /opt/venv/bin/pip install -e . --no-deps

# Default: fail-closed Gate G0 smoke test
CMD ["python", "scripts/smoke_test.py"]
