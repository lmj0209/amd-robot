# Reproduction record

## Gate G0 fields

> Values are filled only from measured RGC output. `TODO` = not yet captured.

- RGC instance: `u-7670-4289a66a` (2026-07-16 probe)
- RGC image: **`amd-oneclick-base:rocm7.2.1-py3.12-v20260416`** — chosen; the
  competition does not mandate a specific image
- RGC image digest: TODO
- GPU: **gfx1100 (RDNA3)**, marketing name "AMD Radeon Graphics", Chip ID 0x744b,
  96 CUs, ~48 GB VRAM → **Radeon PRO W7900** — measured via `rocminfo`
- ROCm kernel module (ROCk): **6.16.13**; HSA runtime 1.18 — measured
- ROCm userspace release: **7.2.1** (`rocm-core 7.2.1.70201-81~24.04`) — measured →
  install `jax[rocm7-local]`; RDNA3 needs `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- CPU: 2× AMD EPYC 9334 (32-core), large RAM — measured
- Python: **3.12.3 (GCC 13.3.0)** — measured
- JAX/JAXLIB: **jax 0.10.2 / jaxlib 0.10.2** (in `.venv`) — measured
- JAX ROCm PJRT/plugin: **jax-rocm7-plugin 0.10.2 / jax-rocm7-pjrt 0.10.2** — measured
- Verified install (2026-07-16): `python -m venv .venv && source .venv/bin/activate &&
  export HSA_OVERRIDE_GFX_VERSION=11.0.0 && pip install -U "jax[rocm7-local]"`
  (pip resolved via tsinghua mirror). `jax.devices()` → `[RocmDevice(id=0)]`,
  `default_backend()` → `gpu` ✅ — JAX-ROCm works on gfx1100
- MuJoCo/Brax: **mujoco 3.10.0 / mujoco-mjx 3.10.0 / brax 0.14.2** (+ flax 0.12.7,
  optax 0.2.8, orbax-checkpoint 0.12.1) — measured; MJX `impl=jax` runs finite on GPU.
  The `Failed to import warp/mujoco_warp` notices are harmless — the Warp backend is
  absent on purpose; we use the JAX backend.
- MuJoCo Playground: installed from git
  (`pip install git+https://github.com/google-deepmind/mujoco_playground.git`);
  `CartpoleBalance` loads with `impl=jax`, reset/step finite — TODO pin exact commit
  in `requirements/rgc.lock`
- Verified lockfile: TODO (`requirements/rgc.lock`)
- `system_info` evidence: TODO
- G0 commit and tag: **`g0-link`** (2026-07-16, on `main`)

## Findings (2026-07-16)

- The RGC base workspace has **no** `jax`, `torch`, `mujoco`, or `brax`
  preinstalled (`pip list` was empty for all of them). The full stack —
  including the JAX ROCm PJRT/plugin for gfx1100 — must be installed by us on
  top of the ROCm runtime.
- `rocminfo` / `amd-smi` work and see the GPU, so the ROCm runtime is functional.
- `groups: cannot find name for group ID 109` is a harmless cosmetic container
  warning and does not affect GPU access.
- **G0 PASSED (2026-07-16)** — `scripts/smoke_test.py` exits 0 on RGC (W7900):
  JAX-ROCm GPU (`rocm:0`, "PJRT C API rocm 70200"), MJX `impl=jax` (256 envs,
  ~97k env-steps/s diagnostic), MuJoCo Playground `CartpoleBalance` (playground
  0.2.0), and a Brax PPO update with checkpoint save+reload. **Required shim:**
  brax calls `jax.device_put_replicated`, removed in jax 0.10.2 — we apply the
  official drop-in (`src/amd_robo/platform/_compat.py`, single-GPU safe) before
  any Brax training. This shim is the basis for a Brax upstream PR (10-pt item).

## Commands

The exact install, smoke, training, evaluation, and benchmark commands are added
only after they run successfully from a clean RGC environment.

## Standing qualification and exact resume

The gfx1100-safe standing run consumes the committed configuration, bounds every
compiled Brax training scan to two steps, and performs evaluation with a
sequential Python loop. The assembled robot must be loaded through
`assets/menagerie/go2_z1/scene_mjx.xml`: the robot-only XML has no floor.
Never replace MuJoCo's compiled `qpos0` with the home keyframe; joint coordinates
are defined relative to that reference. On this gfx1100 stack the measured
control-kernel limit is five physics substeps (`ctrl_dt=0.01`); ten substeps
failed during CompileAndLoad with `ROCM_ERROR_ILLEGAL_ADDRESS`.

The validated short GPU qualification used:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LLVM_PATH=/opt/rocm/llvm
export HIP_DEVICE_LIB_PATH=/opt/rocm-7.2.1/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/workspace/.venv/bin/python scripts/standing_learn_smoke.py \
  --config configs/evidence/standing_ground_2026-07-18.yaml \
  --num-timesteps 5120 \
  --episode-length 128 \
  --skip-eval \
  --params-out /workspace/evidence/standing_ground_gpu_5120.params

/workspace/.venv/bin/python scripts/standing_learn_smoke.py \
  --config configs/evidence/standing_ground_2026-07-18.yaml \
  --eval-only \
  --params-in /workspace/evidence/standing_ground_gpu_5120.params
```

The eight-repeat evaluation accepted the zero-residual home-PD baseline
(25,600 control transitions, zero done/non-finite/height-outlier events) and
rejected the short trained parameters because reward and tilt were worse. A
long standing run is therefore not part of the current plan; Stage 1 proceeds
to command-conditioned locomotion.

For any future long standing diagnostic, save the complete training session:

```bash
/workspace/.venv/bin/python scripts/standing_learn_smoke.py \
  --config configs/standing.yaml \
  --training-state-dir /workspace/checkpoints/standing
```

Resume by pointing at one immutable step directory.  `--num-timesteps` is the
number of additional environment steps in the resumed process:

```bash
/workspace/.venv/bin/python scripts/standing_learn_smoke.py \
  --config configs/standing.yaml \
  --num-timesteps 5120 \
  --resume-training-state \
    /workspace/checkpoints/standing/step_000000005120 \
  --training-state-dir /workspace/checkpoints/standing-resumed
```

These checkpoints contain the complete Brax learner state, MJX rollout state,
and learner/environment PRNG keys.  Restoring learner parameters and optimizer
state without rollout and PRNG state is not a supported resume mode.
