# Reproduction record

## Gate G0 fields

> Values are filled only from measured RGC output. Unavailable provider fields
> are identified explicitly instead of being guessed.

- RGC provider instance: ephemeral identifier intentionally omitted
- RGC image: **`amd-oneclick-base:rocm7.2.1-py3.12-v20260416`** — chosen; the
  competition does not mandate a specific image
- RGC image digest: **unavailable from inside the instance**; recorded as null
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
  (`google-deepmind/mujoco_playground@43d180a226da3aae091d918b63c06c3a343519ad`);
  version 0.2.0, `CartpoleBalance` loads with `impl=jax`, reset/step finite
- Verified lockfile: `requirements/rgc.lock`, captured from the fixed RGC venv
  on 2026-07-21
- `system_info` evidence: captured at commit `18bb8a6` on 2026-07-20;
  image name and digest remain null because RGC does not expose them inside
  the instance.
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
  any Brax training. This project-local compatibility shim is separate from
  the public Brax pmap-axis contribution in
  https://github.com/google/brax/pull/674, which is awaiting maintainer review.

## Commands

The exact 36 redistributable Go2/Z1 meshes required by the assembled model are
already tracked. To audit or refresh them from the pinned upstream commit:

```bash
bash scripts/fetch_menagerie.sh
```

This optional script defaults to Menagerie commit
`71f066ad0be9cd271f7ed58c030243ef157af9f4`, uses sparse checkout, and fails if
a requested full commit resolves differently. A clean checkout can load the
model without running it. Exact install, smoke, training, evaluation, and
benchmark commands below are retained only after measured use.

## Formal ROCm benchmark

The measured Push scan uses `scripts/rocm_benchmark.py`. The worker places
parameters on the target device before timing, separates reset compilation
from the cold target-kernel call, warms up ten steps, synchronizes every
timing boundary, and rejects an unexpected backend or non-finite result.

Example combined point:

```bash
/workspace/.venv/bin/python scripts/rocm_benchmark.py \
  --mode combined \
  --config configs/push_stage2_near_field_solver16_qualification.yaml \
  --params-in /path/to/push_nearfield_v3_params \
  --batch-size 2048 \
  --steps-per-repeat 100 \
  --warmup-steps 10 \
  --repeats 5 \
  --latency-samples 50 \
  --expected-backend gpu
```

For a same-instance CPU point, use the identical command with
`JAX_PLATFORMS=cpu`, batch one, and `--expected-backend cpu`. Run each point
in a fresh process and sample `amd-smi metric` concurrently for GPU points.

Validate and combine raw GPU and CPU directories:

```bash
python scripts/aggregate_rocm_benchmark.py \
  --input-dir /path/to/gpu-raw \
  --input-dir /path/to/cpu-raw \
  --output-csv benchmark_summary.csv \
  --output-json benchmark_summary.json \
  --expected-commit 18bb8a6462ba510a55825723b4ccb4a0e3f3743c
```

The 2026-07-20 scan completed all 21 formal points with exit code zero and
finite results. Combined Push throughput was `49.812`, `2,334.474`,
`5,297.430`, `7,156.298`, `7,404.157`, and `7,632.187 transitions/s` at GPU
batches 1, 64, 256, 1,024, 2,048, and 4,096. Same-instance CPU batch-one
combined throughput was `38.804 transitions/s`. Batch 4,096 cold compilation
was `358.407 s`, so batch 1,024–2,048 is preferred for iteration despite the
slightly higher 4,096 steady-state result.

## Push determinism audit

Before changing the Push controller or training a randomized task, audit the
accepted inference parameters with the same solver-16 configuration and seed.
The audit starts every policy run from one immutable reset, interleaves trained
and zero-residual policies, and writes compressed step traces plus a fail-closed
JSON manifest.

Run a short contract check before the full episode:

```bash
/workspace/.venv/bin/python scripts/locomotion_learn_smoke.py \
  --task push \
  --config configs/push_stage2_near_field_solver16_qualification.yaml \
  --eval-only \
  --params-in /path/to/push_nearfield_v3_params \
  --eval-num-envs 1 \
  --eval-num-steps 20 \
  --determinism-repeats 2 \
  --determinism-audit-dir /workspace/evidence/push-audit-smoke
```

The qualification path is deliberately single-environment. On the measured
gfx1100 stack, a 16-environment Push audit changed discrete outcomes between
identical interleaved runs, while single-environment GPU runs and a 16-env CPU
control were stable. The first formal process records all 4,608 control steps:

```bash
/workspace/.venv/bin/python scripts/locomotion_learn_smoke.py \
  --task push \
  --config configs/push_stage2_near_field_solver16_qualification.yaml \
  --eval-only \
  --params-in /path/to/push_nearfield_v3_params \
  --eval-num-envs 1 \
  --eval-num-steps 4608 \
  --determinism-repeats 3 \
  --determinism-audit-dir /workspace/evidence/push-audit-a
```

Run a fresh process against that immutable reference:

```bash
/workspace/.venv/bin/python scripts/locomotion_learn_smoke.py \
  --task push \
  --config configs/push_stage2_near_field_solver16_qualification.yaml \
  --eval-only \
  --params-in /path/to/push_nearfield_v3_params \
  --eval-num-envs 1 \
  --eval-num-steps 4608 \
  --determinism-repeats 3 \
  --determinism-reference-dir /workspace/evidence/push-audit-a \
  --determinism-audit-dir /workspace/evidence/push-audit-b
```

Each output directory is single-use. The manifest binds the commit, config and
parameter SHA256, seed, device, policy order, initial-state fingerprint,
per-environment discrete outcomes, trace digests, and first exact/1e-6
divergence. Do not open randomized training while identical inputs produce
order-dependent success, terminal, abnormal, or maximum-phase classifications.

Batched Push execution is retained only to reproduce and diagnose the gfx1100
boundary. A determinism audit is already explicit diagnostic intent. Any other
batched Push evaluation must add `--allow-batched-push-eval`, prints
`qualification_evidence=false`, and must not be reported as qualification data.
Regular Push evaluation fails closed unless `--eval-num-envs 1` is selected.

## Chunked single-environment Push matrix

The bounded JIT evaluator keeps the qualification semantics at one environment
while compiling two control steps at a time. It is a separate implementation
from the sequential audit above and must first reproduce its discrete outcome
and safety classifications. A single cross-validation episode is:

```bash
export JAX_COMPILATION_CACHE_DIR=/workspace/evidence/jax-push-qualification-cache

/workspace/.venv/bin/python scripts/locomotion_learn_smoke.py \
  --task push \
  --config configs/push_stage2_near_field_solver16_position_x_2mm_diagnostic.yaml \
  --eval-only \
  --params-in /path/to/push_nearfield_v3_params \
  --eval-policy trained \
  --eval-num-envs 1 \
  --eval-num-steps 4608 \
  --eval-seed 777 \
  --eval-implementation chunked_jit \
  --eval-chunk-steps 2 \
  --eval-output-dir /workspace/evidence/push-chunked-cross-validation-777
```

The output directory must not exist. The manifest binds the commit, config and
parameter hashes, seed, JAX backend, the single ROCm device, every registered
safety gate, and the final task decision. A persistent compilation cache only
avoids repeating the measured multi-minute gfx1100 compilation; it does not
change the seed, state shape, device count, or qualification gates.

After cross-validation, predeclare a held-out seed range and run each episode
in a fresh process. This example reserves seeds `2026072200..2026072299`, which
do not overlap the training seed or development seeds 777/778:

```bash
/workspace/.venv/bin/python scripts/push_qualification_matrix.py \
  --config configs/push_stage2_near_field_solver16_position_x_2mm_diagnostic.yaml \
  --params-in /path/to/push_nearfield_v3_params \
  --output-dir /workspace/evidence/push-2mm-heldout-100 \
  --seed-start 2026072200 \
  --seed-count 100 \
  --num-steps 4608 \
  --chunk-steps 2
```

The launcher never overwrites an attempt. It validates each per-seed exit code
and manifest before aggregation, reports all gate failures, and exits nonzero
unless the complete matrix passes. If the launcher is interrupted, repeat the
same command with `--resume`; the immutable specification must still match the
commit, inputs, seed range, and execution shape exactly.

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
