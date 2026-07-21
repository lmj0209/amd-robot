# ROCm-Accelerated Quadruped Mobile Manipulation with MJX

> **Status:** Gates G0 and G1 have passed on one Radeon PRO W7900 with ROCm.
> The fixed-near-field Push MVP has a reproducible successful rollout and
> stable discrete outcomes in isolated single-environment fresh-process
> audits. Batched gfx1100 Push execution is diagnostic-only; see
> [Measured Push result](#measured-push-result).

**Author:** limengjin — solo developer.

## Project

A quadruped-with-arm robot learns to approach a box and push it into a goal
zone: **Approach → Align → Push → Hold**. The locked stack is MuJoCo MJX with
its JAX implementation, a MuJoCo Playground-style environment, Brax PPO, and
one AMD Radeon GPU through ROCm.

The current validated demonstration uses a fixed near-field box. A `±0.01 m`
randomized-box candidate was rejected after an independent repeat, so the
repository does not claim randomized-task qualification.

The action contract is always 19-dimensional: 12 leg joints, 6 arm joints, and
1 coupled gripper action. The Push MVP may mask the gripper but never changes
the policy interface.

## Repository status

- `src/amd_robo/contracts.py` freezes cross-stage interfaces.
- `configs/` contains versioned run inputs.
- `tests/` runs CPU-safe contract tests.
- `scripts/smoke_test.py` is a fail-closed RGC Gate G0 check.
- `scripts/system_info.py` creates immutable benchmark fingerprints.
- `scripts/rocm_benchmark.py` separates reset JIT, cold target-kernel JIT, and
  synchronized steady-state timing for policy, environment, and combined
  execution.
- `scripts/aggregate_rocm_benchmark.py` validates exit codes, finite results,
  commit identity, and `amd-smi` telemetry before producing CSV/JSON evidence.
- `scripts/push_rollout_render.py` renders one deterministic Push rollout and
  records every frame, phase transition, metric, seed, and artifact hash.
- `assets/manifest.yaml` and `artifacts/manifest.json` track external files.

The selected robot is Go2 + Z1/gripper. No ATEC, Unitree B2, or AgileX Piper
asset is included in this repository.

## Local contract tests

These tests do not prove ROCm support:

```bash
python -m pip install pytest pyyaml
python -m pip install -e . --no-deps
python -m pytest
```

All 36 Go2/Z1 meshes referenced by the assembled model are included, so a
clean checkout does not need a network fetch to load the robot. The optional
`scripts/fetch_menagerie.sh` refresh script defaults to the commit pinned in
`assets/manifest.yaml` and fails if a requested 40-character commit resolves
differently. Retained license copies live in `assets/licenses/`.

## Gate G0 on RGC

Do not install a generic JAX build over the RGC runtime. Follow
`requirements/README.md`, install the AMD PJRT/plugin/JAX/JAXLIB combination for
the exact RGC image and GPU, then install the project without replacing it.

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export LLVM_PATH=/opt/rocm/llvm
export HIP_DEVICE_LIB_PATH=/opt/rocm-7.2.1/lib/llvm/lib/clang/22/lib/amdgcn/bitcode
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/workspace/.venv/bin/python scripts/system_info.py \
  --image-name amd-oneclick-base:rocm7.2.1-py3.12-v20260416 \
  --config configs/smoke.yaml
/workspace/.venv/bin/python scripts/smoke_test.py
```

RGC did not expose the immutable base-image digest inside this instance, so
the evidence records that field as `null` instead of inventing a value.

The smoke command returns 0 only if all of these pass:

1. JAX positively identifies an AMD ROCm GPU and `amd-smi` works.
2. MJX-JAX runs a finite single and 256-environment batched rollout.
3. `CartpoleBalance` loads through MuJoCo Playground with `impl=jax`.
4. Brax PPO completes an update and saves and reloads a checkpoint.

`--skip-ppo` is diagnostic only and exits 2; it never passes Gate G0.

## Measured Push result

The current deterministic trained policy was evaluated on RGC in two fresh
processes. Each process ran three exact-repeat audits with one environment,
`seed=777`, 4,608 control steps, and solver iterations 16:

| Metric | Process A | Fresh process B |
|---|---:|---:|
| Trained task success | `3/3` | `3/3` |
| Baseline task success | `0/3` | `0/3` |
| Trained maximum object speed | `0.441433 m/s` | `0.383896 m/s` |
| Baseline maximum object speed | `0.351248 m/s` | `0.277368 m/s` |
| Trained final goal distance | `0.079316 m` | `0.074502 m` |

Within each process, repeated traces were exact. Continuous trajectories
diverged across fresh processes near first box motion, but success, terminal,
phase, and abnormal outcomes matched. A legacy 20-environment run reported
`20/20` success and one speed exceedance; it is no longer qualification
evidence because repeated gfx1100 batched execution changed both trajectories
and discrete outcomes. Regular Push evaluation now fails closed unless
`--eval-num-envs 1` is selected.

The first object-speed governor experiment reduced measured peaks to
`0.223634–0.251639 m/s`, but a fresh-process A/B pair changed from success to
a Push-phase failure. It is retained as a rejected diagnostic and is not the
MVP controller. The frozen candidate remains fixed-v3 without the governor.

## Reproduce one unedited rollout

The policy parameter file is external to Git. The validated parameter SHA-256
is:

```text
f033f9ff1ff23304b05ddeec9346d4f681ec5f4dbe265c5000804317916f9d0b
```

Render seed index 0 from the same 20-key pool:

```bash
export MUJOCO_GL=egl

/workspace/.venv/bin/python scripts/push_rollout_render.py \
  --config configs/push_stage2_near_field_solver16_qualification.yaml \
  --params-in /path/to/push_nearfield_v3_params \
  --output-dir /tmp/push-rollout \
  --seed 777 \
  --seed-pool-size 20 \
  --seed-index 0 \
  --num-steps 4608 \
  --render-stride 5 \
  --width 960 \
  --height 540
```

The command refuses to overwrite an existing output directory. It writes JPEG
frames plus `manifest.json`. At a 0.01 s control timestep and stride 5, encode
the complete frame sequence at 20 fps:

```bash
ffmpeg -framerate 20 \
  -i /tmp/push-rollout/frame_%06d.jpg \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  push-rollout.mp4
```

The measured rollout completed at step 4,060 with final goal error
`0.079138 m`, maximum object speed `0.305667 m/s`, zero illegal/non-finite
events, and 813 frames (`40.65 s`).

## Measured ROCm performance

The formal scan used one W7900, ROCm 7.2.1, JAX 0.10.2, commit `18bb8a6`,
the solver-16 Push environment, at least ten warmup steps, and five
synchronized repeats of 100 steps per point. All 21 GPU/CPU points exited
zero and reported finite results.

| Backend | Batch | Environment steps/s | Policy + environment steps/s |
|---|---:|---:|---:|
| CPU | 1 | `40.160` | `38.804` |
| GPU | 1 | `51.177` | `49.812` |
| GPU | 64 | `2,679.273` | `2,334.474` |
| GPU | 256 | `5,262.366` | `5,297.430` |
| GPU | 1,024 | `7,186.548` | `7,156.298` |
| GPU | 2,048 | `7,301.583` | `7,404.157` |
| GPU | 4,096 | `7,533.651` | `7,632.187` |

Batch 4,096 has the highest measured throughput, but it improves combined
throughput only 3.08% over batch 2,048 while cold target-kernel compilation
rises from `102.777 s` to `358.407 s` and peak sampled VRAM rises from
`9,339 MB` to `18,567 MB`. Batch 1,024–2,048 is the practical iteration
range. Pure deterministic policy throughput scales from `8,231` inferences/s
at GPU batch one to `8.20M` inferences/s at batch 4,096; the same-instance CPU
is faster for the unbatched policy call.

The selected Push fixed-v3 training run processed 73,728 transitions in
`345.761 s`: `213.234 transitions/s` including cold compilation and
`383.168 SPS` in the final reused host call. It used `training_scan=1`;
large fused PPO scans remain outside the validated gfx1100 boundary.

Run one formal point:

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

Aggregate separate GPU and CPU raw directories:

```bash
python scripts/aggregate_rocm_benchmark.py \
  --input-dir /path/to/gpu-raw \
  --input-dir /path/to/cpu-raw \
  --output-csv benchmark_summary.csv \
  --output-json benchmark_summary.json \
  --expected-commit 18bb8a6462ba510a55825723b4ccb4a0e3f3743c
```

## Reproduction and results

Environment locks, train/evaluate/benchmark commands, checkpoints, videos, and
additional measured results are recorded in `docs/REPRODUCTION.md`.

## Submission materials

The English [technical report](report/TECHNICAL_REPORT.md) and
[four-minute video script](report/VIDEO_SCRIPT.md) bind every submission claim
to the measured rollout and benchmark evidence. The review-video renderer
checks the decoded output frame count so the complete 813-frame rollout cannot
be shortened silently.

## Upstream contribution

The public [Brax PR #674](https://github.com/google/brax/pull/674) makes PPO's
initial `NamedSharding` use the same `i` mesh axis as its training `pmap`.
The stored [patch](patches/brax-main-pmap-axis.patch) and
[validation record](patches/brax-main-pmap-axis-pr.md) include the upstream
base, submitted commit, CPU test, and single-W7900 ROCm results. The Google CLA
and automated change check passed after the PR was opened on July 21, 2026. It
is awaiting maintainer review and is not claimed as accepted or merged.

## License

Project code is MIT licensed. Third-party software, robot models, and assets are
tracked in `THIRD_PARTY_NOTICES.md` and `assets/manifest.yaml`.
