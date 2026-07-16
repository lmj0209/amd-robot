# CLAUDE.md (repo: amd-robo)

Deliverable repo for AMD Radeon Hackathon 2026-07, Track 3.
**English only** for all evaluator-facing content (README, report, code comments
that matter, commit messages).

## Hard rules (do not violate)
- ROCm-first, **single** Radeon GPU. No NVIDIA/CUDA-specific code. No multi-GPU/RCCL.
- Stack: MuJoCo MJX (JAX) + MuJoCo Playground MjxEnv + Brax PPO.
  **No** Isaac/Lab, **no** PureJaxRL hand-rolled PPO, **no** Genesis as primary
  fallback, **no** MJX-Warp.
- Numbers must be **measured on RGC**. Write "target" until measured. Never
  fabricate versions/image tags — leave `TODO` until confirmed.
- **19-dim action** (12 leg + 6 arm + 1 gripper), frozen for checkpoint
  compatibility. Push MVP may mask the gripper dim but must not change it.
- Termination: never by base height alone — always include tilt + illegal contact.
- Keep the repo minimal: no scratch files, no logs, no large artifacts (see
  `.gitignore`), no Chinese. One experiment per folder under `experiments/`.
- Do not version files by filename (`v2`, `bak_`); use git.

## Branching
- `main`: always reproducible; tag at each gate: `g0-link g1-model g2-mvp
  g3-evidence g4-candidate g5-final`.
- `dev`: integration.
- Feature branches per task: `phase1/model-b2piper`, `phase1/model-go2z1`,
  `phase2/locomotion`, `phase2/push-mvp`, `phase3/benchmark`, `phase3/upstream-pr`.
- Conventional commits, English messages (`feat:` `fix:` `docs:` `bench:` `exp:`).

## Before editing here
Read the workspace planning docs (Chinese, not committed):
`../0_规划/计划.md` (current phase + gates) and
`../0_规划/四足带臂_MJX_ROCM_方案设计.md` (task/reward/curriculum).

## Benchmark discipline
JIT warmup separate from steady-state timing; `block_until_ready()`; inputs on
device before timing; ≥5 repeats per config; record OOM/NaN (not just best);
CPU baseline on the same instance; `amd-smi` raw logs; keep CSV + command +
commit + seed.

## Status
Phase 0 not started. Versions/images are `TODO` — do not hardcode until confirmed
on RGC.
