# Project rules

This is the canonical rule set for human and AI contributors.

## Hard constraints

1. Use one AMD Radeon GPU through ROCm. Do not add CUDA/NVIDIA-specific paths or
   multi-GPU/RCCL requirements.
2. Use MuJoCo MJX with `impl=jax`, MuJoCo Playground-style `MjxEnv`, and Brax
   PPO. Do not replace the stack with Isaac, PureJaxRL, Genesis, or MJX-Warp.
3. Keep the action interface at 19 values: legs `[0:12]`, arm `[12:18]`, and
   gripper `[18:19]`.
4. Termination must include tilt and illegal-contact checks; base height alone
   is not a fall detector.
5. Treat performance numbers as unverified until measured on RGC with a commit,
   config hash, seed, system fingerprint, command, and raw output.
6. Never guess an RGC image, digest, dependency version, asset license, or
   benchmark result. Use `TODO` until verified.
7. Keep evaluator-facing content and code comments in English.
8. Do not commit secrets, raw logs, checkpoints, videos, or unapproved assets.
   Track external artifacts with a URL and SHA256 manifest.

## Engineering contracts

- `src/amd_robo/contracts.py` is the source of truth for action, task phase,
  observation, info, and termination contracts.
- Every environment must support reset/step under JIT and VMAP and store physics
  in `State.data` as `mjx.Data`.
- Every formal run must use committed configs and produce immutable evidence.
- Tests must be added with implementation changes. A skipped ROCm/PPO check can
  never count as a passed gate.

## Git and scope

- `main` must remain reproducible; `dev` is the integration branch.
- Use small feature branches and English conventional commits.
- Do not create filename versions such as `v2`, `old`, or `backup`.
- Prefer one tested vertical slice over many empty modules.
