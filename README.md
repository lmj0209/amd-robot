# ROCm-Accelerated Quadruped Mobile Manipulation with MJX

> **Status: Phase 0 (environment bring-up) — not started.**
> All performance numbers and dependency versions below are **placeholders** until
> verified on Radeon Cloud (RGC). Nothing here is a measured result yet.

## What

A quadruped-with-arm robot learns to **approach a randomized box and push it into a
goal zone** (Approach → Align → Push → Hold). Trained with **Brax PPO** on
**MuJoCo MJX (JAX)** physics, accelerated on a single **AMD Radeon GPU via ROCm**.

## Stack (locked)

| Layer | Choice |
|---|---|
| Physics | MuJoCo MJX, JAX implementation |
| Environment | MuJoCo Playground-style MjxEnv (`State.data = mjx.Data`) |
| RL | Brax `training.agents.ppo` |
| Acceleration | JAX ROCm plugin + XLA, single Radeon GPU |

Excluded by design: Isaac Gym/Lab, hand-rolled PureJxRL PPO, Genesis as primary
fallback, MJX-Warp. Single GPU only (no multi-GPU / RCCL).

## Reproduce

> TODO (fill after Phase 0 / Gate G0):
> - RGC container image name + digest
> - Exact pinned versions (see `requirements.txt`)
> - `python scripts/system_info.py` → hardware/software fingerprint
> - `python scripts/smoke_test.py` → one-command G0 smoke test
> - Train / evaluate / benchmark commands
> - Checkpoint location or download

## Results

> TODO after Gate G3: success rate over ≥100 eval episodes, ROCm benchmark tables
> (sim throughput, PPO throughput, inference p50/p95, JIT cold-start), training
> wall-clock to target success rate. All from RGC, with raw CSV retained.

## License

MIT for project code. Robot/asset licenses are tracked per-component in
`THIRD_PARTY_NOTICES.md`.
