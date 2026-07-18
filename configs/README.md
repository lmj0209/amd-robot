# Configuration policy

Configuration files are evaluator-facing inputs and must be committed with each
formal run. A run records the config path, SHA256, Git commit, seed, checkpoint,
and RGC system fingerprint.

- `env.yaml`: stable environment and interface contract.
- `train.yaml`: PPO defaults and curriculum stages.
- `standing.yaml`: safe replay defaults for the completed standing
  qualification and ROCm guardrails, including the five-physics-substep
  control-kernel limit measured on gfx1100.
- `evidence/`: immutable input snapshots for measured runs. The July 18
  standing run loaded the archived configuration and overrode the logged
  timestep and episode-length arguments.
- `smoke.yaml`: Gate G0 diagnostic sizes.

Measured values replace `TODO` only after verification on RGC.
