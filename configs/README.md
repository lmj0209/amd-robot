# Configuration policy

Configuration files are evaluator-facing inputs and must be committed with each
formal run. A run records the config path, SHA256, Git commit, seed, checkpoint,
and RGC system fingerprint.

- `env.yaml`: stable environment and interface contract.
- `train.yaml`: PPO defaults and curriculum stages.
- `standing.yaml`: safe replay defaults for the completed standing
  qualification and ROCm guardrails, including the five-physics-substep
  control-kernel limit measured on gfx1100.
- `locomotion.yaml`: current Phase-2 forward-velocity curriculum. The arm and
  gripper remain masked at their home controls while the policy keeps the
  frozen 19-dimensional action interface.
- `locomotion_stage1_low_speed.yaml`: the first `0.1 m/s` qualification slice
  of the staged `0.1 -> 0.2 -> 0.4 m/s` command curriculum. It resumes the
  complete accepted v14 training session.
- `locomotion_stage1_low_speed_sensitive.yaml`: the second `0.1 m/s`
  qualification. It narrows the bounded tracking kernel after the first slice
  showed that standing still already saturated the broad tracking reward.
- `evidence/`: immutable input snapshots for measured runs. The July 18
  standing and locomotion runs retain the exact configurations identified by
  their logged SHA256 values.
- `smoke.yaml`: Gate G0 diagnostic sizes.

Measured values replace `TODO` only after verification on RGC.
