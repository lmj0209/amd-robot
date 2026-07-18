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
- `locomotion_stage1_forward_extension.yaml`: a short, reward-compatible
  continuation of the faster v13 session at its original `0.2-0.6 m/s`
  command range. It tests whether more on-objective training improves the
  measured `0.4 m/s` gait before committing a larger budget.
- `locomotion_stage1_action_scale_qualification.yaml`: a fresh-policy
  qualification that changes the leg residual scale from `0.25` to the
  measured-safe `0.3` while reducing leg Kp from `50` to `40`, keeping the
  approximate peak residual PD force unchanged. The official Go1 value `0.5`
  and an intermediate `0.35` failed this robot's random-action finite-state
  gate.
- `evidence/`: immutable input snapshots for measured runs. The July 18
  standing and locomotion runs retain the exact configurations identified by
  their logged SHA256 values.
- `smoke.yaml`: Gate G0 diagnostic sizes.

Measured values replace `TODO` only after verification on RGC.
