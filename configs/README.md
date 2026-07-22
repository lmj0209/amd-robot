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
- `push_stage2_near_field_solver16_governor_qualification.yaml`: rejected O2
  diagnostic that filters the executed Push command between `0.10` and
  `0.20 m/s` object speed. It passed measured speed limits but changed trained
  success across fresh processes, so it is not a continuation source.
- `push_stage2_near_field_solver16_governor025_qualification.yaml`: rejected
  single-variable diagnostic that changes only the governor stop speed to
  `0.25 m/s`. It did not reach Hold and is not a continuation source.
- `push_stage2_near_field_solver16_position_x_2mm_diagnostic.yaml`: frozen-v3
  evaluation-only probe that changes only the initial box x-offset range to
  `[-0.002, 0.002]` metres. Run one environment per fresh process with
  `scripts/push_qualification_matrix.py`; it is not qualification evidence
  until the predeclared seed sweep is completed, passes every gate, and is
  archived.
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
- `locomotion_stage1_tracking_qualification.yaml`: a fresh-policy
  single-variable qualification that raises only the linear tracking scale
  from `2.0` to `3.0` while retaining the Kp40/scale0.30 safety contract. The
  measured v33 run remained finite but increased tilt without establishing a
  normal gait, so its policy and session are not continuation sources.
- `locomotion_stage1_network_probe.yaml`: a 4,096-step compile and finite-update
  gate for the upstream Go1/ATEC-sized `[512, 256, 128]` policy and value
  networks. It retains the 256-environment, five-substep, host-scan-two ROCm
  contract and does not qualify a locomotion policy.
- `locomotion_stage1_network_qualification.yaml`: the 524,288-step fresh-policy
  qualification promoted after the large-network compile gate passed. It
  changes only network capacity from the Kp40/scale0.30 v28 qualification and
  retains full-session checkpoints and per-foot sequential evaluation.
- `locomotion_stage1_network_extension.yaml`: a same-objective continuation
  from the complete large-network qualification session to 5,242,880 total
  steps. It changes only the additional training budget and retains
  per-update KL/standard-deviation stop gates.
- `locomotion_stage1_adaptive_kl_qualification.yaml`: a fresh 524,288-step
  qualification after the fixed-rate network extension crossed its KL stop
  gate. It changes only the Brax learning-rate schedule from `NONE` to
  `ADAPTIVE_KL`, matching the stability mechanism used by the ATEC RSL-RL
  velocity baseline while preserving the ROCm host-loop contract.
- `locomotion_stage1_low_lr_qualification.yaml`: a fresh 524,288-step
  qualification that returns to the fixed-rate optimizer and changes only the
  learning rate from `3e-5` to `1e-5`. It tests whether controlled exploration
  can be retained after the fixed-rate extension collapsed and the adaptive
  qualification produced unsafe, uncoordinated foot motion.
- `locomotion_stage1_low_lr_extension.yaml`: a same-objective continuation
  from the complete low-rate qualification to 5,242,880 total steps. It
  changes only the additional training budget and retains the full-session
  checkpoints plus KL, policy-standard-deviation, finite-state, and per-foot
  gait acceptance gates.
- `locomotion_stage1_trot_qualification.yaml`: a fresh 524,288-step
  qualification after the large-network low-rate extension still converged
  to continuous four-foot contact. It retains the low-rate safety settings,
  adds an observed 0.5-second gait phase, and alternates diagonal FL+RR and
  FR+RL contact/swing targets. The frozen action interface remains 19-D.
- `locomotion_stage1_trot_timing_qualification.yaml`: replaces the rejected
  instantaneous contact-match reward with a continuous contact/air-time
  consistency score. It synchronizes FL+RR and FR+RL internally while
  rewarding opposition across the two diagonal groups. All PPO, control, and
  ROCm guardrail settings remain identical to the first trot qualification.
- `locomotion_stage1_trot_dwell_qualification.yaml`: closes the high-frequency
  contact-chatter loophole observed in the timing qualification. The timing
  score ramps from zero to full value only after one synchronized diagonal
  swing pair remains airborne for 0.1 seconds.
- `locomotion_stage1_trot_dwell_extension.yaml`: restores the complete dwell
  qualification session and adds 1,572,864 transitions, reaching 2,097,152
  total steps without changing the environment, reward, PPO, or ROCm
  guardrail contract.
- `locomotion_stage1_crawl_reference_qualification.yaml`: replaces the closed
  pure-trot-reward line with a measured four-beat FL-RR-FR-RL joint reference.
  The reference shifts load into the three-foot support triangle before each
  swing, while the policy retains the frozen 19-D interface and learns only a
  residual. Zero command disables the reference. CPU MuJoCo qualification
  bounds the reference at a 4.0-second cycle, 0.08-radian stride,
  0.06-radian body shift, and 0.45-radian knee lift.
  A 0.07-second sustained-air gate prevents contact-solver flicker from being
  counted or rewarded as a completed swing.
- `locomotion_stage1_crawl_residual_qualification.yaml`: fresh-policy safety
  qualification after the first crawl-residual policy became unsafe. It
  changes only the policy residual range from +/-0.3 to +/-0.1 radians (plus
  the run label), leaving the reference, command, reward, PPO, evaluation, and
  ROCm contracts unchanged.
- `locomotion_stage1_crawl_low_speed_qualification.yaml`: fresh-policy crawl
  curriculum after the safe residual policy still saturated against the
  0.4 m/s evaluation command and suppressed the front-foot swings. It changes
  only the sampled command range from 0.2-0.6 to 0.02-0.06 m/s and the fixed
  evaluation command from 0.4 to 0.04 m/s (plus the run label).
- `locomotion_stage1_crawl_tracking_qualification.yaml`: fresh low-speed
  qualification after the broad 0.25 tracking kernel gave almost identical
  reward at zero and 0.04 m/s. It changes only `tracking_sigma` to 0.0025
  (plus the run label), so the low-speed policy receives a measurable forward
  velocity gradient without changing its control authority.
- `evidence/`: immutable input snapshots for measured runs. The July 18
  standing and locomotion runs retain the exact configurations identified by
  their logged SHA256 values.
- `smoke.yaml`: Gate G0 diagnostic sizes.

Measured values replace `TODO` only after verification on RGC.
