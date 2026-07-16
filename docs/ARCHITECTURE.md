# Architecture

The project has four boundaries:

1. MuJoCo/MJX owns models, dynamics, contacts, actuators, and sensors.
2. The environment owns reset, randomization, observations, phases, rewards,
   termination, and metrics.
3. Brax owns PPO rollout and optimization. It does not own physics state.
4. Platform utilities own ROCm validation, system fingerprints, and benchmark
   synchronization.

The stable task sequence is Approach, Align, Push, and Hold. Policy actions are
always 19-dimensional. `State.info` is initialized with every required key during
reset. Critic-only information must not leak into policy observations.

Robot-specific implementations are added only after the model gate. Until then,
the platform smoke uses an official Playground environment and a minimal MJX
contact model; neither is presented as the project task.
