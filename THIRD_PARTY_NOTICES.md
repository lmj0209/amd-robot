# Third-Party Notices

This project uses third-party software and assets. Sources, licenses, and any
modifications are listed below. **Update this file whenever a new dependency or
asset is introduced.** Do not ship any asset whose license/redistribution terms
are unconfirmed.

## Software

| Component | Version | License | Source | Modification |
|---|---|---|---|---|
| MuJoCo (incl. MJX) | TODO | Apache-2.0 | https://github.com/google-deepmind/mujoco | none |
| Brax | TODO | Apache-2.0 | https://github.com/google/brax | none |
| MuJoCo Playground | TODO | Apache-2.0 | https://github.com/google-deepmind/mujoco_playground | none |
| JAX | TODO | Apache-2.0 | https://github.com/google/jax | none |
| JAX ROCm PJRT | TODO | TODO_VERIFY | AMD repository for the selected RGC ROCm release | none |
| JAX ROCm plugin | TODO | TODO_VERIFY | AMD repository for the selected RGC ROCm release | none |

> Pin the complete compatible set in `requirements/rgc.lock` after Phase 0
> verification on RGC.

## Robot / Asset Models

> Robot chosen at the model gate (2026-07-20): **B2Piper** (primary) OR
> **Unitree Go2 + Z1/gripper** (fallback). Fill the active route after G1.

| Asset | License | Source | Redistribution allowed? | Modification |
|---|---|---|---|---|
| Unitree Go2 (fallback) | BSD-3-Clause | MuJoCo Menagerie | TODO | mount Z1 on base |
| Unitree Z1 + gripper (fallback) | BSD-3-Clause | MuJoCo Menagerie | TODO | attach to Go2 |
| Unitree B2 + AgileX Piper (primary) | **MUST confirm** | ATEC assets / vendor | TODO | USD→MJCF conversion |

## Policy weights

- Trained from random initialization (no transfer) unless stated otherwise in the report.
- Released under MIT together with the network definition, training config, and
  the reproduction recipe.
