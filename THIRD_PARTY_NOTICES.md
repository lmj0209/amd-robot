# Third-Party Notices

This project uses third-party software and assets. Sources, versions, licenses,
and local modifications are listed below. The versions match the validated
Radeon PRO W7900 / ROCm 7.2.1 environment recorded in
`requirements/rgc.lock`.

## Software

| Component | Version | License | Source | Modification |
|---|---|---|---|---|
| MuJoCo + MJX | 3.10.0 | Apache-2.0 | https://github.com/google-deepmind/mujoco | none |
| Brax | 0.14.2 | Apache-2.0 | https://github.com/google/brax | local TrainingState API patch; PPO math unchanged |
| MuJoCo Playground | 0.2.0 at `43d180a226da3aae091d918b63c06c3a343519ad` | Apache-2.0 | https://github.com/google-deepmind/mujoco_playground | none |
| JAX + JAXLIB | 0.10.2 | Apache-2.0 | https://github.com/jax-ml/jax | local compatibility shim for removed replication helper |
| JAX ROCm PJRT | 0.10.2 | Apache-2.0 | https://github.com/jax-ml/jax | none |
| JAX ROCm plugin | 0.10.2 | Apache-2.0 | https://github.com/jax-ml/jax | none |

The Brax release patch is stored in
`patches/brax-0.14.2-training-state.patch`. The separate pmap-axis improvement
was submitted upstream as https://github.com/google/brax/pull/674 and is not
claimed as accepted or merged.

## Robot / Asset Models

| Asset | License | Source | Redistribution allowed? | Modification |
|---|---|---|---|---|
| Unitree Go2 | BSD-3-Clause | MuJoCo Menagerie at `71f066ad0be9cd271f7ed58c030243ef157af9f4` | yes | Z1 mount and MJX scene generation |
| Unitree Z1 + gripper | BSD-3-Clause | MuJoCo Menagerie at `71f066ad0be9cd271f7ed58c030243ef157af9f4` | yes | attached to Go2 with a coupled gripper actuator |
| Derived Go2 + Z1 MJCF | BSD-3-Clause | generated from the two pinned models | yes | 19-DoF action contract and task sites |

The required Unitree license texts are retained in `assets/licenses/`. The
complete fetched source-model directories are gitignored;
`scripts/fetch_menagerie.sh` fetches only the two selected directories at the
pinned commit. No ATEC, Unitree B2, or AgileX Piper asset is included or
distributed by this repository.

## Policy weights

- Trained from random initialization (no transfer) unless stated otherwise in the report.
- Released under MIT together with the network definition, training config, and
  the reproduction recipe.
