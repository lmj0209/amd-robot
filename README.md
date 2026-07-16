# ROCm-Accelerated Quadruped Mobile Manipulation with MJX

> **Status:** the engineering scaffold is ready; Gate G0 has not passed yet.
> No dependency version, performance number, or success rate is a measured RGC
> result until it is accompanied by a system fingerprint and raw evidence.

## Project

A quadruped-with-arm robot learns to approach a randomized box and push it into
a goal zone: **Approach → Align → Push → Hold**. The locked stack is MuJoCo MJX
with its JAX implementation, a MuJoCo Playground-style environment, Brax PPO,
and one AMD Radeon GPU through ROCm.

The action contract is always 19-dimensional: 12 leg joints, 6 arm joints, and
1 coupled gripper action. The Push MVP may mask the gripper but never changes
the policy interface.

## Repository status

- `src/amd_robo/contracts.py` freezes cross-stage interfaces.
- `configs/` contains versioned run inputs.
- `tests/` runs CPU-safe contract tests.
- `scripts/smoke_test.py` is a fail-closed RGC Gate G0 check.
- `scripts/system_info.py` creates immutable benchmark fingerprints.
- `assets/manifest.yaml` and `artifacts/manifest.json` track external files.

The B2Piper model is preferred. If it does not pass the model gate by
2026-07-20, the project switches to Go2 + Z1/gripper without changing the
MJX/JAX/ROCm stack.

## Local contract tests

These tests do not prove ROCm support:

```bash
python -m pip install pytest pyyaml
python -m pip install -e . --no-deps
python -m pytest
```

## Gate G0 on RGC

Do not install a generic JAX build over the RGC runtime. Follow
`requirements/README.md`, install the AMD PJRT/plugin/JAX/JAXLIB combination for
the exact RGC image and GPU, then install the project without replacing it.

```bash
python scripts/system_info.py \
  --image-name TODO_FROM_RGC \
  --image-digest TODO_FROM_RGC \
  --config configs/smoke.yaml
python scripts/smoke_test.py
```

The smoke command returns 0 only if all of these pass:

1. JAX positively identifies an AMD ROCm GPU and `amd-smi` works.
2. MJX-JAX runs a finite single and 256-environment batched rollout.
3. `CartpoleBalance` loads through MuJoCo Playground with `impl=jax`.
4. Brax PPO completes an update and saves and reloads a checkpoint.

`--skip-ppo` is diagnostic only and exits 2; it never passes Gate G0.

## Reproduction and results

Environment locks, train/evaluate/benchmark commands, checkpoints, videos, and
measured results are added only after their gates. See `docs/REPRODUCTION.md`.

## License

Project code is MIT licensed. Third-party software, robot models, and assets are
tracked in `THIRD_PARTY_NOTICES.md` and `assets/manifest.yaml`.
