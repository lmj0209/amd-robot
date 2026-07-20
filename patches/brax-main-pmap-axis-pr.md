# Brax PR draft: match PPO input sharding to the pmap axis

Proposed title:

```text
Fix PPO input sharding to match pmap axis
```

Upstream base: `google/brax@1aa46f127bc4208d0c6de350a58ef00c8f01b0d9`

Prepared commit: `0802964232f42a6732c4c27236bbad785f3ab452`

## Summary

Pass PPO's `_PMAP_AXIS_NAME` to `bcast_local_devices` when placing the initial
training state.

The new `jax.pmap` implementation returns arrays with a `NamedSharding` whose
mesh axis matches the pmap axis name (`i` in PPO). The current initial state
uses `_device_put_sharded` instead. The shapes and devices are the same, but the
sharding objects differ, so the first pmap result changes the input signature
seen by the next host call.

This change:

- lets `bcast_local_devices` accept an axis name while retaining its existing
  default;
- reuses that helper in PPO instead of duplicating its implementation;
- initializes PPO state with the same `i` axis used by the training pmap; and
- adds a focused test for the requested sharding axis.

## Validation

Validated with Python 3.12.3, JAX/JAXLIB 0.10.2, Brax 0.14.2 runtime
dependencies, and the current upstream files.

```text
JAX_PLATFORMS=cpu python -m brax.training.pmap_test
Ran 1 test in 0.141s
OK

CPU minimal PPO:
PPO_SMOKE_PASSED

Radeon PRO W7900 / gfx1100 / ROCm 7.2.1:
input  = NamedSharding(mesh=Mesh('i': 1), spec=P('i',))
output = NamedSharding(mesh=Mesh('i': 1), spec=P('i',))
same=True

Two ROCm PPO host calls:
step=8  delta=8.857824 s
step=16 delta=0.004462 s
ROCM_TWO_HOST_CALLS_PASSED
```

The default `_device_put_sharded` value remains unchanged for other callers.

## Submission checklist

- [x] Patch is based on current `google/brax` main.
- [x] Focused test passes on the CPU backend.
- [x] Minimal PPO smoke passes on CPU and ROCm.
- [x] Input/output sharding equality is measured on ROCm.
- [ ] Contributor License Agreement confirmed by the submitting account.
- [ ] Branch pushed to the submitter's Brax fork.
- [ ] Pull request opened and URL recorded.
