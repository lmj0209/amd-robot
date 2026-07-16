# Reproduction record

## Gate G0 fields

> Values are filled only from measured RGC output. `TODO` = not yet captured.

- RGC instance: `u-7670-4289a66a` (2026-07-16 probe)
- RGC image name: TODO
- RGC image digest: TODO
- GPU: **gfx1100 (RDNA3)**, 1 GPU visible via `amd-smi` (BDF 0000:03:00.0) — measured
- GPU model / VRAM: TODO (gfx1100 is consistent with Radeon PRO W7900-class per
  hello-rocm docs, but not yet confirmed from the instance itself)
- ROCm version: TODO — run `rocminfo --version`
- Python: **3.12.3 (GCC 13.3.0)** — measured
- JAX/JAXLIB: TODO (not preinstalled)
- JAX ROCm PJRT/plugin: TODO (must install the build matching gfx1100)
- MuJoCo/Brax/Playground: TODO (not preinstalled)
- Verified lockfile: TODO (`requirements/rgc.lock`)
- `system_info` evidence: TODO
- G0 commit and tag: TODO

## Findings (2026-07-16)

- The RGC base workspace has **no** `jax`, `torch`, `mujoco`, or `brax`
  preinstalled (`pip list` was empty for all of them). The full stack —
  including the JAX ROCm PJRT/plugin for gfx1100 — must be installed by us on
  top of the ROCm runtime.
- `rocminfo` / `amd-smi` work and see the GPU, so the ROCm runtime is functional.
- `groups: cannot find name for group ID 109` is a harmless cosmetic container
  warning and does not affect GPU access.

## Commands

The exact install, smoke, training, evaluation, and benchmark commands are added
only after they run successfully from a clean RGC environment.
