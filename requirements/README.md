# Dependency workflow

The JAX ROCm runtime is platform-specific. A generic `pip install jax` is not
proof that JAX is using an AMD GPU.

## Before Gate G0

1. Create an RGC instance and record its image name, digest, GPU, and ROCm.
2. Follow the AMD installation instructions for that exact ROCm release and GPU
   architecture. Install matching PJRT, plugin, JAXLIB, and JAX packages.
3. Install the application packages from `base.in` without replacing the
   verified JAX stack.
4. Install this project in editable mode: `python -m pip install -e . --no-deps`.
5. Run `python scripts/smoke_test.py`.

## After Gate G0

Export a complete, exact lock to `rgc.lock`; keep the image digest in
`docs/REPRODUCTION.md`; recreate the environment from scratch; rerun the smoke
test. `requirements.txt` remains a local CPU bootstrap and must not be cited as
the RGC environment specification.
