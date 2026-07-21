# Dependency workflow

The JAX ROCm runtime is platform-specific. A generic `pip install jax` is not
proof that JAX is using an AMD GPU.

## Before Gate G0

1. Create an RGC instance and record its image name, GPU, and ROCm. Record an
   image digest when the provider exposes one; this RGC instance did not.
2. Follow the AMD installation instructions for that exact ROCm release and GPU
   architecture. Install matching PJRT, plugin, JAXLIB, and JAX packages.
3. Install the application packages from `base.in` without replacing the
   verified JAX stack.
4. Install this project in editable mode: `python -m pip install -e . --no-deps`.
5. Run `python scripts/smoke_test.py`.

## Validated lock

`rgc.lock` records the exact environment measured on the Radeon PRO W7900 on
2026-07-21. It was derived from `/workspace/.venv/bin/python -m pip freeze
--all`; the editable private-repository requirement was removed, and the
Playground installation proxy was normalized to the same official upstream
commit.

Recreate the ROCm-specific core before applying the complete lock:

```bash
python -m venv /workspace/.venv
/workspace/.venv/bin/python -m pip install \
  "jax[rocm7-local]==0.10.2"
/workspace/.venv/bin/python -m pip install \
  -r requirements/rgc.lock
/workspace/.venv/bin/python -m pip install -e . --no-deps
```

Then apply the documented Brax 0.14.2 patch, fetch the pinned Menagerie assets,
and rerun `scripts/smoke_test.py`. A generic CPU/GPU `requirements.txt` must not
be cited as the RGC environment specification.
