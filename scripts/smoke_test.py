#!/usr/bin/env python3
"""G0 smoke test — verify the MJX / JAX / ROCm link on RGC.

Gate G0 (see ../0_规划/计划.md) requires:
  - JAX sees a GPU (ROCm) device,
  - exact versions are printed,
  - a batched MJX physics step runs over many parallel envs with no NaN.

This script is intentionally defensive: it reports whatever is available even
before versions are pinned. It is NOT a performance benchmark.

Run:  python scripts/smoke_test.py
Exit code 1 if a hard check fails.

Phase 0 TODO: after versions are confirmed, add a short Brax PPO update check
in `_ppo_check()` (API depends on the pinned Brax version).
"""
from __future__ import annotations

import json
import sys
import time
import traceback


def _version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        return f"NOT INSTALLED ({type(e).__name__}: {e})"


def _check_jax() -> dict:
    import jax

    devices = jax.devices()
    backend = jax.default_backend()
    print("== jax ==")
    print("default_backend:", backend)
    print("devices:", devices)
    gpu_ok = any(
        "gpu" in str(d).lower() or "rocm" in str(d).lower() for d in devices
    )
    if not gpu_ok:
        print("WARN: no GPU device visible to JAX (expected a ROCm GPU on RGC).")
    return {
        "jax_backend": backend,
        "jax_devices": [str(d) for d in devices],
        "gpu_visible": gpu_ok,
    }


# Minimal free-body model — independent of the (not-yet-built) robot MJCF.
_BOX_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _mjx_batch_test(n_envs: int = 256, n_steps: int = 1000) -> dict:
    """Run n_envs parallel MJX envs for n_steps; report NaN + rough throughput."""
    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.mjx as mjx

    model = mujoco.MjModel.from_xml_string(_BOX_XML)
    mxm = mjx.put_model(model)
    d0 = mjx.make_data(mxm)

    # Replicate one data into a batch of n_envs (real copy, not a broadcast view).
    data = jax.tree_util.tree_map(
        lambda x: jnp.array(jnp.broadcast_to(x, (n_envs,) + x.shape)), d0
    )

    @jax.jit
    def step_batch(d):
        return jax.vmap(lambda dd: mjx.step(mxm, dd))(d)

    data = step_batch(data)  # compile + warmup
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), data)

    t0 = time.time()
    for _ in range(n_steps):
        data = step_batch(data)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), data)
    dt = time.time() - t0

    has_nan = bool(jnp.isnan(data.qpos).any())
    return {
        "n_envs": n_envs,
        "n_steps": n_steps,
        "wall_s": round(dt, 3),
        "env_steps_per_s": round(n_envs * n_steps / dt, 0),
        "has_nan": has_nan,
    }


def _ppo_check() -> str:
    # TODO(Phase 0): run one short Brax PPO update once the Brax version is pinned.
    return "skipped (Brax PPO check TODO until Brax version pinned)"


def main() -> int:
    info = {
        "python": sys.version.split()[0],
        "mujoco": _version("mujoco"),
        "brax": _version("brax"),
        "jax": _version("jax"),
        "jaxlib": _version("jaxlib"),
    }
    print("== versions ==")
    print(json.dumps(info, indent=2))

    try:
        info.update(_check_jax())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: jax check failed")
        return 1

    try:
        mjx_res = _mjx_batch_test()
        print("== MJX batched step ==")
        print(json.dumps(mjx_res, indent=2))
        info["mjx_batch"] = mjx_res
        if mjx_res["has_nan"]:
            print("FAIL: NaN detected in MJX batch run")
            return 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("FAIL: MJX batched step failed")
        return 1

    info["ppo_check"] = _ppo_check()
    print("== PPO ==", info["ppo_check"])

    print("\nG0 smoke test PASSED" if info["gpu_visible"] else "\nG0 smoke test PASSED (but no GPU — fix before proceeding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
