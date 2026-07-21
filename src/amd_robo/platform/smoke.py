"""Fail-closed Gate G0 validation for JAX, MJX, Playground, and Brax PPO."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import traceback
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any


class SmokeFailure(RuntimeError):
    """A hard Gate G0 requirement was not met."""


def _distribution_versions() -> dict[str, str]:
    requested = {
        "jax",
        "jaxlib",
        "mujoco",
        "brax",
        "playground",
        "ml-collections",
    }
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in requested or name.startswith("jax-rocm"):
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def _run(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _backend_platform_version() -> str:
    try:
        from jax.extend import backend

        return str(backend.get_backend().platform_version)
    except Exception:  # noqa: BLE001
        return "unknown"


def _check_rocm_device() -> dict[str, Any]:
    import jax

    devices = jax.devices()
    platform_version = _backend_platform_version()
    details = [
        {
            "platform": getattr(device, "platform", "unknown"),
            "device_kind": getattr(device, "device_kind", "unknown"),
            "description": str(device),
        }
        for device in devices
    ]
    identity = " ".join(
        [platform_version]
        + [f"{item['device_kind']} {item['description']}" for item in details]
    ).lower()
    amd_identity = any(marker in identity for marker in ("amd", "rocm", "gfx"))
    nvidia_identity = any(marker in identity for marker in ("nvidia", "cuda"))

    if jax.default_backend() != "gpu" or not devices:
        raise SmokeFailure("JAX is not using a GPU backend")
    if not amd_identity or nvidia_identity:
        raise SmokeFailure(
            "The visible JAX GPU was not positively identified as an AMD ROCm device"
        )

    amd_smi = shutil.which("amd-smi")
    if amd_smi is None:
        raise SmokeFailure("amd-smi is not available in the RGC runtime")
    smi = _run([amd_smi, "list", "--json"])
    if smi["returncode"] != 0:
        smi = _run([amd_smi, "static", "--json"])
    if smi["returncode"] != 0:
        raise SmokeFailure(f"amd-smi could not query the device: {smi['stderr']}")

    return {
        "default_backend": jax.default_backend(),
        "platform_version": platform_version,
        "devices": details,
        "amd_smi_command": smi["command"],
    }


def _block_tree(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _tree_is_finite(tree: Any) -> bool:
    import jax
    import jax.numpy as jnp

    checks = []
    for leaf in jax.tree_util.tree_leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.inexact):
            checks.append(jnp.all(jnp.isfinite(leaf)))
    if not checks:
        return True
    value = jnp.all(jnp.stack(checks))
    value.block_until_ready()
    return bool(value)


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


def _check_mjx(n_envs: int, n_steps: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    cpu_model = mujoco.MjModel.from_xml_string(_BOX_XML)
    model = mjx.put_model(cpu_model, impl="jax")
    data = mjx.make_data(cpu_model, impl="jax")
    implementation = getattr(getattr(model, "impl", None), "value", "jax")
    if implementation != "jax":
        raise SmokeFailure(f"MJX implementation is {implementation!r}, not 'jax'")

    single_step = jax.jit(lambda d: mjx.step(model, d))
    single = single_step(data)
    _block_tree(single)

    batched = jax.tree_util.tree_map(
        lambda value: jnp.array(
            jnp.broadcast_to(value, (n_envs,) + value.shape), copy=True
        ),
        data,
    )

    def batch_step(current):
        return jax.vmap(lambda one: mjx.step(model, one))(current)

    @jax.jit
    def rollout(current):
        def scan_step(carry, _):
            next_data = batch_step(carry)
            return next_data, None

        return jax.lax.scan(scan_step, current, None, length=n_steps)[0]

    warm = rollout(batched)
    _block_tree(warm)
    start = time.perf_counter()
    result = rollout(batched)
    _block_tree(result)
    elapsed = time.perf_counter() - start
    if not _tree_is_finite(result):
        raise SmokeFailure("Non-finite values were found in the MJX data tree")

    return {
        "implementation": implementation,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "finite": True,
        "diagnostic_wall_seconds": round(elapsed, 4),
        "diagnostic_env_steps_per_second": round(n_envs * n_steps / elapsed),
        "performance_evidence": False,
    }


def _load_playground_environment():
    from mujoco_playground import registry

    name = "CartpoleBalance"
    config = registry.get_default_config(name)
    env = registry.load(name, config=config, config_overrides={"impl": "jax"})
    return name, env


def _check_playground() -> tuple[dict[str, Any], Any]:
    import jax
    import jax.numpy as jnp

    name, env = _load_playground_environment()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    action = jnp.zeros((env.action_size,), dtype=jnp.float32)
    next_state = jax.jit(env.step)(state, action)
    _block_tree(next_state)
    if not _tree_is_finite(next_state.data):
        raise SmokeFailure("Playground reset/step produced non-finite MJX state")
    implementation = getattr(getattr(env.mjx_model, "impl", None), "value", "jax")
    if implementation != "jax":
        raise SmokeFailure(
            f"Playground loaded MJX implementation {implementation!r}, not 'jax'"
        )
    return {
        "environment": name,
        "action_size": env.action_size,
        "implementation": implementation,
        "reset_step_finite": True,
    }, env


def _check_ppo(env: Any) -> dict[str, Any]:
    from brax.training.agents.ppo import checkpoint
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    with tempfile.TemporaryDirectory(prefix="amd_robo_g0_") as tmp:
        _, params, metrics = ppo.train(
            environment=env,
            num_timesteps=32,
            max_devices_per_host=1,
            num_envs=8,
            episode_length=64,
            action_repeat=1,
            learning_rate=3e-4,
            entropy_cost=1e-3,
            discounting=0.97,
            unroll_length=4,
            batch_size=2,
            num_minibatches=4,
            num_updates_per_batch=1,
            normalize_observations=True,
            num_evals=1,
            num_eval_envs=4,
            run_evals=False,
            seed=0,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            save_checkpoint_path=tmp,
        )
        _block_tree(params)
        configs = list(Path(tmp).rglob("ppo_network_config.json"))
        if not configs:
            raise SmokeFailure("Brax PPO completed but did not save a checkpoint")
        checkpoint_dir = configs[-1].parent
        restored = checkpoint.load(checkpoint_dir)
        _block_tree(restored)
        if len(__import__("jax").tree_util.tree_leaves(restored)) == 0:
            raise SmokeFailure("The saved PPO checkpoint could not be restored")
        return {
            "training_steps": 32,
            "checkpoint_saved_and_loaded": True,
            "metrics": sorted(metrics.keys()),
        }


def gate_exit_code(checks: dict[str, dict[str, Any]], skipped: set[str]) -> int:
    """Return 0 only when every hard G0 check ran and passed."""

    required = {"rocm", "mjx", "playground", "ppo"}
    if skipped or set(checks) != required:
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Run a partial diagnostic. The command exits 2 and does not pass G0.",
    )
    args = parser.parse_args(argv)
    if args.n_envs < 1 or args.n_steps < 1:
        parser.error("--n-envs and --n-steps must be positive")

    # brax <-> jax 0.10 compat shim (required for the PPO check on our stack).
    from amd_robo.platform import _compat

    _compat.apply_brax_compat()

    report: dict[str, Any] = {
        "versions": _distribution_versions(),
        "checks": {},
        "skipped": [],
    }
    try:
        report["checks"]["rocm"] = _check_rocm_device()
        report["checks"]["mjx"] = _check_mjx(args.n_envs, args.n_steps)
        playground_result, env = _check_playground()
        report["checks"]["playground"] = playground_result
        if args.skip_ppo:
            report["skipped"].append("ppo")
        else:
            report["checks"]["ppo"] = _check_ppo(env)
    except (SmokeFailure, ModuleNotFoundError) as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "FAILED"
        print(json.dumps(report, indent=2, default=str))
        return 1
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "FAILED"
        print(json.dumps(report, indent=2, default=str))
        return 1

    code = gate_exit_code(report["checks"], set(report["skipped"]))
    report["status"] = "PASSED" if code == 0 else "PARTIAL"
    print(json.dumps(report, indent=2, default=str))
    if code == 0:
        print("G0 smoke test PASSED")
    else:
        print("G0 smoke test PARTIAL: skipped checks prevent Gate G0 acceptance")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
