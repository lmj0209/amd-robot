#!/usr/bin/env python3
"""Run one fail-closed MJX/Policy benchmark point in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from locomotion_learn_smoke import _load_config, _make_env  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1),
    )
    return ordered[index]


def _timed_call(fn, argument):
    start = time.perf_counter()
    result = fn(argument)
    _block_tree(result)
    return result, time.perf_counter() - start


def _run_steps(fn, state, num_steps: int):
    for _ in range(num_steps):
        state = fn(state)
    _block_tree(state)
    return state


def _build_policy(config: dict, env, params_path: Path):
    from amd_robo.platform import _compat

    _compat.apply_brax_compat()
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks

    ppo_config = config["ppo"]
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        policy_hidden_layer_sizes=tuple(
            ppo_config.get("policy_hidden_layer_sizes", (32, 32, 32, 32))
        ),
        value_hidden_layer_sizes=tuple(
            ppo_config.get(
                "value_hidden_layer_sizes",
                (256, 256, 256, 256, 256),
            )
        ),
    )
    params = brax_model.load_params(str(params_path))
    params_start = time.perf_counter()
    params = jax.device_put(params)
    _block_tree(params)
    params_device_put_s = time.perf_counter() - params_start
    policy = ppo_networks.make_inference_fn(networks)(
        params,
        deterministic=True,
    )
    policy_key = jax.device_put(jax.random.PRNGKey(0))
    return policy, policy_key, params_device_put_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("env", "policy", "combined"), required=True)
    parser.add_argument(
        "--config",
        default="configs/push_stage2_near_field_solver16_qualification.yaml",
    )
    parser.add_argument("--params-in")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps-per-repeat", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--latency-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--expected-backend",
        choices=("gpu", "cpu"),
        required=True,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.steps_per_repeat <= 0 or args.warmup_steps < 0:
        parser.error("steps per repeat must be positive and warmup non-negative")
    if args.repeats < 5:
        parser.error("formal benchmark requires at least five repeats")
    if args.latency_samples < 5:
        parser.error("latency benchmark requires at least five samples")
    if args.mode != "env" and not args.params_in:
        parser.error("policy and combined modes require --params-in")

    backend = jax.default_backend()
    if backend != args.expected_backend:
        raise RuntimeError(
            f"expected JAX backend {args.expected_backend}, got {backend}"
        )
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible device, got {devices}")

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        parser.error(f"config does not exist: {config_path}")
    params_path = None if args.params_in is None else Path(args.params_in).resolve()
    if params_path is not None and not params_path.is_file():
        parser.error(f"params do not exist: {params_path}")

    config, config_sha256 = _load_config(str(config_path))
    evaluation = config["manual_evaluation"]
    env = _make_env(
        config,
        command_override=evaluation["fixed_command"],
        randomize_reset=False,
        task="push",
    )
    keys = jax.device_put(
        jax.random.split(jax.random.PRNGKey(args.seed), args.batch_size)
    )
    reset_fn = jax.jit(jax.vmap(env.reset))
    state, reset_compile_s = _timed_call(reset_fn, keys)

    result = {
        "schema_version": 1,
        "status": "ok",
        "mode": args.mode,
        "git_commit": _git_commit(),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_sha256,
        "params_sha256": None if params_path is None else _sha256(params_path),
        "backend": backend,
        "device": str(devices[0]),
        "jax_version": jax.__version__,
        "batch_size": args.batch_size,
        "solver_iterations": int(env.mj_model.opt.iterations),
        "control_timestep": env.dt,
        "physics_substeps": env.n_substeps,
        "reset_compile_s": reset_compile_s,
        "warmup_steps": args.warmup_steps,
        "repeats": args.repeats,
        "steps_per_repeat": args.steps_per_repeat,
        "latency_samples": args.latency_samples,
    }

    if args.mode == "env":
        actions = jax.device_put(
            jnp.zeros((args.batch_size, env.action_size), dtype=jnp.float32)
        )
        step_fn = jax.jit(jax.vmap(env.step))

        def benchmark_step(current):
            return step_fn(current, actions)

        state, cold_compile_s = _timed_call(benchmark_step, state)
        state = _run_steps(benchmark_step, state, args.warmup_steps)
        repeat_s = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            state = _run_steps(benchmark_step, state, args.steps_per_repeat)
            repeat_s.append(time.perf_counter() - start)
        transitions = args.batch_size * args.steps_per_repeat
        throughput = [transitions / duration for duration in repeat_s]
        result.update(
            {
                "cold_compile_s": cold_compile_s,
                "repeat_s": repeat_s,
                "throughput_transitions_s": throughput,
                "throughput_mean": statistics.fmean(throughput),
                "throughput_min": min(throughput),
                "throughput_max": max(throughput),
            }
        )
    else:
        assert params_path is not None
        policy, policy_key, params_device_put_s = _build_policy(
            config,
            env,
            params_path,
        )
        policy_fn = jax.jit(lambda obs: policy(obs, policy_key)[0])
        if args.mode == "policy":
            action, cold_compile_s = _timed_call(policy_fn, state.obs)
            for _ in range(args.warmup_steps):
                action = policy_fn(state.obs)
            _block_tree(action)
            latency_s = []
            for _ in range(args.latency_samples):
                start = time.perf_counter()
                action = policy_fn(state.obs)
                _block_tree(action)
                latency_s.append(time.perf_counter() - start)
            throughput = [args.batch_size / duration for duration in latency_s]
            result.update(
                {
                    "params_device_put_s": params_device_put_s,
                    "cold_compile_s": cold_compile_s,
                    "latency_s": latency_s,
                    "latency_p50_ms": _percentile(latency_s, 0.50) * 1000.0,
                    "latency_p95_ms": _percentile(latency_s, 0.95) * 1000.0,
                    "latency_mean_ms": statistics.fmean(latency_s) * 1000.0,
                    "throughput_inferences_s_mean": statistics.fmean(throughput),
                }
            )
        else:
            step_fn = jax.vmap(env.step)

            @jax.jit
            def benchmark_step(current):
                action = policy(current.obs, policy_key)[0]
                return step_fn(current, action)

            state, cold_compile_s = _timed_call(benchmark_step, state)
            state = _run_steps(benchmark_step, state, args.warmup_steps)
            repeat_s = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                state = _run_steps(benchmark_step, state, args.steps_per_repeat)
                repeat_s.append(time.perf_counter() - start)
            transitions = args.batch_size * args.steps_per_repeat
            throughput = [transitions / duration for duration in repeat_s]
            result.update(
                {
                    "params_device_put_s": params_device_put_s,
                    "cold_compile_s": cold_compile_s,
                    "repeat_s": repeat_s,
                    "throughput_transitions_s": throughput,
                    "throughput_mean": statistics.fmean(throughput),
                    "throughput_min": min(throughput),
                    "throughput_max": max(throughput),
                }
            )

    finite = bool(
        jnp.all(jnp.isfinite(state.data.qpos))
        & jnp.all(jnp.isfinite(state.data.qvel))
        & jnp.all(jnp.isfinite(state.obs))
        & jnp.all(jnp.isfinite(state.reward))
    )
    result["finite"] = finite
    result["done_count_final_state"] = int(jnp.sum(state.done))
    if not finite:
        result["status"] = "nonfinite"

    print(f"ROCM_BENCHMARK {json.dumps(result, sort_keys=True)}", flush=True)
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
