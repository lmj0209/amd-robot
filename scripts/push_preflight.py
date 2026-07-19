#!/usr/bin/env python3
"""Compile and run the near-field Push-to-Goal environment sequentially."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from amd_robo.envs.go2_z1_push import Go2Z1PushEnv  # noqa: E402
from amd_robo.platform.smoke import _block_tree  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.num_steps <= 0:
        parser.error("num-envs and num-steps must be positive")

    env = Go2Z1PushEnv()
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    state = reset_fn(keys)
    _block_tree(state)

    initial_object_position = state.info["object_pos"]
    initial_goal_distance = jnp.mean(state.metrics["object_to_goal_distance"])
    initial_prepush_distance = jnp.mean(state.metrics["base_to_prepush_distance"])
    actions = jnp.zeros((args.num_envs, env.action_size))
    done_count = jnp.zeros((), dtype=jnp.int32)
    illegal_contact_count = jnp.zeros((), dtype=jnp.int32)
    nonfinite_state_count = jnp.zeros((), dtype=jnp.int32)
    three_or_more_contact_total = jnp.zeros(())
    max_tilt_deg = jnp.zeros(())

    for _ in range(args.num_steps):
        state = step_fn(state, actions)
        done_count += jnp.sum(state.done.astype(jnp.int32))
        illegal_contact_count += jnp.sum(
            state.metrics["illegal_contact"].astype(jnp.int32)
        )
        nonfinite_state_count += jnp.sum(
            state.metrics["nonfinite_state"].astype(jnp.int32)
        )
        three_or_more_contact_total += jnp.mean(
            jnp.sum(state.info["last_contact"], axis=-1) >= 3
        )
        max_tilt_deg = jnp.maximum(
            max_tilt_deg, jnp.max(state.metrics["tilt_deg"])
        )

    _block_tree(state)
    object_displacement = jnp.linalg.norm(
        state.info["object_pos"][:, :2] - initial_object_position[:, :2],
        axis=-1,
    )
    result = {
        "backend": jax.default_backend(),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "transitions": args.num_envs * args.num_steps,
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "physics_substeps": env.n_substeps,
        "solver_iterations": int(env.mj_model.opt.iterations),
        "initial_base_to_prepush_distance": float(initial_prepush_distance),
        "final_base_to_prepush_distance": float(
            jnp.mean(state.metrics["base_to_prepush_distance"])
        ),
        "initial_object_to_goal_distance": float(initial_goal_distance),
        "final_object_to_goal_distance": float(
            jnp.mean(state.metrics["object_to_goal_distance"])
        ),
        "mean_object_displacement": float(jnp.mean(object_displacement)),
        "max_object_displacement": float(jnp.max(object_displacement)),
        "three_or_more_contact_fraction": float(
            three_or_more_contact_total / args.num_steps
        ),
        "max_tilt_deg": float(max_tilt_deg),
        "done_count": int(done_count),
        "illegal_contact_count": int(illegal_contact_count),
        "nonfinite_state_count": int(nonfinite_state_count),
        "nonfinite_observations": int(jnp.sum(~jnp.isfinite(state.obs))),
        "minimum_crawl_ik_reachable_fraction": float(
            jnp.min(state.metrics["crawl_ik_reachable_fraction"])
        ),
    }
    print(f"PUSH_PREFLIGHT {json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
