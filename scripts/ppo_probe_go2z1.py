#!/usr/bin/env python3
"""Probe whether Brax PPO can train Go2Z1Env on gfx1100 (scan-based rollout).

Brax PPO collects rollouts with ``lax.scan`` over ``unroll_length`` env.step
calls. ``Go2Z1Env.step`` unrolls its ``n_substeps`` physics steps as a fixed
Python loop (no inner scan) because ``vmap x lax.scan x mjx.step`` on go2_z1
segfaults on gfx1100 (see the env docstring). The open G2 question is whether
Brax's OUTER scan over the (substep-unrolled) env.step is itself stable. A
segfault (process exit 139) means Brax cannot train this model as-is.

Usage::

    python scripts/ppo_probe_go2z1.py --unroll-length 4

Prints PPO_PROBE_PASSED on success. Sweep --unroll-length to map the ceiling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unroll-length", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=512)
    parser.add_argument("--episode-length", type=int, default=64)
    args = parser.parse_args()

    import jax

    from amd_robo.platform import _compat

    _compat.apply_brax_compat()
    from amd_robo.envs.go2_z1 import Go2Z1Env
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    # batch_size x num_minibatches must equal num_envs (Brax PPO constraint).
    batch_size = args.num_envs // 4
    num_minibatches = args.num_envs // batch_size

    env = Go2Z1Env()
    print(
        f"unroll={args.unroll_length} num_envs={args.num_envs} "
        f"batch_size={batch_size} num_minibatches={num_minibatches} "
        f"num_timesteps={args.num_timesteps}",
        flush=True,
    )

    _, params, metrics = ppo.train(
        environment=env,
        num_timesteps=args.num_timesteps,
        max_devices_per_host=1,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=1,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        discounting=0.97,
        unroll_length=args.unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=1,
        normalize_observations=True,
        num_evals=1,
        num_eval_envs=4,
        run_evals=False,
        seed=0,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
    print("PPO_PROBE_PASSED", flush=True)
    print("metrics:", sorted(metrics.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
