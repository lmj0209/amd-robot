#!/usr/bin/env python3
"""Compile and run a finite-step Go2+Z1 locomotion gate.

The probe intentionally uses a sequential Python loop around one reusable
``jit(vmap(env.step))`` kernel. This is the measured gfx1100-safe rollout shape;
it does not hide the environment step inside a long fused scan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from amd_robo.envs.go2_z1_locomotion import Go2Z1LocomotionEnv  # noqa: E402
from amd_robo.platform.smoke import _block_tree  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--action-seed", type=int, default=9)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--leg-kp", type=float)
    parser.add_argument("--gait-cycle-time", type=float)
    parser.add_argument("--trot-contact-scale", type=float, default=0.0)
    parser.add_argument("--trot-swing-height-cost-scale", type=float, default=0.0)
    parser.add_argument("--trot-timing-scale", type=float, default=0.0)
    parser.add_argument("--trot-timing-std", type=float, default=0.1)
    parser.add_argument("--trot-timing-max-error", type=float, default=0.2)
    parser.add_argument("--trot-timing-min-air-time", type=float, default=0.0)
    parser.add_argument("--crawl-reference", action="store_true")
    parser.add_argument("--crawl-stride", type=float, default=0.08)
    parser.add_argument("--crawl-shift", type=float, default=0.06)
    parser.add_argument("--crawl-lift", type=float, default=0.45)
    parser.add_argument("--crawl-min-air-time", type=float, default=0.07)
    parser.add_argument("--command-x", type=float)
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help="Use the standing baseline instead of fixed random actions.",
    )
    parser.add_argument(
        "--resample-actions",
        action="store_true",
        help="Sample independent uniform actions at every control step.",
    )
    parser.add_argument(
        "--training-wrapper",
        action="store_true",
        help="Include the Brax episode/autoreset wrapper used during PPO.",
    )
    args = parser.parse_args()
    if (
        args.num_envs <= 0
        or args.num_steps <= 0
        or args.action_scale <= 0.0
        or (args.leg_kp is not None and args.leg_kp <= 0.0)
        or (args.gait_cycle_time is not None and args.gait_cycle_time <= 0.0)
        or args.trot_contact_scale < 0.0
        or args.trot_swing_height_cost_scale < 0.0
        or args.trot_timing_scale < 0.0
        or args.trot_timing_std <= 0.0
        or args.trot_timing_max_error <= 0.0
        or args.trot_timing_min_air_time < 0.0
        or args.crawl_stride < 0.0
        or args.crawl_shift <= 0.0
        or args.crawl_lift <= 0.0
        or args.crawl_min_air_time <= 0.0
    ):
        parser.error(
            "--num-envs, --num-steps, --action-scale, --leg-kp, and "
            "--gait-cycle-time must be positive when provided; trot reward "
            "scales must be non-negative and trot timing shape parameters "
            "must be positive; minimum air time and crawl stride must be "
            "non-negative; crawl shift and lift must be positive"
        )
    if args.gait_cycle_time is None and (
        args.trot_contact_scale > 0.0
        or args.trot_swing_height_cost_scale > 0.0
        or args.trot_timing_scale > 0.0
    ):
        parser.error("trot reward scales require --gait-cycle-time")
    if args.crawl_reference and args.gait_cycle_time is None:
        parser.error("--crawl-reference requires --gait-cycle-time")
    if args.crawl_reference and (
        args.trot_contact_scale > 0.0
        or args.trot_swing_height_cost_scale > 0.0
        or args.trot_timing_scale > 0.0
    ):
        parser.error("--crawl-reference cannot be combined with trot rewards")
    if args.zero_actions and args.resample_actions:
        parser.error("--zero-actions and --resample-actions are mutually exclusive")

    env = Go2Z1LocomotionEnv(
        action_scale=args.action_scale,
        leg_kp=args.leg_kp,
        gait_cycle_time=args.gait_cycle_time,
        trot_contact_scale=args.trot_contact_scale,
        trot_swing_height_cost_scale=args.trot_swing_height_cost_scale,
        trot_timing_scale=args.trot_timing_scale,
        trot_timing_std=args.trot_timing_std,
        trot_timing_max_error=args.trot_timing_max_error,
        trot_timing_min_air_time=args.trot_timing_min_air_time,
        crawl_reference_enabled=args.crawl_reference,
        crawl_stride=args.crawl_stride,
        crawl_shift=args.crawl_shift,
        crawl_lift=args.crawl_lift,
        crawl_min_air_time=args.crawl_min_air_time,
        command_override=(
            None if args.command_x is None else (args.command_x, 0.0, 0.0)
        ),
        randomize_reset=args.command_x is None,
    )
    rollout_env = env
    if args.training_wrapper:
        from mujoco_playground import wrapper

        rollout_env = wrapper.wrap_for_brax_training(
            env, episode_length=args.num_steps + 1, action_repeat=1
        )
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    if args.training_wrapper:
        reset_fn = jax.jit(rollout_env.reset)
        step_fn = jax.jit(rollout_env.step)
    else:
        reset_fn = jax.jit(jax.vmap(rollout_env.reset))
        step_fn = jax.jit(jax.vmap(rollout_env.step))
    state = reset_fn(keys)
    if args.zero_actions:
        actions = jnp.zeros((args.num_envs, env.action_size))
    else:
        actions = jax.random.uniform(
            jax.random.PRNGKey(args.action_seed),
            (args.num_envs, env.action_size),
            minval=-1.0,
            maxval=1.0,
        )

    done_count = 0
    illegal_contact_count = 0
    nonfinite_state_count = 0
    nonfinite_rewards = 0
    nonfinite_observations = 0
    action_key = jax.random.PRNGKey(args.action_seed)
    max_abs_observation = float(jnp.max(jnp.abs(state.obs)))
    min_reward = float("inf")
    max_reward = float("-inf")
    max_abs_actuator_force = 0.0
    reward_metric_names = tuple(
        key for key in state.metrics if key.startswith("reward/")
    )
    max_abs_reward_components = {key: 0.0 for key in reward_metric_names}
    max_feet_air_time = 0.0
    max_swing_peak = 0.0
    previous_contact = None
    liftoff_count = jnp.zeros(4, dtype=jnp.int32)
    touchdown_count = jnp.zeros(4, dtype=jnp.int32)
    contact_duty_total = jnp.zeros(4)
    three_or_more_contact_total = jnp.zeros(())
    all_four_contact_total = jnp.zeros(())
    crawl_swing_sample_count = jnp.zeros(())
    crawl_active_off_total = jnp.zeros(())
    crawl_stance_three_total = jnp.zeros(())
    sustained_swing_count = jnp.zeros(4, dtype=jnp.int32)
    for _ in range(args.num_steps):
        if args.resample_actions:
            action_key, step_action_key = jax.random.split(action_key)
            actions = jax.random.uniform(
                step_action_key,
                (args.num_envs, env.action_size),
                minval=-1.0,
                maxval=1.0,
            )
        state = step_fn(state, actions)
        done_count += int(jnp.sum(state.done).block_until_ready())
        illegal_contact_count += int(
            jnp.sum(state.metrics["illegal_contact"]).block_until_ready()
        )
        nonfinite_state_count += int(
            jnp.sum(state.metrics["nonfinite_state"]).block_until_ready()
        )
        nonfinite_rewards += int(
            jnp.sum(~jnp.isfinite(state.reward)).block_until_ready()
        )
        nonfinite_observations += int(
            jnp.sum(~jnp.isfinite(state.obs)).block_until_ready()
        )
        max_abs_observation = max(
            max_abs_observation,
            float(jnp.max(jnp.abs(state.obs)).block_until_ready()),
        )
        min_reward = min(min_reward, float(jnp.min(state.reward).block_until_ready()))
        max_reward = max(max_reward, float(jnp.max(state.reward).block_until_ready()))
        max_abs_actuator_force = max(
            max_abs_actuator_force,
            float(jnp.max(jnp.abs(state.data.actuator_force)).block_until_ready()),
        )
        for key in reward_metric_names:
            max_abs_reward_components[key] = max(
                max_abs_reward_components[key],
                float(jnp.max(jnp.abs(state.metrics[key])).block_until_ready()),
            )
        max_feet_air_time = max(
            max_feet_air_time,
            float(jnp.max(state.info["feet_air_time"]).block_until_ready()),
        )
        max_swing_peak = max(
            max_swing_peak,
            float(jnp.max(state.info["swing_peak"]).block_until_ready()),
        )
        foot_contact = state.info["last_contact"]
        if previous_contact is not None:
            liftoff_count += jnp.sum(
                previous_contact & ~foot_contact,
                axis=0,
                dtype=jnp.int32,
            )
            touchdown_count += jnp.sum(
                ~previous_contact & foot_contact,
                axis=0,
                dtype=jnp.int32,
            )
        previous_contact = foot_contact
        contact_duty_total += jnp.mean(foot_contact, axis=0)
        contact_count = jnp.sum(foot_contact, axis=-1)
        three_or_more_contact_total += jnp.mean(contact_count >= 3)
        all_four_contact_total += jnp.mean(contact_count == 4)
        if args.crawl_reference:
            sustained_swing_count += jnp.sum(
                state.info["crawl_sustained_touchdown"],
                axis=0,
                dtype=jnp.int32,
            )
            active_leg = state.info["crawl_active_leg"]
            active_contact = jnp.take_along_axis(
                foot_contact,
                active_leg[:, None],
                axis=1,
            )[:, 0]
            swing_window = state.info["crawl_swing_window"]
            crawl_swing_sample_count += jnp.sum(swing_window)
            crawl_active_off_total += jnp.sum(swing_window & ~active_contact)
            crawl_stance_three_total += jnp.sum(
                swing_window & ((contact_count - active_contact) == 3)
            )
    _block_tree(state)

    crawl_swing_denominator = jnp.maximum(crawl_swing_sample_count, 1.0)
    report = {
        "transitions": args.num_envs * args.num_steps,
        "observation_size": int(state.obs.shape[-1]),
        "action_size": env.action_size,
        "control_timestep": env.dt,
        "physics_substeps": env.n_substeps,
        "action_scale": args.action_scale,
        "leg_kp": env._leg_kp,
        "gait_cycle_time": args.gait_cycle_time,
        "crawl_reference": args.crawl_reference,
        "crawl_stride": args.crawl_stride,
        "crawl_shift": args.crawl_shift,
        "crawl_lift": args.crawl_lift,
        "crawl_min_air_time": args.crawl_min_air_time,
        "command_x_override": args.command_x,
        "done_count": done_count,
        "illegal_contact_count": illegal_contact_count,
        "nonfinite_state_count": nonfinite_state_count,
        "nonfinite_rewards": nonfinite_rewards,
        "nonfinite_observations": nonfinite_observations,
        "max_abs_observation": max_abs_observation,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "max_abs_actuator_force": max_abs_actuator_force,
        "max_abs_reward_components": max_abs_reward_components,
        "max_feet_air_time": max_feet_air_time,
        "max_swing_peak": max_swing_peak,
        "liftoffs_per_env": [
            float(value) / args.num_envs for value in liftoff_count
        ],
        "touchdowns_per_env": [
            float(value) / args.num_envs for value in touchdown_count
        ],
        "sustained_swings_per_env": [
            float(value) / args.num_envs for value in sustained_swing_count
        ],
        "foot_contact_duty": [
            float(value) / args.num_steps for value in contact_duty_total
        ],
        "three_or_more_contact_fraction": float(
            three_or_more_contact_total / args.num_steps
        ),
        "all_four_contact_fraction": float(
            all_four_contact_total / args.num_steps
        ),
        "crawl_active_off_fraction": float(
            crawl_active_off_total / crawl_swing_denominator
        ),
        "crawl_stance_three_fraction": float(
            crawl_stance_three_total / crawl_swing_denominator
        ),
        "mean_reward": float(jnp.mean(state.reward)),
        "mean_command_x": float(jnp.mean(state.metrics["command_x"])),
        "mean_forward_velocity": float(
            jnp.mean(state.metrics["base_forward_velocity"])
        ),
        "mean_tracking_error": float(jnp.mean(state.metrics["tracking_linear_error"])),
        "zero_actions": args.zero_actions,
        "resample_actions": args.resample_actions,
        "training_wrapper": args.training_wrapper,
    }
    print(f"LOCOMOTION_PREFLIGHT {json.dumps(report, sort_keys=True)}", flush=True)
    return int(
        report["observation_size"] != env.observation_size
        or report["action_size"] != 19
        or report["physics_substeps"] > 5
        or nonfinite_rewards > 0
        or nonfinite_observations > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
