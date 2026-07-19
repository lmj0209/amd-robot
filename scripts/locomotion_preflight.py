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
    parser.add_argument("--crawl-shift-end-fraction", type=float, default=0.3)
    parser.add_argument("--crawl-lift-start-fraction", type=float, default=0.3)
    parser.add_argument("--crawl-lift-end-fraction", type=float, default=0.8)
    parser.add_argument("--crawl-foot-space", action="store_true")
    parser.add_argument("--crawl-foot-step-length", type=float, default=0.08)
    parser.add_argument("--crawl-foot-clearance", type=float, default=0.04)
    parser.add_argument("--crawl-body-shift-x", type=float, default=0.02)
    parser.add_argument("--crawl-body-shift-y", type=float, default=0.02)
    parser.add_argument("--crawl-min-air-time", type=float, default=0.07)
    parser.add_argument("--crawl-pose-reference", action="store_true")
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
        or args.crawl_foot_step_length <= 0.0
        or args.crawl_foot_clearance <= 0.0
        or args.crawl_body_shift_x < 0.0
        or args.crawl_body_shift_y < 0.0
        or not (
            0.0
            < args.crawl_shift_end_fraction
            <= args.crawl_lift_start_fraction
            < args.crawl_lift_end_fraction
            < 1.0
        )
    ):
        parser.error(
            "--num-envs, --num-steps, --action-scale, --leg-kp, and "
            "--gait-cycle-time must be positive when provided; trot reward "
            "scales must be non-negative and trot timing shape parameters "
            "must be positive; minimum air time and crawl stride must be "
            "non-negative; crawl shift and lift must be positive; crawl "
            "timing must satisfy 0 < shift end <= lift start < lift end < 1"
        )
    if args.gait_cycle_time is None and (
        args.trot_contact_scale > 0.0
        or args.trot_swing_height_cost_scale > 0.0
        or args.trot_timing_scale > 0.0
    ):
        parser.error("trot reward scales require --gait-cycle-time")
    if args.crawl_reference and args.gait_cycle_time is None:
        parser.error("--crawl-reference requires --gait-cycle-time")
    if args.crawl_pose_reference and not args.crawl_reference:
        parser.error("--crawl-pose-reference requires --crawl-reference")
    if args.crawl_foot_space and not args.crawl_reference:
        parser.error("--crawl-foot-space requires --crawl-reference")
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
        crawl_shift_end_fraction=args.crawl_shift_end_fraction,
        crawl_lift_start_fraction=args.crawl_lift_start_fraction,
        crawl_lift_end_fraction=args.crawl_lift_end_fraction,
        crawl_foot_space_enabled=args.crawl_foot_space,
        crawl_foot_step_length=args.crawl_foot_step_length,
        crawl_foot_clearance=args.crawl_foot_clearance,
        crawl_body_shift_x=args.crawl_body_shift_x,
        crawl_body_shift_y=args.crawl_body_shift_y,
        crawl_min_air_time=args.crawl_min_air_time,
        crawl_pose_reference_enabled=args.crawl_pose_reference,
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
    initial_x = state.data.qpos[:, 0]
    if args.zero_actions:
        actions = jnp.zeros((args.num_envs, env.action_size))
    else:
        actions = jax.random.uniform(
            jax.random.PRNGKey(args.action_seed),
            (args.num_envs, env.action_size),
            minval=-1.0,
            maxval=1.0,
        )

    done_count = jnp.zeros((), dtype=jnp.int32)
    illegal_contact_count = jnp.zeros((), dtype=jnp.int32)
    nonfinite_state_count = jnp.zeros((), dtype=jnp.int32)
    nonfinite_rewards = jnp.zeros((), dtype=jnp.int32)
    nonfinite_observations = jnp.zeros((), dtype=jnp.int32)
    action_key = jax.random.PRNGKey(args.action_seed)
    max_abs_observation = jnp.max(jnp.abs(state.obs))
    min_reward = jnp.asarray(jnp.inf)
    max_reward = jnp.asarray(-jnp.inf)
    max_abs_actuator_force = jnp.zeros(())
    forward_velocity_total = jnp.zeros(())
    tilt_total = jnp.zeros(())
    max_tilt_deg = jnp.zeros(())
    reward_metric_names = tuple(
        key for key in state.metrics if key.startswith("reward/")
    )
    max_abs_reward_components = {
        key: jnp.zeros(()) for key in reward_metric_names
    }
    max_feet_air_time = jnp.zeros(())
    max_swing_peak = jnp.zeros(())
    crawl_reference_error_total = jnp.zeros(())
    max_crawl_reference_error = jnp.zeros(())
    has_crawl_reference_error = "crawl_reference_error_rms" in state.metrics
    minimum_crawl_ik_reachable = jnp.ones(())
    has_crawl_ik_reachable = "crawl_ik_reachable_fraction" in state.metrics
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
        done_count += jnp.sum(state.done.astype(jnp.int32))
        illegal_contact_count += jnp.sum(
            state.metrics["illegal_contact"].astype(jnp.int32)
        )
        nonfinite_state_count += jnp.sum(
            state.metrics["nonfinite_state"].astype(jnp.int32)
        )
        nonfinite_rewards += jnp.sum(
            ~jnp.isfinite(state.reward), dtype=jnp.int32
        )
        nonfinite_observations += jnp.sum(
            ~jnp.isfinite(state.obs), dtype=jnp.int32
        )
        max_abs_observation = jnp.maximum(
            max_abs_observation,
            jnp.max(jnp.abs(state.obs)),
        )
        min_reward = jnp.minimum(min_reward, jnp.min(state.reward))
        max_reward = jnp.maximum(max_reward, jnp.max(state.reward))
        max_abs_actuator_force = jnp.maximum(
            max_abs_actuator_force,
            jnp.max(jnp.abs(state.data.actuator_force)),
        )
        forward_velocity_total += jnp.mean(
            state.metrics["base_forward_velocity"]
        )
        tilt_total += jnp.mean(state.metrics["tilt_deg"])
        max_tilt_deg = jnp.maximum(
            max_tilt_deg,
            jnp.max(state.metrics["tilt_deg"]),
        )
        for key in reward_metric_names:
            max_abs_reward_components[key] = jnp.maximum(
                max_abs_reward_components[key],
                jnp.max(jnp.abs(state.metrics[key])),
            )
        max_feet_air_time = jnp.maximum(
            max_feet_air_time,
            jnp.max(state.info["feet_air_time"]),
        )
        max_swing_peak = jnp.maximum(
            max_swing_peak,
            jnp.max(state.info["swing_peak"]),
        )
        if has_crawl_reference_error:
            crawl_reference_error = state.metrics["crawl_reference_error_rms"]
            crawl_reference_error_total += jnp.mean(crawl_reference_error)
            max_crawl_reference_error = jnp.maximum(
                max_crawl_reference_error,
                jnp.max(crawl_reference_error),
            )
        if has_crawl_ik_reachable:
            minimum_crawl_ik_reachable = jnp.minimum(
                minimum_crawl_ik_reachable,
                jnp.min(state.metrics["crawl_ik_reachable_fraction"]),
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
    _block_tree(
        (
            state,
            done_count,
            illegal_contact_count,
            nonfinite_state_count,
            nonfinite_rewards,
            nonfinite_observations,
            max_abs_observation,
            min_reward,
            max_reward,
            max_abs_actuator_force,
            forward_velocity_total,
            tilt_total,
            max_tilt_deg,
            max_abs_reward_components,
            max_feet_air_time,
            max_swing_peak,
            crawl_reference_error_total,
            max_crawl_reference_error,
            minimum_crawl_ik_reachable,
        )
    )

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
        "crawl_shift_end_fraction": args.crawl_shift_end_fraction,
        "crawl_lift_start_fraction": args.crawl_lift_start_fraction,
        "crawl_lift_end_fraction": args.crawl_lift_end_fraction,
        "crawl_foot_space": args.crawl_foot_space,
        "crawl_foot_step_length": args.crawl_foot_step_length,
        "crawl_foot_clearance": args.crawl_foot_clearance,
        "crawl_body_shift_x": args.crawl_body_shift_x,
        "crawl_body_shift_y": args.crawl_body_shift_y,
        "crawl_min_air_time": args.crawl_min_air_time,
        "crawl_pose_reference": args.crawl_pose_reference,
        "command_x_override": args.command_x,
        "done_count": int(done_count),
        "illegal_contact_count": int(illegal_contact_count),
        "nonfinite_state_count": int(nonfinite_state_count),
        "nonfinite_rewards": int(nonfinite_rewards),
        "nonfinite_observations": int(nonfinite_observations),
        "max_abs_observation": float(max_abs_observation),
        "min_reward": float(min_reward),
        "max_reward": float(max_reward),
        "max_abs_actuator_force": float(max_abs_actuator_force),
        "max_abs_reward_components": {
            key: float(value) for key, value in max_abs_reward_components.items()
        },
        "max_feet_air_time": float(max_feet_air_time),
        "max_swing_peak": float(max_swing_peak),
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
        "trajectory_mean_forward_velocity": float(
            forward_velocity_total / args.num_steps
        ),
        "mean_forward_displacement": float(
            jnp.mean(state.data.qpos[:, 0] - initial_x)
        ),
        "trajectory_mean_tilt_deg": float(tilt_total / args.num_steps),
        "max_tilt_deg": float(max_tilt_deg),
        "mean_tracking_error": float(jnp.mean(state.metrics["tracking_linear_error"])),
        "zero_actions": args.zero_actions,
        "resample_actions": args.resample_actions,
        "training_wrapper": args.training_wrapper,
    }
    if has_crawl_reference_error:
        report["mean_crawl_reference_error_rms"] = float(
            crawl_reference_error_total / args.num_steps
        )
        report["max_crawl_reference_error_rms"] = float(
            max_crawl_reference_error
        )
    if has_crawl_ik_reachable:
        report["minimum_crawl_ik_reachable_fraction"] = float(
            minimum_crawl_ik_reachable
        )
    print(f"LOCOMOTION_PREFLIGHT {json.dumps(report, sort_keys=True)}", flush=True)
    return int(
        report["observation_size"] != env.observation_size
        or report["action_size"] != 19
        or report["physics_substeps"] > 5
        or report["nonfinite_rewards"] > 0
        or report["nonfinite_observations"] > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
