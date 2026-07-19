#!/usr/bin/env python3
"""ROCm-safe Brax PPO smoke for command-conditioned Go2+Z1 locomotion."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from amd_robo.envs.go2_z1_locomotion import (  # noqa: E402
    FOOT_SITE_NAMES,
    Go2Z1LocomotionEnv,
)
from amd_robo.training.host_loop import plan_brax_host_loop  # noqa: E402
from amd_robo.training.learner_checkpoint import (  # noqa: E402
    load_training_session_checkpoint,
    make_training_session_checkpoint_callback,
)


def _load_config(path: str) -> tuple[dict, str]:
    import yaml

    payload = Path(path).read_bytes()
    return yaml.safe_load(payload), hashlib.sha256(payload).hexdigest()


def _make_env(
    config: dict,
    *,
    command_override=None,
    randomize_reset: bool | None = None,
) -> Go2Z1LocomotionEnv:
    environment = config["environment"]
    reward = config["reward"]
    if randomize_reset is None:
        randomize_reset = environment["randomized_reset"]
    return Go2Z1LocomotionEnv(
        ctrl_dt=environment["control_timestep"],
        action_scale=environment.get("action_scale", 0.25),
        leg_kp=environment.get("leg_kp"),
        leg_kd=environment.get("leg_kd"),
        foot_condim=environment["foot_condim"],
        bound_observations=environment["bounded_observations"],
        command_x_range=environment["command_x_range"],
        zero_command_probability=environment["zero_command_probability"],
        command_override=command_override,
        randomize_reset=randomize_reset,
        gait_cycle_time=environment.get("gait_cycle_time"),
        tracking_sigma=reward["tracking_sigma"],
        tracking_linear_velocity_scale=reward["tracking_linear_velocity_scale"],
        tracking_angular_velocity_scale=reward["tracking_angular_velocity_scale"],
        pose_scale=reward["pose_scale"],
        moving_pose_multiplier=reward.get("moving_pose_multiplier", 1.0),
        vertical_velocity_cost_scale=reward["vertical_velocity_cost_scale"],
        angular_velocity_xy_cost_scale=reward["angular_velocity_xy_cost_scale"],
        orientation_cost_scale=reward["orientation_cost_scale"],
        stand_still_cost_scale=reward["stand_still_cost_scale"],
        torque_cost_scale=reward["torque_cost_scale"],
        action_rate_cost_scale=reward["action_rate_cost_scale"],
        action_magnitude_cost_scale=reward["action_magnitude_cost_scale"],
        arm_action_magnitude_cost_scale=reward["arm_action_magnitude_cost_scale"],
        feet_slip_cost_scale=reward["feet_slip_cost_scale"],
        feet_clearance_cost_scale=reward["feet_clearance_cost_scale"],
        feet_height_cost_scale=reward["feet_height_cost_scale"],
        feet_air_time_scale=reward["feet_air_time_scale"],
        max_foot_height=reward["max_foot_height"],
        trot_contact_scale=reward.get("trot_contact_scale", 0.0),
        trot_swing_height_cost_scale=reward.get(
            "trot_swing_height_cost_scale", 0.0
        ),
        trot_timing_scale=reward.get("trot_timing_scale", 0.0),
        trot_timing_std=reward.get("trot_timing_std", 0.1),
        trot_timing_max_error=reward.get("trot_timing_max_error", 0.2),
        trot_timing_min_air_time=reward.get("trot_timing_min_air_time", 0.0),
        crawl_reference_enabled=environment.get(
            "crawl_reference_enabled", False
        ),
        crawl_stride=environment.get("crawl_stride", 0.08),
        crawl_shift=environment.get("crawl_shift", 0.06),
        crawl_lift=environment.get("crawl_lift", 0.45),
        crawl_shift_end_fraction=environment.get(
            "crawl_shift_end_fraction", 0.3
        ),
        crawl_lift_start_fraction=environment.get(
            "crawl_lift_start_fraction", 0.3
        ),
        crawl_lift_end_fraction=environment.get(
            "crawl_lift_end_fraction", 0.8
        ),
        crawl_foot_space_enabled=environment.get(
            "crawl_foot_space_enabled", False
        ),
        crawl_foot_step_length=environment.get(
            "crawl_foot_step_length", 0.08
        ),
        crawl_foot_clearance=environment.get(
            "crawl_foot_clearance", 0.04
        ),
        crawl_body_shift_x=environment.get("crawl_body_shift_x", 0.02),
        crawl_body_shift_y=environment.get("crawl_body_shift_y", 0.02),
        crawl_min_air_time=environment.get("crawl_min_air_time", 0.07),
        crawl_pose_reference_enabled=environment.get(
            "crawl_pose_reference_enabled", False
        ),
        termination_cost_scale=reward["termination_cost_scale"],
        illegal_contact_cost_scale=reward["illegal_contact_cost_scale"],
        workspace_limit=environment["workspace_limit"],
    )


def _sequential_eval(env, action_fns, *, n_envs: int, n_steps: int, seed: int):
    from mujoco_playground import wrapper

    wrapped = wrapper.wrap_for_brax_training(
        env, episode_length=n_steps + 1, action_repeat=1
    )
    reset_fn = jax.jit(wrapped.reset)
    step_fn = jax.jit(wrapped.step)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    initial_state = reset_fn(keys)

    results = {}
    for name, action_fn in action_fns.items():
        state = initial_state
        initial_x = state.data.qpos[:, 0]
        reward_metric_names = tuple(
            key for key in state.metrics if key.startswith("reward/")
        )
        reward_component_totals = {
            key: jnp.zeros(()) for key in reward_metric_names
        }
        reward_total = jnp.zeros(())
        forward_velocity_total = jnp.zeros(())
        tracking_error_total = jnp.zeros(())
        tilt_total = jnp.zeros(())
        crawl_reference_error_total = jnp.zeros(())
        has_crawl_reference_error = (
            "crawl_reference_error_rms" in state.metrics
        )
        done_count = jnp.zeros((), dtype=jnp.int32)
        illegal_contact_count = jnp.zeros((), dtype=jnp.int32)
        nonfinite_state_count = jnp.zeros((), dtype=jnp.int32)
        action_square_total = jnp.zeros(())
        leg_action_square_total = jnp.zeros(())
        arm_action_square_total = jnp.zeros(())
        saturated_action_total = jnp.zeros(())
        saturated_leg_action_total = jnp.zeros(())
        contact_duty_total = jnp.zeros(len(FOOT_SITE_NAMES))
        liftoff_count = jnp.zeros(len(FOOT_SITE_NAMES), dtype=jnp.int32)
        touchdown_count = jnp.zeros(len(FOOT_SITE_NAMES), dtype=jnp.int32)
        sustained_swing_count = jnp.zeros(
            len(FOOT_SITE_NAMES), dtype=jnp.int32
        )
        foot_height_total = jnp.zeros(len(FOOT_SITE_NAMES))
        foot_height_max = jnp.full((len(FOOT_SITE_NAMES),), -jnp.inf)
        completed_swing_air_time_total = jnp.zeros(len(FOOT_SITE_NAMES))
        completed_swing_peak_total = jnp.zeros(len(FOOT_SITE_NAMES))
        completed_swing_peak_max = jnp.full(
            (len(FOOT_SITE_NAMES),), -jnp.inf
        )
        diagonal_pair_mismatch_total = jnp.zeros(())
        diagonal_group_opposition_total = jnp.zeros(())
        diagonal_two_contact_total = jnp.zeros(())
        adjacent_two_contact_total = jnp.zeros(())
        lateral_two_contact_total = jnp.zeros(())
        front_hind_two_contact_total = jnp.zeros(())
        all_four_contact_total = jnp.zeros(())
        zero_contact_total = jnp.zeros(())
        previous_contact = None
        for _ in range(n_steps):
            prior_air_time = state.info["feet_air_time"]
            prior_swing_peak = state.info["swing_peak"]
            actions = action_fn(state.obs)
            state = step_fn(state, actions)
            reward_total += jnp.mean(state.reward)
            for key in reward_metric_names:
                reward_component_totals[key] += jnp.mean(state.metrics[key])
            forward_velocity_total += jnp.mean(state.metrics["base_forward_velocity"])
            tracking_error_total += jnp.mean(state.metrics["tracking_linear_error"])
            tilt_total += jnp.mean(state.metrics["tilt_deg"])
            if has_crawl_reference_error:
                crawl_reference_error_total += jnp.mean(
                    state.metrics["crawl_reference_error_rms"]
                )
            done_count += jnp.sum(state.done.astype(jnp.int32))
            illegal_contact_count += jnp.sum(
                state.metrics["illegal_contact"].astype(jnp.int32)
            )
            nonfinite_state_count += jnp.sum(
                state.metrics["nonfinite_state"].astype(jnp.int32)
            )
            action_square_total += jnp.mean(actions * actions)
            leg_action_square_total += jnp.mean(actions[:, :12] ** 2)
            arm_action_square_total += jnp.mean(actions[:, 12:] ** 2)
            saturated_action_total += jnp.mean(jnp.abs(actions) >= 0.95)
            saturated_leg_action_total += jnp.mean(
                jnp.abs(actions[:, :12]) >= 0.95
            )
            foot_contact = state.info["last_contact"]
            if "crawl_sustained_touchdown" in state.info:
                sustained_swing_count += jnp.sum(
                    state.info["crawl_sustained_touchdown"],
                    axis=0,
                    dtype=jnp.int32,
                )
            contact_duty_total += jnp.mean(foot_contact, axis=0)
            foot_height = state.data.site_xpos[:, env._foot_site_ids, -1]
            foot_height_total += jnp.mean(foot_height, axis=0)
            foot_height_max = jnp.maximum(
                foot_height_max, jnp.max(foot_height, axis=0)
            )
            if previous_contact is not None:
                liftoff = previous_contact & ~foot_contact
                touchdown = ~previous_contact & foot_contact
                liftoff_count += jnp.sum(
                    liftoff, axis=0, dtype=jnp.int32
                )
                touchdown_count += jnp.sum(
                    touchdown, axis=0, dtype=jnp.int32
                )
                completed_air_time = prior_air_time + env.dt
                completed_swing_peak = jnp.maximum(
                    prior_swing_peak, foot_height
                )
                completed_swing_air_time_total += jnp.sum(
                    jnp.where(touchdown, completed_air_time, 0.0), axis=0
                )
                completed_swing_peak_total += jnp.sum(
                    jnp.where(touchdown, completed_swing_peak, 0.0), axis=0
                )
                completed_swing_peak_max = jnp.maximum(
                    completed_swing_peak_max,
                    jnp.max(
                        jnp.where(touchdown, completed_swing_peak, -jnp.inf),
                        axis=0,
                    ),
                )

            fl, fr, rl, rr = (foot_contact[:, index] for index in range(4))
            contact_count = jnp.sum(foot_contact, axis=-1)
            diagonal_two = (contact_count == 2) & ((fl & rr) | (fr & rl))
            adjacent_two = (contact_count == 2) & ~diagonal_two
            lateral_two = (contact_count == 2) & ((fl & rl) | (fr & rr))
            front_hind_two = (contact_count == 2) & ((fl & fr) | (rl & rr))
            diagonal_pair_mismatch_total += jnp.mean(
                (
                    jnp.logical_xor(fl, rr).astype(jnp.float32)
                    + jnp.logical_xor(fr, rl).astype(jnp.float32)
                )
                / 2.0
            )
            diagonal_group_opposition_total += jnp.mean(
                (
                    jnp.logical_xor(fl, fr).astype(jnp.float32)
                    + jnp.logical_xor(rl, rr).astype(jnp.float32)
                )
                / 2.0
            )
            diagonal_two_contact_total += jnp.mean(diagonal_two)
            adjacent_two_contact_total += jnp.mean(adjacent_two)
            lateral_two_contact_total += jnp.mean(lateral_two)
            front_hind_two_contact_total += jnp.mean(front_hind_two)
            all_four_contact_total += jnp.mean(contact_count == 4)
            zero_contact_total += jnp.mean(contact_count == 0)
            previous_contact = foot_contact
        results[name] = {
            "mean_reward": float(reward_total / n_steps),
            "mean_forward_velocity": float(forward_velocity_total / n_steps),
            "mean_tracking_error": float(tracking_error_total / n_steps),
            "mean_tilt_deg": float(tilt_total / n_steps),
            "mean_forward_displacement": float(
                jnp.mean(state.data.qpos[:, 0] - initial_x)
            ),
            "action_rms": float(jnp.sqrt(action_square_total / n_steps)),
            "leg_action_rms": float(jnp.sqrt(leg_action_square_total / n_steps)),
            "arm_action_rms": float(jnp.sqrt(arm_action_square_total / n_steps)),
            "action_saturation_fraction": float(
                saturated_action_total / n_steps
            ),
            "leg_action_saturation_fraction": float(
                saturated_leg_action_total / n_steps
            ),
            "done_count": int(done_count),
            "illegal_contact_count": int(illegal_contact_count),
            "nonfinite_state_count": int(nonfinite_state_count),
            "gait_diagnostics_valid": int(done_count) == 0,
            "diagonal_pair_mismatch_fraction": float(
                diagonal_pair_mismatch_total / n_steps
            ),
            "diagonal_group_opposition_fraction": float(
                diagonal_group_opposition_total / n_steps
            ),
            "diagonal_two_contact_fraction": float(
                diagonal_two_contact_total / n_steps
            ),
            "adjacent_two_contact_fraction": float(
                adjacent_two_contact_total / n_steps
            ),
            "lateral_two_contact_fraction": float(
                lateral_two_contact_total / n_steps
            ),
            "front_hind_two_contact_fraction": float(
                front_hind_two_contact_total / n_steps
            ),
            "all_four_contact_fraction": float(
                all_four_contact_total / n_steps
            ),
            "zero_contact_fraction": float(zero_contact_total / n_steps),
        }
        if has_crawl_reference_error:
            results[name]["mean_crawl_reference_error_rms"] = float(
                crawl_reference_error_total / n_steps
            )
        for index, foot_name in enumerate(FOOT_SITE_NAMES):
            completed_swing_count = jnp.maximum(touchdown_count[index], 1)
            results[name].update(
                {
                    f"{foot_name}_contact_duty": float(
                        contact_duty_total[index] / n_steps
                    ),
                    f"{foot_name}_liftoffs_per_env": float(
                        liftoff_count[index] / n_envs
                    ),
                    f"{foot_name}_touchdowns_per_env": float(
                        touchdown_count[index] / n_envs
                    ),
                    f"{foot_name}_sustained_swings_per_env": float(
                        sustained_swing_count[index] / n_envs
                    ),
                    f"{foot_name}_completed_swing_mean_air_time": float(
                        completed_swing_air_time_total[index]
                        / completed_swing_count
                    ),
                    f"{foot_name}_completed_swing_mean_peak": float(
                        completed_swing_peak_total[index]
                        / completed_swing_count
                    ),
                    f"{foot_name}_completed_swing_max_peak": float(
                        jnp.where(
                            touchdown_count[index] > 0,
                            completed_swing_peak_max[index],
                            0.0,
                        )
                    ),
                    f"{foot_name}_mean_height": float(
                        foot_height_total[index] / n_steps
                    ),
                    f"{foot_name}_max_height": float(foot_height_max[index]),
                }
            )
        results[name].update(
            {
                f"mean_{key.replace('/', '_')}": float(total / n_steps)
                for key, total in reward_component_totals.items()
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/locomotion.yaml")
    parser.add_argument("--num-timesteps", type=int)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--learning-rate-schedule")
    parser.add_argument("--desired-kl", type=float)
    parser.add_argument("--no-normalize-observations", action="store_true")
    parser.add_argument("--params-in")
    parser.add_argument("--params-out")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--training-state-dir")
    parser.add_argument("--resume-training-state")
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument(
        "--migrate-missing-nonfinite-state",
        action="store_true",
        help=(
            "Migrate a pre-guard full-session checkpoint whose only missing "
            "metric is the derived nonfinite_state value and its episode sum."
        ),
    )
    parser.add_argument(
        "--migrate-missing-arm-action-magnitude",
        action="store_true",
        help=(
            "Also migrate a full-session checkpoint created before the "
            "reward/arm_action_magnitude metric was introduced."
        ),
    )
    args = parser.parse_args()

    config, config_sha256 = _load_config(args.config)
    ppo_config = config["ppo"]
    guardrails = config["rocm_guardrails"]
    evaluation = config["manual_evaluation"]
    checkpoint = config["checkpoint"]
    checkpoint_interval_steps = (
        checkpoint["interval_steps"]
        if args.checkpoint_interval_steps is None
        else args.checkpoint_interval_steps
    )
    num_timesteps = (
        ppo_config["num_timesteps"]
        if args.num_timesteps is None
        else args.num_timesteps
    )
    episode_length = (
        ppo_config["episode_length"]
        if args.episode_length is None
        else args.episode_length
    )
    num_envs = ppo_config["num_envs"] if args.num_envs is None else args.num_envs
    batch_size = (
        ppo_config["batch_size"] if args.batch_size is None else args.batch_size
    )
    num_minibatches = (
        ppo_config["num_minibatches"]
        if args.num_minibatches is None
        else args.num_minibatches
    )
    learning_rate = (
        ppo_config["learning_rate"]
        if args.learning_rate is None
        else args.learning_rate
    )
    learning_rate_schedule = (
        ppo_config["learning_rate_schedule"]
        if args.learning_rate_schedule is None
        else args.learning_rate_schedule
    )
    desired_kl = (
        ppo_config["desired_kl"] if args.desired_kl is None else args.desired_kl
    )
    normalize_observations = (
        ppo_config["normalize_observations"] and not args.no_normalize_observations
    )
    policy_hidden_layer_sizes = tuple(
        ppo_config.get("policy_hidden_layer_sizes", (32, 32, 32, 32))
    )
    value_hidden_layer_sizes = tuple(
        ppo_config.get("value_hidden_layer_sizes", (256, 256, 256, 256, 256))
    )
    if args.eval_only:
        eval_sources = int(bool(args.params_in)) + int(
            bool(args.resume_training_state)
        )
        if eval_sources != 1:
            parser.error(
                "--eval-only requires exactly one of --params-in or "
                "--resume-training-state"
            )
        num_timesteps = 0
    if args.params_in and args.resume_training_state:
        parser.error("--params-in and --resume-training-state are mutually exclusive")
    if (
        args.migrate_missing_nonfinite_state
        or args.migrate_missing_arm_action_magnitude
    ) and not args.resume_training_state:
        parser.error(
            "training-session metric migration requires --resume-training-state"
        )
    if args.eval_only and (args.skip_eval or args.training_state_dir):
        parser.error(
            "--eval-only cannot be combined with checkpoint output or --skip-eval"
        )
    if (
        num_timesteps < 0
        or episode_length <= 0
        or learning_rate <= 0.0
        or num_envs <= 0
        or batch_size <= 0
        or num_minibatches <= 0
        or checkpoint_interval_steps <= 0
        or not policy_hidden_layer_sizes
        or not value_hidden_layer_sizes
        or any(size <= 0 for size in policy_hidden_layer_sizes)
        or any(size <= 0 for size in value_hidden_layer_sizes)
    ):
        parser.error(
            "timesteps must be non-negative; episode length and learning rate "
            "and PPO batch, network, and checkpoint dimensions must be positive"
        )
    if batch_size * num_minibatches % num_envs:
        parser.error("batch_size * num_minibatches must be divisible by num_envs")

    from amd_robo.platform import _compat

    _compat.apply_brax_compat()
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    env = _make_env(config)
    max_substeps = guardrails["max_physics_substeps_per_control"]
    if env.n_substeps > max_substeps:
        raise ValueError(
            f"physics substeps per control ({env.n_substeps}) exceed the ROCm "
            f"guardrail ({max_substeps})"
        )

    env_steps_per_training_step = (
        batch_size * ppo_config["unroll_length"] * num_minibatches
    )
    actual_timesteps = host_calls = training_scan = 0
    brax_num_evals = 1
    if not args.eval_only:
        host_loop = plan_brax_host_loop(
            num_timesteps=num_timesteps,
            env_steps_per_training_step=env_steps_per_training_step,
            max_training_steps_per_call=guardrails["max_training_steps_per_host_call"],
        )
        actual_timesteps = host_loop.actual_timesteps
        host_calls = host_loop.host_calls
        training_scan = host_loop.training_steps_per_call
        brax_num_evals = host_loop.brax_num_evals

    print(
        "locomotion smoke: "
        f"num_envs={num_envs} batch_size={batch_size} "
        f"num_minibatches={num_minibatches} "
        f"episode_length={episode_length} "
        f"num_timesteps={num_timesteps} "
        f"actual_timesteps={actual_timesteps} "
        f"host_calls={host_calls} training_scan={training_scan} "
        f"control_timestep={env.dt} physics_substeps={env.n_substeps} "
        f"action_size={env.action_size} observation_size={env.observation_size} "
        f"learning_rate={learning_rate} "
        f"learning_rate_schedule={learning_rate_schedule} "
        f"desired_kl={desired_kl} "
        f"normalize_observations={normalize_observations} "
        f"policy_hidden_layer_sizes={policy_hidden_layer_sizes} "
        f"value_hidden_layer_sizes={value_hidden_layer_sizes} "
        f"checkpoint_interval_steps={checkpoint_interval_steps} "
        f"config={args.config} config_sha256={config_sha256} "
        f"seed={config['seed']} "
        f"matmul_precision="
        f"{os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', 'default')} "
        "run_evals=False",
        flush=True,
    )

    def progress(step, training_metrics):
        fields = []
        for key in (
            "training/walltime",
            "training/sps",
            "training/kl_mean",
            "training/learning_rate",
            "training/policy_dist_max_loc",
            "training/policy_dist_min_std",
            "training/policy_loss",
            "training/total_loss",
            "training/v_loss",
        ):
            if key in training_metrics:
                fields.append(f"{key}={training_metrics[key]}")
        print(
            f"LOCOMOTION_TRAINING_PROGRESS step={step} {' '.join(fields)}",
            flush=True,
        )

    restore_params = brax_model.load_params(args.params_in) if args.params_in else None
    if args.params_in:
        print(f"PARAMS_LOADED path={args.params_in}", flush=True)

    def training_session_fn(*unused):
        return None

    restore_training_session_fn = None
    supports_training_session = (
        "training_session_fn" in inspect.signature(ppo.train).parameters
    )
    if (
        args.training_state_dir or args.resume_training_state
    ) and not supports_training_session:
        raise RuntimeError(
            "Brax training-session API patch is missing; rerun scripts/rgc_setup.sh"
        )
    if args.training_state_dir:
        metadata = {
            "config": args.config,
            "config_sha256": config_sha256,
            "num_timesteps": actual_timesteps,
            "episode_length": episode_length,
            "num_envs": num_envs,
            "batch_size": batch_size,
            "num_minibatches": num_minibatches,
            "control_timestep": env.dt,
            "physics_substeps": env.n_substeps,
            "host_calls": host_calls,
            "training_scan": training_scan,
            "learning_rate": learning_rate,
            "learning_rate_schedule": learning_rate_schedule,
            "desired_kl": desired_kl,
            "normalize_observations": normalize_observations,
            "policy_hidden_layer_sizes": policy_hidden_layer_sizes,
            "value_hidden_layer_sizes": value_hidden_layer_sizes,
            "checkpoint_interval_steps": checkpoint_interval_steps,
            "seed": config["seed"],
        }
        training_session_fn = make_training_session_checkpoint_callback(
            args.training_state_dir,
            interval_steps=checkpoint_interval_steps,
            metadata=metadata,
            announce=lambda message: print(message, flush=True),
        )
    if args.resume_training_state:

        def restore_training_session_fn(template):
            migration_metric_names = []
            if args.migrate_missing_arm_action_magnitude:
                migration_metric_names.append("reward/arm_action_magnitude")
            if args.migrate_missing_nonfinite_state:
                migration_metric_names.append("nonfinite_state")
            if migration_metric_names:
                training_state, env_state, local_key, key_envs = template
                episode_metrics = env_state.info.get("episode_metrics", {})
                missing_current_metrics = [
                    name
                    for name in migration_metric_names
                    if name not in env_state.metrics or name not in episode_metrics
                ]
                if missing_current_metrics:
                    raise RuntimeError(
                        "current training-session template is missing migration "
                        f"targets: {missing_current_metrics}"
                    )
                initialized_metrics = {
                    name: env_state.metrics[name]
                    for name in migration_metric_names
                }
                initialized_episode_metrics = {
                    name: episode_metrics[name]
                    for name in migration_metric_names
                }
                migration_metric_set = set(migration_metric_names)
                legacy_metrics = {
                    name: value
                    for name, value in env_state.metrics.items()
                    if name not in migration_metric_set
                }
                legacy_episode_metrics = {
                    name: value
                    for name, value in episode_metrics.items()
                    if name not in migration_metric_set
                }
                legacy_template = (
                    training_state,
                    env_state.replace(
                        metrics=legacy_metrics,
                        info={
                            **env_state.info,
                            "episode_metrics": legacy_episode_metrics,
                        },
                    ),
                    local_key,
                    key_envs,
                )
                manifest = json.loads(
                    (
                        Path(args.resume_training_state)
                        / "manifest.json"
                    ).read_text()
                )
                current_leaf_count = len(jax.tree_util.tree_leaves(template))
                legacy_leaf_count = len(jax.tree_util.tree_leaves(legacy_template))
                expected_added_leaves = 2 * len(migration_metric_names)
                if (
                    current_leaf_count
                    != legacy_leaf_count + expected_added_leaves
                    or manifest.get("leaf_count") != legacy_leaf_count
                ):
                    raise RuntimeError(
                        "refusing training-session migration: expected only "
                        f"{migration_metric_names} and their episode sums, "
                        f"manifest={manifest.get('leaf_count')} "
                        f"legacy={legacy_leaf_count} current={current_leaf_count}"
                    )
                restored_legacy = load_training_session_checkpoint(
                    args.resume_training_state,
                    training_session_template=legacy_template,
                )
                (
                    restored_training_state,
                    restored_env_state,
                    restored_local_key,
                    restored_key_envs,
                ) = restored_legacy
                restored = (
                    restored_training_state,
                    restored_env_state.replace(
                        metrics={
                            **restored_env_state.metrics,
                            **initialized_metrics,
                        },
                        info={
                            **restored_env_state.info,
                            "episode_metrics": {
                                **restored_env_state.info["episode_metrics"],
                                **initialized_episode_metrics,
                            },
                        },
                    ),
                    restored_local_key,
                    restored_key_envs,
                )
                if (
                    jax.tree_util.tree_structure(restored)
                    != jax.tree_util.tree_structure(template)
                ):
                    raise RuntimeError(
                        "migrated training-session PyTree does not match "
                        "the current template"
                    )
                print(
                    "TRAINING_SESSION_MIGRATED "
                    f"added_metrics={','.join(migration_metric_names)} "
                    f"added_leaves={expected_added_leaves} "
                    f"legacy_leaves={legacy_leaf_count} "
                    f"current_leaves={current_leaf_count}",
                    flush=True,
                )
            else:
                restored = load_training_session_checkpoint(
                    args.resume_training_state,
                    training_session_template=template,
                )
            print(
                f"TRAINING_SESSION_LOADED path={args.resume_training_state}",
                flush=True,
            )
            return restored

    training_state_kwargs = {}
    if supports_training_session:
        training_state_kwargs = {
            "training_session_fn": training_session_fn,
            "restore_training_session_fn": restore_training_session_fn,
        }

    event_prefix = "EVAL_ONLY" if args.eval_only else "LOCOMOTION_TRAINING"
    print(f"{event_prefix}_START timesteps={num_timesteps}", flush=True)
    make_policy, params, metrics = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        max_devices_per_host=1,
        num_envs=num_envs,
        episode_length=episode_length,
        action_repeat=1,
        learning_rate=learning_rate,
        learning_rate_schedule=learning_rate_schedule,
        learning_rate_schedule_min_lr=min(1e-5, learning_rate),
        learning_rate_schedule_max_lr=learning_rate,
        desired_kl=desired_kl,
        entropy_cost=ppo_config["entropy_cost"],
        discounting=ppo_config["discounting"],
        unroll_length=ppo_config["unroll_length"],
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=ppo_config["num_updates_per_batch"],
        max_grad_norm=ppo_config["max_grad_norm"],
        normalize_observations=normalize_observations,
        network_factory=functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
        ),
        num_evals=brax_num_evals,
        num_eval_envs=4,
        run_evals=False,
        progress_fn=progress,
        seed=config["seed"],
        restore_params=restore_params,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **training_state_kwargs,
    )
    print(f"{event_prefix}_DONE", flush=True)
    for key in sorted(metrics):
        print(f"METRIC {key}: {metrics[key]}", flush=True)
    if args.params_out:
        brax_model.save_params(args.params_out, params)
        print(f"PARAMS_SAVED path={args.params_out}", flush=True)
    if args.skip_eval:
        print("LOCOMOTION_SMOKE_DONE eval=skipped", flush=True)
        return 0

    policy = make_policy(params, deterministic=True)
    trained_action = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])

    def zero_action(obs):
        return jnp.zeros((obs.shape[0], env.action_size))

    eval_env = _make_env(
        config,
        command_override=evaluation["fixed_command"],
        randomize_reset=False,
    )
    print("LOCOMOTION_EVAL_START implementation=sequential_python_loop", flush=True)
    results = _sequential_eval(
        eval_env,
        {"baseline": zero_action, "trained": trained_action},
        n_envs=evaluation["num_envs"],
        n_steps=evaluation["num_steps"],
        seed=evaluation["seed"],
    )
    for policy_name, policy_metrics in results.items():
        print(
            f"LOCOMOTION_EVAL policy={policy_name} "
            + " ".join(f"{key}={value}" for key, value in policy_metrics.items()),
            flush=True,
        )
    print("LOCOMOTION_SMOKE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
