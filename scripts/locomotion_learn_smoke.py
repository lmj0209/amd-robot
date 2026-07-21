#!/usr/bin/env python3
"""ROCm-safe Brax PPO smoke for command-conditioned Go2+Z1 locomotion."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from amd_robo.contracts import TaskPhase  # noqa: E402
from amd_robo.envs.go2_z1_locomotion import (  # noqa: E402
    FOOT_SITE_NAMES,
    Go2Z1LocomotionEnv,
)
from amd_robo.envs.go2_z1_push import Go2Z1PushEnv  # noqa: E402
from amd_robo.evaluation.determinism import (  # noqa: E402
    first_trace_divergence,
    normalized_discrete_outcome,
    push_evaluation_execution_mode,
    sha256_path,
    trace_summary,
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
    task: str = "locomotion",
) -> Go2Z1LocomotionEnv:
    environment = config["environment"]
    reward = config["reward"]
    if randomize_reset is None:
        randomize_reset = environment["randomized_reset"]
    if task == "push" and command_override is None:
        command_override = (0.025, 0.0, 0.0)
    env_class = (
        Go2Z1PushEnv if task == "push" else Go2Z1LocomotionEnv
    )
    task_kwargs = config.get("push", {}) if task == "push" else {}
    return env_class(
        ctrl_dt=environment["control_timestep"],
        action_scale=environment.get("action_scale", 0.25),
        arm_action_scale=environment.get("arm_action_scale"),
        leg_kp=environment.get("leg_kp"),
        leg_kd=environment.get("leg_kd"),
        arm_kp=environment.get("arm_kp"),
        arm_kd=environment.get("arm_kd"),
        mask_arm=environment.get("mask_arm", True),
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
        **task_kwargs,
    )


def _support_contact_masks(foot_contact):
    """Returns support-count masks for a batch of four-foot contacts."""
    contact_count = jnp.sum(foot_contact, axis=-1, dtype=jnp.int32)
    return (
        contact_count,
        contact_count >= 3,
        contact_count == 4,
        contact_count == 0,
    )


def _push_failure_counts(max_phase, success) -> dict[str, int]:
    """Classify one fixed-horizon Push episode per environment."""
    failed = ~jnp.asarray(success, dtype=bool)
    max_phase = jnp.asarray(max_phase, dtype=jnp.int32)
    return {
        f"push_failure_{phase.name.lower()}_count": int(
            jnp.sum(failed & (max_phase == int(phase)))
        )
        for phase in TaskPhase
    }


def _audit_fields(state, actions) -> dict[str, jax.Array]:
    """Selects state needed to locate the first meaningful rollout split."""

    return {
        "action": actions,
        "command": state.info["command"],
        "ctrl": state.data.ctrl,
        "done": state.done,
        "end_effector_distance": state.metrics[
            "end_effector_to_push_distance"
        ],
        "object_displacement": state.metrics["object_displacement"],
        "object_goal_distance": state.metrics["object_to_goal_distance"],
        "object_qpos": state.info["object_qpos"],
        "object_qvel": state.info["object_qvel"],
        "obs": state.obs,
        "phase": state.info["phase"],
        "qacc": state.data.qacc,
        "qpos": state.data.qpos,
        "qvel": state.data.qvel,
        "reward": state.reward,
        "sensordata": state.data.sensordata,
        "steps": state.info["steps"],
        "success": state.metrics["success"],
    }


def _initial_audit_fields(state, keys) -> dict[str, np.ndarray]:
    fields = {
        "command": state.info["command"],
        "env_rng": state.info["rng"],
        "input_key": keys,
        "object_qpos": state.info["object_qpos"],
        "object_qvel": state.info["object_qvel"],
        "obs": state.obs,
        "phase": state.info["phase"],
        "qpos": state.data.qpos,
        "qvel": state.data.qvel,
    }
    return {
        name: np.asarray(jax.device_get(value))[None, ...]
        for name, value in fields.items()
    }


def _policy_audit_sequence(action_fns, repeats: int):
    if repeats < 2:
        raise ValueError("determinism audit requires at least two repeats")
    counters = {"baseline": 0, "trained": 0}
    sequence = []
    for repeat in range(repeats):
        order = (
            ("trained", "baseline")
            if repeat % 2 == 0
            else ("baseline", "trained")
        )
        for policy_name in order:
            run_index = counters[policy_name]
            counters[policy_name] += 1
            sequence.append(
                (
                    f"{policy_name}_{run_index}",
                    policy_name,
                    action_fns[policy_name],
                )
            )
    return sequence


def _discrete_push_outcome(run: dict[str, object]) -> dict[str, object]:
    return normalized_discrete_outcome(
        run,
        (
            "success_by_env",
            "terminal_by_env",
            "abnormal_by_env",
            "max_phase_by_env",
        ),
    )


def _comparison_report(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    sampled_control_steps: list[int],
) -> dict[str, object]:
    exact = first_trace_divergence(reference, candidate)
    tolerant = first_trace_divergence(
        reference,
        candidate,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )

    def serialize(divergence):
        if divergence is None:
            return None
        report = divergence.to_dict()
        report["sampled_control_step"] = sampled_control_steps[
            divergence.step_index
        ]
        return report

    return {
        "exact_match": exact is None,
        "match_at_1e-6": tolerant is None,
        "first_exact_divergence": serialize(exact),
        "first_1e-6_divergence": serialize(tolerant),
    }


def _load_npz_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _push_determinism_audit(
    env,
    action_fns,
    *,
    output_dir: str | Path,
    reference_dir: str | Path | None,
    config_path: str,
    config_sha256: str,
    params_path: str,
    n_envs: int,
    n_steps: int,
    seed: int,
    repeats: int,
    trace_stride: int,
) -> dict[str, object]:
    """Runs interleaved policies from one immutable reset and saves traces."""

    from mujoco_playground import wrapper

    if trace_stride <= 0:
        raise ValueError("determinism trace stride must be positive")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {destination}")
    destination.mkdir(parents=True)

    wrapped = wrapper.wrap_for_brax_training(
        env,
        episode_length=n_steps + 1,
        action_repeat=1,
        full_reset=True,
    )
    reset_fn = jax.jit(wrapped.reset)
    step_fn = jax.jit(wrapped.step)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    initial_state = reset_fn(keys)
    initial_fields = _initial_audit_fields(initial_state, keys)
    initial_summary = trace_summary(initial_fields)
    initial_trace_file = "initial_state.npz"
    np.savez_compressed(destination / initial_trace_file, **initial_fields)
    sampled_control_steps = list(range(1, n_steps + 1, trace_stride))
    if sampled_control_steps[-1] != n_steps:
        sampled_control_steps.append(n_steps)

    commit = _git_head()
    params_sha256 = sha256_path(params_path)
    external_manifest = None
    external_initial_comparison = None
    external_traces = {}
    external_outcomes = {}
    if reference_dir is not None:
        reference_root = Path(reference_dir)
        external_manifest = json.loads(
            (reference_root / "manifest.json").read_text()
        )
        expected_contract = {
            "commit": commit,
            "config_sha256": config_sha256,
            "params_sha256": params_sha256,
            "seed": seed,
            "num_envs": n_envs,
            "num_steps": n_steps,
            "trace_stride": trace_stride,
        }
        actual_contract = {
            name: external_manifest[name] for name in expected_contract
        }
        if actual_contract != expected_contract:
            raise ValueError(
                "external audit contract differs: "
                f"expected={expected_contract} actual={actual_contract}"
            )
        external_initial_trace = _load_npz_trace(
            reference_root / external_manifest["initial_state_trace_file"]
        )
        external_initial_comparison = _comparison_report(
            external_initial_trace,
            initial_fields,
            [0],
        )
        for policy_name, reference_name in external_manifest[
            "policy_references"
        ].items():
            run = next(
                item
                for item in external_manifest["runs"]
                if item["name"] == reference_name
            )
            external_traces[policy_name] = _load_npz_trace(
                reference_root / run["trace_file"]
            )
            external_outcomes[policy_name] = _discrete_push_outcome(run)

    sequence = _policy_audit_sequence(action_fns, repeats)
    policy_reference_traces = {}
    policy_reference_outcomes = {}
    policy_reference_names = {}
    runs = []
    for run_name, policy_name, action_fn in sequence:
        state = initial_state
        active = jnp.ones((n_envs,), dtype=bool)
        success = jnp.zeros((n_envs,), dtype=bool)
        terminal = jnp.zeros((n_envs,), dtype=bool)
        max_phase = jnp.full(
            (n_envs,), int(TaskPhase.APPROACH), dtype=jnp.int32
        )
        max_object_speed = jnp.zeros((n_envs,))
        peak_step = jnp.full((n_envs,), -1, dtype=jnp.int32)
        first_motion_step = jnp.full((n_envs,), -1, dtype=jnp.int32)
        final_goal_distance = state.metrics["object_to_goal_distance"]
        trace_lists = None

        for control_step in range(1, n_steps + 1):
            actions = action_fn(state.obs)
            state = step_fn(state, actions)
            current_active = active
            phase = state.metrics["task_phase"].astype(jnp.int32)
            max_phase = jnp.where(
                current_active, jnp.maximum(max_phase, phase), max_phase
            )
            object_speed = jnp.linalg.norm(
                state.info["object_qvel"][:, :2], axis=-1
            )
            speed_peak = current_active & (object_speed > max_object_speed)
            peak_step = jnp.where(speed_peak, control_step, peak_step)
            max_object_speed = jnp.where(
                current_active,
                jnp.maximum(max_object_speed, object_speed),
                max_object_speed,
            )
            first_motion_step = jnp.where(
                current_active
                & (first_motion_step < 0)
                & (state.metrics["object_displacement"] >= 1e-3),
                control_step,
                first_motion_step,
            )
            final_goal_distance = jnp.where(
                current_active,
                state.metrics["object_to_goal_distance"],
                final_goal_distance,
            )
            terminal_now = current_active & state.done.astype(bool)
            success = success | (
                terminal_now & (state.metrics["success"] > 0.0)
            )
            terminal = terminal | terminal_now
            active = current_active & ~terminal_now

            if control_step % trace_stride == 1 % trace_stride or (
                control_step == n_steps
            ):
                fields = _audit_fields(state, actions)
                if trace_lists is None:
                    trace_lists = {name: [] for name in fields}
                for name, value in fields.items():
                    trace_lists[name].append(value)

        if trace_lists is None:
            raise RuntimeError("determinism audit did not sample any trace fields")
        trace = {
            name: np.stack(
                [np.asarray(jax.device_get(value)) for value in values],
                axis=0,
            )
            for name, values in trace_lists.items()
        }
        if next(iter(trace.values())).shape[0] != len(sampled_control_steps):
            raise RuntimeError("determinism audit sampled unexpected step count")
        trace_file = f"{run_name}.npz"
        np.savez_compressed(destination / trace_file, **trace)

        run = {
            "name": run_name,
            "policy": policy_name,
            "trace_file": trace_file,
            "trace": trace_summary(trace),
            "success_by_env": tuple(int(value) for value in success),
            "terminal_by_env": tuple(int(value) for value in terminal),
            "abnormal_by_env": tuple(
                int(value) for value in (terminal & ~success)
            ),
            "horizon_incomplete_by_env": tuple(
                int(value) for value in active
            ),
            "max_phase_by_env": tuple(int(value) for value in max_phase),
            "max_object_speed_by_env": tuple(
                float(value) for value in max_object_speed
            ),
            "peak_step_by_env": tuple(int(value) for value in peak_step),
            "first_motion_step_by_env": tuple(
                int(value) for value in first_motion_step
            ),
            "final_goal_distance_by_env": tuple(
                float(value) for value in final_goal_distance
            ),
        }
        if policy_name not in policy_reference_traces:
            policy_reference_traces[policy_name] = trace
            policy_reference_outcomes[policy_name] = _discrete_push_outcome(run)
            policy_reference_names[policy_name] = run_name
            run["in_process_comparison"] = None
        else:
            run["in_process_comparison"] = _comparison_report(
                policy_reference_traces[policy_name],
                trace,
                sampled_control_steps,
            )
        if policy_name in external_traces:
            run["external_comparison"] = _comparison_report(
                external_traces[policy_name],
                trace,
                sampled_control_steps,
            )
            run["external_discrete_outcome_match"] = (
                _discrete_push_outcome(run)
                == external_outcomes[policy_name]
            )
        else:
            run["external_comparison"] = None
            run["external_discrete_outcome_match"] = None
        runs.append(run)

    in_process_outcome_match = all(
        _discrete_push_outcome(run)
        == policy_reference_outcomes[run["policy"]]
        for run in runs
    )
    in_process_exact_match = all(
        run["in_process_comparison"] is None
        or run["in_process_comparison"]["exact_match"]
        for run in runs
    )
    external_outcome_match = (
        None
        if external_manifest is None
        else all(run["external_discrete_outcome_match"] for run in runs)
    )
    external_exact_match = (
        None
        if external_manifest is None
        else all(run["external_comparison"]["exact_match"] for run in runs)
    )
    manifest = {
        "schema_version": 1,
        "kind": "push_determinism_audit",
        "commit": commit,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "params_path": params_path,
        "params_sha256": params_sha256,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "seed": seed,
        "num_envs": n_envs,
        "num_steps": n_steps,
        "trace_stride": trace_stride,
        "repeats_per_policy": repeats,
        "sampled_control_steps": sampled_control_steps,
        "initial_state": initial_summary,
        "initial_state_trace_file": initial_trace_file,
        "external_initial_state_comparison": external_initial_comparison,
        "policy_references": policy_reference_names,
        "runs": runs,
        "in_process_discrete_outcome_match": in_process_outcome_match,
        "in_process_exact_trace_match": in_process_exact_match,
        "external_discrete_outcome_match": external_outcome_match,
        "external_exact_trace_match": external_exact_match,
        "gate_pass": (
            in_process_outcome_match
            and (
                external_outcome_match
                if external_outcome_match is not None
                else True
            )
            and (
                external_initial_comparison["exact_match"]
                if external_initial_comparison is not None
                else True
            )
        ),
    }
    manifest_tmp = destination / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_tmp.replace(destination / "manifest.json")
    return manifest


def _sequential_eval(
    env,
    action_fns,
    *,
    n_envs: int,
    n_steps: int,
    seed: int,
    full_reset: bool = False,
):
    from mujoco_playground import wrapper

    wrapped = wrapper.wrap_for_brax_training(
        env,
        episode_length=n_steps + 1,
        action_repeat=1,
        full_reset=full_reset,
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
        max_tilt_deg = jnp.zeros(())
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
        three_or_more_contact_total = jnp.zeros(())
        all_four_contact_total = jnp.zeros(())
        zero_contact_total = jnp.zeros(())
        previous_contact = None
        push_eval = "task_phase" in state.metrics
        if push_eval:
            push_active = jnp.ones((n_envs,), dtype=bool)
            push_success = jnp.zeros((n_envs,), dtype=bool)
            push_terminal = jnp.zeros((n_envs,), dtype=bool)
            push_max_phase = jnp.full(
                (n_envs,), int(TaskPhase.APPROACH), dtype=jnp.int32
            )
            push_last_goal_distance = state.metrics["object_to_goal_distance"]
            push_min_goal_distance = push_last_goal_distance
            push_last_end_effector_distance = state.metrics[
                "end_effector_to_push_distance"
            ]
            push_distance_total = jnp.zeros(())
            push_active_steps = jnp.zeros(())
            push_max_object_speed = jnp.zeros(())
            push_min_object_height = jnp.full((), jnp.inf)
            push_max_object_height = jnp.full((), -jnp.inf)
            push_max_object_speed_by_env = jnp.zeros((n_envs,))
            push_peak_step_by_env = jnp.full(
                (n_envs,), -1, dtype=jnp.int32
            )
            push_peak_prior_speed_by_env = jnp.zeros((n_envs,))
            push_peak_prior_end_effector_distance_by_env = jnp.zeros(
                (n_envs,)
            )
            push_peak_end_effector_distance_by_env = jnp.zeros((n_envs,))
            push_peak_object_displacement_by_env = jnp.zeros((n_envs,))
            push_peak_leg_action_rms_by_env = jnp.zeros((n_envs,))
            push_min_object_height_by_env = jnp.full((n_envs,), jnp.inf)
            push_max_object_height_by_env = jnp.full((n_envs,), -jnp.inf)
            push_initial_object_x = state.info["object_pos"][:, 0]
            push_initial_object_y = state.info["object_pos"][:, 1]
            push_initial_object_x_min = jnp.min(push_initial_object_x)
            push_initial_object_x_max = jnp.max(push_initial_object_x)
            push_initial_object_y_min = jnp.min(push_initial_object_y)
            push_initial_object_y_max = jnp.max(push_initial_object_y)
            first_phase_steps = {
                phase: jnp.full((n_envs,), -1, dtype=jnp.int32)
                for phase in TaskPhase
            }
            first_phase_steps[TaskPhase.APPROACH] = jnp.zeros(
                (n_envs,), dtype=jnp.int32
            )
        for _ in range(n_steps):
            prior_air_time = state.info["feet_air_time"]
            prior_swing_peak = state.info["swing_peak"]
            if push_eval:
                prior_object_speed = jnp.linalg.norm(
                    state.info["object_qvel"][:, :2], axis=-1
                )
                prior_end_effector_distance = state.metrics[
                    "end_effector_to_push_distance"
                ]
            actions = action_fn(state.obs)
            state = step_fn(state, actions)
            reward_total += jnp.mean(state.reward)
            for key in reward_metric_names:
                reward_component_totals[key] += jnp.mean(state.metrics[key])
            forward_velocity_total += jnp.mean(state.metrics["base_forward_velocity"])
            tracking_error_total += jnp.mean(state.metrics["tracking_linear_error"])
            tilt_total += jnp.mean(state.metrics["tilt_deg"])
            max_tilt_deg = jnp.maximum(
                max_tilt_deg, jnp.max(state.metrics["tilt_deg"])
            )
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
            (
                contact_count,
                three_or_more_contact,
                all_four_contact,
                zero_contact,
            ) = _support_contact_masks(foot_contact)
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
            three_or_more_contact_total += jnp.mean(three_or_more_contact)
            all_four_contact_total += jnp.mean(all_four_contact)
            zero_contact_total += jnp.mean(zero_contact)
            previous_contact = foot_contact
            if push_eval:
                active = push_active
                phase = state.metrics["task_phase"].astype(jnp.int32)
                push_max_phase = jnp.where(
                    active, jnp.maximum(push_max_phase, phase), push_max_phase
                )
                for task_phase in TaskPhase:
                    reached_now = (
                        active
                        & (phase >= int(task_phase))
                        & (first_phase_steps[task_phase] < 0)
                    )
                    first_phase_steps[task_phase] = jnp.where(
                        reached_now,
                        state.info["steps"].astype(jnp.int32),
                        first_phase_steps[task_phase],
                )
                goal_distance = state.metrics["object_to_goal_distance"]
                object_speed = jnp.linalg.norm(
                    state.info["object_qvel"][:, :2], axis=-1
                )
                object_height = state.metrics["object_height"]
                push_last_goal_distance = jnp.where(
                    active, goal_distance, push_last_goal_distance
                )
                push_min_goal_distance = jnp.where(
                    active,
                    jnp.minimum(push_min_goal_distance, goal_distance),
                    push_min_goal_distance,
                )
                push_last_end_effector_distance = jnp.where(
                    active,
                    state.metrics["end_effector_to_push_distance"],
                    push_last_end_effector_distance,
                )
                push_distance_total += jnp.sum(
                    jnp.where(active, goal_distance, 0.0)
                )
                push_active_steps += jnp.sum(active)
                push_max_object_speed = jnp.maximum(
                    push_max_object_speed,
                    jnp.max(jnp.where(active, object_speed, 0.0)),
                )
                push_min_object_height = jnp.minimum(
                    push_min_object_height,
                    jnp.min(jnp.where(active, object_height, jnp.inf)),
                )
                push_max_object_height = jnp.maximum(
                    push_max_object_height,
                    jnp.max(jnp.where(active, object_height, -jnp.inf)),
                )
                new_speed_peak = active & (
                    object_speed > push_max_object_speed_by_env
                )
                push_peak_step_by_env = jnp.where(
                    new_speed_peak,
                    state.info["steps"].astype(jnp.int32),
                    push_peak_step_by_env,
                )
                push_peak_prior_speed_by_env = jnp.where(
                    new_speed_peak,
                    prior_object_speed,
                    push_peak_prior_speed_by_env,
                )
                push_peak_prior_end_effector_distance_by_env = jnp.where(
                    new_speed_peak,
                    prior_end_effector_distance,
                    push_peak_prior_end_effector_distance_by_env,
                )
                push_peak_end_effector_distance_by_env = jnp.where(
                    new_speed_peak,
                    state.metrics["end_effector_to_push_distance"],
                    push_peak_end_effector_distance_by_env,
                )
                push_peak_object_displacement_by_env = jnp.where(
                    new_speed_peak,
                    state.metrics["object_displacement"],
                    push_peak_object_displacement_by_env,
                )
                push_peak_leg_action_rms_by_env = jnp.where(
                    new_speed_peak,
                    jnp.sqrt(jnp.mean(actions[:, :12] ** 2, axis=-1)),
                    push_peak_leg_action_rms_by_env,
                )
                push_max_object_speed_by_env = jnp.where(
                    active,
                    jnp.maximum(push_max_object_speed_by_env, object_speed),
                    push_max_object_speed_by_env,
                )
                push_min_object_height_by_env = jnp.where(
                    active,
                    jnp.minimum(push_min_object_height_by_env, object_height),
                    push_min_object_height_by_env,
                )
                push_max_object_height_by_env = jnp.where(
                    active,
                    jnp.maximum(push_max_object_height_by_env, object_height),
                    push_max_object_height_by_env,
                )
                terminal_now = active & state.done.astype(bool)
                push_success = push_success | (
                    terminal_now & (state.metrics["success"] > 0.0)
                )
                push_terminal = push_terminal | terminal_now
                push_active = active & ~terminal_now
        results[name] = {
            "mean_reward": float(reward_total / n_steps),
            "mean_forward_velocity": float(forward_velocity_total / n_steps),
            "mean_tracking_error": float(tracking_error_total / n_steps),
            "mean_tilt_deg": float(tilt_total / n_steps),
            "max_tilt_deg": float(max_tilt_deg),
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
            "gait_diagnostics_valid": (
                int(done_count) == int(jnp.sum(push_success))
                if push_eval
                else int(done_count) == 0
            ),
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
            "three_or_more_contact_fraction": float(
                three_or_more_contact_total / n_steps
            ),
            "all_four_contact_fraction": float(
                all_four_contact_total / n_steps
            ),
            "zero_contact_fraction": float(zero_contact_total / n_steps),
        }
        if push_eval:
            push_failure_counts = _push_failure_counts(
                push_max_phase, push_success
            )
            results[name].update(
                {
                    "push_episode_count": n_envs,
                    "push_success_count": int(jnp.sum(push_success)),
                    "push_success_rate": float(jnp.mean(push_success)),
                    "push_terminal_count": int(jnp.sum(push_terminal)),
                    "push_abnormal_termination_count": max(
                        int(done_count) - int(jnp.sum(push_success)), 0
                    ),
                    "push_horizon_incomplete_count": int(jnp.sum(push_active)),
                    "push_reached_align_count": int(
                        jnp.sum(push_max_phase >= int(TaskPhase.ALIGN))
                    ),
                    "push_reached_push_count": int(
                        jnp.sum(push_max_phase >= int(TaskPhase.PUSH))
                    ),
                    "push_reached_hold_count": int(
                        jnp.sum(push_max_phase >= int(TaskPhase.HOLD))
                    ),
                    "push_mean_final_goal_distance": float(
                        jnp.mean(push_last_goal_distance)
                    ),
                    "push_mean_min_goal_distance": float(
                        jnp.mean(push_min_goal_distance)
                    ),
                    "push_mean_object_to_goal_distance": float(
                        push_distance_total / jnp.maximum(push_active_steps, 1)
                    ),
                    "push_max_object_speed": float(push_max_object_speed),
                    "push_min_object_height": float(push_min_object_height),
                    "push_max_object_height": float(push_max_object_height),
                    "push_initial_object_x_min": float(
                        push_initial_object_x_min
                    ),
                    "push_initial_object_x_max": float(
                        push_initial_object_x_max
                    ),
                    "push_initial_object_y_min": float(
                        push_initial_object_y_min
                    ),
                    "push_initial_object_y_max": float(
                        push_initial_object_y_max
                    ),
                    "push_initial_object_x_by_env": tuple(
                        float(value) for value in push_initial_object_x
                    ),
                    "push_success_by_env": tuple(
                        int(value) for value in push_success
                    ),
                    "push_max_phase_by_env": tuple(
                        int(value) for value in push_max_phase
                    ),
                    "push_final_goal_distance_by_env": tuple(
                        float(value) for value in push_last_goal_distance
                    ),
                    "push_final_end_effector_distance_by_env": tuple(
                        float(value) for value in push_last_end_effector_distance
                    ),
                    "push_max_object_speed_by_env": tuple(
                        float(value) for value in push_max_object_speed_by_env
                    ),
                    "push_peak_step_by_env": tuple(
                        int(value) for value in push_peak_step_by_env
                    ),
                    "push_peak_prior_speed_by_env": tuple(
                        float(value)
                        for value in push_peak_prior_speed_by_env
                    ),
                    "push_peak_prior_end_effector_distance_by_env": tuple(
                        float(value)
                        for value in (
                            push_peak_prior_end_effector_distance_by_env
                        )
                    ),
                    "push_peak_end_effector_distance_by_env": tuple(
                        float(value)
                        for value in push_peak_end_effector_distance_by_env
                    ),
                    "push_peak_object_displacement_by_env": tuple(
                        float(value)
                        for value in push_peak_object_displacement_by_env
                    ),
                    "push_peak_leg_action_rms_by_env": tuple(
                        float(value)
                        for value in push_peak_leg_action_rms_by_env
                    ),
                    "push_min_object_height_by_env": tuple(
                        float(value) for value in push_min_object_height_by_env
                    ),
                    "push_max_object_height_by_env": tuple(
                        float(value) for value in push_max_object_height_by_env
                    ),
                    **push_failure_counts,
                }
            )
            for task_phase in TaskPhase:
                reached = first_phase_steps[task_phase] >= 0
                results[name][
                    f"push_first_{task_phase.name.lower()}_step_mean"
                ] = float(
                    jnp.where(
                        jnp.any(reached),
                        jnp.sum(
                            jnp.where(
                                reached, first_phase_steps[task_phase], 0
                            )
                        )
                        / jnp.maximum(jnp.sum(reached), 1),
                        -1.0,
                    )
                )
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
    parser.add_argument(
        "--task",
        choices=("locomotion", "push"),
        default="locomotion",
    )
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
    parser.add_argument(
        "--eval-policy",
        choices=("both", "baseline", "trained"),
        default="both",
    )
    parser.add_argument("--eval-num-envs", type=int)
    parser.add_argument("--eval-num-steps", type=int)
    parser.add_argument(
        "--allow-batched-push-eval",
        action="store_true",
        help=(
            "Allow diagnostic-only batched Push evaluation. Results are not "
            "qualification evidence on gfx1100."
        ),
    )
    parser.add_argument(
        "--determinism-audit-dir",
        help=(
            "Run an interleaved Push-policy determinism audit and write a "
            "new manifest/trace directory."
        ),
    )
    parser.add_argument(
        "--determinism-reference-dir",
        help=(
            "Compare the new determinism audit against a compatible fresh-"
            "process audit directory."
        ),
    )
    parser.add_argument(
        "--determinism-repeats",
        type=int,
        default=3,
        help="Number of interleaved repeats for each Push policy.",
    )
    parser.add_argument(
        "--determinism-trace-stride",
        type=int,
        default=1,
        help="Record every Nth control step, always including the final step.",
    )
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
    eval_num_envs = (
        evaluation["num_envs"]
        if args.eval_num_envs is None
        else args.eval_num_envs
    )
    eval_num_steps = (
        evaluation["num_steps"]
        if args.eval_num_steps is None
        else args.eval_num_steps
    )
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
    if args.determinism_reference_dir and not args.determinism_audit_dir:
        parser.error(
            "--determinism-reference-dir requires --determinism-audit-dir"
        )
    if args.determinism_audit_dir:
        if args.task != "push":
            parser.error("determinism audit is only available for --task push")
        if not args.eval_only:
            parser.error("determinism audit requires --eval-only")
        if args.skip_eval:
            parser.error("determinism audit cannot be combined with --skip-eval")
        if args.eval_policy != "both":
            parser.error("determinism audit requires --eval-policy both")
        if not args.params_in:
            parser.error(
                "determinism audit requires frozen inference --params-in"
            )
        if args.determinism_repeats < 2:
            parser.error("--determinism-repeats must be at least 2")
        if args.determinism_trace_stride <= 0:
            parser.error("--determinism-trace-stride must be positive")
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
        or eval_num_envs <= 0
        or eval_num_steps <= 0
        or not policy_hidden_layer_sizes
        or not value_hidden_layer_sizes
        or any(size <= 0 for size in policy_hidden_layer_sizes)
        or any(size <= 0 for size in value_hidden_layer_sizes)
    ):
        parser.error(
            "timesteps must be non-negative; episode length, evaluation size, "
            "learning rate, PPO batch, network, and checkpoint dimensions must "
            "be positive"
        )
    push_eval_mode = None
    if args.task == "push" and not args.skip_eval:
        try:
            push_eval_mode = push_evaluation_execution_mode(
                eval_num_envs,
                determinism_audit=bool(args.determinism_audit_dir),
                allow_batched_diagnostic=args.allow_batched_push_eval,
            )
        except ValueError as error:
            parser.error(str(error))
    elif args.allow_batched_push_eval:
        parser.error(
            "--allow-batched-push-eval requires an enabled Push evaluation"
        )
    if batch_size * num_minibatches % num_envs:
        parser.error("batch_size * num_minibatches must be divisible by num_envs")

    from amd_robo.platform import _compat

    _compat.apply_brax_compat()
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    env = _make_env(config, task=args.task)
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
        f"task={args.task} "
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
            "task": args.task,
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
    if push_eval_mode == "batched_diagnostic":
        print(
            "PUSH_BATCHED_EVAL_DIAGNOSTIC_ONLY "
            f"num_envs={eval_num_envs} qualification_evidence=false",
            flush=True,
        )
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
        wrap_env_fn=functools.partial(
            wrapper.wrap_for_brax_training,
            full_reset=args.task == "push",
        ),
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

    action_fns = {
        "baseline": zero_action,
        "trained": trained_action,
    }
    if args.eval_policy != "both":
        action_fns = {
            args.eval_policy: action_fns[args.eval_policy],
        }
    eval_env = _make_env(
        config,
        command_override=evaluation["fixed_command"],
        randomize_reset=False,
        task=args.task,
    )
    if args.determinism_audit_dir:
        print(
            "PUSH_DETERMINISM_AUDIT_START "
            f"num_envs={eval_num_envs} num_steps={eval_num_steps} "
            f"seed={evaluation['seed']} repeats={args.determinism_repeats} "
            f"trace_stride={args.determinism_trace_stride}",
            flush=True,
        )
        manifest = _push_determinism_audit(
            eval_env,
            action_fns,
            output_dir=args.determinism_audit_dir,
            reference_dir=args.determinism_reference_dir,
            config_path=args.config,
            config_sha256=config_sha256,
            params_path=args.params_in,
            n_envs=eval_num_envs,
            n_steps=eval_num_steps,
            seed=evaluation["seed"],
            repeats=args.determinism_repeats,
            trace_stride=args.determinism_trace_stride,
        )
        for run in manifest["runs"]:
            comparison = run["in_process_comparison"]
            print(
                "PUSH_DETERMINISM_RUN "
                f"name={run['name']} policy={run['policy']} "
                f"trace_sha256={run['trace']['sha256']} "
                f"success_by_env={run['success_by_env']} "
                f"max_phase_by_env={run['max_phase_by_env']} "
                f"in_process_exact="
                f"{None if comparison is None else comparison['exact_match']}",
                flush=True,
            )
            if (
                comparison is not None
                and comparison["first_exact_divergence"] is not None
            ):
                print(
                    "PUSH_DETERMINISM_DIVERGENCE "
                    f"name={run['name']} "
                    + " ".join(
                        f"{key}={value}"
                        for key, value in comparison[
                            "first_exact_divergence"
                        ].items()
                    ),
                    flush=True,
                )
        print(
            "PUSH_DETERMINISM_AUDIT_DONE "
            f"gate_pass={manifest['gate_pass']} "
            f"in_process_discrete_outcome_match="
            f"{manifest['in_process_discrete_outcome_match']} "
            f"in_process_exact_trace_match="
            f"{manifest['in_process_exact_trace_match']} "
            f"external_discrete_outcome_match="
            f"{manifest['external_discrete_outcome_match']} "
            f"manifest={Path(args.determinism_audit_dir) / 'manifest.json'}",
            flush=True,
        )
        return 0
    print(
        "LOCOMOTION_EVAL_START implementation=sequential_python_loop "
        f"policies={','.join(action_fns)} num_envs={eval_num_envs} "
        f"num_steps={eval_num_steps}",
        flush=True,
    )
    results = _sequential_eval(
        eval_env,
        action_fns,
        n_envs=eval_num_envs,
        n_steps=eval_num_steps,
        seed=evaluation["seed"],
        full_reset=args.task == "push",
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
