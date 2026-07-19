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

from amd_robo.contracts import TaskPhase  # noqa: E402
from amd_robo.envs.go2_z1_push import Go2Z1PushEnv  # noqa: E402
from amd_robo.platform.smoke import _block_tree  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--approach-stop-distance", type=float)
    parser.add_argument("--solver-iterations", type=int)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.num_steps <= 0:
        parser.error("num-envs and num-steps must be positive")
    if args.approach_stop_distance is not None and args.approach_stop_distance <= 0.0:
        parser.error("approach-stop-distance must be positive")
    if args.solver_iterations is not None and args.solver_iterations <= 0:
        parser.error("solver-iterations must be positive")

    env_kwargs = {}
    if args.approach_stop_distance is not None:
        env_kwargs["approach_stop_distance"] = args.approach_stop_distance
    if args.solver_iterations is not None:
        env_kwargs["solver_iterations"] = args.solver_iterations
    env = Go2Z1PushEnv(**env_kwargs)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
    state = reset_fn(keys)
    _block_tree(state)

    initial_object_position = state.info["object_pos"]
    initial_goal_distance = jnp.mean(state.metrics["object_to_goal_distance"])
    initial_prepush_distance = jnp.mean(state.metrics["base_to_prepush_distance"])
    initial_end_effector_distance = jnp.mean(
        state.metrics["end_effector_to_push_distance"]
    )
    actions = jnp.zeros((args.num_envs, env.action_size))
    done_count = jnp.zeros((), dtype=jnp.int32)
    illegal_contact_count = jnp.zeros((), dtype=jnp.int32)
    nonfinite_state_count = jnp.zeros((), dtype=jnp.int32)
    three_or_more_contact_total = jnp.zeros(())
    robot_box_contact_total = jnp.zeros(())
    first_align_step = jnp.asarray(args.num_steps + 1, dtype=jnp.int32)
    base_to_prepush_at_align = jnp.asarray(jnp.nan)
    first_push_step = jnp.asarray(args.num_steps + 1, dtype=jnp.int32)
    end_effector_to_target_at_push = jnp.asarray(jnp.nan)
    first_hold_step = jnp.asarray(args.num_steps + 1, dtype=jnp.int32)
    first_success_step = jnp.asarray(args.num_steps + 1, dtype=jnp.int32)
    first_robot_box_contact_step = jnp.asarray(args.num_steps + 1, dtype=jnp.int32)
    base_to_prepush_at_first_contact = jnp.asarray(jnp.nan)
    first_object_motion_step = jnp.asarray(args.num_steps + 1, dtype=jnp.int32)
    base_to_prepush_at_first_motion = jnp.asarray(jnp.nan)
    minimum_robot_box_signed_margin = jnp.asarray(jnp.inf)
    max_tilt_deg = jnp.zeros(())
    max_object_height = jnp.max(state.info["object_pos"][:, 2])
    max_object_speed = jnp.zeros(())
    minimum_goal_distance = initial_goal_distance
    minimum_prepush_distance = initial_prepush_distance
    minimum_end_effector_distance = initial_end_effector_distance

    for step_index in range(args.num_steps):
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
        any_align = jnp.any(state.info["phase"] >= int(TaskPhase.ALIGN))
        first_align_now = any_align & (first_align_step > args.num_steps)
        first_align_step = jnp.where(
            first_align_now,
            step_index + 1,
            first_align_step,
        )
        base_to_prepush_at_align = jnp.where(
            first_align_now,
            jnp.mean(state.metrics["base_to_prepush_distance"]),
            base_to_prepush_at_align,
        )
        any_push = jnp.any(state.info["phase"] >= int(TaskPhase.PUSH))
        first_push_now = any_push & (first_push_step > args.num_steps)
        first_push_step = jnp.where(
            first_push_now,
            step_index + 1,
            first_push_step,
        )
        end_effector_to_target_at_push = jnp.where(
            first_push_now,
            jnp.mean(state.metrics["end_effector_to_push_distance"]),
            end_effector_to_target_at_push,
        )
        any_hold = jnp.any(state.info["phase"] >= int(TaskPhase.HOLD))
        first_hold_step = jnp.where(
            any_hold & (first_hold_step > args.num_steps),
            step_index + 1,
            first_hold_step,
        )
        any_success = jnp.any(state.metrics["success"] > 0.0)
        first_success_step = jnp.where(
            any_success & (first_success_step > args.num_steps),
            step_index + 1,
            first_success_step,
        )
        contact = state.data._impl.contact
        geom1, geom2 = contact.geom[:, :, 0], contact.geom[:, :, 1]
        active = contact.dist < contact.includemargin
        box_contact = (geom1 == env._box_geom_id) | (geom2 == env._box_geom_id)
        other_geom = jnp.where(geom1 == env._box_geom_id, geom2, geom1)
        robot_box_pair = box_contact & (other_geom != env._floor_geom_id)
        robot_box_contact = jnp.any(active & robot_box_pair, axis=-1)
        minimum_robot_box_signed_margin = jnp.minimum(
            minimum_robot_box_signed_margin,
            jnp.min(
                jnp.where(
                    robot_box_pair,
                    contact.dist - contact.includemargin,
                    jnp.inf,
                )
            ),
        )
        robot_box_contact_total += jnp.mean(robot_box_contact)
        any_robot_box_contact = jnp.any(robot_box_contact)
        first_contact_now = any_robot_box_contact & (
            first_robot_box_contact_step > args.num_steps
        )
        first_robot_box_contact_step = jnp.where(
            first_contact_now,
            step_index + 1,
            first_robot_box_contact_step,
        )
        base_to_prepush_at_first_contact = jnp.where(
            first_contact_now,
            jnp.mean(state.metrics["base_to_prepush_distance"]),
            base_to_prepush_at_first_contact,
        )
        object_motion = jnp.any(
            jnp.linalg.norm(
                state.info["object_pos"][:, :2] - initial_object_position[:, :2],
                axis=-1,
            )
            > 1.0e-3
        )
        first_motion_now = object_motion & (first_object_motion_step > args.num_steps)
        first_object_motion_step = jnp.where(
            first_motion_now,
            step_index + 1,
            first_object_motion_step,
        )
        base_to_prepush_at_first_motion = jnp.where(
            first_motion_now,
            jnp.mean(state.metrics["base_to_prepush_distance"]),
            base_to_prepush_at_first_motion,
        )
        max_tilt_deg = jnp.maximum(max_tilt_deg, jnp.max(state.metrics["tilt_deg"]))
        max_object_height = jnp.maximum(
            max_object_height, jnp.max(state.info["object_pos"][:, 2])
        )
        max_object_speed = jnp.maximum(
            max_object_speed,
            jnp.max(jnp.linalg.norm(state.info["object_qvel"][:, :3], axis=-1)),
        )
        minimum_goal_distance = jnp.minimum(
            minimum_goal_distance,
            jnp.mean(state.metrics["object_to_goal_distance"]),
        )
        minimum_prepush_distance = jnp.minimum(
            minimum_prepush_distance,
            jnp.mean(state.metrics["base_to_prepush_distance"]),
        )
        minimum_end_effector_distance = jnp.minimum(
            minimum_end_effector_distance,
            jnp.mean(state.metrics["end_effector_to_push_distance"]),
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
        "approach_stop_distance": float(env._approach_stop_distance),
        "initial_base_to_prepush_distance": float(initial_prepush_distance),
        "minimum_base_to_prepush_distance": float(minimum_prepush_distance),
        "final_base_to_prepush_distance": float(
            jnp.mean(state.metrics["base_to_prepush_distance"])
        ),
        "initial_object_to_goal_distance": float(initial_goal_distance),
        "minimum_object_to_goal_distance": float(minimum_goal_distance),
        "final_object_to_goal_distance": float(
            jnp.mean(state.metrics["object_to_goal_distance"])
        ),
        "initial_end_effector_to_push_distance": float(initial_end_effector_distance),
        "minimum_end_effector_to_push_distance": float(minimum_end_effector_distance),
        "final_end_effector_to_push_distance": float(
            jnp.mean(state.metrics["end_effector_to_push_distance"])
        ),
        "final_object_position": [
            float(value) for value in jnp.mean(state.info["object_pos"], axis=0)
        ],
        "max_object_height": float(max_object_height),
        "max_object_speed": float(max_object_speed),
        "mean_object_displacement": float(jnp.mean(object_displacement)),
        "max_object_displacement": float(jnp.max(object_displacement)),
        "robot_box_contact_fraction": float(robot_box_contact_total / args.num_steps),
        "final_phase": int(jnp.max(state.info["phase"])),
        "final_align_progress": float(jnp.mean(state.metrics["align_progress"])),
        "first_align_step": (
            None if int(first_align_step) > args.num_steps else int(first_align_step)
        ),
        "base_to_prepush_at_align": (
            None
            if int(first_align_step) > args.num_steps
            else float(base_to_prepush_at_align)
        ),
        "first_push_step": (
            None if int(first_push_step) > args.num_steps else int(first_push_step)
        ),
        "end_effector_to_target_at_push": (
            None
            if int(first_push_step) > args.num_steps
            else float(end_effector_to_target_at_push)
        ),
        "first_hold_step": (
            None if int(first_hold_step) > args.num_steps else int(first_hold_step)
        ),
        "first_success_step": (
            None
            if int(first_success_step) > args.num_steps
            else int(first_success_step)
        ),
        "final_success_hold_count": int(jnp.max(state.info["success_count"])),
        "success": bool(jnp.any(state.metrics["success"] > 0.0)),
        "first_robot_box_contact_step": (
            None
            if int(first_robot_box_contact_step) > args.num_steps
            else int(first_robot_box_contact_step)
        ),
        "base_to_prepush_at_first_contact": (
            None
            if int(first_robot_box_contact_step) > args.num_steps
            else float(base_to_prepush_at_first_contact)
        ),
        "minimum_robot_box_signed_margin": float(minimum_robot_box_signed_margin),
        "first_object_motion_step": (
            None
            if int(first_object_motion_step) > args.num_steps
            else int(first_object_motion_step)
        ),
        "base_to_prepush_at_first_motion": (
            None
            if int(first_object_motion_step) > args.num_steps
            else float(base_to_prepush_at_first_motion)
        ),
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
