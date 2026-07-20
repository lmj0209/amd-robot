from __future__ import annotations

import jax
import jax.numpy as jnp

from amd_robo.contracts import TaskPhase
from amd_robo.envs.go2_z1_push import Go2Z1PushEnv
from amd_robo.platform.smoke import _block_tree, _tree_is_finite


def test_push_reset_uses_named_keyframe_and_exposes_task_state():
    env = Go2Z1PushEnv()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    _block_tree(state)

    assert env.action_size == 19
    assert env.observation_size == 87
    assert state.obs.shape == (87,)
    assert env._home_keyframe == "push_home"
    assert env._joint_qpos_indices.shape == (19,)
    assert env._joint_dof_indices.shape == (19,)
    assert not jnp.any(env._joint_qpos_indices == env._box_qpos_adr)
    assert state.info["phase"] == int(TaskPhase.APPROACH)
    assert state.info["align_steps"] == 0
    assert state.info["push_steps"] == 0
    assert state.metrics["align_progress"] == 0.0
    assert state.metrics["success"] == 0.0
    assert state.metrics["task_reward"] == 0.0
    assert all(
        state.metrics[f"reward/task_{name}"] == 0.0 for name in env._TASK_REWARD_NAMES
    )
    assert jnp.allclose(
        state.info["object_qpos"],
        jnp.asarray([0.8, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]),
    )
    assert jnp.allclose(state.info["object_pos"], jnp.asarray([0.8, 0.0, 0.1]))
    assert jnp.allclose(
        state.info["end_effector_pos"],
        jnp.asarray([-0.15487455, 0.0, 0.6098599]),
        atol=1.0e-5,
    )
    assert jnp.allclose(
        state.info["push_contact_pos"],
        jnp.asarray([0.695, 0.0, 0.1]),
    )
    assert jnp.allclose(state.info["prepush_pos"], jnp.asarray([0.5, 0.0, 0.015]))
    assert jnp.allclose(state.info["goal_pos"], jnp.asarray([1.2, 0.0, 0.005]))
    assert jnp.isclose(state.metrics["base_to_prepush_distance"], 0.5)
    assert jnp.isclose(state.metrics["object_to_goal_distance"], 0.4)
    assert state.metrics["end_effector_to_push_distance"] > 0.9
    assert _tree_is_finite(state.data)


def test_push_step_allows_box_floor_contact_and_stays_finite():
    env = Go2Z1PushEnv()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert nxt.metrics["illegal_contact"] == 0.0
    assert nxt.metrics["nonfinite_state"] == 0.0
    assert nxt.metrics["object_displacement"] < 1.0e-3
    assert nxt.info["phase"] == int(TaskPhase.APPROACH)
    assert jnp.isclose(nxt.info["command"][0], 0.025)
    assert jnp.allclose(nxt.data.ctrl[12:], env._home_ctrl[12:])
    assert _tree_is_finite(nxt.data)


def test_push_reset_randomizes_object_x_and_moves_prepush_target_with_it():
    env = Go2Z1PushEnv(
        object_position_x_offset_range=(-0.02, 0.02),
        object_position_y_offset_range=(0.0, 0.0),
    )
    keys = jax.random.split(jax.random.PRNGKey(20260719), 8)
    reset = jax.jit(jax.vmap(env.reset))
    state = reset(keys)
    repeated = reset(keys)
    _block_tree(state)
    _block_tree(repeated)

    object_x = state.info["object_pos"][:, 0]
    object_y = state.info["object_pos"][:, 1]
    assert jnp.all(object_x >= 0.78)
    assert jnp.all(object_x <= 0.82)
    assert jnp.ptp(object_x) > 0.01
    assert jnp.allclose(object_y, 0.0)
    assert jnp.allclose(state.info["prepush_pos"][:, 0], object_x - 0.3)
    assert jnp.allclose(state.info["prepush_pos"][:, 1], object_y)
    assert jnp.allclose(state.metrics["object_displacement"], 0.0, atol=1.0e-6)
    assert jnp.allclose(state.info["object_qpos"], repeated.info["object_qpos"])
    assert _tree_is_finite(state.data)


def test_push_enters_align_and_stops_crawl_before_contact():
    env = Go2Z1PushEnv(approach_stop_distance=1.0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert nxt.info["phase"] == int(TaskPhase.ALIGN)
    assert nxt.info["align_steps"] == 1
    assert nxt.info["push_steps"] == 0
    assert jnp.allclose(nxt.info["command"], 0.0)
    assert nxt.metrics["task_phase"] == float(TaskPhase.ALIGN)
    assert 0.0 < nxt.metrics["align_progress"] < 1.0
    assert not jnp.allclose(nxt.data.ctrl[12:18], env._home_ctrl[12:18])
    assert nxt.metrics["object_displacement"] < 1.0e-3
    assert jnp.isfinite(nxt.reward)
    assert jnp.isfinite(nxt.metrics["task_reward"])
    assert _tree_is_finite(nxt.data)


def test_push_align_reference_reaches_the_audited_arm_target():
    env = Go2Z1PushEnv()
    state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        info={
            **state.info,
            "phase": jnp.asarray(int(TaskPhase.ALIGN)),
            "align_steps": jnp.asarray(
                round(env._align_duration / env.dt),
                dtype=jnp.int32,
            ),
        }
    )

    reference = env._task_actuator_reference(state)

    assert jnp.allclose(reference[:12], 0.0)
    assert jnp.allclose(reference[18], 0.0)
    assert jnp.allclose(
        env._home_ctrl[12:18] + reference[12:18],
        env._align_arm_joint_target,
    )


def test_push_policy_residual_is_active_only_in_push_phase():
    env = Go2Z1PushEnv()
    state = env.reset(jax.random.PRNGKey(0))

    for phase in (TaskPhase.APPROACH, TaskPhase.ALIGN, TaskPhase.HOLD):
        staged = state.replace(
            info={
                **state.info,
                "phase": jnp.asarray(int(phase)),
            }
        )
        assert jnp.all(env._task_policy_action_mask(staged) == 0.0)

    pushing = state.replace(
        info={
            **state.info,
            "phase": jnp.asarray(int(TaskPhase.PUSH)),
        }
    )
    assert jnp.all(env._task_policy_action_mask(pushing) == 1.0)


def test_push_command_ramp_uses_smoothstep_before_full_speed():
    env = Go2Z1PushEnv(push_command_ramp_duration=1.0)
    state = env.reset(jax.random.PRNGKey(0))

    def command_at(push_steps):
        staged = state.replace(
            info={
                **state.info,
                "phase": jnp.asarray(int(TaskPhase.PUSH)),
                "push_steps": jnp.asarray(push_steps, dtype=jnp.int32),
            }
        )
        return env._push_command_for_state(staged)

    assert jnp.allclose(command_at(0), 0.0)
    assert jnp.allclose(command_at(50), env._push_command * 0.5)
    assert jnp.allclose(command_at(100), env._push_command)
    assert jnp.allclose(command_at(200), env._push_command)


def test_push_align_can_synchronize_crawl_phase_before_contact():
    env = Go2Z1PushEnv(
        approach_stop_distance=1.0,
        align_gait_phase_sync=True,
    )
    state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(
        info={
            **state.info,
            "gait_phase": jnp.asarray(1.75, dtype=jnp.float32),
        }
    )
    nxt = env.step(state, jnp.zeros(env.action_size))

    assert nxt.info["phase"] == int(TaskPhase.ALIGN)
    assert jnp.isclose(
        nxt.info["gait_phase"],
        2.0 * jnp.pi * env.dt / env._gait_cycle_time,
    )


def test_push_align_can_synchronize_to_a_nonzero_crawl_phase():
    env = Go2Z1PushEnv(
        approach_stop_distance=1.0,
        align_gait_phase_sync=True,
        align_gait_phase_fraction=0.85,
    )
    state = env.reset(jax.random.PRNGKey(0))
    nxt = env.step(state, jnp.zeros(env.action_size))

    expected = 2.0 * jnp.pi * (
        0.85 + env.dt / env._gait_cycle_time
    )
    assert nxt.info["phase"] == int(TaskPhase.ALIGN)
    assert jnp.isclose(nxt.info["gait_phase"], expected)


def test_push_align_phase_fraction_rejects_invalid_values():
    for fraction in (-0.01, 1.0):
        with pytest.raises(
            ValueError,
            match=r"align gait phase fraction must be in \[0, 1\)",
        ):
            Go2Z1PushEnv(align_gait_phase_fraction=fraction)


def test_push_align_can_synchronize_phase_once_on_entry():
    env = Go2Z1PushEnv(
        approach_stop_distance=1.0,
        align_entry_gait_phase_fraction=0.35,
    )
    state = env.reset(jax.random.PRNGKey(0))
    first = env.step(state, jnp.zeros(env.action_size))
    second = env.step(first, jnp.zeros(env.action_size))

    phase_increment = 2.0 * jnp.pi * env.dt / env._gait_cycle_time
    assert first.info["phase"] == int(TaskPhase.ALIGN)
    assert jnp.isclose(
        first.info["gait_phase"],
        2.0 * jnp.pi * 0.35 + phase_increment,
    )
    assert jnp.isclose(
        second.info["gait_phase"],
        first.info["gait_phase"] + phase_increment,
    )


def test_push_align_entry_phase_sync_rejects_invalid_or_conflicting_values():
    for fraction in (-0.01, 1.0):
        with pytest.raises(
            ValueError,
            match=r"align entry gait phase fraction must be in \[0, 1\)",
        ):
            Go2Z1PushEnv(align_entry_gait_phase_fraction=fraction)

    with pytest.raises(
        ValueError,
        match="continuous and entry-only gait phase sync conflict",
    ):
        Go2Z1PushEnv(
            align_gait_phase_sync=True,
            align_entry_gait_phase_fraction=0.35,
        )


def test_push_box_contact_time_constant_can_be_softened():
    env = Go2Z1PushEnv(push_box_solref_timeconst=0.04)

    assert env.mj_model.geom_solref[env._box_geom_id, 0] == pytest.approx(0.04)
    assert env.mj_model.geom_solref[env._box_geom_id, 1] == pytest.approx(1.0)


def test_push_box_contact_time_constant_must_be_positive():
    with pytest.raises(
        ValueError,
        match="push box solref time constant must be positive",
    ):
        Go2Z1PushEnv(push_box_solref_timeconst=0.0)


def test_push_task_reward_is_phase_gated_and_bounded():
    env = Go2Z1PushEnv()
    previous = env.reset(jax.random.PRNGKey(0))
    current = previous.replace(
        info={
            **previous.info,
            "phase": jnp.asarray(int(TaskPhase.APPROACH)),
        },
        metrics={
            **previous.metrics,
            "base_to_prepush_distance": (
                previous.metrics["base_to_prepush_distance"] - 0.01
            ),
        },
    )

    components = env._task_reward_components(previous, current)

    assert jnp.isclose(components["approach_progress"], 0.5)
    assert components["align_progress"] == 0.0
    assert components["push_progress"] == 0.0
    assert components["hold"] == 0.0
    assert components["success_bonus"] == 0.0


def test_push_object_costs_are_normalized_to_their_limits():
    env = Go2Z1PushEnv()
    previous = env.reset(jax.random.PRNGKey(0))
    current = previous.replace(
        info={
            **previous.info,
            "phase": jnp.asarray(int(TaskPhase.PUSH)),
            "object_qvel": previous.info["object_qvel"].at[0].set(1.0),
        },
        metrics={
            **previous.metrics,
            "object_height": jnp.asarray(0.15),
        },
    )

    components = env._task_reward_components(previous, current)

    assert jnp.isclose(components["object_speed"], 1.0)
    assert jnp.isclose(components["object_height"], 2.25)


def test_push_enters_push_after_completed_alignment():
    env = Go2Z1PushEnv(align_distance_threshold=2.0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = state.replace(
        info={
            **state.info,
            "phase": jnp.asarray(int(TaskPhase.ALIGN)),
            "align_steps": jnp.asarray(
                round(env._align_duration / env.dt),
                dtype=jnp.int32,
            ),
        }
    )
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert nxt.info["phase"] == int(TaskPhase.PUSH)
    assert nxt.info["align_steps"] == round(env._align_duration / env.dt)
    assert nxt.info["push_steps"] == 1
    assert jnp.allclose(nxt.info["command"], env._push_command)
    assert _tree_is_finite(nxt.data)


def test_push_requires_a_sustained_hold_for_success():
    env = Go2Z1PushEnv(goal_threshold=1.0, success_hold_steps=1)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state = state.replace(
        info={
            **state.info,
            "phase": jnp.asarray(int(TaskPhase.PUSH)),
            "align_steps": jnp.asarray(
                round(env._align_duration / env.dt),
                dtype=jnp.int32,
            ),
        }
    )
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert nxt.info["phase"] == int(TaskPhase.HOLD)
    assert nxt.info["success_count"] == 1
    assert nxt.metrics["success"] == 1.0
    assert nxt.done == 1.0
    assert jnp.allclose(nxt.info["command"], 0.0)


def test_push_reset_and_step_are_finite_under_vmap():
    env = Go2Z1PushEnv()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    state = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((2, env.action_size))
    nxt = jax.jit(jax.vmap(env.step))(state, actions)
    _block_tree(nxt)

    assert nxt.obs.shape == (2, env.observation_size)
    assert jnp.all(nxt.metrics["illegal_contact"] == 0.0)
    assert jnp.all(nxt.metrics["nonfinite_state"] == 0.0)
    assert _tree_is_finite(nxt.data)
