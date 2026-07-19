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
    assert state.metrics["align_progress"] == 0.0
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


def test_push_enters_align_and_stops_crawl_before_contact():
    env = Go2Z1PushEnv(approach_stop_distance=1.0)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert nxt.info["phase"] == int(TaskPhase.ALIGN)
    assert nxt.info["align_steps"] == 1
    assert jnp.allclose(nxt.info["command"], 0.0)
    assert nxt.metrics["task_phase"] == float(TaskPhase.ALIGN)
    assert 0.0 < nxt.metrics["align_progress"] < 1.0
    assert not jnp.allclose(nxt.data.ctrl[12:18], env._home_ctrl[12:18])
    assert nxt.metrics["object_displacement"] < 1.0e-3
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
