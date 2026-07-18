"""Phase-2 locomotion environment contract and finite-step checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amd_robo.envs.go2_z1_locomotion import Go2Z1LocomotionEnv
from amd_robo.platform.smoke import _block_tree, _tree_is_finite


def test_locomotion_reset_adds_bounded_forward_command() -> None:
    env = Go2Z1LocomotionEnv()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    assert env.action_size == 19
    assert state.obs.shape == (73,)
    assert state.info["command"].shape == (3,)
    assert state.info["feet_air_time"].shape == (4,)
    assert state.info["last_contact"].shape == (4,)
    assert state.info["swing_peak"].shape == (4,)
    assert 0.0 <= state.info["command"][0] <= 0.6
    assert jnp.allclose(state.info["command"][1:], 0.0)


def test_locomotion_zero_command_preserves_standing_and_arm_mask() -> None:
    env = Go2Z1LocomotionEnv(
        command_override=(0.0, 0.0, 0.0),
        randomize_reset=False,
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    action = jnp.ones(env.action_size)
    nxt = jax.jit(env.step)(state, action)
    _block_tree(nxt)

    assert _tree_is_finite(nxt.data)
    assert jnp.isfinite(nxt.reward)
    assert jnp.allclose(nxt.info["last_action"][12:], 1.0)
    assert jnp.allclose(nxt.data.ctrl[12:], env._home_ctrl[12:])
    assert nxt.metrics["illegal_contact"] == 0.0
    assert nxt.metrics["nonfinite_state"] == 0.0
    assert 0.0 <= nxt.reward <= 100.0
    assert jnp.isfinite(nxt.metrics["reward/feet_air_time"])
    assert jnp.isfinite(nxt.metrics["reward/feet_clearance"])
    assert nxt.metrics["reward/arm_action_magnitude"] < 0.0
