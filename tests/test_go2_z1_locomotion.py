"""Phase-2 locomotion environment contract and finite-step checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

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


def test_trot_phase_is_opt_in_and_preserves_the_action_contract() -> None:
    legacy = Go2Z1LocomotionEnv(randomize_reset=False)
    legacy_state = jax.jit(legacy.reset)(jax.random.PRNGKey(0))

    assert legacy.observation_size == 73
    assert "gait_phase" not in legacy_state.info
    assert "reward/trot_contact" not in legacy_state.metrics

    env = Go2Z1LocomotionEnv(
        command_override=(0.4, 0.0, 0.0),
        randomize_reset=False,
        gait_cycle_time=0.5,
        trot_contact_scale=0.5,
        trot_swing_height_cost_scale=0.2,
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert env.action_size == 19
    assert env.observation_size == 75
    assert state.obs.shape == (75,)
    assert state.info["gait_phase"] == 0.0
    assert nxt.info["gait_phase"] > state.info["gait_phase"]
    assert jnp.isfinite(nxt.metrics["reward/trot_contact"])
    assert jnp.isfinite(nxt.metrics["reward/trot_swing_height"])
    assert _tree_is_finite(nxt.data)


def test_trot_phase_alternates_diagonal_contact_targets() -> None:
    first = Go2Z1LocomotionEnv._desired_trot_contact(jnp.asarray(0.0))
    second = Go2Z1LocomotionEnv._desired_trot_contact(jnp.asarray(jnp.pi))

    assert jnp.array_equal(first, jnp.asarray([True, False, False, True]))
    assert jnp.array_equal(second, jnp.asarray([False, True, True, False]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"gait_cycle_time": 0.0}, "gait_cycle_time must be positive"),
        ({"trot_contact_scale": -0.1}, "trot reward scales must be non-negative"),
        (
            {"trot_swing_height_cost_scale": 0.1},
            "trot reward scales require gait_cycle_time",
        ),
    ),
)
def test_trot_parameters_reject_invalid_combinations(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Go2Z1LocomotionEnv(**kwargs)
