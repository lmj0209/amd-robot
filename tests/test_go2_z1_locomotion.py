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
    assert "feet_contact_time" not in legacy_state.info
    assert "reward/trot_timing" not in legacy_state.metrics

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
    assert "feet_contact_time" not in nxt.info
    assert "reward/trot_timing" not in nxt.metrics


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
        ({"trot_timing_std": 0.0}, "trot timing std and max error must be positive"),
        (
            {"trot_timing_min_air_time": -0.1},
            "trot timing minimum air time must be non-negative",
        ),
        (
            {"trot_timing_scale": 1.0},
            "trot reward scales require gait_cycle_time",
        ),
        (
            {"crawl_reference_enabled": True},
            "crawl reference requires gait_cycle_time",
        ),
        (
            {"crawl_pose_reference_enabled": True},
            "crawl pose reference requires crawl reference",
        ),
        (
            {"crawl_foot_space_enabled": True},
            "foot-space crawl requires crawl reference",
        ),
        (
            {"crawl_stride": -0.1},
            "crawl stride must be non-negative",
        ),
        (
            {
                "crawl_shift_end_fraction": 0.4,
                "crawl_lift_start_fraction": 0.3,
            },
            "crawl timing must satisfy",
        ),
    ),
)
def test_trot_parameters_reject_invalid_combinations(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Go2Z1LocomotionEnv(**kwargs)


def test_trot_timing_is_opt_in_and_finite() -> None:
    env = Go2Z1LocomotionEnv(
        command_override=(0.4, 0.0, 0.0),
        randomize_reset=False,
        gait_cycle_time=0.5,
        trot_timing_scale=1.0,
        trot_timing_std=0.1,
        trot_timing_max_error=0.2,
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert state.info["feet_contact_time"].shape == (4,)
    assert nxt.info["feet_contact_time"].shape == (4,)
    assert jnp.isfinite(nxt.metrics["reward/trot_timing"])


def test_trot_timing_prefers_alternating_diagonal_pairs_over_standing() -> None:
    alternating = Go2Z1LocomotionEnv._trot_timing_score(
        air_time=jnp.asarray([0.0, 0.2, 0.2, 0.0]),
        contact_time=jnp.asarray([0.2, 0.0, 0.0, 0.2]),
        std=0.1,
        max_error=0.2,
    )
    standing = Go2Z1LocomotionEnv._trot_timing_score(
        air_time=jnp.zeros(4),
        contact_time=jnp.full(4, 0.2),
        std=0.1,
        max_error=0.2,
    )

    assert jnp.isclose(alternating, 1.0)
    assert alternating > standing


def test_trot_timing_dwell_gate_rejects_fast_contact_chatter() -> None:
    chatter = Go2Z1LocomotionEnv._trot_timing_score(
        air_time=jnp.asarray([0.0, 0.01, 0.01, 0.0]),
        contact_time=jnp.asarray([0.01, 0.0, 0.0, 0.01]),
        std=0.1,
        max_error=0.2,
        min_air_time=0.1,
    )
    sustained = Go2Z1LocomotionEnv._trot_timing_score(
        air_time=jnp.asarray([0.0, 0.1, 0.1, 0.0]),
        contact_time=jnp.asarray([0.1, 0.0, 0.0, 0.1]),
        std=0.1,
        max_error=0.2,
        min_air_time=0.1,
    )

    assert jnp.isclose(chatter, 0.1)
    assert jnp.isclose(sustained, 1.0)


def test_crawl_reference_is_opt_in_and_preserves_raw_policy_action() -> None:
    env = Go2Z1LocomotionEnv(
        command_override=(0.4, 0.0, 0.0),
        randomize_reset=False,
        gait_cycle_time=4.0,
        crawl_reference_enabled=True,
        crawl_stride=0.08,
        crawl_shift=0.06,
        crawl_lift=0.45,
        leg_kp=50.0,
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    action = jnp.ones(env.action_size)
    nxt = jax.jit(env.step)(state, action)
    _block_tree(nxt)

    assert env.action_size == 19
    assert env.observation_size == 75
    assert state.obs.shape == (75,)
    assert state.info["crawl_reference"].shape == (12,)
    assert state.info["crawl_sustained_touchdown"].shape == (4,)
    assert jnp.any(jnp.abs(state.info["crawl_reference"]) > 0.0)
    assert jnp.allclose(nxt.info["last_action"], action)
    assert jnp.allclose(nxt.data.ctrl[12:], env._home_ctrl[12:])
    assert jnp.isfinite(nxt.metrics["crawl_reference_rms"])
    assert _tree_is_finite(nxt.data)


def test_crawl_reference_is_disabled_for_zero_command() -> None:
    env = Go2Z1LocomotionEnv(
        command_override=(0.0, 0.0, 0.0),
        randomize_reset=False,
        gait_cycle_time=4.0,
        crawl_reference_enabled=True,
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))

    assert jnp.allclose(state.info["crawl_reference"], 0.0)
    assert jnp.allclose(state.data.qpos, env._home_qpos)
    assert jnp.allclose(state.data.ctrl, env._home_ctrl)


def test_foot_space_crawl_is_opt_in_reachable_and_preserves_contracts() -> None:
    env = Go2Z1LocomotionEnv(
        command_override=(0.04, 0.0, 0.0),
        randomize_reset=False,
        gait_cycle_time=4.0,
        crawl_reference_enabled=True,
        crawl_foot_space_enabled=True,
        crawl_foot_step_length=0.08,
        crawl_foot_clearance=0.04,
        crawl_body_shift_x=0.02,
        crawl_body_shift_y=0.02,
        crawl_pose_reference_enabled=True,
        leg_kp=50.0,
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)

    assert env.action_size == 19
    assert env.observation_size == 75
    assert state.info["crawl_foot_targets"].shape == (4, 3)
    assert jnp.all(state.info["crawl_ik_reachable"])
    assert jnp.array_equal(
        state.info["last_contact"], env._foot_floor_contacts(state.data)
    )
    assert nxt.metrics["crawl_ik_reachable_fraction"] == 1.0
    assert jnp.all(jnp.isfinite(nxt.info["crawl_reference"]))
    assert jnp.allclose(nxt.data.ctrl[12:], env._home_ctrl[12:])
    assert _tree_is_finite(nxt.data)


def test_crawl_pose_reference_is_opt_in_and_retargets_pose_reward() -> None:
    common = {
        "command_override": (0.04, 0.0, 0.0),
        "randomize_reset": False,
        "gait_cycle_time": 4.0,
        "crawl_reference_enabled": True,
        "crawl_stride": 0.08,
        "crawl_shift": 0.06,
        "crawl_lift": 0.45,
        "leg_kp": 50.0,
    }
    legacy = Go2Z1LocomotionEnv(**common)
    reference = Go2Z1LocomotionEnv(
        **common,
        crawl_pose_reference_enabled=True,
    )
    key = jax.random.PRNGKey(0)
    legacy_state = jax.jit(legacy.reset)(key)
    reference_state = jax.jit(reference.reset)(key)
    action = jnp.zeros(reference.action_size)
    legacy_next = jax.jit(legacy.step)(legacy_state, action)
    reference_next = jax.jit(reference.step)(reference_state, action)
    _block_tree((legacy_next, reference_next))

    assert "crawl_reference_error_rms" not in legacy_next.metrics
    assert jnp.isfinite(reference_next.metrics["crawl_reference_error_rms"])
    assert (
        reference_next.metrics["reward/pose"]
        > legacy_next.metrics["reward/pose"]
    )


def test_crawl_schedule_uses_four_beat_sequence() -> None:
    phases = jnp.asarray([0.0, 0.5 * jnp.pi, jnp.pi, 1.5 * jnp.pi])
    active_legs = [
        int(Go2Z1LocomotionEnv._crawl_schedule(phase)[0]) for phase in phases
    ]

    assert active_legs == [0, 3, 1, 2]


def test_crawl_schedule_timing_window_is_opt_in() -> None:
    phase = jnp.asarray(2.0 * jnp.pi * 0.0625)
    _, default_window = Go2Z1LocomotionEnv._crawl_schedule(phase)
    _, wide_window = Go2Z1LocomotionEnv._crawl_schedule(phase, 0.2, 0.9)

    assert not bool(default_window)
    assert bool(wide_window)


def test_crawl_reference_rejects_trot_reward_combination() -> None:
    with pytest.raises(
        ValueError,
        match="crawl reference cannot be combined with trot rewards",
    ):
        Go2Z1LocomotionEnv(
            gait_cycle_time=4.0,
            crawl_reference_enabled=True,
            trot_contact_scale=0.1,
        )


def test_crawl_reference_requires_positive_sustained_air_time() -> None:
    with pytest.raises(
        ValueError,
        match="minimum air time must be positive",
    ):
        Go2Z1LocomotionEnv(crawl_min_air_time=0.0)
