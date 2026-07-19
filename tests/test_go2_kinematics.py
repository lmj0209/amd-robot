"""Analytic Go2 leg kinematics checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amd_robo.envs.go2_kinematics import (
    GO2_HOME_FOOT_POSITIONS,
    GO2_HOME_LEG_ANGLES,
    go2_foot_space_crawl_reference,
    go2_foot_space_crawl_targets,
    go2_leg_forward_kinematics,
    go2_leg_inverse_kinematics,
)


def test_home_leg_kinematics_round_trip() -> None:
    angles, reachable = jax.jit(go2_leg_inverse_kinematics)(
        GO2_HOME_FOOT_POSITIONS
    )

    assert jnp.all(reachable)
    assert jnp.allclose(angles, GO2_HOME_LEG_ANGLES, atol=1.0e-6)


def test_leg_kinematics_round_trip_for_reachable_foot_targets() -> None:
    offsets = jnp.asarray(
        [
            [0.03, 0.01, 0.02],
            [-0.02, -0.01, 0.03],
            [0.02, 0.01, 0.01],
            [-0.03, -0.01, 0.02],
        ]
    )
    targets = GO2_HOME_FOOT_POSITIONS + offsets
    angles, reachable = jax.jit(go2_leg_inverse_kinematics)(targets)
    recovered = jax.jit(go2_leg_forward_kinematics)(angles)

    assert jnp.all(reachable)
    assert jnp.allclose(recovered, targets, atol=1.0e-6)


def test_leg_inverse_kinematics_rejects_unreachable_targets() -> None:
    targets = GO2_HOME_FOOT_POSITIONS.at[:, 2].set(-1.0)
    angles, reachable = jax.jit(go2_leg_inverse_kinematics)(targets)

    assert not jnp.any(reachable)
    assert jnp.all(jnp.isfinite(angles))


def _foot_space_reference(phase: jax.Array):
    return go2_foot_space_crawl_reference(
        phase,
        step_length=0.08,
        foot_clearance=0.04,
        body_shift_x=0.02,
        body_shift_y=0.02,
        shift_end_fraction=0.25,
        lift_start_fraction=0.3,
        lift_end_fraction=0.8,
    )


def test_foot_space_crawl_reference_is_reachable_over_a_cycle() -> None:
    phases = jnp.linspace(0.0, 2.0 * jnp.pi, 65, endpoint=False)
    references, targets, reachable = jax.jit(jax.vmap(_foot_space_reference))(
        phases
    )

    assert references.shape == (65, 12)
    assert targets.shape == (65, 4, 3)
    assert jnp.all(reachable)
    assert jnp.all(jnp.isfinite(references))


def test_foot_space_crawl_swings_forward_and_lifts_active_foot() -> None:
    start = go2_foot_space_crawl_targets(
        jnp.asarray(0.0),
        step_length=0.08,
        foot_clearance=0.04,
        body_shift_x=0.0,
        body_shift_y=0.0,
        shift_end_fraction=0.25,
        lift_start_fraction=0.3,
        lift_end_fraction=0.8,
    )
    before_lift = go2_foot_space_crawl_targets(
        jnp.asarray(2.0 * jnp.pi * 0.05),
        step_length=0.08,
        foot_clearance=0.04,
        body_shift_x=0.0,
        body_shift_y=0.0,
        shift_end_fraction=0.25,
        lift_start_fraction=0.3,
        lift_end_fraction=0.8,
    )
    middle = go2_foot_space_crawl_targets(
        jnp.asarray(2.0 * jnp.pi * 0.125),
        step_length=0.08,
        foot_clearance=0.04,
        body_shift_x=0.0,
        body_shift_y=0.0,
        shift_end_fraction=0.25,
        lift_start_fraction=0.3,
        lift_end_fraction=0.8,
    )
    end = go2_foot_space_crawl_targets(
        jnp.asarray(2.0 * jnp.pi * 0.249),
        step_length=0.08,
        foot_clearance=0.04,
        body_shift_x=0.0,
        body_shift_y=0.0,
        shift_end_fraction=0.25,
        lift_start_fraction=0.3,
        lift_end_fraction=0.8,
    )

    assert jnp.isclose(start[0, 0], before_lift[0, 0])
    assert start[0, 0] < end[0, 0]
    assert middle[0, 2] > GO2_HOME_FOOT_POSITIONS[0, 2] + 0.03
