"""Analytic Go2 leg kinematics checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from amd_robo.envs.go2_kinematics import (
    GO2_HOME_FOOT_POSITIONS,
    GO2_HOME_LEG_ANGLES,
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
