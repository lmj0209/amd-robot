"""Analytic Go2 leg kinematics used by the foot-space crawl reference."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

GO2_HIP_OFFSET = 0.0955
GO2_THIGH_LENGTH = 0.213
GO2_FOOT_X_OFFSET = -0.002
GO2_CALF_LENGTH = 0.213

GO2_SIDE_SIGNS = jnp.asarray([1.0, -1.0, 1.0, -1.0])
GO2_HOME_LEG_ANGLES = jnp.asarray(
    [
        [0.0, 0.9, -1.8],
        [0.0, 0.9, -1.8],
        [0.0, 0.9, -1.8],
        [0.0, 0.9, -1.8],
    ]
)
GO2_LEG_CTRL_MIN = jnp.asarray([-0.9472, -1.4, -2.6227])
GO2_LEG_CTRL_MAX = jnp.asarray([0.9472, 2.5, -0.84776])

_GO2_EFFECTIVE_CALF_LENGTH = math.hypot(
    GO2_CALF_LENGTH, GO2_FOOT_X_OFFSET
)
_GO2_FOOT_ANGLE_OFFSET = math.atan2(
    -GO2_FOOT_X_OFFSET, GO2_CALF_LENGTH
)


def go2_leg_forward_kinematics(joint_angles: jax.Array) -> jax.Array:
    """Return foot positions relative to each hip joint.

    Args:
        joint_angles: Array ending in ``(4, 3)`` ordered as
            ``[abduction, thigh, calf]`` for FL, FR, RL, RR.
    """

    angles = jnp.asarray(joint_angles)
    abduction = angles[..., :, 0]
    thigh = angles[..., :, 1]
    calf = angles[..., :, 2]
    effective_calf_angle = thigh + calf + _GO2_FOOT_ANGLE_OFFSET

    sagittal_x = (
        -GO2_THIGH_LENGTH * jnp.sin(thigh)
        - _GO2_EFFECTIVE_CALF_LENGTH * jnp.sin(effective_calf_angle)
    )
    sagittal_z = (
        -GO2_THIGH_LENGTH * jnp.cos(thigh)
        - _GO2_EFFECTIVE_CALF_LENGTH * jnp.cos(effective_calf_angle)
    )
    lateral = GO2_SIDE_SIGNS * GO2_HIP_OFFSET
    foot_y = jnp.cos(abduction) * lateral - jnp.sin(abduction) * sagittal_z
    foot_z = jnp.sin(abduction) * lateral + jnp.cos(abduction) * sagittal_z
    return jnp.stack([sagittal_x, foot_y, foot_z], axis=-1)


def go2_leg_inverse_kinematics(
    foot_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve the below-hip knee-backward branch of the Go2 leg IK.

    Returns:
        ``(joint_angles, reachable)`` where ``reachable`` has one boolean per
        leg and checks geometry plus actuator control limits.
    """

    foot = jnp.asarray(foot_positions)
    foot_x = foot[..., :, 0]
    foot_y = foot[..., :, 1]
    foot_z = foot[..., :, 2]
    lateral = GO2_SIDE_SIGNS * GO2_HIP_OFFSET

    sagittal_z_sq = foot_y * foot_y + foot_z * foot_z - GO2_HIP_OFFSET**2
    sagittal_z = -jnp.sqrt(jnp.maximum(sagittal_z_sq, 1.0e-12))
    abduction = jnp.arctan2(foot_z, foot_y) - jnp.arctan2(
        sagittal_z, lateral
    )
    abduction = jnp.arctan2(jnp.sin(abduction), jnp.cos(abduction))

    distance_sq = foot_x * foot_x + sagittal_z * sagittal_z
    cosine_knee = (
        distance_sq
        - GO2_THIGH_LENGTH**2
        - _GO2_EFFECTIVE_CALF_LENGTH**2
    ) / (2.0 * GO2_THIGH_LENGTH * _GO2_EFFECTIVE_CALF_LENGTH)
    effective_knee = -jnp.arccos(jnp.clip(cosine_knee, -1.0, 1.0))
    foot_direction = jnp.arctan2(-foot_x, -sagittal_z)
    thigh = foot_direction - jnp.arctan2(
        _GO2_EFFECTIVE_CALF_LENGTH * jnp.sin(effective_knee),
        GO2_THIGH_LENGTH
        + _GO2_EFFECTIVE_CALF_LENGTH * jnp.cos(effective_knee),
    )
    calf = effective_knee - _GO2_FOOT_ANGLE_OFFSET
    angles = jnp.stack([abduction, thigh, calf], axis=-1)

    minimum_reach = abs(GO2_THIGH_LENGTH - _GO2_EFFECTIVE_CALF_LENGTH)
    maximum_reach = GO2_THIGH_LENGTH + _GO2_EFFECTIVE_CALF_LENGTH
    distance = jnp.sqrt(jnp.maximum(distance_sq, 0.0))
    geometry_reachable = (
        (sagittal_z_sq >= 0.0)
        & (distance >= minimum_reach)
        & (distance <= maximum_reach)
    )
    within_limits = jnp.all(
        (angles >= GO2_LEG_CTRL_MIN) & (angles <= GO2_LEG_CTRL_MAX),
        axis=-1,
    )
    return angles.astype(jnp.float32), geometry_reachable & within_limits


GO2_HOME_FOOT_POSITIONS = go2_leg_forward_kinematics(GO2_HOME_LEG_ANGLES)
