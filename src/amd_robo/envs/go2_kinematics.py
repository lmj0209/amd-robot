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

_CRAWL_SEQUENCE = jnp.asarray([0, 3, 1, 2], dtype=jnp.int32)
_CRAWL_PHASE_OFFSETS = jnp.asarray([0.0, 0.5, 0.75, 0.25])
_CRAWL_FORE_AFT_SIGNS = jnp.asarray([1.0, 1.0, -1.0, -1.0])

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


def _smoothstep(value: jax.Array) -> jax.Array:
    value = jnp.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def go2_foot_space_crawl_targets(
    gait_phase: jax.Array,
    *,
    step_length: float,
    foot_clearance: float,
    body_shift_x: float,
    body_shift_y: float,
    shift_end_fraction: float,
    lift_start_fraction: float,
    lift_end_fraction: float,
) -> jax.Array:
    """Build FL-RR-FR-RL foot targets relative to the four hip joints."""

    cycle_position = jnp.mod(gait_phase, 2.0 * jnp.pi) / (2.0 * jnp.pi)
    quarter_position = 4.0 * cycle_position
    slot = jnp.floor(quarter_position).astype(jnp.int32)
    quarter_phase = quarter_position - jnp.floor(quarter_position)
    active_leg = _CRAWL_SEQUENCE[slot]
    previous_leg = _CRAWL_SEQUENCE[jnp.mod(slot - 1, 4)]

    shift_blend = _smoothstep(quarter_phase / shift_end_fraction)
    previous_shift_x = _CRAWL_FORE_AFT_SIGNS[previous_leg] * body_shift_x
    active_shift_x = _CRAWL_FORE_AFT_SIGNS[active_leg] * body_shift_x
    previous_shift_y = GO2_SIDE_SIGNS[previous_leg] * body_shift_y
    active_shift_y = GO2_SIDE_SIGNS[active_leg] * body_shift_y
    common_shift_x = (
        (1.0 - shift_blend) * previous_shift_x
        + shift_blend * active_shift_x
    )
    common_shift_y = (
        (1.0 - shift_blend) * previous_shift_y
        + shift_blend * active_shift_y
    )

    leg_phase = jnp.mod(cycle_position - _CRAWL_PHASE_OFFSETS, 1.0)
    slot_phase = leg_phase / 0.25
    lift_duration = lift_end_fraction - lift_start_fraction
    swing_progress = _smoothstep(
        (slot_phase - lift_start_fraction) / lift_duration
    )
    stance_progress = _smoothstep((leg_phase - 0.25) / 0.75)
    half_step = 0.5 * step_length
    foot_x = jnp.where(
        leg_phase < 0.25,
        -half_step + step_length * swing_progress,
        half_step - step_length * stance_progress,
    )

    lift_progress = jnp.clip(
        (quarter_phase - lift_start_fraction) / lift_duration,
        0.0,
        1.0,
    )
    lift_window = (quarter_phase >= lift_start_fraction) & (
        quarter_phase < lift_end_fraction
    )
    foot_z = (
        foot_clearance
        * jnp.square(jnp.sin(jnp.pi * lift_progress))
        * lift_window.astype(jnp.float32)
        * (jnp.arange(4) == active_leg).astype(jnp.float32)
    )
    offsets = jnp.stack(
        [
            foot_x + common_shift_x,
            jnp.full(4, common_shift_y),
            foot_z,
        ],
        axis=-1,
    )
    return (GO2_HOME_FOOT_POSITIONS + offsets).astype(jnp.float32)


def go2_foot_space_crawl_reference(
    gait_phase: jax.Array,
    *,
    step_length: float,
    foot_clearance: float,
    body_shift_x: float,
    body_shift_y: float,
    shift_end_fraction: float,
    lift_start_fraction: float,
    lift_end_fraction: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return joint deltas, foot targets, and per-leg IK reachability."""

    targets = go2_foot_space_crawl_targets(
        gait_phase,
        step_length=step_length,
        foot_clearance=foot_clearance,
        body_shift_x=body_shift_x,
        body_shift_y=body_shift_y,
        shift_end_fraction=shift_end_fraction,
        lift_start_fraction=lift_start_fraction,
        lift_end_fraction=lift_end_fraction,
    )
    angles, reachable = go2_leg_inverse_kinematics(targets)
    reference = (angles - GO2_HOME_LEG_ANGLES).reshape(12)
    return reference.astype(jnp.float32), targets, reachable
