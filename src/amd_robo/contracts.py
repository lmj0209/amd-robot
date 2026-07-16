"""Stable project contracts shared by environments, training, and tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Sized


class TaskPhase(IntEnum):
    """Push-to-Goal task phases stored as JAX-compatible integer values."""

    APPROACH = 0
    ALIGN = 1
    PUSH = 2
    HOLD = 3


@dataclass(frozen=True)
class ActionLayout:
    """Frozen policy action layout used by every curriculum stage."""

    leg: slice = slice(0, 12)
    arm: slice = slice(12, 18)
    gripper: slice = slice(18, 19)
    size: int = 19

    def covered_indices(self) -> tuple[int, ...]:
        return tuple(range(self.leg.start, self.leg.stop)) + tuple(
            range(self.arm.start, self.arm.stop)
        ) + tuple(range(self.gripper.start, self.gripper.stop))


ACTION_LAYOUT: Final = ActionLayout()

REQUIRED_INFO_KEYS: Final = frozenset(
    {
        "rng",
        "phase",
        "goal_pos",
        "last_action",
        "success_count",
    }
)

POLICY_OBSERVATION_FIELDS: Final = (
    "base_linear_velocity",
    "base_angular_velocity",
    "projected_gravity",
    "joint_position_error",
    "joint_velocity",
    "last_action",
    "object_relative_position",
    "goal_relative_position",
    "end_effector_relative_position",
    "phase",
)

CRITIC_EXTRA_FIELDS: Final = (
    "object_world_pose",
    "goal_world_position",
    "contact_summary",
    "randomized_dynamics",
)

REQUIRED_TERMINATION_SIGNALS: Final = frozenset(
    {
        "tilt",
        "illegal_contact",
        "workspace_bounds",
        "non_finite_state",
        "timeout",
        "success_hold",
    }
)


def validate_action(action: Sized) -> None:
    """Raise when an action violates the frozen 19-dimensional interface."""

    if len(action) != ACTION_LAYOUT.size:
        raise ValueError(
            f"Expected {ACTION_LAYOUT.size} action values, got {len(action)}"
        )
