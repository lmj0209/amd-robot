from __future__ import annotations

import pytest

from amd_robo.contracts import (
    ACTION_LAYOUT,
    REQUIRED_INFO_KEYS,
    REQUIRED_TERMINATION_SIGNALS,
    ActionLayout,
    TaskPhase,
    validate_action,
)


def test_action_layout_is_contiguous_and_frozen_at_19() -> None:
    assert ActionLayout() == ACTION_LAYOUT
    assert ACTION_LAYOUT.size == 19
    assert ACTION_LAYOUT.covered_indices() == tuple(range(19))
    assert ACTION_LAYOUT.leg == slice(0, 12)
    assert ACTION_LAYOUT.arm == slice(12, 18)
    assert ACTION_LAYOUT.gripper == slice(18, 19)


def test_action_validation_rejects_dimension_drift() -> None:
    validate_action([0.0] * 19)
    with pytest.raises(ValueError, match="Expected 19"):
        validate_action([0.0] * 18)


def test_task_phase_values_are_stable() -> None:
    assert [phase.value for phase in TaskPhase] == [0, 1, 2, 3]


def test_state_and_termination_contracts_include_safety_fields() -> None:
    assert {"rng", "phase", "goal_pos", "last_action"} <= REQUIRED_INFO_KEYS
    assert {"tilt", "illegal_contact", "non_finite_state"} <= (
        REQUIRED_TERMINATION_SIGNALS
    )
