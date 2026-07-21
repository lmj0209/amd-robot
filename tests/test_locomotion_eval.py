import jax.numpy as jnp

from amd_robo.contracts import TaskPhase
from scripts.locomotion_learn_smoke import (
    _push_failure_counts,
    _resolve_evaluation_seed,
    _support_contact_masks,
)


def test_support_contact_masks_cover_zero_through_four_contacts():
    foot_contact = jnp.asarray(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    contact_count, three_or_more, all_four, zero = _support_contact_masks(
        foot_contact
    )

    assert contact_count.tolist() == [0, 1, 2, 3, 4]
    assert three_or_more.tolist() == [False, False, False, True, True]
    assert all_four.tolist() == [False, False, False, False, True]
    assert zero.tolist() == [True, False, False, False, False]


def test_push_failure_counts_use_the_furthest_phase_reached():
    counts = _push_failure_counts(
        jnp.asarray(
            [
                int(TaskPhase.APPROACH),
                int(TaskPhase.ALIGN),
                int(TaskPhase.PUSH),
                int(TaskPhase.HOLD),
                int(TaskPhase.HOLD),
            ]
        ),
        jnp.asarray([False, False, False, False, True]),
    )

    assert counts == {
        "push_failure_approach_count": 1,
        "push_failure_align_count": 1,
        "push_failure_push_count": 1,
        "push_failure_hold_count": 1,
    }


def test_evaluation_seed_override_does_not_change_the_committed_default():
    assert _resolve_evaluation_seed(777, None) == 777
    assert _resolve_evaluation_seed(777, 20260721) == 20260721
