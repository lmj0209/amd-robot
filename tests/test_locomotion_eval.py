import jax.numpy as jnp

from scripts.locomotion_learn_smoke import _support_contact_masks


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
