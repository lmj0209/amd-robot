import jax.numpy as jnp

from amd_robo.contracts import TaskPhase
from scripts.locomotion_learn_smoke import (
    _build_direct_eval_policy,
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


def test_direct_eval_policy_uses_committed_network_shapes_and_params():
    calls = {}

    class FakeNetworks:
        @staticmethod
        def make_ppo_networks(observation_size, action_size, **kwargs):
            calls["network"] = (observation_size, action_size, kwargs)
            return "networks"

        @staticmethod
        def make_inference_fn(networks):
            calls["inference_networks"] = networks

            def bind(params, *, deterministic):
                calls["bind"] = (params, deterministic)
                return "policy"

            return bind

    class FakeEnv:
        observation_size = 87
        action_size = 19

    params = object()
    policy = _build_direct_eval_policy(
        FakeNetworks,
        FakeEnv(),
        {
            "policy_hidden_layer_sizes": [512, 256, 128],
            "value_hidden_layer_sizes": [512, 256, 128],
        },
        params,
    )

    assert policy == "policy"
    assert calls["network"] == (
        87,
        19,
        {
            "policy_hidden_layer_sizes": (512, 256, 128),
            "value_hidden_layer_sizes": (512, 256, 128),
        },
    )
    assert calls["inference_networks"] == "networks"
    assert calls["bind"] == (params, True)
