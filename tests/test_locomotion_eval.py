import jax.numpy as jnp
import pytest

from amd_robo.contracts import TaskPhase
from scripts.locomotion_learn_smoke import (
    _build_direct_eval_policy,
    _json_compatible,
    _push_failure_counts,
    _push_qualification_gates,
    _resolve_evaluation_seed,
    _support_contact_masks,
    _validate_chunked_push_eval,
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


def _passing_push_qualification_result() -> dict[str, object]:
    return {
        "push_episode_count": 1,
        "push_success_count": 1,
        "push_terminal_count": 1,
        "push_abnormal_termination_count": 0,
        "push_final_goal_distance_by_env": (0.05,),
        "push_max_object_speed_by_env": (0.3,),
        "push_min_object_height_by_env": (0.09,),
        "push_max_object_height_by_env": (0.11,),
        "max_tilt_deg": 8.0,
        "max_abs_action": 0.8,
        "illegal_contact_count": 0,
        "workspace_bounds_count": 0,
        "nonfinite_state_count": 0,
        "nonfinite_action_count": 0,
        "action_saturation_count": 0,
    }


def test_chunked_push_qualification_shape_is_fail_closed():
    _validate_chunked_push_eval(n_envs=1, n_steps=4608, chunk_steps=8)

    with pytest.raises(ValueError, match="exactly one environment"):
        _validate_chunked_push_eval(n_envs=2, n_steps=4608, chunk_steps=8)
    with pytest.raises(ValueError, match="steps must be positive"):
        _validate_chunked_push_eval(n_envs=1, n_steps=0, chunk_steps=8)
    with pytest.raises(ValueError, match=r"must be in \[1, 64\]"):
        _validate_chunked_push_eval(n_envs=1, n_steps=4608, chunk_steps=65)


def test_push_qualification_gates_accept_a_safe_success():
    gates = _push_qualification_gates(
        _passing_push_qualification_result(),
        goal_threshold=0.08,
        object_speed_limit=0.5,
        object_height_tolerance=0.02,
    )

    assert gates
    assert all(gates.values())


def test_qualification_json_replaces_nonfinite_values_with_null():
    assert _json_compatible(
        {"finite": 1.0, "values": (float("nan"), float("inf"))}
    ) == {"finite": 1.0, "values": [None, None]}


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        ("push_success_count", 0, "task_success"),
        ("push_final_goal_distance_by_env", (0.081,), "goal_distance"),
        ("push_max_object_speed_by_env", (0.501,), "object_speed"),
        ("push_min_object_height_by_env", (0.079,), "object_height"),
        ("illegal_contact_count", 1, "illegal_contact"),
        ("workspace_bounds_count", 1, "workspace_bounds"),
        ("nonfinite_state_count", 1, "finite_state"),
        ("nonfinite_action_count", 1, "finite_action"),
        ("action_saturation_count", 1, "unsaturated_action"),
        ("max_tilt_deg", float("nan"), "finite_summary"),
    ],
)
def test_push_qualification_gates_reject_each_registered_failure(
    field: str,
    value: object,
    failed_gate: str,
):
    result = _passing_push_qualification_result()
    result[field] = value

    gates = _push_qualification_gates(
        result,
        goal_threshold=0.08,
        object_speed_limit=0.5,
        object_height_tolerance=0.02,
    )

    assert gates[failed_gate] is False
