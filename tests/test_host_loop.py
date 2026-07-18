import pytest

from amd_robo.training.host_loop import plan_brax_host_loop


@pytest.mark.parametrize(
    (
        "num_timesteps",
        "expected_host_calls",
        "expected_training_steps_per_call",
        "expected_actual_timesteps",
    ),
    (
        (512, 1, 2, 512),
        (5_120, 10, 2, 5_120),
        (20_480, 40, 2, 20_480),
        (5_000, 10, 2, 5_120),
        (100, 1, 1, 256),
    ),
)
def test_plan_bounds_compiled_scan_and_covers_requested_timesteps(
    num_timesteps,
    expected_host_calls,
    expected_training_steps_per_call,
    expected_actual_timesteps,
):
    plan = plan_brax_host_loop(
        num_timesteps=num_timesteps,
        env_steps_per_training_step=256,
        max_training_steps_per_call=2,
    )

    assert plan.host_calls == expected_host_calls
    assert plan.training_steps_per_call == expected_training_steps_per_call
    assert plan.brax_num_evals == expected_host_calls + 1
    assert plan.actual_timesteps == expected_actual_timesteps
    assert plan.actual_timesteps >= num_timesteps
    assert plan.training_steps_per_call <= 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("num_timesteps", 0),
        ("env_steps_per_training_step", 0),
        ("max_training_steps_per_call", 0),
    ),
)
def test_plan_rejects_nonpositive_inputs(field, value):
    kwargs = {
        "num_timesteps": 512,
        "env_steps_per_training_step": 256,
        "max_training_steps_per_call": 2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        plan_brax_host_loop(**kwargs)
