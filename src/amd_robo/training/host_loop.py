"""Plan bounded Brax PPO scans driven by its existing host loop."""

from __future__ import annotations

from dataclasses import dataclass


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class BraxHostLoopPlan:
    """A host-loop layout for one call to ``brax.ppo.train``.

    Brax keeps ``TrainingState`` and the environment state alive across
    evaluation iterations. With ``run_evals=False``, those iterations are a
    convenient host loop around a fixed-size compiled ``training_epoch``.
    """

    requested_timesteps: int
    env_steps_per_training_step: int
    max_training_steps_per_call: int
    host_calls: int
    training_steps_per_call: int
    brax_num_evals: int
    actual_timesteps: int


def plan_brax_host_loop(
    *,
    num_timesteps: int,
    env_steps_per_training_step: int,
    max_training_steps_per_call: int,
) -> BraxHostLoopPlan:
    """Choose ``num_evals`` so each compiled Brax scan stays bounded.

    Brax 0.14.2 calculates the scan length as::

        ceil(num_timesteps / ((num_evals - 1) * env_steps_per_training_step))

    when ``num_evals > 1``. It then invokes that compiled epoch from a Python
    loop ``num_evals - 1`` times while carrying the complete ``TrainingState``.
    """

    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if env_steps_per_training_step <= 0:
        raise ValueError("env_steps_per_training_step must be positive")
    if max_training_steps_per_call <= 0:
        raise ValueError("max_training_steps_per_call must be positive")

    requested_training_steps = _ceil_div(num_timesteps, env_steps_per_training_step)
    host_calls = _ceil_div(requested_training_steps, max_training_steps_per_call)
    training_steps_per_call = _ceil_div(
        num_timesteps, host_calls * env_steps_per_training_step
    )
    if training_steps_per_call > max_training_steps_per_call:
        raise AssertionError("host-loop plan exceeded the requested scan bound")

    return BraxHostLoopPlan(
        requested_timesteps=num_timesteps,
        env_steps_per_training_step=env_steps_per_training_step,
        max_training_steps_per_call=max_training_steps_per_call,
        host_calls=host_calls,
        training_steps_per_call=training_steps_per_call,
        brax_num_evals=host_calls + 1,
        actual_timesteps=(
            host_calls * training_steps_per_call * env_steps_per_training_step
        ),
    )
