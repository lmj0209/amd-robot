from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_locomotion_config_is_rocm_safe_and_preserves_action_contract() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "locomotion.yaml").read_text())

    assert config["status"] == "locomotion_gait_curriculum"
    assert config["algorithm"] == "brax_ppo"
    assert config["environment"]["control_timestep"] == 0.01
    assert config["environment"]["foot_condim"] == 6
    assert config["environment"]["command_x_range"] == [0.2, 0.6]
    assert config["environment"]["zero_command_probability"] == 0.1
    assert config["reward"]["profile"] == "forward_gait_curriculum_v4"
    assert config["reward"]["tracking_linear_velocity_scale"] == 5.0
    assert config["reward"]["orientation_cost_scale"] == 5.0
    assert config["reward"]["feet_clearance_cost_scale"] == 0.5
    assert config["reward"]["arm_action_magnitude_cost_scale"] == 0.01
    assert config["reward"]["feet_air_time_scale"] == 0.1
    assert config["reward"]["max_foot_height"] == 0.1
    assert config["rocm_guardrails"] == {
        "max_physics_substeps_per_control": 5,
        "max_training_steps_per_host_call": 2,
        "brax_run_evals": False,
    }
    assert config["ppo"]["num_timesteps"] == 2097152
    assert config["ppo"]["num_envs"] == 256
    assert config["ppo"]["batch_size"] == 64
    assert config["ppo"]["learning_rate"] == 0.00003
    assert config["ppo"]["learning_rate_schedule"] == "NONE"
    assert config["ppo"]["normalize_observations"] is False
    assert config["ppo"]["entropy_cost"] == 0.01
    assert config["checkpoint"]["interval_steps"] == 524288
    assert config["manual_evaluation"]["fixed_command"] == [0.4, 0.0, 0.0]


def test_training_defaults_match_the_active_locomotion_stage() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text())

    assert config["ppo"] == {
        "num_timesteps": 2097152,
        "num_envs": 256,
        "episode_length": 256,
        "learning_rate": 0.00003,
        "checkpoint_interval": 524288,
    }


def test_low_speed_qualification_preserves_safe_v14_settings() -> None:
    current = yaml.safe_load(
        (REPO_ROOT / "configs" / "locomotion.yaml").read_text()
    )
    low_speed = yaml.safe_load(
        (REPO_ROOT / "configs" / "locomotion_stage1_low_speed.yaml").read_text()
    )

    assert low_speed["status"] == "locomotion_stage1_low_speed_qualification"
    assert low_speed["environment"]["command_x_range"] == [0.08, 0.12]
    assert low_speed["manual_evaluation"]["fixed_command"] == [0.1, 0.0, 0.0]
    assert low_speed["ppo"]["num_timesteps"] == 524288
    assert low_speed["checkpoint"]["scope"] == "full_training_session"
    assert low_speed["checkpoint"]["includes_rollout_state"] is True
    assert low_speed["rocm_guardrails"] == current["rocm_guardrails"]

    for key, value in current["reward"].items():
        if key != "profile":
            assert low_speed["reward"][key] == value


def test_sensitive_low_speed_qualification_changes_only_the_intended_knobs() -> None:
    first = yaml.safe_load(
        (REPO_ROOT / "configs" / "locomotion_stage1_low_speed.yaml").read_text()
    )
    sensitive = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_low_speed_sensitive.yaml"
        ).read_text()
    )

    assert sensitive["reward"]["tracking_sigma"] == 0.025
    assert sensitive["ppo"]["learning_rate"] == 0.00001
    assert sensitive["ppo"]["desired_kl"] == 0.005
    assert sensitive["environment"] == first["environment"]
    assert sensitive["manual_evaluation"] == first["manual_evaluation"]
    assert sensitive["rocm_guardrails"] == first["rocm_guardrails"]

    for section in ("reward", "ppo"):
        first_values = dict(first[section])
        sensitive_values = dict(sensitive[section])
        for key in (
            ("profile", "tracking_sigma")
            if section == "reward"
            else ("learning_rate", "desired_kl")
        ):
            first_values.pop(key)
            sensitive_values.pop(key)
        assert sensitive_values == first_values
