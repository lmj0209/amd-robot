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


def test_forward_extension_is_reward_compatible_with_v13() -> None:
    archived = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "evidence"
            / "locomotion_gait_v13_2026-07-18.yaml"
        ).read_text()
    )
    extension = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_forward_extension.yaml"
        ).read_text()
    )

    assert extension["environment"] == archived["environment"]
    assert extension["manual_evaluation"] == archived["manual_evaluation"]
    assert extension["reward"]["arm_action_magnitude_cost_scale"] == 0.0
    extension_reward = dict(extension["reward"])
    archived_reward = dict(archived["reward"])
    extension_reward.pop("profile")
    extension_reward.pop("arm_action_magnitude_cost_scale")
    archived_reward.pop("profile")
    assert extension_reward == archived_reward
    assert extension["ppo"]["num_timesteps"] == 524288


def test_action_scale_qualification_changes_only_the_control_range() -> None:
    extension = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_forward_extension.yaml"
        ).read_text()
    )
    action_scale = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_action_scale_qualification.yaml"
        ).read_text()
    )

    assert "action_scale" not in extension["environment"]
    assert action_scale["environment"]["action_scale"] == 0.3
    assert action_scale["environment"]["leg_kp"] == 40.0
    action_environment = dict(action_scale["environment"])
    action_environment.pop("action_scale")
    action_environment.pop("leg_kp")
    assert action_environment == extension["environment"]

    action_reward = dict(action_scale["reward"])
    extension_reward = dict(extension["reward"])
    action_reward.pop("profile")
    extension_reward.pop("profile")
    assert action_reward == extension_reward
    assert action_scale["ppo"] == extension["ppo"]


def test_pose_qualification_changes_only_the_moving_pose_reward() -> None:
    action_scale = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_action_scale_qualification.yaml"
        ).read_text()
    )
    pose = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_pose_qualification.yaml"
        ).read_text()
    )

    assert pose["reward"]["moving_pose_multiplier"] == 0.2
    assert pose["environment"] == action_scale["environment"]
    assert pose["ppo"] == action_scale["ppo"]
    assert pose["manual_evaluation"] == action_scale["manual_evaluation"]
    assert pose["rocm_guardrails"] == action_scale["rocm_guardrails"]

    pose_reward = dict(pose["reward"])
    action_reward = dict(action_scale["reward"])
    pose_reward.pop("profile")
    pose_reward.pop("moving_pose_multiplier")
    action_reward.pop("profile")
    assert pose_reward == action_reward


def test_mid_pose_qualification_changes_only_the_multiplier() -> None:
    aggressive = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_pose_qualification.yaml"
        ).read_text()
    )
    mid = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_pose_mid_qualification.yaml"
        ).read_text()
    )

    assert aggressive["reward"]["moving_pose_multiplier"] == 0.2
    assert mid["reward"]["moving_pose_multiplier"] == 0.6
    for section in ("environment", "ppo", "manual_evaluation", "rocm_guardrails"):
        assert mid[section] == aggressive[section]

    aggressive_reward = dict(aggressive["reward"])
    mid_reward = dict(mid["reward"])
    for key in ("profile", "moving_pose_multiplier"):
        aggressive_reward.pop(key)
        mid_reward.pop(key)
    assert mid_reward == aggressive_reward


def test_tracking_qualification_changes_only_linear_tracking_scale() -> None:
    baseline = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_action_scale_qualification.yaml"
        ).read_text()
    )
    tracking = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_tracking_qualification.yaml"
        ).read_text()
    )

    assert baseline["reward"]["tracking_linear_velocity_scale"] == 2.0
    assert tracking["reward"]["tracking_linear_velocity_scale"] == 3.0
    for section in ("environment", "ppo", "manual_evaluation", "rocm_guardrails"):
        assert tracking[section] == baseline[section]

    baseline_reward = dict(baseline["reward"])
    tracking_reward = dict(tracking["reward"])
    for key in ("profile", "tracking_linear_velocity_scale"):
        baseline_reward.pop(key)
        tracking_reward.pop(key)
    assert tracking_reward == baseline_reward


def test_network_probe_changes_only_network_and_probe_budget() -> None:
    baseline = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_action_scale_qualification.yaml"
        ).read_text()
    )
    probe = yaml.safe_load(
        (
            REPO_ROOT / "configs" / "locomotion_stage1_network_probe.yaml"
        ).read_text()
    )

    assert probe["ppo"]["policy_hidden_layer_sizes"] == [512, 256, 128]
    assert probe["ppo"]["value_hidden_layer_sizes"] == [512, 256, 128]
    assert probe["ppo"]["num_timesteps"] == 4096
    assert probe["manual_evaluation"]["enabled"] is False
    assert probe["checkpoint"]["interval_steps"] == 4096
    for section in ("environment", "reward", "rocm_guardrails"):
        baseline_values = dict(baseline[section])
        probe_values = dict(probe[section])
        if section == "reward":
            baseline_values.pop("profile")
            probe_values.pop("profile")
        assert probe_values == baseline_values

    baseline_ppo = dict(baseline["ppo"])
    probe_ppo = dict(probe["ppo"])
    for key in (
        "num_timesteps",
        "policy_hidden_layer_sizes",
        "value_hidden_layer_sizes",
    ):
        baseline_ppo.pop(key, None)
        probe_ppo.pop(key, None)
    assert probe_ppo == baseline_ppo


def test_network_qualification_promotes_only_probe_run_budget() -> None:
    probe = yaml.safe_load(
        (
            REPO_ROOT / "configs" / "locomotion_stage1_network_probe.yaml"
        ).read_text()
    )
    qualification = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_network_qualification.yaml"
        ).read_text()
    )

    assert qualification["ppo"]["num_timesteps"] == 524288
    assert qualification["manual_evaluation"]["enabled"] is True
    assert qualification["checkpoint"]["interval_steps"] == 262144
    for section in ("environment", "rocm_guardrails"):
        assert qualification[section] == probe[section]

    for section, changed_keys in (
        ("reward", {"profile"}),
        ("ppo", {"num_timesteps"}),
        ("manual_evaluation", {"enabled"}),
        ("checkpoint", {"interval_steps"}),
    ):
        probe_values = dict(probe[section])
        qualification_values = dict(qualification[section])
        for key in changed_keys:
            probe_values.pop(key)
            qualification_values.pop(key)
        assert qualification_values == probe_values


def test_network_extension_changes_only_additional_budget() -> None:
    qualification = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_network_qualification.yaml"
        ).read_text()
    )
    extension = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_network_extension.yaml"
        ).read_text()
    )

    assert extension["ppo"]["num_timesteps"] == 4718592
    for section in (
        "environment",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert extension[section] == qualification[section]

    for section, changed_keys in (
        ("reward", {"profile"}),
        ("ppo", {"num_timesteps"}),
    ):
        qualification_values = dict(qualification[section])
        extension_values = dict(extension[section])
        for key in changed_keys:
            qualification_values.pop(key)
            extension_values.pop(key)
        assert extension_values == qualification_values


def test_adaptive_kl_qualification_changes_only_lr_schedule() -> None:
    fixed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_network_qualification.yaml"
        ).read_text()
    )
    adaptive = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_adaptive_kl_qualification.yaml"
        ).read_text()
    )

    assert fixed["ppo"]["learning_rate_schedule"] == "NONE"
    assert adaptive["ppo"]["learning_rate_schedule"] == "ADAPTIVE_KL"
    for section in (
        "environment",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert adaptive[section] == fixed[section]

    for section, changed_keys in (
        ("reward", {"profile"}),
        ("ppo", {"learning_rate_schedule"}),
    ):
        fixed_values = dict(fixed[section])
        adaptive_values = dict(adaptive[section])
        for key in changed_keys:
            fixed_values.pop(key)
            adaptive_values.pop(key)
        assert adaptive_values == fixed_values


def test_low_lr_qualification_changes_only_learning_rate() -> None:
    fixed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_network_qualification.yaml"
        ).read_text()
    )
    low_lr = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_low_lr_qualification.yaml"
        ).read_text()
    )

    assert fixed["ppo"]["learning_rate"] == 0.00003
    assert low_lr["ppo"]["learning_rate"] == 0.00001
    for section in (
        "environment",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert low_lr[section] == fixed[section]

    for section, changed_keys in (
        ("reward", {"profile"}),
        ("ppo", {"learning_rate"}),
    ):
        fixed_values = dict(fixed[section])
        low_lr_values = dict(low_lr[section])
        for key in changed_keys:
            fixed_values.pop(key)
            low_lr_values.pop(key)
        assert low_lr_values == fixed_values


def test_low_lr_extension_changes_only_additional_budget() -> None:
    qualification = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_low_lr_qualification.yaml"
        ).read_text()
    )
    extension = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_low_lr_extension.yaml"
        ).read_text()
    )

    assert extension["ppo"]["num_timesteps"] == 4718592
    for section in (
        "environment",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert extension[section] == qualification[section]

    for section, changed_keys in (
        ("reward", {"profile"}),
        ("ppo", {"num_timesteps"}),
    ):
        qualification_values = dict(qualification[section])
        extension_values = dict(extension[section])
        for key in changed_keys:
            qualification_values.pop(key)
            extension_values.pop(key)
        assert extension_values == qualification_values


def test_trot_qualification_adds_only_phase_and_trot_rewards() -> None:
    baseline = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_low_lr_qualification.yaml"
        ).read_text()
    )
    trot = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_qualification.yaml"
        ).read_text()
    )

    assert trot["environment"]["gait_cycle_time"] == 0.5
    assert trot["reward"]["trot_contact_scale"] == 0.5
    assert trot["reward"]["trot_swing_height_cost_scale"] == 0.2
    for section in (
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert trot[section] == baseline[section]

    baseline_environment = dict(baseline["environment"])
    trot_environment = dict(trot["environment"])
    trot_environment.pop("gait_cycle_time")
    assert trot_environment == baseline_environment

    baseline_reward = dict(baseline["reward"])
    trot_reward = dict(trot["reward"])
    for key in (
        "profile",
        "trot_contact_scale",
        "trot_swing_height_cost_scale",
    ):
        trot_reward.pop(key)
    baseline_reward.pop("profile")
    assert trot_reward == baseline_reward


def test_trot_timing_qualification_changes_only_contact_shaping() -> None:
    phase = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_qualification.yaml"
        ).read_text()
    )
    timing = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_timing_qualification.yaml"
        ).read_text()
    )

    for section in (
        "environment",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert timing[section] == phase[section]

    assert timing["reward"]["trot_contact_scale"] == 0.0
    assert timing["reward"]["trot_timing_scale"] == 1.0
    assert timing["reward"]["trot_timing_std"] == 0.1
    assert timing["reward"]["trot_timing_max_error"] == 0.2

    phase_reward = dict(phase["reward"])
    timing_reward = dict(timing["reward"])
    for key in (
        "profile",
        "trot_contact_scale",
    ):
        phase_reward.pop(key)
        timing_reward.pop(key)
    for key in (
        "trot_timing_scale",
        "trot_timing_std",
        "trot_timing_max_error",
    ):
        timing_reward.pop(key)
    assert timing_reward == phase_reward


def test_crawl_reference_qualification_changes_only_required_contracts() -> None:
    baseline = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_low_lr_qualification.yaml"
        ).read_text()
    )
    crawl = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_reference_qualification.yaml"
        ).read_text()
    )

    assert crawl["environment"]["gait_cycle_time"] == 4.0
    assert crawl["environment"]["crawl_reference_enabled"] is True
    assert crawl["environment"]["crawl_stride"] == 0.08
    assert crawl["environment"]["crawl_shift"] == 0.06
    assert crawl["environment"]["crawl_lift"] == 0.45
    assert crawl["environment"]["crawl_min_air_time"] == 0.07
    assert crawl["environment"]["leg_kp"] == 50.0
    assert crawl["ppo"]["episode_length"] == 512
    assert crawl["manual_evaluation"]["num_steps"] == 800
    assert crawl["rocm_guardrails"] == baseline["rocm_guardrails"]
    assert crawl["checkpoint"] == baseline["checkpoint"]

    baseline_environment = dict(baseline["environment"])
    crawl_environment = dict(crawl["environment"])
    for key in (
        "gait_cycle_time",
        "crawl_reference_enabled",
        "crawl_stride",
        "crawl_shift",
        "crawl_lift",
        "crawl_min_air_time",
    ):
        crawl_environment.pop(key)
    baseline_environment["leg_kp"] = 50.0
    assert crawl_environment == baseline_environment

    baseline_reward = dict(baseline["reward"])
    crawl_reward = dict(crawl["reward"])
    baseline_reward.pop("profile")
    crawl_reward.pop("profile")
    assert crawl_reward == baseline_reward

    baseline_ppo = dict(baseline["ppo"])
    crawl_ppo = dict(crawl["ppo"])
    baseline_ppo["episode_length"] = 512
    assert crawl_ppo == baseline_ppo

    baseline_evaluation = dict(baseline["manual_evaluation"])
    crawl_evaluation = dict(crawl["manual_evaluation"])
    baseline_evaluation["num_steps"] = 800
    assert crawl_evaluation == baseline_evaluation


def test_foot_space_reference_qualification_freezes_cpu_gate_parameters() -> None:
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_foot_space_reference_qualification.yaml"
        ).read_text()
    )

    environment = config["environment"]
    assert config["status"] == (
        "locomotion_stage1_foot_space_reference_qualification"
    )
    assert environment["control_timestep"] == 0.01
    assert environment["action_scale"] == 0.1
    assert environment["leg_kp"] == 150.0
    assert environment["leg_kd"] == 8.0
    assert environment["foot_condim"] == 6
    assert environment["gait_cycle_time"] == 4.0
    assert environment["crawl_reference_enabled"] is True
    assert environment["crawl_foot_space_enabled"] is True
    assert environment["crawl_foot_step_length"] == 0.1
    assert environment["crawl_foot_clearance"] == 0.06
    assert environment["crawl_body_shift_x"] == 0.016
    assert environment["crawl_body_shift_y"] == 0.02
    assert environment["crawl_shift_end_fraction"] == 0.25
    assert environment["crawl_lift_start_fraction"] == 0.45
    assert environment["crawl_lift_end_fraction"] == 0.85
    assert environment["crawl_pose_reference_enabled"] is True
    assert config["manual_evaluation"]["fixed_command"] == [0.025, 0.0, 0.0]
    assert config["rocm_guardrails"]["max_training_steps_per_host_call"] == 1
    assert config["checkpoint"]["scope"] == "full_training_session"


def test_crawl_residual_qualification_changes_only_action_scale() -> None:
    reference = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_reference_qualification.yaml"
        ).read_text()
    )
    residual = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_residual_qualification.yaml"
        ).read_text()
    )

    assert residual["environment"]["action_scale"] == 0.1
    assert reference["environment"]["action_scale"] == 0.3
    for section in (
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert residual[section] == reference[section]

    reference_environment = dict(reference["environment"])
    residual_environment = dict(residual["environment"])
    reference_environment["action_scale"] = 0.1
    assert residual_environment == reference_environment

    reference_reward = dict(reference["reward"])
    residual_reward = dict(residual["reward"])
    reference_reward.pop("profile")
    residual_reward.pop("profile")
    assert residual_reward == reference_reward


def test_crawl_low_speed_changes_only_command_curriculum() -> None:
    residual = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_residual_qualification.yaml"
        ).read_text()
    )
    low_speed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_low_speed_qualification.yaml"
        ).read_text()
    )

    assert low_speed["environment"]["command_x_range"] == [0.02, 0.06]
    assert low_speed["manual_evaluation"]["fixed_command"] == [0.04, 0.0, 0.0]
    for section in ("ppo", "rocm_guardrails", "checkpoint"):
        assert low_speed[section] == residual[section]

    residual_environment = dict(residual["environment"])
    low_speed_environment = dict(low_speed["environment"])
    residual_environment["command_x_range"] = [0.02, 0.06]
    assert low_speed_environment == residual_environment

    residual_evaluation = dict(residual["manual_evaluation"])
    low_speed_evaluation = dict(low_speed["manual_evaluation"])
    residual_evaluation["fixed_command"] = [0.04, 0.0, 0.0]
    assert low_speed_evaluation == residual_evaluation

    residual_reward = dict(residual["reward"])
    low_speed_reward = dict(low_speed["reward"])
    residual_reward.pop("profile")
    low_speed_reward.pop("profile")
    assert low_speed_reward == residual_reward


def test_crawl_tracking_qualification_changes_only_tracking_sigma() -> None:
    low_speed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_low_speed_qualification.yaml"
        ).read_text()
    )
    tracking = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_tracking_qualification.yaml"
        ).read_text()
    )

    for section in (
        "environment",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert tracking[section] == low_speed[section]
    assert tracking["reward"]["tracking_sigma"] == 0.0025

    low_speed_reward = dict(low_speed["reward"])
    tracking_reward = dict(tracking["reward"])
    low_speed_reward["tracking_sigma"] = 0.0025
    low_speed_reward.pop("profile")
    tracking_reward.pop("profile")
    assert tracking_reward == low_speed_reward


def test_crawl_pose_reference_adds_only_the_reference_target_switch() -> None:
    tracking = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_tracking_qualification.yaml"
        ).read_text()
    )
    reference = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_pose_reference_qualification.yaml"
        ).read_text()
    )

    for section in (
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert reference[section] == tracking[section]

    tracking_environment = dict(tracking["environment"])
    reference_environment = dict(reference["environment"])
    reference_environment.pop("crawl_pose_reference_enabled")
    assert reference["environment"]["crawl_pose_reference_enabled"] is True
    assert reference_environment == tracking_environment

    tracking_reward = dict(tracking["reward"])
    reference_reward = dict(reference["reward"])
    tracking_reward.pop("profile")
    reference_reward.pop("profile")
    assert reference_reward == tracking_reward


def test_crawl_residual_regularization_changes_only_action_cost() -> None:
    reference = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_pose_reference_qualification.yaml"
        ).read_text()
    )
    regularized = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_crawl_residual_regularized_qualification.yaml"
        ).read_text()
    )

    for section in (
        "environment",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert regularized[section] == reference[section]

    assert reference["reward"]["action_magnitude_cost_scale"] == 0.001
    assert regularized["reward"]["action_magnitude_cost_scale"] == 0.01
    reference_reward = dict(reference["reward"])
    regularized_reward = dict(regularized["reward"])
    reference_reward["action_magnitude_cost_scale"] = 0.01
    reference_reward.pop("profile")
    regularized_reward.pop("profile")
    assert regularized_reward == reference_reward


def test_trot_dwell_qualification_adds_only_minimum_air_time() -> None:
    timing = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_timing_qualification.yaml"
        ).read_text()
    )
    dwell = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_dwell_qualification.yaml"
        ).read_text()
    )

    for section in (
        "environment",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert dwell[section] == timing[section]

    assert dwell["reward"]["trot_timing_min_air_time"] == 0.1
    timing_reward = dict(timing["reward"])
    dwell_reward = dict(dwell["reward"])
    timing_reward.pop("profile")
    dwell_reward.pop("profile")
    dwell_reward.pop("trot_timing_min_air_time")
    assert dwell_reward == timing_reward


def test_trot_dwell_extension_changes_only_additional_budget() -> None:
    qualification = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_dwell_qualification.yaml"
        ).read_text()
    )
    extension = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "locomotion_stage1_trot_dwell_extension.yaml"
        ).read_text()
    )

    assert extension["ppo"]["num_timesteps"] == 1572864
    for section in (
        "environment",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert extension[section] == qualification[section]

    qualification_reward = dict(qualification["reward"])
    extension_reward = dict(extension["reward"])
    qualification_reward.pop("profile")
    extension_reward.pop("profile")
    assert extension_reward == qualification_reward

    qualification_ppo = dict(qualification["ppo"])
    extension_ppo = dict(extension["ppo"])
    qualification_ppo.pop("num_timesteps")
    extension_ppo.pop("num_timesteps")
    assert extension_ppo == qualification_ppo
