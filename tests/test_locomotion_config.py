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


def test_push_near_field_qualification_spans_the_physical_oracle() -> None:
    config = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_qualification.yaml"
        ).read_text()
    )

    assert config["status"] == "push_stage2_near_field_qualification"
    assert config["environment"]["randomized_reset"] is False
    assert config["environment"]["action_scale"] == 0.1
    assert config["push"]["goal_threshold"] == 0.08
    assert config["push"]["success_hold_steps"] == 100
    assert config["push"]["object_speed_cost_scale"] == 20.0
    assert config["push"]["object_height_tolerance"] == 0.02
    assert config["curriculum"] == {
        "deterministic_phases": ["APPROACH", "ALIGN", "HOLD"],
        "learned_residual_phases": ["PUSH"],
    }
    assert config["reward"]["profile"] == "push_stage2_near_field_push_only_v3"
    assert config["ppo"]["episode_length"] == 4608
    assert config["ppo"]["num_timesteps"] == (
        config["ppo"]["num_envs"] * config["ppo"]["episode_length"]
    )
    assert config["rocm_guardrails"]["max_training_steps_per_host_call"] == 1
    assert config["manual_evaluation"]["num_steps"] == 4608
    assert config["checkpoint"]["scope"] == "full_training_session"


def test_push_position_x_qualification_changes_only_initial_object_x() -> None:
    fixed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_qualification.yaml"
        ).read_text()
    )
    randomized = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_qualification.yaml"
        ).read_text()
    )

    assert randomized["status"] == "push_stage2_position_x_qualification"
    assert randomized["push"]["object_position_x_offset_range"] == [-0.02, 0.02]
    assert randomized["push"]["object_position_y_offset_range"] == [0.0, 0.0]
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert randomized[section] == fixed[section]

    fixed_push = dict(fixed["push"])
    randomized_push = dict(randomized["push"])
    randomized_push.pop("object_position_x_offset_range")
    randomized_push.pop("object_position_y_offset_range")
    assert randomized_push == fixed_push

    fixed_reward = dict(fixed["reward"])
    randomized_reward = dict(randomized["reward"])
    fixed_reward.pop("profile")
    randomized_reward.pop("profile")
    assert randomized_reward == fixed_reward


def test_push_position_x_1cm_qualification_halves_only_the_probe_range() -> None:
    probe = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_qualification.yaml"
        ).read_text()
    )
    qualification = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_qualification.yaml"
        ).read_text()
    )

    assert qualification["status"] == "push_stage2_position_x_1cm_qualification"
    assert qualification["push"]["object_position_x_offset_range"] == [-0.01, 0.01]
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert qualification[section] == probe[section]

    probe_push = dict(probe["push"])
    qualification_push = dict(qualification["push"])
    probe_push.pop("object_position_x_offset_range")
    qualification_push.pop("object_position_x_offset_range")
    assert qualification_push == probe_push

    probe_reward = dict(probe["reward"])
    qualification_reward = dict(qualification["reward"])
    probe_reward.pop("profile")
    qualification_reward.pop("profile")
    assert qualification_reward == probe_reward


def test_push_position_x_align85_changes_only_the_measured_align_gate() -> None:
    position = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_qualification.yaml"
        ).read_text()
    )
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )

    assert align85["status"] == (
        "push_stage2_position_x_1cm_align85_qualification"
    )
    assert position["push"]["align_distance_threshold"] == 0.08
    assert align85["push"]["align_distance_threshold"] == 0.085
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert align85[section] == position[section]

    position_push = dict(position["push"])
    align85_push = dict(align85["push"])
    position_push.pop("align_distance_threshold")
    align85_push.pop("align_distance_threshold")
    assert align85_push == position_push

    position_reward = dict(position["reward"])
    align85_reward = dict(align85["reward"])
    position_reward.pop("profile")
    align85_reward.pop("profile")
    assert align85_reward == position_reward


def test_push_position_x_ramp_changes_only_push_command_startup() -> None:
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )
    ramp_cases = (
        (
            "push_stage2_position_x_1cm_align85_ramp_qualification.yaml",
            "push_stage2_position_x_1cm_align85_ramp_qualification",
            1.0,
        ),
        (
            "push_stage2_position_x_1cm_align85_ramp050_qualification.yaml",
            "push_stage2_position_x_1cm_align85_ramp050_qualification",
            0.5,
        ),
        (
            "push_stage2_position_x_1cm_align85_ramp075_qualification.yaml",
            "push_stage2_position_x_1cm_align85_ramp075_qualification",
            0.75,
        ),
    )

    for config_name, status, duration in ramp_cases:
        ramp = yaml.safe_load((REPO_ROOT / "configs" / config_name).read_text())

        assert ramp["status"] == status
        assert ramp["push"]["push_command_ramp_duration"] == duration
        for section in (
            "environment",
            "curriculum",
            "ppo",
            "rocm_guardrails",
            "manual_evaluation",
            "checkpoint",
        ):
            assert ramp[section] == align85[section]

        align85_push = dict(align85["push"])
        ramp_push = dict(ramp["push"])
        ramp_push.pop("push_command_ramp_duration")
        assert ramp_push == align85_push

        align85_reward = dict(align85["reward"])
        ramp_reward = dict(ramp["reward"])
        align85_reward.pop("profile")
        ramp_reward.pop("profile")
        assert ramp_reward == align85_reward


def test_push_position_x_safety_margin_changes_only_penalty_thresholds() -> None:
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )
    margin = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_safety_margin_qualification.yaml"
        ).read_text()
    )

    assert margin["status"] == (
        "push_stage2_position_x_1cm_align85_safety_margin_qualification"
    )
    assert margin["push"]["object_speed_limit"] == 0.35
    assert margin["push"]["object_height_tolerance"] == 0.015
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert margin[section] == align85[section]

    align85_push = dict(align85["push"])
    margin_push = dict(margin["push"])
    for key in ("object_speed_limit", "object_height_tolerance"):
        align85_push.pop(key)
        margin_push.pop(key)
    assert margin_push == align85_push

    align85_reward = dict(align85["reward"])
    margin_reward = dict(margin["reward"])
    align85_reward.pop("profile")
    margin_reward.pop("profile")
    assert margin_reward == align85_reward


def test_push_position_x_phase_sync_changes_only_align_phase_timing() -> None:
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )
    phase_sync = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_sync_qualification.yaml"
        ).read_text()
    )

    assert phase_sync["status"] == (
        "push_stage2_position_x_1cm_align85_phase_sync_qualification"
    )
    assert phase_sync["push"]["align_gait_phase_sync"] is True
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert phase_sync[section] == align85[section]

    align85_push = dict(align85["push"])
    phase_sync_push = dict(phase_sync["push"])
    phase_sync_push.pop("align_gait_phase_sync")
    assert phase_sync_push == align85_push

    align85_reward = dict(align85["reward"])
    phase_sync_reward = dict(phase_sync["reward"])
    align85_reward.pop("profile")
    phase_sync_reward.pop("profile")
    assert phase_sync_reward == align85_reward


def test_push_position_x_residual050_changes_only_action_scale() -> None:
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )
    residual050 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_residual050_qualification.yaml"
        ).read_text()
    )

    assert residual050["status"] == (
        "push_stage2_position_x_1cm_align85_residual050_qualification"
    )
    assert residual050["environment"]["action_scale"] == 0.05
    for section in (
        "push",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert residual050[section] == align85[section]

    align85_environment = dict(align85["environment"])
    residual050_environment = dict(residual050["environment"])
    align85_environment.pop("action_scale")
    residual050_environment.pop("action_scale")
    assert residual050_environment == align85_environment

    align85_reward = dict(align85["reward"])
    residual050_reward = dict(residual050["reward"])
    align85_reward.pop("profile")
    residual050_reward.pop("profile")
    assert residual050_reward == align85_reward


def test_push_position_x_phase_sync085_changes_only_align_phase_timing() -> None:
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )
    phase_sync = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_sync085_qualification.yaml"
        ).read_text()
    )

    assert phase_sync["status"] == (
        "push_stage2_position_x_1cm_align85_phase_sync085_qualification"
    )
    assert phase_sync["push"]["align_gait_phase_sync"] is True
    assert phase_sync["push"]["align_gait_phase_fraction"] == 0.85
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert phase_sync[section] == align85[section]

    align85_push = dict(align85["push"])
    phase_sync_push = dict(phase_sync["push"])
    phase_sync_push.pop("align_gait_phase_sync")
    phase_sync_push.pop("align_gait_phase_fraction")
    assert phase_sync_push == align85_push

    align85_reward = dict(align85["reward"])
    phase_sync_reward = dict(phase_sync["reward"])
    align85_reward.pop("profile")
    phase_sync_reward.pop("profile")
    assert phase_sync_reward == align85_reward


def test_push_position_x_phase_entry035_changes_only_align_phase_timing() -> None:
    align85 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_qualification.yaml"
        ).read_text()
    )
    phase_entry = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_entry035_qualification.yaml"
        ).read_text()
    )

    assert phase_entry["status"] == (
        "push_stage2_position_x_1cm_align85_phase_entry035_qualification"
    )
    assert phase_entry["push"]["align_entry_gait_phase_fraction"] == 0.35
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert phase_entry[section] == align85[section]

    align85_push = dict(align85["push"])
    phase_entry_push = dict(phase_entry["push"])
    phase_entry_push.pop("align_entry_gait_phase_fraction")
    assert phase_entry_push == align85_push

    align85_reward = dict(align85["reward"])
    phase_entry_reward = dict(phase_entry["reward"])
    align85_reward.pop("profile")
    phase_entry_reward.pop("profile")
    assert phase_entry_reward == align85_reward


def test_push_position_x_phase_entry035_residual095_changes_only_scale() -> None:
    phase_entry = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_entry035_qualification.yaml"
        ).read_text()
    )
    residual095 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align85_"
                "phase_entry035_residual095_qualification.yaml"
            )
        ).read_text()
    )

    assert residual095["status"] == (
        "push_stage2_position_x_1cm_align85_"
        "phase_entry035_residual095_qualification"
    )
    assert residual095["environment"]["action_scale"] == 0.095
    for section in (
        "push",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert residual095[section] == phase_entry[section]

    phase_entry_environment = dict(phase_entry["environment"])
    residual095_environment = dict(residual095["environment"])
    phase_entry_environment.pop("action_scale")
    residual095_environment.pop("action_scale")
    assert residual095_environment == phase_entry_environment

    phase_entry_reward = dict(phase_entry["reward"])
    residual095_reward = dict(residual095["reward"])
    phase_entry_reward.pop("profile")
    residual095_reward.pop("profile")
    assert residual095_reward == phase_entry_reward


def test_push_position_x_contact_soft040_changes_only_box_contact() -> None:
    phase_entry = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_entry035_qualification.yaml"
        ).read_text()
    )
    contact_soft = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align85_"
                "phase_entry035_contact_soft040_qualification.yaml"
            )
        ).read_text()
    )

    assert contact_soft["status"] == (
        "push_stage2_position_x_1cm_align85_"
        "phase_entry035_contact_soft040_qualification"
    )
    assert contact_soft["push"]["push_box_solref_timeconst"] == 0.04
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert contact_soft[section] == phase_entry[section]

    phase_entry_push = dict(phase_entry["push"])
    contact_soft_push = dict(contact_soft["push"])
    contact_soft_push.pop("push_box_solref_timeconst")
    assert contact_soft_push == phase_entry_push

    phase_entry_reward = dict(phase_entry["reward"])
    contact_soft_reward = dict(contact_soft["reward"])
    phase_entry_reward.pop("profile")
    contact_soft_reward.pop("profile")
    assert contact_soft_reward == phase_entry_reward


def test_push_position_x_pad_soft040_changes_only_pad_contact() -> None:
    phase_entry = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_entry035_qualification.yaml"
        ).read_text()
    )
    pad_soft = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align85_"
                "phase_entry035_pad_soft040_qualification.yaml"
            )
        ).read_text()
    )

    assert pad_soft["status"] == (
        "push_stage2_position_x_1cm_align85_"
        "phase_entry035_pad_soft040_qualification"
    )
    assert pad_soft["push"]["push_pad_solref_timeconst"] == 0.04
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert pad_soft[section] == phase_entry[section]

    phase_entry_push = dict(phase_entry["push"])
    pad_soft_push = dict(pad_soft["push"])
    pad_soft_push.pop("push_pad_solref_timeconst")
    assert pad_soft_push == phase_entry_push

    phase_entry_reward = dict(phase_entry["reward"])
    pad_soft_reward = dict(pad_soft["reward"])
    phase_entry_reward.pop("profile")
    pad_soft_reward.pop("profile")
    assert pad_soft_reward == phase_entry_reward


def test_push_position_x_arm_compliance_changes_only_arm_pd() -> None:
    phase_entry = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_entry035_qualification.yaml"
        ).read_text()
    )
    arm_compliance = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align85_"
                "phase_entry035_arm_compliance_qualification.yaml"
            )
        ).read_text()
    )

    assert arm_compliance["status"] == (
        "push_stage2_position_x_1cm_align85_"
        "phase_entry035_arm_compliance_qualification"
    )
    assert arm_compliance["environment"]["arm_kp"] == 300.0
    assert arm_compliance["environment"]["arm_kd"] == 30.0
    for section in (
        "push",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert arm_compliance[section] == phase_entry[section]

    phase_entry_environment = dict(phase_entry["environment"])
    arm_compliance_environment = dict(arm_compliance["environment"])
    arm_compliance_environment.pop("arm_kp")
    arm_compliance_environment.pop("arm_kd")
    assert arm_compliance_environment == phase_entry_environment

    phase_entry_reward = dict(phase_entry["reward"])
    arm_compliance_reward = dict(arm_compliance["reward"])
    phase_entry_reward.pop("profile")
    arm_compliance_reward.pop("profile")
    assert arm_compliance_reward == phase_entry_reward


def test_push_position_x_arm_compliance500_changes_only_arm_pd() -> None:
    phase_entry = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_position_x_1cm_align85_phase_entry035_qualification.yaml"
        ).read_text()
    )
    arm_compliance = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align85_"
                "phase_entry035_arm_compliance500_qualification.yaml"
            )
        ).read_text()
    )

    assert arm_compliance["status"] == (
        "push_stage2_position_x_1cm_align85_"
        "phase_entry035_arm_compliance500_qualification"
    )
    assert arm_compliance["environment"]["arm_kp"] == 500.0
    assert arm_compliance["environment"]["arm_kd"] == 50.0
    for section in (
        "push",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert arm_compliance[section] == phase_entry[section]

    phase_entry_environment = dict(phase_entry["environment"])
    arm_compliance_environment = dict(arm_compliance["environment"])
    arm_compliance_environment.pop("arm_kp")
    arm_compliance_environment.pop("arm_kd")
    assert arm_compliance_environment == phase_entry_environment

    phase_entry_reward = dict(phase_entry["reward"])
    arm_compliance_reward = dict(arm_compliance["reward"])
    phase_entry_reward.pop("profile")
    arm_compliance_reward.pop("profile")
    assert arm_compliance_reward == phase_entry_reward


def test_push_position_x_compliant_arm_align90_changes_only_align_gate() -> None:
    arm_compliance = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align85_"
                "phase_entry035_arm_compliance_qualification.yaml"
            )
        ).read_text()
    )
    align90 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_"
                "phase_entry035_arm_compliance_qualification.yaml"
            )
        ).read_text()
    )

    assert align90["status"] == (
        "push_stage2_position_x_1cm_align90_"
        "phase_entry035_arm_compliance_qualification"
    )
    assert align90["environment"]["arm_kp"] == 300.0
    assert align90["environment"]["arm_kd"] == 30.0
    assert align90["push"]["align_distance_threshold"] == 0.09
    for section in (
        "environment",
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert align90[section] == arm_compliance[section]

    arm_compliance_push = dict(arm_compliance["push"])
    align90_push = dict(align90["push"])
    arm_compliance_push.pop("align_distance_threshold")
    align90_push.pop("align_distance_threshold")
    assert align90_push == arm_compliance_push

    arm_compliance_reward = dict(arm_compliance["reward"])
    align90_reward = dict(align90["reward"])
    arm_compliance_reward.pop("profile")
    align90_reward.pop("profile")
    assert align90_reward == arm_compliance_reward


def test_push_position_x_arm_residual_changes_only_arm_authority() -> None:
    compliant = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_"
                "phase_entry035_arm_compliance_qualification.yaml"
            )
        ).read_text()
    )
    arm_residual = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_"
                "phase_entry035_arm_residual_qualification.yaml"
            )
        ).read_text()
    )

    assert arm_residual["status"] == (
        "push_stage2_position_x_1cm_align90_"
        "phase_entry035_arm_residual_qualification"
    )
    assert arm_residual["environment"]["arm_action_scale"] == 0.02
    assert arm_residual["environment"]["mask_arm"] is False
    assert arm_residual["push"]["push_arm_residual_enabled"] is True
    assert arm_residual["reward"]["arm_action_magnitude_cost_scale"] == 0.01
    for section in (
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert arm_residual[section] == compliant[section]

    compliant_environment = dict(compliant["environment"])
    arm_residual_environment = dict(arm_residual["environment"])
    arm_residual_environment.pop("arm_action_scale")
    arm_residual_environment.pop("mask_arm")
    assert arm_residual_environment == compliant_environment

    compliant_push = dict(compliant["push"])
    arm_residual_push = dict(arm_residual["push"])
    arm_residual_push.pop("push_arm_residual_enabled")
    assert arm_residual_push == compliant_push

    compliant_reward = dict(compliant["reward"])
    arm_residual_reward = dict(arm_residual["reward"])
    compliant_reward.pop("profile")
    arm_residual_reward.pop("profile")
    compliant_reward["arm_action_magnitude_cost_scale"] = 0.01
    assert arm_residual_reward == compliant_reward


def test_push_position_x_arm_ee_x_projects_joint_authority() -> None:
    joint_residual = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_"
                "phase_entry035_arm_residual_qualification.yaml"
            )
        ).read_text()
    )
    ee_x = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_"
                "phase_entry035_arm_ee_x_qualification.yaml"
            )
        ).read_text()
    )

    assert ee_x["status"] == (
        "push_stage2_position_x_1cm_align90_"
        "phase_entry035_arm_ee_x_qualification"
    )
    assert ee_x["push"]["push_arm_residual_mode"] == "ee_x"
    assert ee_x["push"]["push_arm_ee_x_range"] == 0.005
    for section in (
        "curriculum",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert ee_x[section] == joint_residual[section]

    joint_environment = dict(joint_residual["environment"])
    ee_x_environment = dict(ee_x["environment"])
    joint_environment.pop("arm_action_scale")
    assert ee_x_environment == joint_environment

    joint_push = dict(joint_residual["push"])
    ee_x_push = dict(ee_x["push"])
    ee_x_push.pop("push_arm_residual_mode")
    ee_x_push.pop("push_arm_ee_x_range")
    assert ee_x_push == joint_push

    joint_reward = dict(joint_residual["reward"])
    ee_x_reward = dict(ee_x["reward"])
    joint_reward.pop("profile")
    ee_x_reward.pop("profile")
    assert ee_x_reward == joint_reward


def test_push_hold_entry_decay_changes_only_command_transition() -> None:
    base = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_"
                "phase_entry035_arm_ee_x_qualification.yaml"
            )
        ).read_text()
    )
    decay = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / (
                "push_stage2_position_x_1cm_align90_phase_entry035_"
                "arm_ee_x_hold_decay050_qualification.yaml"
            )
        ).read_text()
    )

    assert decay["status"] == (
        "push_stage2_position_x_1cm_align90_phase_entry035_"
        "arm_ee_x_hold_decay050_qualification"
    )
    assert decay["push"]["hold_entry_command_decay_duration"] == 0.5
    for section in (
        "environment",
        "curriculum",
        "reward",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert decay[section] == base[section]

    base_push = dict(base["push"])
    decay_push = dict(decay["push"])
    decay_push.pop("hold_entry_command_decay_duration")
    assert decay_push == base_push


def test_push_near_field_solver16_changes_only_solver_iterations() -> None:
    base = yaml.safe_load(
        (
            REPO_ROOT / "configs" / "push_stage2_near_field_qualification.yaml"
        ).read_text()
    )
    solver16 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_qualification.yaml"
        ).read_text()
    )

    assert solver16["status"] == "push_stage2_near_field_solver16_qualification"
    assert solver16["push"]["solver_iterations"] == 16
    for section in (
        "environment",
        "curriculum",
        "reward",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert solver16[section] == base[section]

    base_push = dict(base["push"])
    solver16_push = dict(solver16["push"])
    solver16_push.pop("solver_iterations")
    assert solver16_push == base_push


def test_push_near_field_governor_changes_only_command_envelope() -> None:
    solver16 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_qualification.yaml"
        ).read_text()
    )
    governor = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_governor_qualification.yaml"
        ).read_text()
    )

    assert governor["status"] == (
        "push_stage2_near_field_solver16_governor_qualification"
    )
    assert governor["push"]["object_speed_governor_start"] == 0.1
    assert governor["push"]["object_speed_governor_stop"] == 0.2
    assert governor["manual_evaluation"]["num_envs"] == 1
    assert governor["manual_evaluation"]["repeat_count"] == 20
    for section in (
        "environment",
        "curriculum",
        "reward",
        "ppo",
        "rocm_guardrails",
        "checkpoint",
    ):
        assert governor[section] == solver16[section]

    solver16_push = dict(solver16["push"])
    governor_push = dict(governor["push"])
    governor_push.pop("object_speed_governor_start")
    governor_push.pop("object_speed_governor_stop")
    assert governor_push == solver16_push

    solver16_evaluation = dict(solver16["manual_evaluation"])
    governor_evaluation = dict(governor["manual_evaluation"])
    for key in ("num_envs", "repeat_count"):
        solver16_evaluation.pop(key)
        governor_evaluation.pop(key)
    assert governor_evaluation == solver16_evaluation


def test_push_near_field_governor025_changes_only_stop_speed() -> None:
    governor020 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_governor_qualification.yaml"
        ).read_text()
    )
    governor025 = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_governor025_qualification.yaml"
        ).read_text()
    )

    assert governor025["status"] == (
        "push_stage2_near_field_solver16_governor025_qualification"
    )
    assert governor020["push"]["object_speed_governor_stop"] == 0.2
    assert governor025["push"]["object_speed_governor_stop"] == 0.25
    for section in (
        "environment",
        "curriculum",
        "reward",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert governor025[section] == governor020[section]

    governor020_push = dict(governor020["push"])
    governor025_push = dict(governor025["push"])
    governor020_push.pop("object_speed_governor_stop")
    governor025_push.pop("object_speed_governor_stop")
    assert governor025_push == governor020_push


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


def test_push_two_millimetre_probe_changes_only_initial_object_x() -> None:
    fixed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_qualification.yaml"
        ).read_text()
    )
    probe = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_solver16_position_x_2mm_diagnostic.yaml"
        ).read_text()
    )

    assert probe["status"] == (
        "push_stage2_near_field_solver16_position_x_2mm_diagnostic"
    )
    assert probe["push"]["object_position_x_offset_range"] == [-0.002, 0.002]
    assert probe["push"]["object_position_y_offset_range"] == [0.0, 0.0]
    assert probe["manual_evaluation"]["num_envs"] == 1

    for section in (
        "environment",
        "curriculum",
        "reward",
        "ppo",
        "rocm_guardrails",
        "checkpoint",
    ):
        assert probe[section] == fixed[section]

    fixed_push = dict(fixed["push"])
    probe_push = dict(probe["push"])
    probe_push.pop("object_position_x_offset_range")
    probe_push.pop("object_position_y_offset_range")
    assert probe_push == fixed_push

    fixed_evaluation = dict(fixed["manual_evaluation"])
    probe_evaluation = dict(probe["manual_evaluation"])
    fixed_evaluation["num_envs"] = 1
    assert probe_evaluation == fixed_evaluation


def test_push_two_millimetre_training_changes_only_initial_object_x() -> None:
    fixed = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_qualification.yaml"
        ).read_text()
    )
    randomized = yaml.safe_load(
        (
            REPO_ROOT
            / "configs"
            / "push_stage2_near_field_position_x_2mm_training.yaml"
        ).read_text()
    )

    assert randomized["status"] == (
        "push_stage2_near_field_position_x_2mm_training"
    )
    assert randomized["push"]["object_position_x_offset_range"] == [
        -0.002,
        0.002,
    ]
    assert randomized["push"]["object_position_y_offset_range"] == [0.0, 0.0]

    for section in (
        "environment",
        "curriculum",
        "reward",
        "ppo",
        "rocm_guardrails",
        "manual_evaluation",
        "checkpoint",
    ):
        assert randomized[section] == fixed[section]

    fixed_push = dict(fixed["push"])
    randomized_push = dict(randomized["push"])
    randomized_push.pop("object_position_x_offset_range")
    randomized_push.pop("object_position_y_offset_range")
    assert randomized_push == fixed_push
