from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_config_matches_python_contract() -> None:
    from amd_robo.contracts import ACTION_LAYOUT, REQUIRED_TERMINATION_SIGNALS

    config = yaml.safe_load((REPO_ROOT / "configs" / "env.yaml").read_text())
    assert config["implementation"] == "jax"
    assert config["robot"]["route"] == "go2_z1"
    assert config["control"]["simulation_dt"] == 0.002
    assert config["control"]["control_dt"] == 0.01
    assert config["control"]["action_size"] == ACTION_LAYOUT.size
    assert config["control"]["action_layout"] == {
        "legs": [0, 12],
        "arm": [12, 18],
        "gripper": [18, 19],
    }
    assert set(config["termination_requires"]) == REQUIRED_TERMINATION_SIGNALS


def test_smoke_config_is_fail_closed() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "smoke.yaml").read_text())
    assert config == {
        "schema_version": 1,
        "require_rocm": True,
        "require_mjx_impl": "jax",
        "num_envs": 256,
        "num_steps": 1000,
        "require_playground": True,
        "require_ppo_update": True,
        "require_checkpoint_roundtrip": True,
    }


def test_standing_config_preserves_rocm_training_guardrails() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "standing.yaml").read_text())

    assert config["algorithm"] == "brax_ppo"
    assert config["environment"]["control_timestep"] == 0.01
    assert config["environment"]["foot_condim"] == 6
    assert config["reward"] == {
        "profile": "smooth_height_velocity_pose_v1",
        "termination_cost": 2.0,
        "height_sigma": 0.02,
        "linear_velocity_sigma": 0.25,
        "angular_velocity_sigma": 0.25,
        "linear_velocity_scale": 1.0,
        "angular_velocity_scale": 0.5,
        "pose_scale": 0.5,
        "alive_scale": 0.1,
        "action_cost_scale": 0.001,
        "action_rate_cost_scale": 0.01,
    }
    assert config["status"] == "standing_qualified_zero_residual"
    assert config["ppo"]["num_timesteps"] == 5120
    assert config["ppo"]["episode_length"] == 128
    assert config["ppo"]["learning_rate"] == 0.0001
    assert config["rocm_guardrails"] == {
        "max_physics_substeps_per_control": 5,
        "max_training_steps_per_host_call": 2,
        "brax_run_evals": False,
    }
    assert config["manual_evaluation"]["implementation"] == ("sequential_python_loop")
    assert config["manual_evaluation"]["repeat_count"] == 8
    assert config["manual_evaluation"]["height_tolerance"] == 0.2
    assert config["checkpoint"] == {
        "scope": "full_training_session",
        "interval_steps": 5120,
        "includes_rollout_state": True,
        "policy_snapshot_scope": "inference_params",
        "policy_snapshot_interval_steps": 5120,
    }
    assert config["precision"]["status"] == "default_measured_for_diagnostic"
