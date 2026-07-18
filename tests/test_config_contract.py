from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_env_config_matches_python_contract() -> None:
    from amd_robo.contracts import ACTION_LAYOUT, REQUIRED_TERMINATION_SIGNALS

    config = yaml.safe_load((REPO_ROOT / "configs" / "env.yaml").read_text())
    assert config["implementation"] == "jax"
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
    assert config["environment"]["foot_condim"] == 6
    assert config["rocm_guardrails"] == {
        "max_training_steps_per_host_call": 2,
        "brax_run_evals": False,
    }
    assert config["manual_evaluation"]["implementation"] == ("sequential_python_loop")
    assert config["checkpoint"] == {
        "scope": "full_training_session",
        "interval_steps": 5120,
        "includes_rollout_state": True,
    }
    assert config["precision"]["status"] == "pending_multi_seed_ab"
