from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPO_ROOT / "experiments" / "2026-07-22-push-2mm-randomized-training"
)


def _seed_range(spec: dict[str, int]) -> set[int]:
    return set(range(spec["seed_start"], spec["seed_start"] + spec["seed_count"]))


def test_randomized_training_plan_binds_the_committed_config() -> None:
    plan = json.loads((EXPERIMENT_ROOT / "manifest.json").read_text())
    config_path = REPO_ROOT / plan["candidate"]["config_path"]
    config = yaml.safe_load(config_path.read_text())

    assert plan["status"] == "pre_registered"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == plan["candidate"][
        "config_sha256"
    ]
    assert config["seed"] == plan["candidate"]["training_seed"]
    assert config["ppo"]["num_timesteps"] == plan["candidate"]["num_timesteps"]
    assert config["ppo"]["num_envs"] == plan["candidate"]["num_envs"]
    assert config["ppo"]["episode_length"] == plan["candidate"][
        "episode_length"
    ]


def test_randomized_training_plan_keeps_qualification_seeds_blind() -> None:
    plan = json.loads((EXPERIMENT_ROOT / "manifest.json").read_text())
    qualification = plan["qualification"]
    development = set(range(2026072200, 2026072300)) | {777, 778}
    screening = _seed_range(qualification["screening"])
    blind = _seed_range(qualification["blind"])

    assert development.isdisjoint(screening)
    assert development.isdisjoint(blind)
    assert screening.isdisjoint(blind)
    assert max(screening | blind) < 2**31
    assert qualification["screening"]["required_gate_pass_count"] == len(
        screening
    )
    assert qualification["blind"]["required_gate_pass_count"] == len(blind)
    assert qualification["screening"]["required_runtime_failure_count"] == 0
    assert qualification["blind"]["required_runtime_failure_count"] == 0


def test_randomized_training_plan_cannot_relax_frozen_gates() -> None:
    plan = json.loads((EXPERIMENT_ROOT / "manifest.json").read_text())
    rules = plan["decision_rules"]

    assert rules["run_blind_only_after_screening_pass"] is True
    assert rules["promote_only_after_blind_pass"] is True
    assert rules["change_goal_threshold"] is False
    assert rules["change_object_speed_limit"] is False
    assert rules["change_object_height_tolerance"] is False
    assert rules["reuse_development_failure_seeds_for_qualification"] is False
    assert rules["move_existing_submission_tag"] is False
