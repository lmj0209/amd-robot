from __future__ import annotations

import json
from copy import deepcopy

import pytest

from scripts.push_qualification_matrix import (
    _load_valid_attempt,
    _next_attempt,
    _seed_sequence,
    _summary,
    _validate_matrix_spec,
    _validate_seed_manifest,
    _wilson_interval,
)


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "commit": "a" * 40,
        "config_path": "/workspace/amd-robot/configs/push.yaml",
        "config_sha256": "b" * 64,
        "params_name": "push_params",
        "params_sha256": "c" * 64,
        "seeds": [100, 101],
        "num_envs": 1,
        "num_steps": 4608,
        "implementation": "chunked_jit",
        "chunk_steps": 2,
        "policy": "trained",
        "python": "/workspace/.venv/bin/python",
        "runner": "/workspace/amd-robot/scripts/locomotion_learn_smoke.py",
    }


def _trained_result() -> dict[str, object]:
    gates = {
        "task_success": True,
        "successful_terminal": True,
        "finite_state": True,
    }
    return {
        "qualification_gates": gates,
        "qualification_gate_pass": True,
    }


def _manifest() -> dict[str, object]:
    spec = _spec()
    return {
        "schema_version": 1,
        "status": "complete",
        "commit": spec["commit"],
        "config_path": spec["config_path"],
        "config_sha256": spec["config_sha256"],
        "params_path": spec["params_name"],
        "params_sha256": spec["params_sha256"],
        "seed": 100,
        "num_envs": 1,
        "num_steps": 4608,
        "implementation": "chunked_jit",
        "chunk_steps": 2,
        "jax_backend": "gpu",
        "jax_devices": ["rocm:0"],
        "results": {"trained": _trained_result()},
    }


def _record(seed: int, gate_pass: bool = True) -> dict[str, object]:
    return {
        "seed": seed,
        "gate_pass": gate_pass,
        "success": gate_pass,
        "final_goal_distance": 0.05,
        "maximum_object_speed": 0.3,
        "minimum_object_height": 0.09,
        "maximum_object_height": 0.11,
        "maximum_tilt_deg": 8.0,
        "maximum_abs_action": 0.8,
        "illegal_contact_count": 0,
        "workspace_bounds_count": 0,
        "nonfinite_state_count": 0,
        "nonfinite_action_count": 0,
        "action_saturation_count": 0,
    }


def test_seed_sequence_is_predeclared_and_bounded():
    assert _seed_sequence(100, 3) == [100, 101, 102]
    with pytest.raises(ValueError, match="non-negative"):
        _seed_sequence(-1, 3)
    with pytest.raises(ValueError, match="positive"):
        _seed_sequence(100, 0)
    with pytest.raises(ValueError, match="32-bit"):
        _seed_sequence(2**31 - 1, 2)


def test_valid_seed_manifest_is_bound_to_rocm_inputs():
    trained = _validate_seed_manifest(_manifest(), spec=_spec(), seed=100)
    assert trained["qualification_gate_pass"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit", "d" * 40),
        ("config_sha256", "d" * 64),
        ("params_sha256", "d" * 64),
        ("seed", 101),
        ("num_envs", 2),
        ("jax_backend", "cpu"),
        ("jax_devices", ["cuda:0"]),
    ],
)
def test_seed_manifest_rejects_provenance_mismatch(field: str, value: object):
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match="binding failed"):
        _validate_seed_manifest(manifest, spec=_spec(), seed=100)


def test_seed_manifest_accepts_legacy_rocm_device_rendering():
    manifest = _manifest()
    manifest["jax_devices"] = ["RocmDevice(id=0)"]
    trained = _validate_seed_manifest(manifest, spec=_spec(), seed=100)
    assert trained["qualification_gate_pass"] is True


def test_seed_manifest_rejects_inconsistent_gate_rollup():
    manifest = _manifest()
    manifest["results"]["trained"]["qualification_gates"]["finite_state"] = False
    with pytest.raises(ValueError, match="aggregate gate is inconsistent"):
        _validate_seed_manifest(manifest, spec=_spec(), seed=100)


def test_resume_requires_an_identical_specification():
    actual = _spec()
    _validate_matrix_spec(actual, deepcopy(actual))
    expected = deepcopy(actual)
    expected["chunk_steps"] = 8
    with pytest.raises(ValueError, match="chunk_steps"):
        _validate_matrix_spec(actual, expected)


def test_resume_preserves_attempts_and_reuses_only_valid_evidence(tmp_path):
    seed_dir = tmp_path / "seed_0000000100"
    interrupted = _next_attempt(seed_dir)
    assert interrupted.name == "attempt_001"
    assert _load_valid_attempt(seed_dir, spec=_spec(), seed=100) is None

    complete = _next_attempt(seed_dir)
    assert complete.name == "attempt_002"
    (complete / "qualification").mkdir()
    (complete / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (complete / "qualification" / "manifest.json").write_text(
        json.dumps(_manifest()),
        encoding="utf-8",
    )

    valid = _load_valid_attempt(seed_dir, spec=_spec(), seed=100)
    assert valid is not None
    assert valid[0] == complete


def test_summary_is_fail_closed_and_reports_wilson_interval():
    complete = [_record(seed) for seed in range(100)]
    summary = _summary(complete, expected_count=100)
    assert summary["status"] == "pass"
    assert summary["success_rate"] == 1.0
    assert 0.96 < summary["success_rate_wilson_95"][0] < 0.97
    assert summary["success_rate_wilson_95"][1] == pytest.approx(1.0)

    failed = complete[:-1] + [_record(999, gate_pass=False)]
    assert _summary(failed, expected_count=100)["status"] == "fail"
    assert _summary(complete[:99], expected_count=100)["status"] == "incomplete"


def test_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError, match="0 <= successes"):
        _wilson_interval(2, 1)
