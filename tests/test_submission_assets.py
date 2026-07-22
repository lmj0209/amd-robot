from __future__ import annotations

import json
from copy import deepcopy

import pytest

from report.render_submission_assets import (
    HEIGHT,
    WIDTH,
    _load_qualification_matrix,
    _matrix_evaluation_slide,
)
from scripts.push_qualification_matrix import _summary


def _record(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "gate_pass": True,
        "success": True,
        "final_goal_distance": 0.05 + (seed % 10) * 0.001,
        "maximum_object_speed": 0.3 + (seed % 10) * 0.001,
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


def _matrix_payload() -> dict[str, object]:
    seeds = list(range(1000, 1100))
    records = [_record(seed) for seed in seeds]
    summary = _summary(records, expected_count=100)
    return {
        "status": summary["status"],
        "spec": {
            "commit": "a" * 40,
            "config_sha256": "b" * 64,
            "params_sha256": "c" * 64,
            "seeds": seeds,
        },
        "summary": summary,
        "records": records,
    }


def _write_payload(tmp_path, payload: dict[str, object]):
    path = tmp_path / "matrix_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_completed_matrix_drives_a_deterministic_evidence_slide(tmp_path):
    matrix = _load_qualification_matrix(_write_payload(tmp_path, _matrix_payload()))

    assert matrix["status"] == "pass"
    assert matrix["success_count"] == 100
    assert matrix["gate_pass_count"] == 100
    assert matrix["seed_start"] == 1000
    assert matrix["seed_end"] == 1099
    assert _matrix_evaluation_slide(matrix).size == (WIDTH, HEIGHT)


def test_matrix_slide_rejects_incomplete_evidence(tmp_path):
    payload = _matrix_payload()
    payload["records"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        _load_qualification_matrix(_write_payload(tmp_path, payload))


def test_matrix_slide_rejects_inconsistent_safety_totals(tmp_path):
    payload = _matrix_payload()
    payload["summary"]["safety_totals"]["illegal_contact_count"] = 1
    with pytest.raises(ValueError, match="safety totals"):
        _load_qualification_matrix(_write_payload(tmp_path, payload))


def test_matrix_slide_rejects_seed_reordering(tmp_path):
    payload = _matrix_payload()
    payload["records"][0], payload["records"][1] = (
        payload["records"][1],
        payload["records"][0],
    )
    with pytest.raises(ValueError, match="seed order"):
        _load_qualification_matrix(_write_payload(tmp_path, payload))


def test_matrix_slide_rejects_nonfinite_metrics(tmp_path):
    payload = deepcopy(_matrix_payload())
    payload["records"][0]["maximum_object_speed"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _load_qualification_matrix(_write_payload(tmp_path, payload))
