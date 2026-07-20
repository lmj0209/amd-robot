from pathlib import Path

import numpy as np
import pytest

from amd_robo.evaluation.determinism import (
    array_sha256,
    first_trace_divergence,
    sha256_path,
    trace_sha256,
    trace_summary,
)


def test_trace_digest_is_stable_and_preserves_dtype_and_field_name() -> None:
    first = {
        "qpos": np.asarray([[[1.0, 2.0]]], dtype=np.float32),
        "done": np.asarray([[False]]),
    }
    repeated = {
        "done": np.asarray([[False]]),
        "qpos": np.asarray([[[1.0, 2.0]]], dtype=np.float32),
    }
    different_dtype = {
        "qpos": np.asarray([[[1.0, 2.0]]], dtype=np.float64),
        "done": np.asarray([[False]]),
    }

    assert trace_sha256(first) == trace_sha256(repeated)
    assert trace_sha256(first) != trace_sha256(different_dtype)
    assert array_sha256(first["qpos"], name="qpos") != array_sha256(
        first["qpos"], name="observation"
    )
    assert trace_summary(first)["fields"]["qpos"]["shape"] == [1, 1, 2]


def test_first_trace_divergence_reports_step_environment_and_component() -> None:
    reference = {
        "qpos": np.zeros((3, 2, 4), dtype=np.float32),
        "phase": np.zeros((3, 2), dtype=np.int32),
    }
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["qpos"][1, 1, 2] = 0.25
    candidate["phase"][2, 0] = 1

    divergence = first_trace_divergence(reference, candidate)

    assert divergence is not None
    assert divergence.step_index == 1
    assert divergence.environment_index == 1
    assert divergence.field == "qpos"
    assert divergence.component_index == 2
    assert divergence.reference_value == 0.0
    assert divergence.candidate_value == pytest.approx(0.25)
    assert divergence.maximum_absolute_difference == pytest.approx(0.25)


def test_first_trace_divergence_honors_float_tolerance() -> None:
    reference = {"value": np.zeros((2, 1, 1), dtype=np.float32)}
    candidate = {"value": np.full((2, 1, 1), 1e-7, dtype=np.float32)}

    assert first_trace_divergence(reference, candidate) is not None
    assert (
        first_trace_divergence(
            reference,
            candidate,
            absolute_tolerance=1e-6,
        )
        is None
    )


def test_first_trace_divergence_rejects_mismatched_contracts() -> None:
    reference = {"qpos": np.zeros((2, 1, 3), dtype=np.float32)}

    with pytest.raises(ValueError, match="trace fields differ"):
        first_trace_divergence(
            reference,
            {"qvel": np.zeros((2, 1, 3), dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="trace shape differs"):
        first_trace_divergence(
            reference,
            {"qpos": np.zeros((3, 1, 3), dtype=np.float32)},
        )


def test_sha256_path_covers_files_and_directory_layout(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "artifact.bin").write_bytes(b"payload")
    (second / "artifact.bin").write_bytes(b"payload")

    assert sha256_path(first / "artifact.bin") == sha256_path(
        second / "artifact.bin"
    )
    assert sha256_path(first) == sha256_path(second)

    (second / "nested").mkdir()
    (second / "nested" / "artifact.bin").write_bytes(b"payload")
    assert sha256_path(first) != sha256_path(second)
