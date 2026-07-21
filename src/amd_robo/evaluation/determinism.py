"""Stable fingerprints and first-divergence reports for rollout traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TraceDivergence:
    """The first sampled rollout value that differs between two traces."""

    step_index: int
    environment_index: int
    field: str
    component_index: int
    reference_value: float | int | bool
    candidate_value: float | int | bool
    maximum_absolute_difference: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _array_payload(name: str, value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(np.asarray(value))
    metadata = json.dumps(
        {
            "name": name,
            "dtype": array.dtype.str,
            "shape": array.shape,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return metadata + b"\0" + array.tobytes(order="C")


def array_sha256(value: np.ndarray, *, name: str = "array") -> str:
    """Hashes array metadata and C-order bytes without dtype coercion."""

    return hashlib.sha256(_array_payload(name, value)).hexdigest()


def trace_sha256(fields: Mapping[str, np.ndarray]) -> str:
    """Hashes a named trace mapping in stable field order."""

    digest = hashlib.sha256()
    for name in sorted(fields):
        digest.update(_array_payload(name, fields[name]))
        digest.update(b"\0")
    return digest.hexdigest()


def trace_summary(fields: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Returns an aggregate digest plus a digest for every trace field."""

    if not fields:
        raise ValueError("trace fields cannot be empty")
    return {
        "sha256": trace_sha256(fields),
        "fields": {
            name: {
                "sha256": array_sha256(value, name=name),
                "dtype": np.asarray(value).dtype.str,
                "shape": list(np.asarray(value).shape),
            }
            for name, value in sorted(fields.items())
        },
    }


def normalized_discrete_outcome(
    record: Mapping[str, object],
    field_names: tuple[str, ...],
) -> dict[str, object]:
    """Normalizes array-like outcomes across memory and JSON representations."""

    return {name: np.asarray(record[name]).tolist() for name in field_names}


def push_evaluation_execution_mode(
    num_envs: int,
    *,
    determinism_audit: bool,
    allow_batched_diagnostic: bool,
) -> str:
    """Classifies fail-closed Push qualification versus batch diagnostics."""

    if num_envs <= 0:
        raise ValueError("Push evaluation environment count must be positive")
    if num_envs == 1:
        return "single_env_qualification"
    if determinism_audit or allow_batched_diagnostic:
        return "batched_diagnostic"
    raise ValueError(
        "Push qualification requires --eval-num-envs 1 because gfx1100 "
        "batched MJX evaluation changed discrete outcomes; use "
        "--allow-batched-push-eval only for explicit diagnostics"
    )


def _python_scalar(value: np.generic) -> float | int | bool:
    scalar = value.item()
    if isinstance(scalar, (bool, int, float)):
        return scalar
    raise TypeError(f"unsupported trace scalar: {type(scalar)}")


def _maximum_absolute_difference(
    reference: np.ndarray, candidate: np.ndarray
) -> float | None:
    if reference.dtype.kind not in "fc" and candidate.dtype.kind not in "fc":
        return None
    difference = np.abs(
        reference.astype(np.float64) - candidate.astype(np.float64)
    )
    finite = difference[np.isfinite(difference)]
    return float(np.max(finite)) if finite.size else None


def first_trace_divergence(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> TraceDivergence | None:
    """Finds the first differing step, field, environment, and component."""

    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("trace tolerances must be non-negative")
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise ValueError(f"trace fields differ: missing={missing} extra={extra}")

    for field in sorted(reference):
        reference_value = np.asarray(reference[field])
        candidate_value = np.asarray(candidate[field])
        if reference_value.shape != candidate_value.shape:
            raise ValueError(
                f"trace shape differs for {field}: "
                f"{reference_value.shape} != {candidate_value.shape}"
            )
        if reference_value.ndim < 2:
            raise ValueError(
                f"trace field {field} must have step and environment axes"
            )

    step_count = np.asarray(next(iter(reference.values()))).shape[0]
    for field in reference:
        if np.asarray(reference[field]).shape[0] != step_count:
            raise ValueError("reference trace fields have different step counts")
        if np.asarray(candidate[field]).shape[0] != step_count:
            raise ValueError("candidate trace fields have different step counts")

    for step_index in range(step_count):
        for field in sorted(reference):
            reference_step = np.asarray(reference[field])[step_index]
            candidate_step = np.asarray(candidate[field])[step_index]
            if reference_step.shape != candidate_step.shape:
                raise ValueError(
                    f"trace step shape differs for {field}: "
                    f"{reference_step.shape} != {candidate_step.shape}"
                )
            if (
                reference_step.dtype.kind in "fc"
                or candidate_step.dtype.kind in "fc"
            ):
                equal = np.isclose(
                    reference_step,
                    candidate_step,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    equal_nan=True,
                )
            else:
                equal = reference_step == candidate_step
            if np.all(equal):
                continue

            flat_index = int(np.flatnonzero(~equal)[0])
            component_shape = reference_step.shape[1:]
            environment_index = int(
                np.unravel_index(flat_index, reference_step.shape)[0]
            )
            if component_shape:
                component_size = int(np.prod(component_shape))
                component_index = flat_index % component_size
            else:
                component_index = 0
            reference_flat = reference_step.reshape(-1)
            candidate_flat = candidate_step.reshape(-1)
            return TraceDivergence(
                step_index=step_index,
                environment_index=environment_index,
                field=field,
                component_index=component_index,
                reference_value=_python_scalar(reference_flat[flat_index]),
                candidate_value=_python_scalar(candidate_flat[flat_index]),
                maximum_absolute_difference=_maximum_absolute_difference(
                    reference_step, candidate_step
                ),
            )
    return None


def sha256_path(path: str | Path) -> str:
    """Hashes one file or a directory tree with stable relative paths."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        return hashlib.sha256(source.read_bytes()).hexdigest()
    if not source.is_dir():
        raise ValueError(f"unsupported artifact path: {source}")

    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory is empty: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
