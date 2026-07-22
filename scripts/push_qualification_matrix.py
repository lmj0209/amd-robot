#!/usr/bin/env python3
"""Run and aggregate a fail-closed single-environment Push seed matrix.

Each seed is evaluated in a fresh process.  Existing attempts are immutable;
``--resume`` validates completed attempts and creates a new attempt after an
interrupted or failed one.  The launcher itself has no JAX dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _seed_sequence(seed_start: int, seed_count: int) -> list[int]:
    if seed_start < 0:
        raise ValueError("seed start must be non-negative")
    if seed_count <= 0:
        raise ValueError("seed count must be positive")
    seeds = list(range(seed_start, seed_start + seed_count))
    if seeds[-1] > 2**31 - 1:
        raise ValueError("seed sequence exceeds signed 32-bit range")
    return seeds


def _absolute_without_resolving_symlinks(path: str | Path) -> Path:
    """Keeps virtual-environment interpreter symlinks intact."""

    return Path(os.path.abspath(os.fspath(path)))


def _matrix_spec(
    *,
    repo_root: Path,
    config: Path,
    params: Path,
    seeds: list[int],
    num_steps: int,
    chunk_steps: int,
    python: Path,
    runner: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "commit": _git_head(repo_root),
        "config_path": config.as_posix(),
        "config_sha256": _sha256_file(config),
        "params_name": params.name,
        "params_sha256": _sha256_file(params),
        "seeds": seeds,
        "num_envs": 1,
        "num_steps": num_steps,
        "implementation": "chunked_jit",
        "chunk_steps": chunk_steps,
        "policy": "trained",
        "python": str(python),
        "runner": runner.as_posix(),
    }


def _validate_matrix_spec(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual != expected:
        mismatches = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            "resume specification does not match current inputs: "
            + ", ".join(mismatches)
        )


def _validate_seed_manifest(
    manifest: dict[str, Any],
    *,
    spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version": 1,
        "status": "complete",
        "commit": spec["commit"],
        "config_path": spec["config_path"],
        "config_sha256": spec["config_sha256"],
        "params_path": spec["params_name"],
        "params_sha256": spec["params_sha256"],
        "seed": seed,
        "num_envs": 1,
        "num_steps": spec["num_steps"],
        "implementation": "chunked_jit",
        "chunk_steps": spec["chunk_steps"],
        "jax_backend": "gpu",
    }
    mismatches = [
        key
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    ]
    devices = manifest.get("jax_devices")
    if not isinstance(devices, list) or len(devices) != 1:
        mismatches.append("jax_devices")
    elif "rocm" not in str(devices[0]).lower():
        mismatches.append("jax_devices")
    results = manifest.get("results")
    if not isinstance(results, dict) or set(results) != {"trained"}:
        mismatches.append("results")
    if mismatches:
        raise ValueError(
            f"seed {seed} manifest binding failed: "
            + ", ".join(sorted(set(mismatches)))
        )
    trained = results["trained"]
    if not isinstance(trained, dict):
        raise ValueError(f"seed {seed} trained result is not an object")
    gates = trained.get("qualification_gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError(f"seed {seed} qualification gates are missing")
    if not all(isinstance(value, bool) for value in gates.values()):
        raise ValueError(f"seed {seed} qualification gates are not boolean")
    if trained.get("qualification_gate_pass") is not all(gates.values()):
        raise ValueError(f"seed {seed} aggregate gate is inconsistent")
    return trained


def _attempt_directories(seed_dir: Path) -> list[Path]:
    if not seed_dir.exists():
        return []
    return sorted(
        path
        for path in seed_dir.iterdir()
        if path.is_dir() and path.name.startswith("attempt_")
    )


def _load_valid_attempt(
    seed_dir: Path,
    *,
    spec: dict[str, Any],
    seed: int,
) -> tuple[Path, dict[str, Any]] | None:
    for attempt in _attempt_directories(seed_dir):
        exit_path = attempt / "exit_code.txt"
        manifest_path = attempt / "qualification" / "manifest.json"
        if not exit_path.is_file() or not manifest_path.is_file():
            continue
        try:
            if int(exit_path.read_text(encoding="utf-8").strip()) != 0:
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            trained = _validate_seed_manifest(manifest, spec=spec, seed=seed)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        return attempt, trained
    return None


def _next_attempt(seed_dir: Path) -> Path:
    attempts = _attempt_directories(seed_dir)
    next_index = 1
    if attempts:
        try:
            next_index = max(
                int(path.name.removeprefix("attempt_")) for path in attempts
            )
        except ValueError as error:
            raise ValueError(f"invalid attempt directory under {seed_dir}") from error
        next_index += 1
    attempt = seed_dir / f"attempt_{next_index:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    return attempt


def _run_seed(
    *,
    repo_root: Path,
    config: Path,
    params: Path,
    output_root: Path,
    python: Path,
    runner: Path,
    seed: int,
    num_steps: int,
    chunk_steps: int,
) -> tuple[Path, int, float]:
    seed_dir = output_root / "runs" / f"seed_{seed:010d}"
    attempt = _next_attempt(seed_dir)
    qualification_dir = attempt / "qualification"
    command = [
        str(python),
        str(runner),
        "--task",
        "push",
        "--config",
        str(config),
        "--eval-only",
        "--params-in",
        str(params),
        "--eval-policy",
        "trained",
        "--eval-num-envs",
        "1",
        "--eval-num-steps",
        str(num_steps),
        "--eval-seed",
        str(seed),
        "--eval-implementation",
        "chunked_jit",
        "--eval-chunk-steps",
        str(chunk_steps),
        "--eval-output-dir",
        str(qualification_dir),
    ]
    _write_json_atomic(
        attempt / "launch.json",
        {
            "command": command,
            "cwd": str(repo_root),
            "seed": seed,
            "started_unix_seconds": time.time(),
        },
    )
    started = time.perf_counter()
    with (attempt / "run.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    walltime = time.perf_counter() - started
    (attempt / "exit_code.txt").write_text(
        f"{completed.returncode}\n", encoding="utf-8"
    )
    _write_json_atomic(
        attempt / "completion.json",
        {
            "exit_code": completed.returncode,
            "seed": seed,
            "walltime_seconds": walltime,
        },
    )
    return attempt, completed.returncode, walltime


def _optional_first(result: dict[str, Any], field: str) -> Any | None:
    values = result.get(field)
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        return None
    return values[0]


def _result_record(seed: int, attempt: Path, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "attempt": attempt.as_posix(),
        "gate_pass": result["qualification_gate_pass"],
        "gates": result["qualification_gates"],
        "success": result["push_success_count"] == 1,
        "terminal": result["push_terminal_count"] == 1,
        "abnormal": result["push_abnormal_termination_count"] != 0,
        "initial_object_x": result["push_initial_object_x_by_env"][0],
        "initial_object_y": result["push_initial_object_y_by_env"][0],
        "final_goal_distance": result["push_final_goal_distance_by_env"][0],
        "maximum_object_speed": result["push_max_object_speed_by_env"][0],
        "peak_step": _optional_first(result, "push_peak_step_by_env"),
        "peak_phase": _optional_first(result, "push_peak_phase_by_env"),
        "peak_prior_object_speed": _optional_first(
            result, "push_peak_prior_speed_by_env"
        ),
        "peak_object_velocity_x": _optional_first(
            result, "push_peak_object_velocity_x_by_env"
        ),
        "peak_object_velocity_y": _optional_first(
            result, "push_peak_object_velocity_y_by_env"
        ),
        "peak_prior_end_effector_distance": _optional_first(
            result, "push_peak_prior_end_effector_distance_by_env"
        ),
        "peak_end_effector_distance": _optional_first(
            result, "push_peak_end_effector_distance_by_env"
        ),
        "peak_object_displacement": _optional_first(
            result, "push_peak_object_displacement_by_env"
        ),
        "peak_object_height": _optional_first(
            result, "push_peak_object_height_by_env"
        ),
        "peak_goal_distance": _optional_first(
            result, "push_peak_goal_distance_by_env"
        ),
        "peak_leg_action_rms": _optional_first(
            result, "push_peak_leg_action_rms_by_env"
        ),
        "peak_arm_action_rms": _optional_first(
            result, "push_peak_arm_action_rms_by_env"
        ),
        "peak_command_scale": _optional_first(
            result, "push_peak_command_scale_by_env"
        ),
        "minimum_object_height": result["push_min_object_height_by_env"][0],
        "maximum_object_height": result["push_max_object_height_by_env"][0],
        "completion_step": result["push_completion_step_by_env"][0],
        "maximum_tilt_deg": result["max_tilt_deg"],
        "maximum_abs_action": result["max_abs_action"],
        "illegal_contact_count": result["illegal_contact_count"],
        "workspace_bounds_count": result["workspace_bounds_count"],
        "nonfinite_state_count": result["nonfinite_state_count"],
        "nonfinite_action_count": result["nonfinite_action_count"],
        "action_saturation_count": result["action_saturation_count"],
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return centre - radius, centre + radius


def _summary(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    if not records:
        return {
            "expected_episode_count": expected_count,
            "completed_episode_count": 0,
            "status": "incomplete",
        }
    successes = sum(record["success"] for record in records)
    gate_passes = sum(record["gate_pass"] for record in records)
    lower, upper = _wilson_interval(successes, len(records))
    safety_fields = (
        "illegal_contact_count",
        "workspace_bounds_count",
        "nonfinite_state_count",
        "nonfinite_action_count",
        "action_saturation_count",
    )
    return {
        "expected_episode_count": expected_count,
        "completed_episode_count": len(records),
        "status": (
            "pass"
            if len(records) == expected_count and gate_passes == expected_count
            else "fail"
            if len(records) == expected_count
            else "incomplete"
        ),
        "gate_pass_count": gate_passes,
        "success_count": successes,
        "success_rate": successes / len(records),
        "success_rate_wilson_95": [lower, upper],
        "final_goal_distance": {
            "minimum": min(record["final_goal_distance"] for record in records),
            "mean": sum(record["final_goal_distance"] for record in records)
            / len(records),
            "maximum": max(record["final_goal_distance"] for record in records),
        },
        "maximum_object_speed": max(
            record["maximum_object_speed"] for record in records
        ),
        "minimum_object_height": min(
            record["minimum_object_height"] for record in records
        ),
        "maximum_object_height": max(
            record["maximum_object_height"] for record in records
        ),
        "maximum_tilt_deg": max(record["maximum_tilt_deg"] for record in records),
        "maximum_abs_action": max(record["maximum_abs_action"] for record in records),
        "safety_totals": {
            field: sum(record[field] for record in records) for field in safety_fields
        },
        "failed_seeds": [
            record["seed"] for record in records if not record["gate_pass"]
        ],
    }


def _progress_payload(
    *,
    spec: dict[str, Any],
    records: list[dict[str, Any]],
    runtime_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "spec": spec,
        "summary": _summary(records, len(spec["seeds"])),
        "records": records,
        "runtime_failures": runtime_failures,
        "updated_unix_seconds": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--params-in", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seed-count", required=True, type=int)
    parser.add_argument("--num-steps", type=int, default=4608)
    parser.add_argument("--chunk-steps", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--runner",
        default=str(Path(__file__).with_name("locomotion_learn_smoke.py")),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = Path(args.config).resolve()
    params = Path(args.params_in).resolve()
    output_root = Path(args.output_dir).resolve()
    python = _absolute_without_resolving_symlinks(args.python)
    runner = Path(args.runner).resolve()
    seeds = _seed_sequence(args.seed_start, args.seed_count)
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive")
    if not 1 <= args.chunk_steps <= 64:
        parser.error("--chunk-steps must be in [1, 64]")
    for name, path in (
        ("config", config),
        ("params", params),
        ("python", python),
        ("runner", runner),
    ):
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")

    spec = _matrix_spec(
        repo_root=repo_root,
        config=config,
        params=params,
        seeds=seeds,
        num_steps=args.num_steps,
        chunk_steps=args.chunk_steps,
        python=python,
        runner=runner,
    )
    spec_path = output_root / "matrix_spec.json"
    if args.resume:
        if not spec_path.is_file():
            parser.error("--resume requires an existing matrix_spec.json")
        actual_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        _validate_matrix_spec(actual_spec, spec)
    else:
        if output_root.exists():
            parser.error("--output-dir must not exist unless --resume is used")
        output_root.mkdir(parents=True)
        _write_json_atomic(spec_path, spec)

    records: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds, start=1):
        seed_dir = output_root / "runs" / f"seed_{seed:010d}"
        valid = _load_valid_attempt(seed_dir, spec=spec, seed=seed)
        if valid is None:
            print(
                f"PUSH_MATRIX_SEED_START index={index}/{len(seeds)} seed={seed}",
                flush=True,
            )
            attempt, exit_code, walltime = _run_seed(
                repo_root=repo_root,
                config=config,
                params=params,
                output_root=output_root,
                python=python,
                runner=runner,
                seed=seed,
                num_steps=args.num_steps,
                chunk_steps=args.chunk_steps,
            )
            if exit_code != 0:
                runtime_failures.append(
                    {
                        "seed": seed,
                        "attempt": attempt.as_posix(),
                        "exit_code": exit_code,
                        "walltime_seconds": walltime,
                    }
                )
                _write_json_atomic(
                    output_root / "matrix_progress.json",
                    _progress_payload(
                        spec=spec,
                        records=records,
                        runtime_failures=runtime_failures,
                    ),
                )
                print(
                    f"PUSH_MATRIX_SEED_RUNTIME_FAILURE seed={seed} "
                    f"exit_code={exit_code} attempt={attempt}",
                    flush=True,
                )
                if args.stop_on_failure:
                    break
                continue
            manifest_path = attempt / "qualification" / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result = _validate_seed_manifest(manifest, spec=spec, seed=seed)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                runtime_failures.append(
                    {
                        "seed": seed,
                        "attempt": attempt.as_posix(),
                        "exit_code": exit_code,
                        "validation_error": str(error),
                    }
                )
                if args.stop_on_failure:
                    break
                continue
            valid = (attempt, result)
        attempt, result = valid
        record = _result_record(seed, attempt, result)
        records.append(record)
        _write_json_atomic(
            output_root / "matrix_progress.json",
            _progress_payload(
                spec=spec,
                records=records,
                runtime_failures=runtime_failures,
            ),
        )
        print(
            f"PUSH_MATRIX_SEED_DONE index={index}/{len(seeds)} seed={seed} "
            f"gate_pass={record['gate_pass']} attempt={attempt}",
            flush=True,
        )
        if args.stop_on_failure and not record["gate_pass"]:
            break

    final_payload = _progress_payload(
        spec=spec,
        records=records,
        runtime_failures=runtime_failures,
    )
    final_payload["status"] = final_payload["summary"]["status"]
    _write_json_atomic(output_root / "matrix_manifest.json", final_payload)
    print(
        "PUSH_MATRIX_DONE "
        f"status={final_payload['status']} "
        f"completed={len(records)}/{len(seeds)} "
        f"gate_passes={final_payload['summary'].get('gate_pass_count', 0)} "
        f"manifest={output_root / 'matrix_manifest.json'}",
        flush=True,
    )
    return 0 if final_payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
