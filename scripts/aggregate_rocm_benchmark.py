#!/usr/bin/env python3
"""Validate and aggregate ROCm benchmark logs into CSV and JSON evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


RESULT_PREFIX = "ROCM_BENCHMARK "
IDENTITY_FIELDS = ("backend", "mode", "batch_size")
CSV_FIELDS = (
    "backend",
    "mode",
    "batch_size",
    "status",
    "finite",
    "exit_code",
    "git_commit",
    "config_sha256",
    "params_sha256",
    "reset_compile_s",
    "cold_compile_s",
    "steps_per_repeat",
    "repeats",
    "throughput_mean",
    "throughput_min",
    "throughput_max",
    "throughput_inferences_s_mean",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "metric_samples",
    "gfx_activity_mean_pct",
    "gfx_activity_p95_pct",
    "gfx_activity_max_pct",
    "socket_power_mean_w",
    "socket_power_max_w",
    "hotspot_max_c",
    "used_vram_max_mb",
    "log_file",
    "metrics_file",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help="Benchmark directory; repeat for separate GPU and CPU evidence.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-commit")
    return parser.parse_args()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _number(sample: dict[str, Any], *path: str) -> float | None:
    value: Any = sample
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric_summary(path: Path) -> dict[str, int | float | None]:
    if not path.exists():
        return {
            "metric_samples": 0,
            "gfx_activity_mean_pct": None,
            "gfx_activity_p95_pct": None,
            "gfx_activity_max_pct": None,
            "socket_power_mean_w": None,
            "socket_power_max_w": None,
            "hotspot_max_c": None,
            "used_vram_max_mb": None,
        }
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"{path}: expected a JSON array")

    def values(*field_path: str) -> list[float]:
        result = []
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"{path}: metric sample is not an object")
            value = _number(sample, *field_path)
            if value is not None:
                result.append(value)
        return result

    gfx = values("usage", "gfx_activity")
    power = values("power", "socket_power")
    hotspot = values("temperature", "hotspot")
    vram = values("mem_usage", "used_vram")
    return {
        "metric_samples": len(samples),
        "gfx_activity_mean_pct": statistics.fmean(gfx) if gfx else None,
        "gfx_activity_p95_pct": _percentile(gfx, 0.95),
        "gfx_activity_max_pct": max(gfx) if gfx else None,
        "socket_power_mean_w": statistics.fmean(power) if power else None,
        "socket_power_max_w": max(power) if power else None,
        "hotspot_max_c": max(hotspot) if hotspot else None,
        "used_vram_max_mb": max(vram) if vram else None,
    }


def _result_from_log(
    log_path: Path, expected_commit: str | None
) -> dict[str, Any]:
    lines = [
        line.removeprefix(RESULT_PREFIX)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(lines) != 1:
        raise ValueError(
            f"{log_path}: expected exactly one {RESULT_PREFIX.strip()} line, "
            f"found {len(lines)}"
        )
    result = json.loads(lines[0])
    missing = [field for field in IDENTITY_FIELDS if field not in result]
    if missing:
        raise ValueError(f"{log_path}: missing identity fields {missing}")

    exit_path = log_path.with_suffix(".exit")
    if not exit_path.exists():
        raise ValueError(f"{log_path}: missing {exit_path.name}")
    exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    if exit_code != 0:
        raise ValueError(f"{log_path}: nonzero exit code {exit_code}")
    if result.get("status") != "ok" or result.get("finite") is not True:
        raise ValueError(f"{log_path}: benchmark status/finite check failed")
    if expected_commit and result.get("git_commit") != expected_commit:
        raise ValueError(
            f"{log_path}: commit {result.get('git_commit')} does not match "
            f"{expected_commit}"
        )

    metrics_path = log_path.with_name(
        f"{log_path.stem}_amd_smi.json"
    )
    result.update(_metric_summary(metrics_path))
    result["exit_code"] = exit_code
    result["log_file"] = log_path.name
    result["metrics_file"] = metrics_path.name if metrics_path.exists() else None
    return result


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in CSV_FIELDS})


def main() -> int:
    args = _parse_args()
    missing_dirs = [path for path in args.input_dir if not path.is_dir()]
    if missing_dirs:
        raise ValueError(f"input directories not found: {missing_dirs}")
    log_paths = sorted(
        log_path
        for input_dir in args.input_dir
        for log_path in input_dir.glob("*.log")
    )
    results = [
        _result_from_log(path, args.expected_commit)
        for path in log_paths
        if any(line.startswith(RESULT_PREFIX) for line in path.read_text(
            encoding="utf-8"
        ).splitlines())
    ]
    if not results:
        raise ValueError(f"no benchmark results found in {args.input_dir}")
    results.sort(
        key=lambda item: (
            str(item["backend"]),
            str(item["mode"]),
            int(item["batch_size"]),
        )
    )
    identities = [
        tuple(result[field] for field in IDENTITY_FIELDS)
        for result in results
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate backend/mode/batch_size result")

    _write_csv(args.output_csv, results)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result_count": len(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "ROCM_BENCHMARK_AGGREGATED "
        f"count={len(results)} csv={args.output_csv} json={args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
