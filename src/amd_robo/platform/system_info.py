"""Collect an immutable, command-attributed RGC reproducibility fingerprint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def _run(command: list[str], cwd: Path | None = None, timeout: int = 30) -> dict:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _package_versions() -> dict[str, str]:
    wanted = {
        "jax",
        "jaxlib",
        "mujoco",
        "brax",
        "playground",
        "ml-collections",
        "numpy",
    }
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name in wanted or name.startswith("jax-rocm"):
            found[name] = dist.version
    return dict(sorted(found.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info(repo_root: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain=v1"], cwd=repo_root)
    return {
        "commit": commit["stdout"] if commit["returncode"] == 0 else None,
        "dirty": bool(status["stdout"]) if status["returncode"] == 0 else None,
        "status_command": status,
    }


def _jax_info() -> dict[str, Any]:
    try:
        import jax

        try:
            from jax.extend import backend

            platform_version = str(backend.get_backend().platform_version)
        except Exception:  # noqa: BLE001
            platform_version = "unknown"
        return {
            "default_backend": jax.default_backend(),
            "platform_version": platform_version,
            "devices": [
                {
                    "platform": getattr(device, "platform", "unknown"),
                    "device_kind": getattr(device, "device_kind", "unknown"),
                    "description": str(device),
                }
                for device in jax.devices()
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect(
    repo_root: Path,
    image_name: str | None,
    image_digest: str | None,
    config_paths: Sequence[Path],
) -> dict[str, Any]:
    captured = datetime.now(UTC)
    configs = []
    for path in config_paths:
        resolved = path.resolve()
        configs.append(
            {
                "path": str(resolved),
                "exists": resolved.is_file(),
                "sha256": _sha256(resolved) if resolved.is_file() else None,
            }
        )

    return {
        "schema_version": 1,
        "captured_at_utc": captured.isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "container": {
            "image_name": image_name,
            "image_digest": image_digest,
            "note": "Both fields must be supplied from RGC metadata for formal runs.",
        },
        "packages": _package_versions(),
        "jax": _jax_info(),
        "git": _git_info(repo_root),
        "configs": configs,
        "environment": {
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
        },
        "commands": {
            "amd_smi_version": _run(["amd-smi", "version", "--json"]),
            "amd_smi_list": _run(["amd-smi", "list", "--json"]),
            "amd_smi_static": _run(["amd-smi", "static", "--json"]),
            "amd_smi_metric": _run(["amd-smi", "metric", "--json"]),
            "rocm_smi": _run(["rocm-smi", "--showproductname", "--json"]),
            "rocminfo": _run(["rocminfo"]),
            "pip_list": _run(
                [sys.executable, "-m", "pip", "list", "--format=json"]
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-name", default=os.environ.get("RGC_IMAGE_NAME"))
    parser.add_argument("--image-digest", default=os.environ.get("RGC_IMAGE_DIGEST"))
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        type=Path,
        help="Configuration file to hash. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/raw/system"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    info = collect(
        repo_root=repo_root,
        image_name=args.image_name,
        image_digest=args.image_digest,
        config_paths=args.config,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = info["git"]["commit"]
    suffix = commit[:8] if commit else "nogit"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"system_info_{timestamp}_{suffix}.json"
    output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
