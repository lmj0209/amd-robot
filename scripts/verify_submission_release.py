#!/usr/bin/env python3
"""Verify public submission URLs by anonymous full downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlsplit, urlunsplit

REQUIRED_ARTIFACT_FIELDS = {
    "id",
    "kind",
    "url",
    "sha256",
    "bytes",
    "git_commit",
    "config_sha256",
    "license",
}


def _validate_public_https_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"URL must be public HTTPS: {url}")
    if parsed.username or parsed.password:
        raise ValueError(f"URL must not contain credentials: {url}")


def _stable_final_url(url: str) -> str:
    """Keep redirect provenance without persisting expiring query signatures."""
    _validate_public_https_url(url)
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _request(url: str) -> urllib.request.Request:
    _validate_public_https_url(url)
    return urllib.request.Request(
        url,
        headers={
            "Accept": "*/*",
            "Cache-Control": "no-cache",
            "User-Agent": "amd-robo-anonymous-release-verifier/1",
        },
        method="GET",
    )


def _response_status(response: BinaryIO) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    if status != 200:
        raise ValueError(f"expected HTTP 200, received {status}")
    return int(status)


def _check_repository(
    url: str,
    *,
    timeout: float,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Any]:
    started = time.perf_counter()
    with opener(_request(url), timeout=timeout) as response:
        status = _response_status(response)
        sample = response.read(4096)
        final_url = _stable_final_url(response.geturl())
    if not sample:
        raise ValueError("public repository response was empty")
    return {
        "url": url,
        "final_url": final_url,
        "http_status": status,
        "sample_bytes": len(sample),
        "walltime_seconds": time.perf_counter() - started,
    }


def _download_artifact(
    artifact: dict[str, Any],
    *,
    timeout: float,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(
            f"artifact {artifact.get('id', '<unknown>')} missing fields: {missing}"
        )
    expected_sha256 = artifact["sha256"]
    expected_bytes = artifact["bytes"]
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"artifact {artifact['id']} has an invalid SHA-256")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError(f"artifact {artifact['id']} has an invalid byte count")

    digest = hashlib.sha256()
    downloaded_bytes = 0
    started = time.perf_counter()
    with opener(_request(artifact["url"]), timeout=timeout) as response:
        status = _response_status(response)
        final_url = _stable_final_url(response.geturl())
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            downloaded_bytes += len(chunk)
    actual_sha256 = digest.hexdigest()
    if downloaded_bytes != expected_bytes:
        raise ValueError(
            f"artifact {artifact['id']} byte mismatch: "
            f"expected {expected_bytes}, received {downloaded_bytes}"
        )
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"artifact {artifact['id']} SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual_sha256}"
        )
    return {
        "id": artifact["id"],
        "kind": artifact["kind"],
        "url": artifact["url"],
        "final_url": final_url,
        "http_status": status,
        "bytes": downloaded_bytes,
        "sha256": actual_sha256,
        "walltime_seconds": time.perf_counter() - started,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported artifact manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact manifest must contain a non-empty artifact list")
    ids = [artifact.get("id") for artifact in artifacts if isinstance(artifact, dict)]
    if len(ids) != len(artifacts) or len(set(ids)) != len(ids):
        raise ValueError("artifact manifest IDs must be present and unique")
    return manifest


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite release verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/manifest.json", type=Path)
    parser.add_argument(
        "--repository-url",
        default="https://github.com/lmj0209/amd-robot",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.output.exists():
        parser.error("--output must not exist")

    manifest = _load_manifest(args.manifest)
    repository = _check_repository(args.repository_url, timeout=args.timeout)
    artifacts = [
        _download_artifact(artifact, timeout=args.timeout)
        for artifact in manifest["artifacts"]
    ]
    payload = {
        "schema_version": 1,
        "status": "pass",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "anonymous": True,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "repository": repository,
        "artifacts": artifacts,
    }
    _write_json_new(args.output, payload)
    print(
        "SUBMISSION_RELEASE_VERIFIED "
        f"repository_status={repository['http_status']} "
        f"artifacts={len(artifacts)} output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"anonymous release verification failed: HTTP {error.code} {error.url}"
        ) from error
