from __future__ import annotations

import hashlib
import io
import json

import pytest

from scripts.verify_submission_release import (
    _check_repository,
    _download_artifact,
    _load_manifest,
    _validate_public_https_url,
    _write_json_new,
)


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, payload: bytes, url: str = "https://public.example/final"):
        super().__init__(payload)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _artifact(payload: bytes = b"checkpoint") -> dict[str, object]:
    return {
        "id": "policy",
        "kind": "checkpoint",
        "url": "https://public.example/policy",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "license": "MIT",
    }


def test_anonymous_full_download_checks_hash_bytes_and_headers():
    payload = b"checkpoint"
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse(payload)

    result = _download_artifact(_artifact(payload), timeout=5.0, opener=opener)

    assert result["http_status"] == 200
    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    request, timeout = requests[0]
    assert timeout == 5.0
    assert request.get_header("Authorization") is None
    assert request.get_header("User-agent") == (
        "amd-robo-anonymous-release-verifier/1"
    )


def test_release_verifier_drops_ephemeral_redirect_query():
    signed_url = "https://assets.example/policy?sig=temporary#fragment"
    result = _download_artifact(
        _artifact(),
        timeout=5.0,
        opener=lambda request, timeout: FakeResponse(b"checkpoint", signed_url),
    )

    assert result["final_url"] == "https://assets.example/policy"


def test_repository_check_requires_nonempty_http_200():
    result = _check_repository(
        "https://public.example/repository",
        timeout=5.0,
        opener=lambda request, timeout: FakeResponse(b"public repository"),
    )
    assert result["http_status"] == 200
    assert result["sample_bytes"] == len(b"public repository")

    with pytest.raises(ValueError, match="empty"):
        _check_repository(
            "https://public.example/repository",
            timeout=5.0,
            opener=lambda request, timeout: FakeResponse(b""),
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://public.example/file",
        "https://user:password@public.example/file",
        "/local/path",
    ],
)
def test_release_verifier_rejects_nonpublic_urls(url: str):
    with pytest.raises(ValueError, match="URL"):
        _validate_public_https_url(url)


def test_release_verifier_fails_closed_on_byte_or_hash_mismatch():
    with pytest.raises(ValueError, match="byte mismatch"):
        _download_artifact(
            _artifact(b"expected"),
            timeout=5.0,
            opener=lambda request, timeout: FakeResponse(b"short"),
        )

    artifact = _artifact(b"actual")
    artifact["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _download_artifact(
            artifact,
            timeout=5.0,
            opener=lambda request, timeout: FakeResponse(b"actual"),
        )


def test_manifest_requires_unique_artifacts(tmp_path):
    artifact = _artifact()
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [artifact, artifact],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        _load_manifest(path)


def test_release_evidence_is_never_overwritten(tmp_path):
    output = tmp_path / "verification.json"
    _write_json_new(output, {"status": "pass"})
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_json_new(output, {"status": "pass"})
