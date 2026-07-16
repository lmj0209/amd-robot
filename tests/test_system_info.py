from __future__ import annotations

import sys
from pathlib import Path

from amd_robo.platform.system_info import _run, _sha256


def test_command_capture_preserves_command_and_exit_code() -> None:
    result = _run([sys.executable, "-c", "print('ok')"])
    assert result["returncode"] == 0
    assert result["stdout"] == "ok"
    assert result["command"][0] == sys.executable


def test_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    expected = "db9bda4272ee21cda5ff1d213fa8366a2cca3143c5a795bdcd5f75a85106033d"
    assert _sha256(path) == expected
