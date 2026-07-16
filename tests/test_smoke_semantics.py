from __future__ import annotations

from amd_robo.platform.smoke import gate_exit_code


def _passed_checks() -> dict[str, dict]:
    return {name: {} for name in ("rocm", "mjx", "playground", "ppo")}


def test_full_gate_is_success() -> None:
    assert gate_exit_code(_passed_checks(), set()) == 0


def test_skipped_ppo_is_never_success() -> None:
    checks = _passed_checks()
    checks.pop("ppo")
    assert gate_exit_code(checks, {"ppo"}) == 2


def test_missing_required_check_is_never_success() -> None:
    checks = _passed_checks()
    checks.pop("rocm")
    assert gate_exit_code(checks, set()) == 2
