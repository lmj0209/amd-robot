from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_chunked_push_cross_validation_is_not_overclaimed() -> None:
    evidence = json.loads(
        (
            REPO_ROOT
            / "benchmarks"
            / "raw"
            / "push_chunked_cross_validation_2026-07-21.json"
        ).read_text(encoding="utf-8")
    )

    assert evidence["status"] == "pass"
    assert evidence["evidence_class"] == "implementation_cross_validation"
    assert evidence["qualification_claim"] is False
    assert evidence["decision"]["next_gate"] == (
        "predeclared held-out 100-seed matrix"
    )
    assert len(evidence["records"]) == 2
    for record in evidence["records"]:
        assert record["candidate"]["gate_pass"] is True
        assert record["discrete_outcome_match"] is True
        assert record["safety_classification_match"] is True
        assert len(record["candidate"]["manifest_sha256"]) == 64


def test_rendered_submission_cards_use_isolated_evaluation_claims() -> None:
    source = (
        REPO_ROOT / "report" / "render_submission_assets.py"
    ).read_text(encoding="utf-8")

    for withdrawn_claim in (
        '"20/20"',
        '"1/20"',
        '"0.347"',
        '"0.690"',
        "20-episode speed gate failed",
    ):
        assert withdrawn_claim not in source

    for current_claim in (
        "Isolated fresh-process evidence",
        '"3/3"',
        '"0.441"',
        '"0.384"',
        "batched gfx1100 Push is diagnostic-only",
    ):
        assert current_claim in source


def test_submission_materials_include_verified_solo_credit() -> None:
    renderer = (
        REPO_ROOT / "report" / "render_submission_assets.py"
    ).read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    video_script = (
        REPO_ROOT / "report" / "VIDEO_SCRIPT.md"
    ).read_text(encoding="utf-8")

    assert 'CREATOR = "limengjin"' in renderer
    assert 'CREATOR_ROLE = "Solo developer"' in renderer
    assert "limengjin — solo developer" in readme
    assert "limengjin — Solo developer" in video_script


def test_submission_materials_record_public_brax_pr_as_pending() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    technical_report = (
        REPO_ROOT / "report" / "TECHNICAL_REPORT.md"
    ).read_text(encoding="utf-8")
    checklist = (
        REPO_ROOT / "report" / "SUBMISSION_CHECKLIST.md"
    ).read_text(encoding="utf-8")

    for material in (readme, technical_report):
        normalized_material = " ".join(material.split())
        assert "https://github.com/google/brax/pull/674" in normalized_material
        assert "awaiting maintainer review" in normalized_material
        assert "not claimed as accepted or merged" in normalized_material

    normalized_checklist = " ".join(checklist.split())
    assert "public Brax PR #674" in normalized_checklist
    assert "without claiming acceptance or merge" in normalized_checklist
