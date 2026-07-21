from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
