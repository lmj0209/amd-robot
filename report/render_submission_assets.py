"""Render deterministic title cards for the four-minute review video."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

WIDTH = 960
HEIGHT = 540
BACKGROUND = "#0c111b"
PANEL = "#151d2b"
TEXT = "#f3f6fa"
MUTED = "#aeb9c9"
ACCENT = "#ed1c24"
CYAN = "#54c7ec"
GREEN = "#67d391"
YELLOW = "#f4c95d"
CREATOR = "limengjin"
CREATOR_ROLE = "Solo developer"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = (
        ("C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/arialbd.ttf")
        if bold
        else ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf")
    )
    linux = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    for candidate in (*windows, linux):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError("No supported TrueType font was found")


TITLE = _font(42, bold=True)
SUBTITLE = _font(25, bold=True)
BODY = _font(23)
SMALL = _font(17)
METRIC = _font(36, bold=True)


def _canvas(section: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 12, HEIGHT), fill=ACCENT)
    draw.text((42, 24), section.upper(), font=SMALL, fill=CYAN)
    draw.text(
        (42, HEIGHT - 34),
        "AMD Robot Competition | Track 3 Physical AI | measured evidence",
        font=SMALL,
        fill=MUTED,
    )
    return image, draw


def _fit_image(path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    width = box[2] - box[0]
    height = box[3] - box[1]
    source = Image.open(path).convert("RGB")
    return ImageOps.fit(
        source,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )


def _bullets(
    draw: ImageDraw.ImageDraw,
    items: list[str],
    *,
    x: int = 58,
    y: int = 145,
    width: int = 66,
    line_height: int = 31,
) -> None:
    cursor = y
    for item in items:
        lines = textwrap.wrap(item, width=width)
        draw.ellipse((x, cursor + 9, x + 8, cursor + 17), fill=ACCENT)
        for index, line in enumerate(lines):
            draw.text(
                (x + 22, cursor + index * line_height),
                line,
                font=BODY,
                fill=TEXT,
            )
        cursor += max(1, len(lines)) * line_height + 12


def _title_slide(final_frame: Path) -> Image.Image:
    background = _fit_image(final_frame, (0, 0, WIDTH, HEIGHT))
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (5, 10, 18, 172))
    image = Image.alpha_composite(background.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 12, HEIGHT), fill=ACCENT)
    draw.text((54, 92), "ROCm-Accelerated", font=TITLE, fill=TEXT)
    draw.text((54, 145), "Quadruped Mobile Manipulation", font=TITLE, fill=TEXT)
    draw.text(
        (58, 220),
        "MuJoCo MJX + Playground-style MjxEnv + Brax PPO",
        font=SUBTITLE,
        fill=CYAN,
    )
    draw.rounded_rectangle((55, 302, 536, 366), radius=14, fill="#111b28")
    draw.text(
        (78, 319),
        "One Radeon PRO W7900 | 19D control",
        font=BODY,
        fill=TEXT,
    )
    draw.text(
        (58, 465),
        f"{CREATOR} | {CREATOR_ROLE} | AMD Robot Competition | Track 3 Physical AI",
        font=SMALL,
        fill=MUTED,
    )
    return image


def _problem_slide(mid_frame: Path) -> Image.Image:
    image, draw = _canvas("Application")
    draw.text(
        (50, 61),
        "Move a near-field obstruction into a goal zone",
        font=TITLE,
        fill=TEXT,
    )
    image.paste(_fit_image(mid_frame, (510, 132, 920, 410)), (510, 132))
    _bullets(
        draw,
        [
            "Go2 quadruped + Z1 arm and gripper",
            "Approach -> Align -> Push -> Hold",
            "Goal error <= 0.08 m for 100 control steps",
            "Fixed-box, state-based simulation scope",
        ],
        x=55,
        y=145,
        width=36,
    )
    return image


def _architecture_slide() -> Image.Image:
    image, draw = _canvas("Architecture")
    draw.text((50, 61), "Auditable ownership boundaries", font=TITLE, fill=TEXT)
    labels = [
        ("MuJoCo MJX", "dynamics, contact,\nactuators, sensors", CYAN),
        ("MjxEnv", "observation, phases,\nreward, metrics", GREEN),
        ("Brax PPO", "rollouts, residual\npolicy, optimization", YELLOW),
        ("ROCm evidence", "backend checks,\nhashes, benchmarks", ACCENT),
    ]
    for index, (heading, detail, color) in enumerate(labels):
        x0 = 42 + index * 228
        x1 = x0 + 190
        draw.rounded_rectangle(
            (x0, 155, x1, 335),
            radius=16,
            fill=PANEL,
            outline=color,
            width=3,
        )
        draw.text((x0 + 18, 177), heading, font=SUBTITLE, fill=color)
        draw.multiline_text(
            (x0 + 18, 229),
            detail,
            font=SMALL,
            fill=TEXT,
            spacing=8,
        )
        if index < len(labels) - 1:
            draw.line((x1 + 7, 244, x1 + 30, 244), fill=MUTED, width=4)
            draw.polygon(
                ((x1 + 30, 244), (x1 + 20, 237), (x1 + 20, 251)),
                fill=MUTED,
            )
    draw.text((196, 386), "Frozen contract:", font=SUBTITLE, fill=MUTED)
    draw.text(
        (402, 386),
        "87D observation -> 19D action",
        font=SUBTITLE,
        fill=TEXT,
    )
    return image


def _rocm_slide() -> Image.Image:
    image, draw = _canvas("AMD engineering")
    draw.text(
        (50, 61),
        "ROCm-first contact-rich training on gfx1100",
        font=TITLE,
        fill=TEXT,
    )
    _bullets(
        draw,
        [
            "Radeon PRO W7900, ROCm 7.2.1, JAX ROCm plugin 0.10.2",
            "Large fused reverse-mode PPO graphs exposed a compiler stability boundary",
            "Validated path: five physics substeps + host small-update loop",
            "Complete learner, optimizer, rollout, and PRNG state is persisted",
        ],
        y=145,
        width=73,
    )
    return image


def _load_qualification_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = payload.get("spec")
    summary = payload.get("summary")
    records = payload.get("records")
    if not isinstance(spec, dict) or not isinstance(summary, dict):
        raise ValueError("qualification matrix is missing its spec or summary")
    if not isinstance(records, list):
        raise ValueError("qualification matrix records must be a list")

    expected = summary.get("expected_episode_count")
    completed = summary.get("completed_episode_count")
    status = summary.get("status")
    if expected != 100:
        raise ValueError("submission evidence requires exactly 100 episodes")
    if completed != expected or len(records) != expected:
        raise ValueError("qualification matrix is incomplete")
    if status not in {"pass", "fail"} or payload.get("status") != status:
        raise ValueError("qualification matrix status is invalid or inconsistent")
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != expected:
        raise ValueError("qualification matrix seed specification is invalid")
    if [record.get("seed") for record in records] != seeds:
        raise ValueError("qualification matrix records do not match seed order")

    safety_fields = (
        "illegal_contact_count",
        "workspace_bounds_count",
        "nonfinite_state_count",
        "nonfinite_action_count",
        "action_saturation_count",
    )
    numeric_fields = (
        "final_goal_distance",
        "maximum_object_speed",
        "minimum_object_height",
        "maximum_object_height",
    )
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("qualification matrix record must be an object")
        if not isinstance(record.get("gate_pass"), bool) or not isinstance(
            record.get("success"), bool
        ):
            raise ValueError("qualification matrix decisions must be boolean")
        values = [record.get(field) for field in numeric_fields]
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        ):
            raise ValueError("qualification matrix contains non-finite metrics")
        if not all(
            isinstance(record.get(field), int) and record[field] >= 0
            for field in safety_fields
        ):
            raise ValueError("qualification matrix contains invalid safety counts")

    gate_pass_count = sum(record["gate_pass"] for record in records)
    success_count = sum(record["success"] for record in records)
    safety_totals = {
        field: sum(record[field] for record in records) for field in safety_fields
    }
    maximum_object_speed = max(
        record["maximum_object_speed"] for record in records
    )
    maximum_goal_distance = max(
        record["final_goal_distance"] for record in records
    )
    if summary.get("gate_pass_count") != gate_pass_count:
        raise ValueError("qualification matrix gate total is inconsistent")
    if summary.get("success_count") != success_count:
        raise ValueError("qualification matrix success total is inconsistent")
    if summary.get("safety_totals") != safety_totals:
        raise ValueError("qualification matrix safety totals are inconsistent")
    if not math.isclose(
        summary.get("maximum_object_speed", math.nan), maximum_object_speed
    ):
        raise ValueError("qualification matrix speed maximum is inconsistent")
    if not math.isclose(
        summary.get("final_goal_distance", {}).get("maximum", math.nan),
        maximum_goal_distance,
    ):
        raise ValueError("qualification matrix goal maximum is inconsistent")
    wilson = summary.get("success_rate_wilson_95")
    if (
        not isinstance(wilson, list)
        or len(wilson) != 2
        or not all(isinstance(value, (int, float)) for value in wilson)
    ):
        raise ValueError("qualification matrix Wilson interval is missing")

    return {
        "status": status,
        "expected": expected,
        "gate_pass_count": gate_pass_count,
        "success_count": success_count,
        "success_wilson_95": wilson,
        "maximum_object_speed": maximum_object_speed,
        "maximum_goal_distance": maximum_goal_distance,
        "safety_totals": safety_totals,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "commit": spec.get("commit"),
        "config_sha256": spec.get("config_sha256"),
        "params_sha256": spec.get("params_sha256"),
    }


def _evaluation_slide(matrix: dict[str, Any] | None = None) -> Image.Image:
    if matrix is not None:
        return _matrix_evaluation_slide(matrix)
    image, draw = _canvas("Measured task result")
    draw.text(
        (50, 61),
        "Isolated fresh-process evidence",
        font=TITLE,
        fill=TEXT,
    )
    metrics = [
        ("3/3", "trained success, A", GREEN),
        ("3/3", "trained success, B", GREEN),
        ("0.441", "maximum A, m/s", CYAN),
        ("0.384", "maximum B, m/s", CYAN),
    ]
    for index, (value, label, color) in enumerate(metrics):
        x0 = 45 + index * 226
        draw.rounded_rectangle((x0, 155, x0 + 195, 280), radius=15, fill=PANEL)
        draw.text((x0 + 18, 174), value, font=METRIC, fill=color)
        draw.text((x0 + 18, 232), label, font=SMALL, fill=MUTED)
    draw.text(
        (53, 327),
        "Within-process repeats exact | fresh-process discrete outcomes match",
        font=SUBTITLE,
        fill=TEXT,
    )
    draw.rounded_rectangle((50, 383, 910, 438), radius=12, fill="#2d1f16")
    draw.text(
        (72, 398),
        "Disclosure: batched gfx1100 Push is diagnostic-only.",
        font=BODY,
        fill=YELLOW,
    )
    return image


def _matrix_evaluation_slide(matrix: dict[str, Any]) -> Image.Image:
    image, draw = _canvas("Held-out task result")
    draw.text(
        (50, 61),
        "Single-environment +/-2 mm matrix",
        font=TITLE,
        fill=TEXT,
    )
    count = matrix["expected"]
    metrics = [
        (f"{matrix['success_count']}/{count}", "task success", GREEN),
        (
            f"{matrix['gate_pass_count']}/{count}",
            "all gates pass",
            GREEN if matrix["status"] == "pass" else YELLOW,
        ),
        (
            f"{matrix['maximum_object_speed']:.3f}",
            "maximum speed, m/s",
            CYAN,
        ),
        (
            f"{matrix['maximum_goal_distance']:.4f}",
            "worst final error, m",
            CYAN,
        ),
    ]
    for index, (value, label, color) in enumerate(metrics):
        x0 = 45 + index * 226
        draw.rounded_rectangle((x0, 145, x0 + 195, 270), radius=15, fill=PANEL)
        draw.text((x0 + 16, 164), value, font=METRIC, fill=color)
        draw.text((x0 + 16, 222), label, font=SMALL, fill=MUTED)

    lower, upper = matrix["success_wilson_95"]
    safety_total = sum(matrix["safety_totals"].values())
    draw.text(
        (53, 308),
        f"Success 95% Wilson interval: {lower:.3f}-{upper:.3f}",
        font=SUBTITLE,
        fill=TEXT,
    )
    draw.text(
        (53, 350),
        f"Registered safety-event total: {safety_total}",
        font=SUBTITLE,
        fill=GREEN if safety_total == 0 else YELLOW,
    )
    draw.text(
        (53, 393),
        f"Seeds {matrix['seed_start']}-{matrix['seed_end']} | fresh process each",
        font=SMALL,
        fill=MUTED,
    )
    draw.rounded_rectangle((50, 427, 910, 474), radius=12, fill="#2d1f16")
    draw.text(
        (72, 438),
        "One ROCm device, one environment; batched Push remains diagnostic-only.",
        font=SMALL,
        fill=YELLOW,
    )
    return image


def _benchmark_slide(chart: Path) -> Image.Image:
    image, draw = _canvas("ROCm performance")
    draw.text(
        (50, 58),
        "Synchronized policy + MJX throughput",
        font=TITLE,
        fill=TEXT,
    )
    image.paste(_fit_image(chart, (47, 120, 615, 450)), (47, 120))
    draw.text((649, 150), "7,404", font=METRIC, fill=GREEN)
    draw.text((649, 198), "transitions/s", font=SUBTITLE, fill=TEXT)
    draw.text((649, 235), "GPU batch 2,048", font=SMALL, fill=MUTED)
    draw.text((649, 295), "Practical range", font=SUBTITLE, fill=CYAN)
    draw.text((649, 333), "batch 1,024-2,048", font=BODY, fill=TEXT)
    draw.text((649, 382), "21/21 points", font=SUBTITLE, fill=GREEN)
    draw.text((649, 417), "exit 0 and finite", font=SMALL, fill=MUTED)
    return image


def _training_slide() -> Image.Image:
    image, draw = _canvas("Brax PPO")
    draw.text((50, 61), "Selected Push training run", font=TITLE, fill=TEXT)
    metrics = [
        ("73,728", "on-policy transitions"),
        ("345.761 s", "Brax walltime"),
        ("383.168", "final reused host-call SPS"),
        ("0.000192", "final KL"),
        ("0.583383", "minimum policy standard deviation"),
    ]
    y = 138
    for value, label in metrics:
        draw.text((65, y), value, font=SUBTITLE, fill=CYAN)
        draw.text((285, y + 2), label, font=BODY, fill=TEXT)
        y += 59
    return image


def _reproduction_slide() -> Image.Image:
    image, draw = _canvas("Reproducibility")
    draw.text(
        (50, 61),
        "Evidence is tied to code, seed, and hashes",
        font=TITLE,
        fill=TEXT,
    )
    _bullets(
        draw,
        [
            "Versioned YAML config and explicit JAX PRNG keys",
            "Policy SHA-256 and video SHA-256",
            "All 813 rollout frames retained in sequence",
            "Fresh benchmark process, warmup, synchronization, and backend checks",
            "CPU-safe contracts plus exact full-state training resume",
        ],
        y=137,
        width=72,
        line_height=29,
    )
    return image


def _limitations_slide() -> Image.Image:
    image, draw = _canvas("Current boundary")
    draw.text((50, 61), "What this MVP does not claim", font=TITLE, fill=TEXT)
    _bullets(
        draw,
        [
            "No randomized-box qualification",
            "No perception, sim-to-real result, or hardware safety certificate",
            "The held-out 100-episode matrix is not yet complete",
            "Large fused PPO scans are outside the validated gfx1100 boundary",
            "Brax PR #674 is open and has not been reviewed or merged",
        ],
        y=137,
        width=72,
        line_height=29,
    )
    return image


def _closing_slide(final_frame: Path) -> Image.Image:
    image = _fit_image(final_frame, (0, 0, WIDTH, HEIGHT)).convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (5, 10, 18, 168))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 12, HEIGHT), fill=ACCENT)
    draw.text((55, 132), "Physical AI on Radeon + ROCm", font=TITLE, fill=TEXT)
    draw.text(
        (58, 205),
        "Contact-rich manipulation with measured evidence",
        font=SUBTITLE,
        fill=CYAN,
    )
    draw.rounded_rectangle((55, 287, 771, 354), radius=14, fill="#111b28")
    draw.text(
        (78, 306),
        "Fresh-process A/B matched | limitations disclosed",
        font=BODY,
        fill=TEXT,
    )
    draw.text(
        (58, 430),
        f"Created by {CREATOR} | {CREATOR_ROLE}",
        font=SMALL,
        fill=CYAN,
    )
    draw.text(
        (58, 465),
        "Code, report, reproduction commands, and artifact hashes included",
        font=SMALL,
        fill=MUTED,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-chart", type=Path, required=True)
    parser.add_argument("--rollout-mid", type=Path, required=True)
    parser.add_argument("--rollout-final", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for source in (args.benchmark_chart, args.rollout_mid, args.rollout_final):
        if not source.is_file():
            raise FileNotFoundError(source)
    matrix = None
    if args.qualification_manifest is not None:
        if not args.qualification_manifest.is_file():
            raise FileNotFoundError(args.qualification_manifest)
        matrix = _load_qualification_matrix(args.qualification_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    slides = {
        "01_title.png": _title_slide(args.rollout_final),
        "02_application.png": _problem_slide(args.rollout_mid),
        "03_architecture.png": _architecture_slide(),
        "04_rocm_engineering.png": _rocm_slide(),
        "06_evaluation.png": _evaluation_slide(matrix),
        "07_benchmark.png": _benchmark_slide(args.benchmark_chart),
        "08_training.png": _training_slide(),
        "09_reproduction.png": _reproduction_slide(),
        "10_limitations.png": _limitations_slide(),
        "11_closing.png": _closing_slide(args.rollout_final),
    }
    for name, image in slides.items():
        image.save(args.output_dir / name, optimize=True)
        print(args.output_dir / name)

    slides["03_architecture.png"].save(
        args.output_dir / "architecture.png",
        optimize=True,
    )


if __name__ == "__main__":
    main()
