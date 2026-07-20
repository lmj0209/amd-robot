"""Assemble and verify the four-minute submission review video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


FPS = 20
WIDTH = 960
HEIGHT = 540
SLIDES = (
    ("01_title.png", 15),
    ("02_application.png", 20),
    ("03_architecture.png", 25),
    ("04_rocm_engineering.png", 20),
    (None, None),
    ("06_evaluation.png", 24),
    ("07_benchmark.png", 25),
    ("08_training.png", 20),
    ("09_reproduction.png", 20),
    ("10_limitations.png", 20),
    ("11_closing.png", 10),
)


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print(" ".join(command))
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture,
    )


def _frame_count(ffmpeg: str, path: Path) -> int:
    result = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    frames = [
        int(line.removeprefix("frame="))
        for line in result.stdout.splitlines()
        if line.startswith("frame=")
    ]
    if not frames:
        raise RuntimeError(f"ffmpeg did not report a frame count for {path}")
    return frames[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_common() -> list[str]:
    return [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.0",
        "-x264-params",
        "keyint=40:min-keyint=1:scenecut=0",
        "-video_track_timescale",
        "10240",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--slides-dir", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.rollout.is_file():
        raise FileNotFoundError(args.rollout)
    for name, _ in SLIDES:
        if name is not None and not (args.slides_dir / name).is_file():
            raise FileNotFoundError(args.slides_dir / name)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rollout_frames = _frame_count(args.ffmpeg, args.rollout)
    expected_frames = sum(
        int(duration * FPS)
        for _, duration in SLIDES
        if duration is not None
    ) + rollout_frames

    with tempfile.TemporaryDirectory(prefix="amd-track3-video-") as temporary:
        temporary_dir = Path(temporary)
        segments: list[Path] = []
        for index, (name, duration) in enumerate(SLIDES):
            segment = temporary_dir / f"segment_{index:02d}.mp4"
            if name is None:
                command = [
                    args.ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-i",
                    str(args.rollout),
                    "-vf",
                    f"scale={WIDTH}:{HEIGHT}:flags=lanczos",
                    *_encode_common(),
                    str(segment),
                ]
            else:
                command = [
                    args.ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-loop",
                    "1",
                    "-framerate",
                    str(FPS),
                    "-i",
                    str(args.slides_dir / name),
                    "-t",
                    str(duration),
                    "-vf",
                    f"scale={WIDTH}:{HEIGHT}:flags=lanczos",
                    *_encode_common(),
                    str(segment),
                ]
            _run(command)
            segments.append(segment)

        concat_file = temporary_dir / "segments.txt"
        concat_file.write_text(
            "".join(
                f"file '{segment.as_posix()}'\n"
                for segment in segments
            ),
            encoding="utf-8",
        )
        _run(
            [
                args.ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(args.output),
            ]
        )

    output_frames = _frame_count(args.ffmpeg, args.output)
    if output_frames != expected_frames:
        raise RuntimeError(
            f"frame-count mismatch: expected {expected_frames}, "
            f"decoded {output_frames}"
        )

    manifest = {
        "schema_version": 1,
        "fps": FPS,
        "width": WIDTH,
        "height": HEIGHT,
        "slide_duration_seconds": sum(
            duration for _, duration in SLIDES if duration is not None
        ),
        "rollout_frames": rollout_frames,
        "rollout_sha256": _sha256(args.rollout),
        "expected_frames": expected_frames,
        "decoded_output_frames": output_frames,
        "expected_duration_seconds": expected_frames / FPS,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": _sha256(args.output),
        "audio": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(manifest_path)


if __name__ == "__main__":
    main()
