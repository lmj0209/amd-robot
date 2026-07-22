# Submission report

- [Technical report](TECHNICAL_REPORT.md)
- [Video script and storyboard](VIDEO_SCRIPT.md)
- `figures/`: measured rollout and ROCm benchmark figures used by the report
- `render_submission_assets.py`: deterministic 960x540 title-card renderer for
  the review video. Pass `--qualification-manifest` only after the held-out
  matrix finishes; the renderer rejects incomplete, reordered, non-finite, or
  internally inconsistent 100-episode evidence.
- `render_submission_video.py`: frame-count-checked FFmpeg assembler for the
  four-minute review video

The large trained parameters, raw frame archive, and MP4 are external evidence
artifacts. Their hashes and reproduction commands are recorded in the report
and the repository documentation.
