# External artifacts

Large checkpoints and videos are not committed. Every published artifact must
be listed in `manifest.json` with an immutable URL, SHA256, size, source commit,
config hash, and license. A missing checksum makes an artifact non-reproducible.

The repository and release were verified without a GitHub session on
2026-07-22. The compact immutable result is recorded in
`../benchmarks/raw/submission_release_verification_2026-07-22.json`.

Re-run the full-download verifier after any release change and write a new
immutable evidence file:

```bash
python scripts/verify_submission_release.py \
  --manifest artifacts/manifest.json \
  --output /workspace/evidence/submission-release-verification.json
```

The verifier performs full downloads and fails on HTTP status, size, or
SHA-256 mismatch. The output path must not already exist.
