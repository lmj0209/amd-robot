# Submission checklist

## Verified identity and scope

- Competition: AMD Robot Competition, Track 3 Physical AI.
- Project: ROCm-Accelerated Quadruped Mobile Manipulation with MJX.
- Author: `limengjin`.
- Role: solo developer.
- Validated controller: ungoverned fixed-v3 with solver 16.
- Frozen action contract: 19 dimensions.
- Current media format: silent video with complete English on-screen text.

## Public links required before submission

The account owner must create and verify these public URLs. Do not replace a
private resource or publish it without an explicit access decision.

- [ ] Public source repository URL: <https://github.com/lmj0209/amd-robot>
  (anonymous HTTP check returned 404 on 2026-07-21).
- [ ] Public four-minute video URL:
  <https://github.com/lmj0209/amd-robot/releases/download/submission-artifacts-v1/amd_track3_submission_review_v4.mp4>
  (anonymous HTTP check returned 404 on 2026-07-21).
- [ ] Public checkpoint or artifact URL:
  <https://github.com/lmj0209/amd-robot/releases/download/submission-artifacts-v1/push_nearfield_v3_params>
  (anonymous HTTP check returned 404 on 2026-07-21).
- [ ] Verify each URL without GitHub authentication and recheck both downloaded
  files against their frozen SHA-256 values with
  `scripts/verify_submission_release.py`; retain its immutable JSON result.
- [x] Preserve the historical `submission-candidate-v1` tag and the frozen
  video/checkpoint SHA-256 values in `artifacts/manifest.json`; do not move that
  tag. A new candidate tag is required after the current gates pass.

## Evidence rules

- [x] Use isolated one-environment fresh-process Push qualification only.
- [x] Keep batched gfx1100 Push statistics diagnostic-only.
- [x] Keep the complete 813-frame rollout segment unedited.
- [x] Disclose fixed near-field, state-based simulation scope.
- [x] Do not claim randomized-box, perception, sim-to-real, or hardware safety
  qualification.
- [x] Record the explicitly resumed upstream contribution as public Brax PR
  #674, with CLA/check success and without claiming acceptance or merge.

## Final media gate

- [x] Render after the final credit decision is frozen.
- [x] Decode every output frame and compare it with the expected frame count.
- [x] Confirm H.264 High, yuv420p, 960x540, 20 fps, and duration below 4:00.
- [x] Confirm the silent audio decision and inspect the final stream list.
- [x] Inspect the title, rollout entry/exit, evaluation card, limitations, and
  closing frame.
- [x] Save the renderer commit, manifest, SHA256SUMS, and review decision.

## Release gate

- [ ] Rerun the CPU-safe suite from the new public candidate. The previous
  candidate recorded Ruff plus 171 passing tests in GitHub Actions run
  [29827356191](https://github.com/lmj0209/amd-robot/actions/runs/29827356191).
- [ ] Confirm README reproduction commands match the new public candidate.
- [x] Confirm the checkpoint SHA-256 is
  `f033f9ff1ff23304b05ddeec9346d4f681ec5f4dbe265c5000804317916f9d0b`.
- [ ] Repeat the secret/private-URL/temporary-file/withdrawn-claim audit on the
  new candidate.
- [ ] Create a new immutable reviewed-candidate tag only after all public links,
  anonymous downloads, checksums, tests, and evidence gates are valid.
