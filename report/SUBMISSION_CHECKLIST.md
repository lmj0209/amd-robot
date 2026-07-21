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

- [x] Public source repository URL: <https://github.com/lmj0209/amd-robot>
- [x] Public four-minute video URL:
  <https://github.com/lmj0209/amd-robot/releases/download/submission-artifacts-v1/amd_track3_submission_review_v4.mp4>
- [x] Public checkpoint or artifact URL:
  <https://github.com/lmj0209/amd-robot/releases/download/submission-artifacts-v1/push_nearfield_v3_params>
- [x] Verify each URL without GitHub authentication and recheck both downloaded
  files against their frozen SHA-256 values.
- [x] Record the final repository state with tag `submission-candidate-v1` and
  publish the video/checkpoint SHA-256 values in `artifacts/manifest.json`.

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

- [x] Run the CPU-safe test suite from a clean public checkout: Ruff passed and
  all 171 tests passed in GitHub Actions run
  [29827356191](https://github.com/lmj0209/amd-robot/actions/runs/29827356191).
- [x] Confirm README reproduction commands match the final public repository.
- [x] Confirm the checkpoint SHA-256 is
  `f033f9ff1ff23304b05ddeec9346d4f681ec5f4dbe265c5000804317916f9d0b`.
- [x] Confirm no secrets, private URLs, temporary files, or withdrawn claims
  are present.
- [x] Tag the reviewed candidate as `submission-candidate-v1` only after all
  public links and downloaded checksums are valid.
