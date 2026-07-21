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
- [ ] Public four-minute video URL: `TBD`
- [ ] Public checkpoint or artifact URL: `TBD`
- [ ] Verify each URL in a signed-out browser session.
- [ ] Record the final repository commit and video/checkpoint SHA-256 values.

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

- [ ] Run the CPU-safe test suite from a clean checkout.
- [ ] Confirm README reproduction commands match the final public repository.
- [ ] Confirm the checkpoint SHA-256 is
  `f033f9ff1ff23304b05ddeec9346d4f681ec5f4dbe265c5000804317916f9d0b`.
- [ ] Confirm no secrets, private URLs, temporary files, or withdrawn claims
  are present.
- [ ] Tag the reviewed candidate only after all public links are valid.
