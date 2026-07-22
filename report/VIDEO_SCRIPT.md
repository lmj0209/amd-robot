# Four-minute submission video

Target duration: 4:00. The previous review render is withdrawn because its
batched evaluation card is no longer qualification evidence. Regenerate it
from this script with English on-screen text plus the complete unedited
simulation rollout. Optional narration below may be recorded later; it must
not replace or cover the measured on-screen limitations.

Review status: a credited silent v4 candidate was rendered from commit
`4ae1aa5239460224a1b4265ec825916e0f2b8855`. It contains 4,793 decoded frames
over 239.65 seconds, including all 813 rollout frames. Its SHA-256 is
`90e949e693d5c0bf6d72cccb4cdc24412bba441622570418d92d287741736c35`.
The verified credit is `limengjin — Solo developer`. Public repository,
video, and checkpoint URLs all returned HTTP 404 in an anonymous check on
2026-07-21 and remain release blockers. This v4 render also predates the
bounded-JIT seed-matrix evaluator; do not submit it unchanged if that matrix
produces a reportable aggregate result.

| Time | Visual | On-screen message |
|---|---|---|
| 0:00-0:15 | Title and final rollout frame | ROCm-Accelerated Quadruped Mobile Manipulation with MJX |
| 0:15-0:35 | Problem statement and Push frame | Clear a near-field obstruction: Approach, Align, Push, Hold |
| 0:35-1:00 | Architecture diagram | MJX physics -> Playground-style environment -> Brax PPO -> ROCm evidence |
| 1:00-1:20 | AMD engineering card | One W7900; fixed 19D action; gfx1100-safe host small-update loop |
| 1:20-2:01 | Complete 40.65 s rollout | Unedited seed 777, pool 20, index 0 |
| 2:01-2:25 | Evaluation table | Fresh-process A/B: trained 3/3 in both; batched Push is diagnostic-only |
| 2:25-2:50 | Benchmark chart | 7,404 combined transitions/s at practical batch 2,048 |
| 2:50-3:10 | Training card | 73,728 transitions; exact checkpoint scope; final KL and policy std |
| 3:10-3:30 | Reproduction card | Versioned config, seed, hashes, renderer, fail-closed benchmark |
| 3:30-3:50 | Limitations card | Fixed box only; no perception/sim-to-real; randomized candidate rejected |
| 3:50-4:00 | Closing frame | Physical AI on Radeon + ROCm, with measured evidence |

## Optional narration

### 0:00-0:15

This is a ROCm-accelerated physical AI project for AMD Robot Competition
Track 3. A Go2 quadruped with a Z1 arm moves a box into a target zone.

### 0:15-0:35

The scoped application is simple warehouse obstruction clearing. The robot
must approach a fixed near-field box, align, push it toward the goal, and hold
it within eight centimeters for one hundred control steps.

### 0:35-1:00

MuJoCo MJX owns dynamics and contact. A Playground-style environment owns
observations, phase logic, reward, termination, and metrics. Brax owns PPO.
ROCm utilities validate the backend and synchronize every benchmark. The
interface remains nineteen actions: twelve legs, six arm joints, and one
gripper.

### 1:00-1:20

All measured GPU work uses one Radeon PRO W7900 with ROCm 7.2.1. Large fused
reverse-mode graphs were unstable on gfx1100, so training uses a bounded
host small-update loop with complete learner and rollout state persistence.
This keeps MJX and Brax PPO on the required AMD stack.

### 1:20-2:01

This is the complete rendered rollout: seed seven-seven-seven, index zero
from a twenty-key pool. No frames were removed. The policy reaches Hold at
step three thousand nine hundred sixty-one and completes at step four
thousand sixty.

### 2:01-2:25

Across two fresh processes, the trained policy succeeded in all three exact
repeats per process. Maximum object speeds were point-four-four-one and
point-three-eight-four meters per second. Earlier batched Push statistics are
diagnostic-only because repeated gfx eleven-hundred batches changed discrete
outcomes. This remains simulation evidence, not a safety certificate.

### 2:25-2:50

The formal benchmark used fresh processes, separate cold compilation, ten
warmups, and synchronized repeats. Combined policy and MJX throughput reached
seven thousand four hundred four transitions per second at batch two
thousand forty-eight. Batch four thousand ninety-six was only three percent
faster but used much more compile time and memory.

### 2:50-3:10

The selected Push run generated seventy-three thousand seven hundred
twenty-eight transitions. Exact resumes include parameters, optimizer,
rollout state, and both random-number states. The final reused host call
reported three hundred eighty-three SPS and finite optimization metrics.

### 3:10-3:30

Reproduction is based on versioned YAML, explicit seeds, parameter and video
hashes, a renderer that records every frame, and benchmark scripts that fail
on the wrong backend or non-finite output.

### 3:30-3:50

The present boundary is equally important: fixed near-field placement,
state-based simulation, no real-robot transfer, and no safety certificate.
A randomized-box candidate failed its registered gates, and the hundred-
episode expansion was intentionally paused.

### 3:50-4:00

The result is an auditable mobile-manipulation MVP and a measured ROCm path
for contact-rich physical AI on Radeon.

## Final-edit checklist

- Keep the verified `limengjin — Solo developer` credit visible.
- Record narration only if the exact measured claims remain unchanged.
- Keep the complete 813-frame rollout segment intact.
- Do not present the legacy 20-environment batch as qualification evidence.
- Verify the final MP4 by decoding every frame and record its SHA-256.
- After the held-out matrix decision, either preserve the disclosed limitation
  or render a new evidence card from the completed aggregate; never mix v4
  visuals with newer claims.
