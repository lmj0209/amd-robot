# Risk-Aware Hazard Triage and Containment

## Product objective

The project targets hazardous warehouses, industrial facilities, and emergency
response areas where a person should not approach a suspicious battery pack,
chemical container, or object blocking an evacuation aisle.

The Go2 base approaches the active object, the Z1 arm aligns with a safe contact
point, and the controller pushes the object into a containment zone or out of a
clearance corridor. The system runs in MuJoCo MJX on one AMD Radeon GPU through
ROCm.

This is a simulation research prototype. It is not safety certified, is not
validated on a physical robot, and must not claim real hazardous-material
handling capability.

## Scoring-oriented capabilities

The final candidate should demonstrate measurable behavior beyond relabeling
the existing push box:

1. a risk level conditions the task observation and safety envelope;
2. containment and corridor-clearance modes use different physical goals;
3. higher simulated risk produces a more conservative measured speed profile;
4. randomized and held-out layouts are evaluated independently;
5. two active objects can be prioritized and handled sequentially.

The action contract remains 19-dimensional:

- legs: indices 0 through 11;
- arm: indices 12 through 17;
- gripper: index 18.

## Stable baseline

The accepted baseline remains the four-phase
`Approach -> Align -> Push -> Hold` task and its fixed-nearfield v3 inference
parameters. New experiments must not overwrite that artifact.

The solver-16 fixed evaluation completed 20 of 20 task episodes, but one episode
exceeded the internal 0.5 m/s object-speed gate. Randomized placement is not yet
accepted. These limitations remain visible until replaced by stronger measured
evidence.

## Gate sequence

### O1: evaluation determinism

Audit identical checkpoint, config, keys, and initial states across policy
orders, repeated runs, fresh processes, and single versus batched execution.
Record the first divergent step and field. No new training is allowed while
policy order changes discrete episode outcomes without an identified cause.

### O2: risk-aware safety governor

Limit contact approach and push commands using measured relative speed, object
speed, and optionally object acceleration. This is a controller change, not a
solver or threshold change. The first target is zero speed-gate exceedances in
20 fixed episodes without reducing task success.

### O3: randomization

Increase object-x randomization through 0.002 m, 0.005 m, and 0.010 m ranges.
Add y position, mass, and friction only as isolated later variables. Training,
validation, and held-out seed sets must remain separate.

### O4: risk and disposal semantics

Add risk level, task mode, active safety envelope, and relative goal state to
the observation contract. `contain` places the object in a containment zone;
`clear` moves it beyond a corridor boundary. Both results must be derived from
MJX physical state.

### O5: multi-object triage

Start with two objects rather than an eighteen-object collection task. A
higher-risk object is contained first, and a normal obstruction is then cleared
from the corridor. Each object's success and safety metrics are reported
separately.

### O6: perception boundary

First introduce a state-estimator interface and configurable pose/label noise.
RGB-D or semantic perception is a stretch gate after the physical task is
accepted. No CUDA-specific perception path may be introduced.

## Evidence requirements

Formal claims require:

- commit and clean worktree;
- config and parameter SHA256;
- hardware and software fingerprint;
- fixed training, validation, and held-out seeds;
- raw output and explicit process exit status;
- per-episode success, completion time, target error, speed, height, tilt,
  illegal contact, non-finite state, and action saturation;
- an unedited successful rollout and at least one documented failure.

An upstream contribution is a separate competition category. The prepared
patch is not an opened pull request, and no upstream score is claimed unless
publication is explicitly authorized and completed.
