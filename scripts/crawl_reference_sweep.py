#!/usr/bin/env python3
"""Sweep the zero-residual four-beat crawl in native CPU MuJoCo.

This is a fast physics filter before the more expensive sequential MJX gate.
It applies the same FL-RR-FR-RL joint reference as the locomotion environment,
keeps the 19-DoF action contract at zero residual, and reports contact, tilt,
and displacement diagnostics for each candidate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_mjx.xml"

FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")
FOOT_SITE_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
CRAWL_SEQUENCE = np.asarray([0, 3, 1, 2], dtype=np.int32)
CRAWL_PHASE_OFFSETS = np.asarray([0.0, 0.5, 0.75, 0.25])
CRAWL_FORE_AFT_SIGNS = np.asarray([1.0, 1.0, -1.0, -1.0])
CRAWL_LEFT_RIGHT_SIGNS = np.asarray([1.0, -1.0, 1.0, -1.0])


@dataclass(frozen=True)
class Candidate:
    name: str
    cycle_time: float
    stride: float
    shift: float
    lift: float


DEFAULT_CANDIDATES = (
    Candidate("baseline", 4.0, 0.08, 0.06, 0.45),
    Candidate("cycle_2_5", 2.5, 0.08, 0.06, 0.45),
    Candidate("cycle_3_0", 3.0, 0.08, 0.06, 0.45),
    Candidate("cycle_3_5", 3.5, 0.08, 0.06, 0.45),
    Candidate("stride_0_06", 4.0, 0.06, 0.06, 0.45),
    Candidate("stride_0_09", 4.0, 0.09, 0.06, 0.45),
    Candidate("shift_0_05", 4.0, 0.08, 0.05, 0.45),
    Candidate("shift_0_07", 4.0, 0.08, 0.07, 0.45),
    Candidate("lift_0_35", 4.0, 0.08, 0.06, 0.35),
    Candidate("lift_0_55", 4.0, 0.08, 0.06, 0.55),
)


def _smoothstep(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _crawl_reference(time_s: float, candidate: Candidate) -> np.ndarray:
    cycle_position = np.mod(time_s / candidate.cycle_time, 1.0)
    quarter_position = 4.0 * cycle_position
    slot = int(np.floor(quarter_position)) % 4
    quarter_phase = quarter_position - np.floor(quarter_position)
    active_leg = int(CRAWL_SEQUENCE[slot])
    previous_leg = int(CRAWL_SEQUENCE[(slot - 1) % 4])

    shift_blend = _smoothstep(quarter_phase / 0.3)
    previous_pitch = (
        -CRAWL_FORE_AFT_SIGNS[previous_leg] * candidate.shift
    )
    active_pitch = -CRAWL_FORE_AFT_SIGNS[active_leg] * candidate.shift
    previous_hip = CRAWL_LEFT_RIGHT_SIGNS[previous_leg] * candidate.shift
    active_hip = CRAWL_LEFT_RIGHT_SIGNS[active_leg] * candidate.shift
    common_pitch = (
        (1.0 - shift_blend) * previous_pitch + shift_blend * active_pitch
    )
    common_hip = (
        (1.0 - shift_blend) * previous_hip + shift_blend * active_hip
    )

    leg_phase = np.mod(cycle_position - CRAWL_PHASE_OFFSETS, 1.0)
    swing_progress = _smoothstep(leg_phase / 0.25)
    stance_progress = _smoothstep((leg_phase - 0.25) / 0.75)
    stride_pitch = np.where(
        leg_phase < 0.25,
        candidate.stride * (1.0 - 2.0 * swing_progress),
        candidate.stride * (-1.0 + 2.0 * stance_progress),
    )
    thigh = common_pitch + stride_pitch

    lift_progress = np.clip((quarter_phase - 0.3) / 0.5, 0.0, 1.0)
    lift_window = 0.3 <= quarter_phase < 0.8
    knee_lift = np.zeros(4)
    knee_lift[active_leg] = (
        candidate.lift
        * np.square(np.sin(np.pi * lift_progress))
        * float(lift_window)
    )
    calf = (
        0.65 * common_hip * CRAWL_LEFT_RIGHT_SIGNS
        + 0.8 * np.square(thigh)
        - knee_lift
    )
    return np.stack(
        [np.full(4, common_hip), thigh, calf],
        axis=-1,
    ).reshape(12)


def _parse_candidate(value: str) -> Candidate:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "candidate must be NAME,CYCLE_TIME,STRIDE,SHIFT,LIFT"
        )
    try:
        candidate = Candidate(
            name=fields[0],
            cycle_time=float(fields[1]),
            stride=float(fields[2]),
            shift=float(fields[3]),
            lift=float(fields[4]),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if (
        not candidate.name
        or candidate.cycle_time <= 0.0
        or candidate.stride < 0.0
        or candidate.shift <= 0.0
        or candidate.lift <= 0.0
    ):
        raise argparse.ArgumentTypeError(
            "candidate name must be non-empty; cycle, shift, and lift must be "
            "positive; stride must be non-negative"
        )
    return candidate


def _active_contacts(
    data: mujoco.MjData,
    *,
    floor_geom_id: int,
    foot_geom_ids: np.ndarray,
) -> tuple[np.ndarray, bool]:
    foot_contact = np.zeros(4, dtype=bool)
    illegal_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist >= 0.0:
            continue
        if contact.geom1 == floor_geom_id:
            other_geom = contact.geom2
        elif contact.geom2 == floor_geom_id:
            other_geom = contact.geom1
        else:
            continue
        matches = np.flatnonzero(foot_geom_ids == other_geom)
        if matches.size:
            foot_contact[matches] = True
        else:
            illegal_contact = True
    return foot_contact, illegal_contact


def _tilt_deg(data: mujoco.MjData) -> float:
    w, x, y, z = data.qpos[3:7]
    world_up_z = w * w - x * x - y * y + z * z
    return float(np.degrees(np.arccos(np.clip(world_up_z, -1.0, 1.0))))


def _run_candidate(
    xml_path: Path,
    candidate: Candidate,
    *,
    num_cycles: int,
    control_timestep: float,
    leg_kp: float,
    minimum_air_time: float,
    max_tilt_gate_deg: float,
    minimum_support_fraction: float,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.actuator_gainprm[:12, 0] = leg_kp
    model.actuator_biasprm[:12, 1] = -leg_kp
    substeps = round(control_timestep / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, control_timestep):
        raise ValueError("control timestep must be divisible by simulation timestep")

    floor_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    foot_geom_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOM_NAMES
        ]
    )
    foot_site_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in FOOT_SITE_NAMES
        ]
    )
    if floor_geom_id < 0 or np.any(foot_geom_ids < 0) or np.any(foot_site_ids < 0):
        raise ValueError("crawl sweep requires the floor and all foot geoms/sites")

    home_qpos = (
        model.key_qpos[0].copy() if model.nkey > 0 else model.qpos0.copy()
    )
    home_ctrl = (
        model.key_ctrl[0].copy()
        if model.nkey > 0 and model.key_ctrl.shape[1] > 0
        else np.zeros(model.nu)
    )
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]

    data = mujoco.MjData(model)
    initial_reference = _crawl_reference(0.0, candidate)
    data.qpos[:] = home_qpos
    data.qpos[7:19] += initial_reference
    data.ctrl[:] = home_ctrl
    data.ctrl[:12] += initial_reference
    mujoco.mj_forward(model, data)

    initial_x = float(data.qpos[0])
    previous_contact, initial_illegal = _active_contacts(
        data,
        floor_geom_id=floor_geom_id,
        foot_geom_ids=foot_geom_ids,
    )
    feet_air_time = np.zeros(4)
    liftoff_count = np.zeros(4, dtype=np.int32)
    touchdown_count = np.zeros(4, dtype=np.int32)
    sustained_swing_count = np.zeros(4, dtype=np.int32)
    contact_duty_total = np.zeros(4)
    support_count = 0
    illegal_contact_count = int(initial_illegal)
    nonfinite_state_count = 0
    max_tilt_deg = _tilt_deg(data)
    max_foot_height = data.site_xpos[foot_site_ids, 2].copy()

    num_steps = round(num_cycles * candidate.cycle_time / control_timestep)
    for step_index in range(num_steps):
        time_s = step_index * control_timestep
        reference = _crawl_reference(time_s, candidate)
        data.ctrl[:] = home_ctrl
        data.ctrl[:12] += reference
        data.ctrl[:] = np.clip(data.ctrl, ctrl_min, ctrl_max)
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        foot_contact, illegal_contact = _active_contacts(
            data,
            floor_geom_id=floor_geom_id,
            foot_geom_ids=foot_geom_ids,
        )
        liftoff_count += previous_contact & ~foot_contact
        touchdown_count += ~previous_contact & foot_contact
        contact_filter = foot_contact | previous_contact
        next_air_time = feet_air_time + control_timestep
        sustained_swing_count += (
            (next_air_time >= minimum_air_time) & contact_filter
        )
        feet_air_time = next_air_time * ~foot_contact
        previous_contact = foot_contact

        contact_duty_total += foot_contact
        support_count += int(np.count_nonzero(foot_contact) >= 3)
        illegal_contact_count += int(illegal_contact)
        finite = (
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.qacc))
            and np.all(np.isfinite(data.actuator_force))
        )
        nonfinite_state_count += int(not finite)
        max_tilt_deg = max(max_tilt_deg, _tilt_deg(data))
        max_foot_height = np.maximum(
            max_foot_height, data.site_xpos[foot_site_ids, 2]
        )

    duration_s = num_steps * control_timestep
    displacement = float(data.qpos[0] - initial_x)
    support_fraction = support_count / num_steps
    expected_swings = np.full(4, num_cycles)
    accepted = bool(
        illegal_contact_count == 0
        and nonfinite_state_count == 0
        and max_tilt_deg <= max_tilt_gate_deg
        and support_fraction >= minimum_support_fraction
        and np.array_equal(sustained_swing_count, expected_swings)
        and displacement > 0.0
    )
    return {
        **asdict(candidate),
        "backend": "mujoco_cpu",
        "num_cycles": num_cycles,
        "num_steps": num_steps,
        "duration_s": duration_s,
        "physics_substeps": substeps,
        "forward_displacement": displacement,
        "mean_forward_velocity": displacement / duration_s,
        "max_tilt_deg": max_tilt_deg,
        "illegal_contact_count": illegal_contact_count,
        "nonfinite_state_count": nonfinite_state_count,
        "three_or_more_contact_fraction": support_fraction,
        "foot_contact_duty": (contact_duty_total / num_steps).tolist(),
        "liftoffs": liftoff_count.tolist(),
        "touchdowns": touchdown_count.tolist(),
        "sustained_swings": sustained_swing_count.tolist(),
        "max_foot_height": max_foot_height.tolist(),
        "accepted": accepted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        help=(
            "candidate as NAME,CYCLE_TIME,STRIDE,SHIFT,LIFT; repeat the flag "
            "to replace the built-in one-factor sweep"
        ),
    )
    parser.add_argument("--num-cycles", type=int, default=8)
    parser.add_argument("--control-timestep", type=float, default=0.01)
    parser.add_argument("--leg-kp", type=float, default=50.0)
    parser.add_argument("--minimum-air-time", type=float, default=0.07)
    parser.add_argument("--max-tilt-gate-deg", type=float, default=12.0)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.98)
    args = parser.parse_args()
    if (
        args.num_cycles <= 0
        or args.control_timestep <= 0.0
        or args.leg_kp <= 0.0
        or args.minimum_air_time <= 0.0
        or args.max_tilt_gate_deg <= 0.0
        or not 0.0 <= args.minimum_support_fraction <= 1.0
    ):
        parser.error(
            "cycles, timesteps, gains, air time, and tilt gate must be "
            "positive; support fraction must be in [0, 1]"
        )

    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    results = [
        _run_candidate(
            args.xml,
            candidate,
            num_cycles=args.num_cycles,
            control_timestep=args.control_timestep,
            leg_kp=args.leg_kp,
            minimum_air_time=args.minimum_air_time,
            max_tilt_gate_deg=args.max_tilt_gate_deg,
            minimum_support_fraction=args.minimum_support_fraction,
        )
        for candidate in candidates
    ]
    for result in results:
        print(
            f"CRAWL_REFERENCE_CANDIDATE "
            f"{json.dumps(result, sort_keys=True)}",
            flush=True,
        )
    ranking = [
        result["name"]
        for result in sorted(
            (result for result in results if result["accepted"]),
            key=lambda result: result["mean_forward_velocity"],
            reverse=True,
        )
    ]
    print(
        "CRAWL_REFERENCE_SWEEP "
        + json.dumps(
            {
                "backend": "mujoco_cpu",
                "candidate_count": len(results),
                "accepted_count": len(ranking),
                "accepted_ranking": ranking,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
