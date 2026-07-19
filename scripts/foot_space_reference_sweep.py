#!/usr/bin/env python3
"""Sweep the analytic foot-space crawl in native CPU MuJoCo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from amd_robo.envs.go2_kinematics import (  # noqa: E402
    go2_foot_space_crawl_reference,
)
from crawl_reference_sweep import (  # noqa: E402
    CRAWL_SEQUENCE,
    DEFAULT_XML,
    FOOT_GEOM_NAMES,
    FOOT_SITE_NAMES,
    _active_contacts,
    _tilt_deg,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    cycle_time: float
    step_length: float
    foot_clearance: float
    body_shift_x: float
    body_shift_y: float
    shift_end_fraction: float = 0.25
    lift_start_fraction: float = 0.3
    lift_end_fraction: float = 0.8


DEFAULT_CANDIDATES = (
    Candidate("initial", 4.0, 0.08, 0.04, 0.02, 0.02),
    Candidate("no_fore_aft_shift", 4.0, 0.08, 0.04, 0.0, 0.02),
    Candidate("small_shift", 4.0, 0.08, 0.04, 0.005, 0.015),
    Candidate("lateral_only_small", 4.0, 0.08, 0.04, 0.0, 0.015),
    Candidate("step_0_06", 4.0, 0.06, 0.04, 0.0, 0.015),
    Candidate("step_0_10", 4.0, 0.10, 0.04, 0.0, 0.015),
    Candidate("cycle_3_0", 3.0, 0.08, 0.04, 0.0, 0.015),
    Candidate("clearance_0_06", 4.0, 0.08, 0.06, 0.0, 0.015),
)


def _parse_candidate(value: str) -> Candidate:
    fields = value.split(",")
    if len(fields) not in (6, 9):
        raise argparse.ArgumentTypeError(
            "candidate must be NAME,CYCLE,STEP,CLEARANCE,SHIFT_X,SHIFT_Y "
            "or append SHIFT_END,LIFT_START,LIFT_END"
        )
    try:
        timing = (
            (0.25, 0.3, 0.8)
            if len(fields) == 6
            else tuple(float(field) for field in fields[6:9])
        )
        candidate = Candidate(
            name=fields[0],
            cycle_time=float(fields[1]),
            step_length=float(fields[2]),
            foot_clearance=float(fields[3]),
            body_shift_x=float(fields[4]),
            body_shift_y=float(fields[5]),
            shift_end_fraction=timing[0],
            lift_start_fraction=timing[1],
            lift_end_fraction=timing[2],
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if (
        not candidate.name
        or candidate.cycle_time <= 0.0
        or candidate.step_length <= 0.0
        or candidate.foot_clearance <= 0.0
        or not np.isfinite(candidate.body_shift_x)
        or candidate.body_shift_y < 0.0
        or not (
            0.0
            < candidate.shift_end_fraction
            <= candidate.lift_start_fraction
            < candidate.lift_end_fraction
            < 1.0
        )
    ):
        raise argparse.ArgumentTypeError(
            "candidate dimensions must be positive, lateral shift non-negative, "
            "fore-aft shift finite, and timing must satisfy "
            "0 < shift end <= lift start < lift end < 1"
        )
    return candidate


def _reference_sequence(
    candidate: Candidate,
    *,
    num_steps: int,
    control_timestep: float,
) -> tuple[np.ndarray, bool]:
    phases = (
        2.0
        * jnp.pi
        * jnp.arange(num_steps, dtype=jnp.float32)
        * control_timestep
        / candidate.cycle_time
    )

    def reference(phase):
        return go2_foot_space_crawl_reference(
            phase,
            step_length=candidate.step_length,
            foot_clearance=candidate.foot_clearance,
            body_shift_x=candidate.body_shift_x,
            body_shift_y=candidate.body_shift_y,
            shift_end_fraction=candidate.shift_end_fraction,
            lift_start_fraction=candidate.lift_start_fraction,
            lift_end_fraction=candidate.lift_end_fraction,
        )

    references, _, reachable = jax.jit(jax.vmap(reference))(phases)
    return np.asarray(references), bool(jnp.all(reachable))


def _support_margin(
    center_of_mass_xy: np.ndarray,
    foot_positions_xy: np.ndarray,
    excluded_leg: int,
) -> float:
    """Return signed COM distance to the three-foot support boundary."""

    support = np.delete(foot_positions_xy, excluded_leg, axis=0)
    centroid = np.mean(support, axis=0)
    angles = np.arctan2(support[:, 1] - centroid[1], support[:, 0] - centroid[0])
    support = support[np.argsort(angles)]
    edges = np.roll(support, -1, axis=0) - support
    relative_com = center_of_mass_xy - support
    signed_distances = (
        edges[:, 0] * relative_com[:, 1]
        - edges[:, 1] * relative_com[:, 0]
    ) / np.linalg.norm(edges, axis=1)
    return float(np.min(signed_distances))


def _run_candidate(
    xml_path: Path,
    candidate: Candidate,
    *,
    num_cycles: int,
    control_timestep: float,
    settle_time: float,
    leg_kp: float,
    leg_kd: float,
    minimum_air_time: float,
    max_tilt_gate_deg: float,
    minimum_support_fraction: float,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.actuator_gainprm[:12, 0] = leg_kp
    model.actuator_biasprm[:12, 1] = -leg_kp
    model.actuator_biasprm[:12, 2] = -leg_kd
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
    base_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base"
    )
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
    num_steps = round(num_cycles * candidate.cycle_time / control_timestep)
    references, ik_reachable = _reference_sequence(
        candidate,
        num_steps=num_steps,
        control_timestep=control_timestep,
    )

    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    data.qpos[7:19] += references[0]
    data.ctrl[:] = home_ctrl
    data.ctrl[:12] += references[0]
    mujoco.mj_forward(model, data)
    settle_steps = round(settle_time / control_timestep)
    if not np.isclose(settle_steps * control_timestep, settle_time):
        raise ValueError("settle time must be divisible by control timestep")
    for _ in range(settle_steps):
        for _ in range(substeps):
            mujoco.mj_step(model, data)

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
    sustained_swing_by_active_leg = np.zeros((4, 4), dtype=np.int32)
    contact_duty_total = np.zeros(4)
    support_count = 0
    illegal_contact_count = int(initial_illegal)
    nonfinite_state_count = 0
    max_tilt_deg = _tilt_deg(data)
    max_foot_height = data.site_xpos[foot_site_ids, 2].copy()
    pre_lift_support_margin = np.full(4, np.inf)
    active_swing_steps = np.zeros(4, dtype=np.int32)
    active_swing_contact_steps = np.zeros(4, dtype=np.int32)
    active_swing_max_height = np.full(4, -np.inf)

    for step_index, reference in enumerate(references):
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
        quarter_position = (
            4.0 * step_index * control_timestep / candidate.cycle_time
        )
        quarter_phase = quarter_position - np.floor(quarter_position)
        active_leg = CRAWL_SEQUENCE[int(np.floor(quarter_position)) % 4]
        liftoff_count += previous_contact & ~foot_contact
        touchdown_count += ~previous_contact & foot_contact
        contact_filter = foot_contact | previous_contact
        next_air_time = feet_air_time + control_timestep
        sustained_swing = (
            (next_air_time >= minimum_air_time) & contact_filter
        )
        sustained_swing_count += sustained_swing
        sustained_swing_by_active_leg[active_leg] += sustained_swing
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
        if (
            candidate.shift_end_fraction
            <= quarter_phase
            < candidate.lift_start_fraction
        ):
            margin = _support_margin(
                data.subtree_com[base_body_id, :2],
                data.site_xpos[foot_site_ids, :2],
                active_leg,
            )
            pre_lift_support_margin[active_leg] = min(
                pre_lift_support_margin[active_leg], margin
            )
        if (
            candidate.lift_start_fraction
            <= quarter_phase
            < candidate.lift_end_fraction
        ):
            active_swing_steps[active_leg] += 1
            active_swing_contact_steps[active_leg] += int(
                foot_contact[active_leg]
            )
            active_swing_max_height[active_leg] = max(
                active_swing_max_height[active_leg],
                data.site_xpos[foot_site_ids[active_leg], 2],
            )

    duration_s = num_steps * control_timestep
    displacement = float(data.qpos[0] - initial_x)
    support_fraction = support_count / num_steps
    expected_swings = np.full(4, num_cycles)
    accepted = bool(
        ik_reachable
        and illegal_contact_count == 0
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
        "settle_time": settle_time,
        "physics_substeps": substeps,
        "leg_kp": leg_kp,
        "leg_kd": leg_kd,
        "ik_reachable": ik_reachable,
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
        "sustained_swing_by_active_leg": (
            sustained_swing_by_active_leg.tolist()
        ),
        "max_foot_height": max_foot_height.tolist(),
        "pre_lift_support_margin": pre_lift_support_margin.tolist(),
        "active_swing_contact_fraction": (
            active_swing_contact_steps / active_swing_steps
        ).tolist(),
        "active_swing_max_height": active_swing_max_height.tolist(),
        "accepted": accepted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--candidate", action="append", type=_parse_candidate)
    parser.add_argument("--num-cycles", type=int, default=4)
    parser.add_argument("--control-timestep", type=float, default=0.01)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--leg-kp", type=float, default=50.0)
    parser.add_argument("--leg-kd", type=float, default=0.5)
    parser.add_argument("--minimum-air-time", type=float, default=0.07)
    parser.add_argument("--max-tilt-gate-deg", type=float, default=12.0)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.98)
    args = parser.parse_args()
    if (
        args.num_cycles <= 0
        or args.control_timestep <= 0.0
        or args.settle_time < 0.0
        or args.leg_kp <= 0.0
        or args.leg_kd < 0.0
        or args.minimum_air_time <= 0.0
        or args.max_tilt_gate_deg <= 0.0
        or not 0.0 <= args.minimum_support_fraction <= 1.0
    ):
        parser.error("invalid sweep gate or simulation parameter")

    candidates = tuple(args.candidate or DEFAULT_CANDIDATES)
    results = [
        _run_candidate(
            args.xml,
            candidate,
            num_cycles=args.num_cycles,
            control_timestep=args.control_timestep,
            settle_time=args.settle_time,
            leg_kp=args.leg_kp,
            leg_kd=args.leg_kd,
            minimum_air_time=args.minimum_air_time,
            max_tilt_gate_deg=args.max_tilt_gate_deg,
            minimum_support_fraction=args.minimum_support_fraction,
        )
        for candidate in candidates
    ]
    for result in results:
        print(
            "FOOT_SPACE_REFERENCE_CANDIDATE "
            + json.dumps(result, sort_keys=True),
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
        "FOOT_SPACE_REFERENCE_SWEEP "
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
