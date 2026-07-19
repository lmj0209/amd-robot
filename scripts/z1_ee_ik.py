#!/usr/bin/env python3
"""Solve and audit a collision-free Z1 pre-push joint target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_push_mjx.xml"
ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))


def _required_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"required model object not found: {name}")
    return object_id


def _bad_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[int, float]:
    count = 0
    penetration = 0.0
    for contact in data.contact[: data.ncon]:
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body1 != 0 and body2 != 0 and contact.dist < 0.0:
            count += 1
            penetration += -float(contact.dist)
    return count, penetration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--base-x", type=float, default=0.3)
    parser.add_argument("--contact-clearance", type=float, default=0.06)
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.contact_clearance <= 0.0:
        parser.error("contact-clearance must be positive")
    if args.restarts <= 0 or args.iterations <= 0:
        parser.error("restarts and iterations must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    key_id = _required_id(model, mujoco.mjtObj.mjOBJ_KEY, "push_home")
    ee_site_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, "z1_ee")
    contact_site_id = _required_id(model, mujoco.mjtObj.mjOBJ_SITE, "push_contact_site")
    arm_joint_ids = np.asarray(
        [
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ]
    )
    arm_qpos_indices = model.jnt_qposadr[arm_joint_ids]
    arm_dof_indices = model.jnt_dofadr[arm_joint_ids]
    lower = model.jnt_range[arm_joint_ids, 0]
    upper = model.jnt_range[arm_joint_ids, 1]
    home = model.key_qpos[key_id, arm_qpos_indices].copy()

    data.qpos[:] = model.key_qpos[key_id]
    data.qpos[0] = args.base_x
    mujoco.mj_forward(model, data)
    target_position = data.site_xpos[contact_site_id].copy()
    target_position[0] -= args.contact_clearance

    rng = np.random.default_rng(args.seed)
    best = None
    for restart in range(args.restarts + 1):
        data.qpos[:] = model.key_qpos[key_id]
        data.ctrl[:] = model.key_ctrl[key_id]
        data.qpos[0] = args.base_x
        data.qpos[arm_qpos_indices] = (
            home if restart == 0 else rng.uniform(lower, upper)
        )
        for _ in range(args.iterations):
            mujoco.mj_forward(model, data)
            error = target_position - data.site_xpos[ee_site_id]
            if np.linalg.norm(error) < 1.0e-5:
                break
            jacobian_position = np.zeros((3, model.nv))
            jacobian_rotation = np.zeros((3, model.nv))
            mujoco.mj_jacSite(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                ee_site_id,
            )
            jacobian = jacobian_position[:, arm_dof_indices]
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1.0e-4 * np.eye(3),
                error,
            )
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0.08:
                delta *= 0.08 / delta_norm
            data.qpos[arm_qpos_indices] = np.clip(
                data.qpos[arm_qpos_indices] + delta,
                lower,
                upper,
            )

        mujoco.mj_forward(model, data)
        error_norm = float(np.linalg.norm(target_position - data.site_xpos[ee_site_id]))
        bad_count, bad_penetration = _bad_contacts(model, data)
        score = (
            error_norm
            + 10.0 * bad_penetration
            + 0.05 * bad_count
            + 1.0e-4 * float(np.linalg.norm(data.qpos[arm_qpos_indices] - home))
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "error": error_norm,
                "bad_count": bad_count,
                "bad_penetration": bad_penetration,
                "qpos": data.qpos[arm_qpos_indices].copy(),
                "ee_position": data.site_xpos[ee_site_id].copy(),
            }

    interpolation_bad_samples = 0
    interpolation_max_penetration = 0.0
    for fraction in np.linspace(0.0, 1.0, 101):
        data.qpos[:] = model.key_qpos[key_id]
        data.qpos[0] = args.base_x
        data.qpos[arm_qpos_indices] = home + fraction * (best["qpos"] - home)
        mujoco.mj_forward(model, data)
        bad_count, bad_penetration = _bad_contacts(model, data)
        interpolation_bad_samples += int(bad_count > 0)
        interpolation_max_penetration = max(
            interpolation_max_penetration,
            bad_penetration,
        )

    result = {
        "seed": args.seed,
        "base_x": args.base_x,
        "contact_clearance": args.contact_clearance,
        "restarts": args.restarts,
        "iterations": args.iterations,
        "target_position": [float(value) for value in target_position],
        "end_effector_position": [float(value) for value in best["ee_position"]],
        "position_error": best["error"],
        "arm_joint_target": [float(value) for value in best["qpos"]],
        "joint_lower": [float(value) for value in lower],
        "joint_upper": [float(value) for value in upper],
        "target_bad_contact_count": best["bad_count"],
        "target_bad_contact_penetration": best["bad_penetration"],
        "interpolation_samples": 101,
        "interpolation_bad_samples": interpolation_bad_samples,
        "interpolation_max_penetration": interpolation_max_penetration,
    }
    print(f"Z1_EE_IK {json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
