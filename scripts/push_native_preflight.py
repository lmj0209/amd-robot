#!/usr/bin/env python3
"""Run a long-horizon native MuJoCo stability gate for the push scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_push_mjx.xml"


def _required_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"push task object not found: {name}")
    return object_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--num-steps", type=int, default=8000)
    parser.add_argument("--solver-iterations", type=int)
    args = parser.parse_args()
    if args.num_steps <= 0:
        parser.error("num-steps must be positive")
    if args.solver_iterations is not None and args.solver_iterations <= 0:
        parser.error("solver-iterations must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if args.solver_iterations is not None:
        model.opt.iterations = args.solver_iterations
    key_id = _required_id(model, mujoco.mjtObj.mjOBJ_KEY, "push_home")
    box_body_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "push_box_body")
    box_joint_id = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "push_box_joint")
    box_qpos_adr = int(model.jnt_qposadr[box_joint_id])
    box_dof_adr = int(model.jnt_dofadr[box_joint_id])

    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[key_id]
    data.ctrl[:] = model.key_ctrl[key_id]
    mujoco.mj_forward(model, data)
    initial_box_position = data.xpos[box_body_id].copy()
    max_box_height = float(initial_box_position[2])
    max_box_speed = 0.0
    first_motion_step = None

    for step_index in range(args.num_steps):
        mujoco.mj_step(model, data)
        box_position = data.xpos[box_body_id]
        box_speed = np.linalg.norm(data.qvel[box_dof_adr : box_dof_adr + 3])
        displacement = np.linalg.norm(box_position[:2] - initial_box_position[:2])
        max_box_height = max(max_box_height, float(box_position[2]))
        max_box_speed = max(max_box_speed, float(box_speed))
        if first_motion_step is None and displacement > 1.0e-3:
            first_motion_step = step_index + 1

    box_position = data.xpos[box_body_id]
    result = {
        "backend": "native_mujoco",
        "num_steps": args.num_steps,
        "simulated_seconds": args.num_steps * model.opt.timestep,
        "solver_iterations": int(model.opt.iterations),
        "finite_qpos": bool(np.isfinite(data.qpos).all()),
        "finite_qvel": bool(np.isfinite(data.qvel).all()),
        "first_object_motion_step": first_motion_step,
        "final_object_position": [float(value) for value in box_position],
        "object_displacement": float(
            np.linalg.norm(box_position[:2] - initial_box_position[:2])
        ),
        "max_object_height": max_box_height,
        "max_object_speed": max_box_speed,
        "box_qpos_address": box_qpos_adr,
    }
    print(f"PUSH_NATIVE_PREFLIGHT {json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
