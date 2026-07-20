#!/usr/bin/env python3
"""Render one deterministic Push rollout and emit an auditable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
from mujoco import mjx  # noqa: E402
from PIL import Image  # noqa: E402

from amd_robo.contracts import TaskPhase  # noqa: E402
from locomotion_learn_smoke import _load_config, _make_env  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.6, 0.0, 0.25)
    camera.distance = 2.2
    camera.azimuth = 140.0
    camera.elevation = -24.0
    return camera


def _save_frame(
    *,
    renderer: mujoco.Renderer,
    mj_data: mujoco.MjData,
    mj_model: mujoco.MjModel,
    mjx_data,
    camera: mujoco.MjvCamera,
    output_dir: Path,
    frame_index: int,
) -> str:
    mjx.get_data_into(mj_data, mj_model, mjx_data)
    renderer.update_scene(mj_data, camera=camera)
    filename = f"frame_{frame_index:06d}.jpg"
    Image.fromarray(renderer.render()).save(
        output_dir / filename,
        format="JPEG",
        quality=92,
        subsampling=0,
    )
    return filename


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/push_stage2_near_field_solver16_qualification.yaml",
    )
    parser.add_argument("--params-in", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--seed-pool-size", type=int, default=20)
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--render-stride", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    if args.seed_pool_size <= 0:
        parser.error("seed pool size must be positive")
    if not 0 <= args.seed_index < args.seed_pool_size:
        parser.error("seed index must be inside the seed pool")
    if args.render_stride <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("render stride and image dimensions must be positive")

    config_path = Path(args.config).resolve()
    params_path = Path(args.params_in).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not config_path.is_file():
        parser.error(f"config does not exist: {config_path}")
    if not params_path.is_file():
        parser.error(f"params do not exist: {params_path}")
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    config, config_sha256 = _load_config(str(config_path))
    evaluation = config["manual_evaluation"]
    num_steps = (
        evaluation["num_steps"] if args.num_steps is None else args.num_steps
    )
    if num_steps <= 0:
        parser.error("number of steps must be positive")

    from amd_robo.platform import _compat

    _compat.apply_brax_compat()
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks

    env = _make_env(
        config,
        command_override=evaluation["fixed_command"],
        randomize_reset=False,
        task="push",
    )
    ppo_config = config["ppo"]
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        policy_hidden_layer_sizes=tuple(
            ppo_config.get("policy_hidden_layer_sizes", (32, 32, 32, 32))
        ),
        value_hidden_layer_sizes=tuple(
            ppo_config.get(
                "value_hidden_layer_sizes",
                (256, 256, 256, 256, 256),
            )
        ),
    )
    params = brax_model.load_params(str(params_path))
    policy = ppo_networks.make_inference_fn(networks)(
        params,
        deterministic=True,
    )
    policy_key = jax.random.PRNGKey(0)
    policy_fn = jax.jit(lambda obs: policy(obs, policy_key)[0])
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    reset_keys = jax.random.split(
        jax.random.PRNGKey(args.seed),
        args.seed_pool_size,
    )
    reset_key = reset_keys[args.seed_index]
    state = reset_fn(reset_key)
    jax.block_until_ready(state.data.qpos)

    output_dir.mkdir(parents=True)
    mj_data = mujoco.MjData(env.mj_model)
    camera = _camera()
    renderer = mujoco.Renderer(
        env.mj_model,
        height=args.height,
        width=args.width,
    )

    frame_records: list[dict[str, int | str]] = []
    frame_records.append(
        {
            "frame": _save_frame(
                renderer=renderer,
                mj_data=mj_data,
                mj_model=env.mj_model,
                mjx_data=state.data,
                camera=camera,
                output_dir=output_dir,
                frame_index=0,
            ),
            "step": 0,
        }
    )
    first_phase_steps = {phase.name.lower(): None for phase in TaskPhase}
    first_phase_steps[TaskPhase.APPROACH.name.lower()] = 0
    max_object_speed = 0.0
    min_object_height = float(state.metrics["object_height"])
    max_object_height = min_object_height
    max_tilt_deg = 0.0
    illegal_contact_count = 0
    nonfinite_state_count = 0
    action_saturation_count = 0
    final_step = 0
    success = False
    start = time.perf_counter()

    try:
        for step in range(1, num_steps + 1):
            action = policy_fn(state.obs)
            state = step_fn(state, action)
            jax.block_until_ready(state.data.qpos)
            final_step = step

            phase = TaskPhase(int(state.info["phase"]))
            phase_name = phase.name.lower()
            if first_phase_steps[phase_name] is None:
                first_phase_steps[phase_name] = step
            object_speed = float(jnp.linalg.norm(state.info["object_qvel"][:2]))
            object_height = float(state.metrics["object_height"])
            tilt_deg = float(state.metrics["tilt_deg"])
            max_object_speed = max(max_object_speed, object_speed)
            min_object_height = min(min_object_height, object_height)
            max_object_height = max(max_object_height, object_height)
            max_tilt_deg = max(max_tilt_deg, tilt_deg)
            illegal_contact_count += int(state.metrics["illegal_contact"] > 0.0)
            nonfinite_state_count += int(state.metrics["nonfinite_state"] > 0.0)
            action_saturation_count += int(jnp.any(jnp.abs(action) >= 0.999))
            success = bool(state.metrics["success"] > 0.0)

            if step % args.render_stride == 0 or success:
                frame_records.append(
                    {
                        "frame": _save_frame(
                            renderer=renderer,
                            mj_data=mj_data,
                            mj_model=env.mj_model,
                            mjx_data=state.data,
                            camera=camera,
                            output_dir=output_dir,
                            frame_index=len(frame_records),
                        ),
                        "step": step,
                    }
                )
            if bool(state.done):
                break
        if frame_records[-1]["step"] != final_step:
            frame_records.append(
                {
                    "frame": _save_frame(
                        renderer=renderer,
                        mj_data=mj_data,
                        mj_model=env.mj_model,
                        mjx_data=state.data,
                        camera=camera,
                        output_dir=output_dir,
                        frame_index=len(frame_records),
                    ),
                    "step": final_step,
                }
            )
    finally:
        renderer.close()

    walltime = time.perf_counter() - start
    manifest = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": config_sha256,
        "params_sha256": _sha256(params_path),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "seed": args.seed,
        "seed_pool_size": args.seed_pool_size,
        "seed_index": args.seed_index,
        "policy": "trained_deterministic",
        "solver_iterations": int(env.mj_model.opt.iterations),
        "control_timestep": env.dt,
        "render_stride": args.render_stride,
        "video_fps": 1.0 / (env.dt * args.render_stride),
        "width": args.width,
        "height": args.height,
        "requested_steps": num_steps,
        "final_step": final_step,
        "success": success,
        "first_phase_steps": first_phase_steps,
        "final_goal_distance": float(state.metrics["object_to_goal_distance"]),
        "max_object_speed": max_object_speed,
        "min_object_height": min_object_height,
        "max_object_height": max_object_height,
        "max_tilt_deg": max_tilt_deg,
        "illegal_contact_count": illegal_contact_count,
        "nonfinite_state_count": nonfinite_state_count,
        "action_saturation_count": action_saturation_count,
        "frame_count": len(frame_records),
        "frames": frame_records,
        "rollout_and_render_walltime_s": walltime,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PUSH_ROLLOUT_RENDER "
        + json.dumps(
            {
                key: manifest[key]
                for key in (
                    "git_commit",
                    "config_sha256",
                    "params_sha256",
                    "seed",
                    "seed_pool_size",
                    "seed_index",
                    "solver_iterations",
                    "final_step",
                    "success",
                    "max_object_speed",
                    "min_object_height",
                    "max_object_height",
                    "illegal_contact_count",
                    "nonfinite_state_count",
                    "frame_count",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
