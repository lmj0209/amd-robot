"""Near-field Push-to-Goal task state for the qualified Go2+Z1 crawl."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco

from amd_robo.contracts import TaskPhase
from amd_robo.envs.go2_z1 import REPO_ROOT, _rotmat
from amd_robo.envs.go2_z1_locomotion import Go2Z1LocomotionEnv

DEFAULT_PUSH_XML = REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_push_mjx.xml"
PUSH_HOME_KEYFRAME = "push_home"
PUSH_BOX_BODY_NAME = "push_box_body"
PUSH_BOX_JOINT_NAME = "push_box_joint"
PUSH_BOX_GEOM_NAME = "push_box"
PREPUSH_SITE_NAME = "prepush_site"
GOAL_SITE_NAME = "goal_site"
DEFAULT_APPROACH_STOP_DISTANCE = 0.2


class Go2Z1PushEnv(Go2Z1LocomotionEnv):
    """Expose physical object and goal state without changing the 19-D action."""

    _TASK_OBSERVATION_SIZE = 9

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_PUSH_XML,
        approach_stop_distance: float = DEFAULT_APPROACH_STOP_DISTANCE,
        **kwargs,
    ) -> None:
        if approach_stop_distance <= 0.0:
            raise ValueError("approach stop distance must be positive")
        self._approach_stop_distance = float(approach_stop_distance)
        kwargs.setdefault("home_keyframe", PUSH_HOME_KEYFRAME)
        kwargs.setdefault("ctrl_dt", 0.01)
        kwargs.setdefault("action_scale", 0.1)
        kwargs.setdefault("leg_kp", 150.0)
        kwargs.setdefault("leg_kd", 8.0)
        kwargs.setdefault("foot_condim", 6)
        kwargs.setdefault("randomize_reset", False)
        kwargs.setdefault("command_override", (0.025, 0.0, 0.0))
        kwargs.setdefault("gait_cycle_time", 4.0)
        kwargs.setdefault("crawl_reference_enabled", True)
        kwargs.setdefault("crawl_shift_end_fraction", 0.25)
        kwargs.setdefault("crawl_lift_start_fraction", 0.45)
        kwargs.setdefault("crawl_lift_end_fraction", 0.85)
        kwargs.setdefault("crawl_foot_space_enabled", True)
        kwargs.setdefault("crawl_foot_step_length", 0.1)
        kwargs.setdefault("crawl_foot_clearance", 0.06)
        kwargs.setdefault("crawl_body_shift_x", 0.016)
        kwargs.setdefault("crawl_body_shift_y", 0.02)
        kwargs.setdefault("crawl_min_air_time", 0.07)
        kwargs.setdefault("crawl_pose_reference_enabled", True)
        super().__init__(xml_path=xml_path, **kwargs)

        self._box_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY, PUSH_BOX_BODY_NAME
        )
        self._box_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, PUSH_BOX_JOINT_NAME
        )
        self._box_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, PUSH_BOX_GEOM_NAME
        )
        self._prepush_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, PREPUSH_SITE_NAME
        )
        self._goal_site_id = self._required_id(mujoco.mjtObj.mjOBJ_SITE, GOAL_SITE_NAME)
        if self.mj_model.jnt_type[self._box_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("push box joint must be free")
        self._box_qpos_adr = int(self.mj_model.jnt_qposadr[self._box_joint_id])
        self._box_dof_adr = int(self.mj_model.jnt_dofadr[self._box_joint_id])
        self._box_qpos_slice = slice(self._box_qpos_adr, self._box_qpos_adr + 7)
        self._box_qvel_slice = slice(self._box_dof_adr, self._box_dof_adr + 6)
        self._initial_box_qpos = self._home_qpos[self._box_qpos_slice]
        self._allowed_floor_geom_ids = jnp.concatenate(
            [
                self._allowed_floor_geom_ids,
                jnp.asarray([self._box_geom_id], dtype=jnp.int32),
            ]
        )

    def _required_id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.mj_model, object_type, name)
        if object_id < 0:
            raise ValueError(f"push task object not found: {name}")
        return object_id

    @property
    def observation_size(self) -> int:
        return super().observation_size + self._TASK_OBSERVATION_SIZE

    def reset(self, rng: jax.Array):
        return self._with_task_state(super().reset(rng))

    def step(self, state, action: jax.Array):
        prepush_position = state.data.site_xpos[self._prepush_site_id]
        base_to_prepush = jnp.linalg.norm(state.data.qpos[:2] - prepush_position[:2])
        enter_align = (state.info["phase"] == int(TaskPhase.APPROACH)) & (
            base_to_prepush <= self._approach_stop_distance
        )
        phase = jnp.where(
            enter_align,
            int(TaskPhase.ALIGN),
            state.info["phase"],
        )
        command = jnp.where(
            phase == int(TaskPhase.APPROACH),
            state.info["command"],
            jnp.zeros_like(state.info["command"]),
        )
        staged = state.replace(
            info={
                **state.info,
                "phase": phase,
                "command": command,
            }
        )
        return self._with_task_state(super().step(staged, action))

    def _task_vectors(self, data):
        world_to_base = _rotmat(data.qpos[3:7]).T
        base_position = data.qpos[:3]
        object_position = data.xpos[self._box_body_id]
        prepush_position = data.site_xpos[self._prepush_site_id]
        goal_position = data.site_xpos[self._goal_site_id]
        object_relative = world_to_base @ (object_position - base_position)
        prepush_relative = world_to_base @ (prepush_position - base_position)
        goal_relative_object = world_to_base @ (goal_position - object_position)
        return (
            object_position,
            prepush_position,
            goal_position,
            object_relative,
            prepush_relative,
            goal_relative_object,
        )

    def _observation(
        self,
        data,
        last_action,
        phase,
        command: jax.Array | None = None,
        gait_phase: jax.Array | None = None,
    ) -> jax.Array:
        locomotion = super()._observation(
            data,
            last_action,
            phase,
            command,
            gait_phase,
        )
        (
            _,
            _,
            _,
            object_relative,
            prepush_relative,
            goal_relative_object,
        ) = self._task_vectors(data)
        task = jnp.concatenate(
            [object_relative, prepush_relative, goal_relative_object]
        )
        return jnp.clip(jnp.concatenate([locomotion, task]), -10.0, 10.0)

    def _with_task_state(self, state):
        (
            object_position,
            prepush_position,
            goal_position,
            _,
            _,
            _,
        ) = self._task_vectors(state.data)
        object_qpos = state.data.qpos[self._box_qpos_slice]
        object_qvel = state.data.qvel[self._box_qvel_slice]
        info = {
            **state.info,
            "object_qpos": object_qpos,
            "object_qvel": object_qvel,
            "object_pos": object_position,
            "prepush_pos": prepush_position,
            "goal_pos": goal_position,
        }
        metrics = {
            **state.metrics,
            "base_to_prepush_distance": jnp.linalg.norm(
                state.data.qpos[:2] - prepush_position[:2]
            ),
            "object_to_goal_distance": jnp.linalg.norm(
                object_position[:2] - goal_position[:2]
            ),
            "object_displacement": jnp.linalg.norm(
                object_qpos[:2] - self._initial_box_qpos[:2]
            ),
            "object_height": object_position[2],
            "approach_stop_distance": jnp.asarray(self._approach_stop_distance),
            "task_phase": info["phase"].astype(jnp.float32),
        }
        return state.replace(
            obs=self._observation(
                state.data,
                info["last_action"],
                info["phase"],
                info["command"],
                info.get("gait_phase"),
            ),
            metrics=metrics,
            info=info,
        )
