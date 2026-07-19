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
END_EFFECTOR_SITE_NAME = "z1_ee"
PUSH_CONTACT_SITE_NAME = "push_contact_site"
PREPUSH_SITE_NAME = "prepush_site"
GOAL_SITE_NAME = "goal_site"
DEFAULT_APPROACH_STOP_DISTANCE = 0.2
DEFAULT_ALIGN_DURATION = 6.0
DEFAULT_ALIGN_DISTANCE_THRESHOLD = 0.08
DEFAULT_PUSH_COMMAND_X = 0.025
ALIGN_ARM_JOINT_TARGET = (
    2.4480703588935633,
    2.775837254707154,
    -0.5128763925136487,
    -0.46937292541924974,
    0.6170071964054299,
    0.18534777065335487,
)


class Go2Z1PushEnv(Go2Z1LocomotionEnv):
    """Expose physical object and goal state without changing the 19-D action."""

    _TASK_OBSERVATION_SIZE = 12

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_PUSH_XML,
        approach_stop_distance: float = DEFAULT_APPROACH_STOP_DISTANCE,
        align_duration: float = DEFAULT_ALIGN_DURATION,
        align_distance_threshold: float = DEFAULT_ALIGN_DISTANCE_THRESHOLD,
        push_command_x: float = DEFAULT_PUSH_COMMAND_X,
        **kwargs,
    ) -> None:
        if approach_stop_distance <= 0.0:
            raise ValueError("approach stop distance must be positive")
        if align_duration <= 0.0:
            raise ValueError("align duration must be positive")
        if align_distance_threshold <= 0.0:
            raise ValueError("align distance threshold must be positive")
        if push_command_x <= 0.0:
            raise ValueError("push command must be positive")
        self._approach_stop_distance = float(approach_stop_distance)
        self._align_duration = float(align_duration)
        self._align_distance_threshold = float(align_distance_threshold)
        self._push_command = jnp.asarray(
            [push_command_x, 0.0, 0.0],
            dtype=jnp.float32,
        )
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
        self._end_effector_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, END_EFFECTOR_SITE_NAME
        )
        self._push_contact_site_id = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, PUSH_CONTACT_SITE_NAME
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
        self._align_arm_joint_target = jnp.asarray(ALIGN_ARM_JOINT_TARGET)
        if bool(
            jnp.any(self._align_arm_joint_target < self._ctrl_min[12:18])
            | jnp.any(self._align_arm_joint_target > self._ctrl_max[12:18])
        ):
            raise ValueError("align arm target exceeds actuator limits")
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
        state = super().reset(rng)
        state = state.replace(
            info={
                **state.info,
                "align_steps": jnp.asarray(0, dtype=jnp.int32),
                "push_steps": jnp.asarray(0, dtype=jnp.int32),
            }
        )
        return self._with_task_state(state)

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
        align_complete = (
            (phase == int(TaskPhase.ALIGN))
            & (state.info["align_steps"] * self.dt >= self._align_duration)
            & (
                state.metrics["end_effector_to_push_distance"]
                <= self._align_distance_threshold
            )
        )
        phase = jnp.where(
            align_complete,
            int(TaskPhase.PUSH),
            phase,
        )
        command = jnp.where(
            phase == int(TaskPhase.APPROACH),
            state.info["command"],
            jnp.where(
                phase == int(TaskPhase.PUSH),
                self._push_command,
                jnp.zeros_like(state.info["command"]),
            ),
        )
        align_step_limit = round(self._align_duration / self.dt)
        align_steps = jnp.where(
            phase >= int(TaskPhase.ALIGN),
            jnp.minimum(state.info["align_steps"] + 1, align_step_limit),
            0,
        )
        push_steps = jnp.where(
            phase == int(TaskPhase.PUSH),
            state.info["push_steps"] + 1,
            0,
        )
        staged = state.replace(
            info={
                **state.info,
                "phase": phase,
                "command": command,
                "align_steps": align_steps,
                "push_steps": push_steps,
            }
        )
        return self._with_task_state(super().step(staged, action))

    def _task_actuator_reference(self, state) -> jax.Array:
        progress = jnp.clip(
            state.info["align_steps"] * self.dt / self._align_duration,
            0.0,
            1.0,
        )
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        arm_offset = smooth_progress * (
            self._align_arm_joint_target - self._home_ctrl[12:18]
        )
        return jnp.zeros_like(self._home_ctrl).at[12:18].set(arm_offset)

    def _task_vectors(self, data):
        world_to_base = _rotmat(data.qpos[3:7]).T
        base_position = data.qpos[:3]
        object_position = data.xpos[self._box_body_id]
        end_effector_position = data.site_xpos[self._end_effector_site_id]
        push_contact_position = data.site_xpos[self._push_contact_site_id]
        prepush_position = data.site_xpos[self._prepush_site_id]
        goal_position = data.site_xpos[self._goal_site_id]
        object_relative = world_to_base @ (object_position - base_position)
        prepush_relative = world_to_base @ (prepush_position - base_position)
        goal_relative_object = world_to_base @ (goal_position - object_position)
        push_contact_relative_ee = world_to_base @ (
            push_contact_position - end_effector_position
        )
        return (
            object_position,
            end_effector_position,
            push_contact_position,
            prepush_position,
            goal_position,
            object_relative,
            prepush_relative,
            goal_relative_object,
            push_contact_relative_ee,
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
            _,
            _,
            object_relative,
            prepush_relative,
            goal_relative_object,
            push_contact_relative_ee,
        ) = self._task_vectors(data)
        task = jnp.concatenate(
            [
                object_relative,
                prepush_relative,
                goal_relative_object,
                push_contact_relative_ee,
            ]
        )
        return jnp.clip(jnp.concatenate([locomotion, task]), -10.0, 10.0)

    def _with_task_state(self, state):
        (
            object_position,
            end_effector_position,
            push_contact_position,
            prepush_position,
            goal_position,
            _,
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
            "end_effector_pos": end_effector_position,
            "push_contact_pos": push_contact_position,
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
            "end_effector_to_push_distance": jnp.linalg.norm(
                end_effector_position - push_contact_position
            ),
            "approach_stop_distance": jnp.asarray(self._approach_stop_distance),
            "task_phase": info["phase"].astype(jnp.float32),
            "align_progress": jnp.clip(
                info["align_steps"] * self.dt / self._align_duration,
                0.0,
                1.0,
            ),
            "push_steps": info["push_steps"].astype(jnp.float32),
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
