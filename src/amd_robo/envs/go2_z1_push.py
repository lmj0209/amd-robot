"""Near-field Push-to-Goal task state for the qualified Go2+Z1 crawl."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from amd_robo.contracts import ACTION_LAYOUT, TaskPhase
from amd_robo.envs.go2_z1 import REPO_ROOT, _rotmat
from amd_robo.envs.go2_z1_locomotion import Go2Z1LocomotionEnv

DEFAULT_PUSH_XML = REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_push_mjx.xml"
PUSH_HOME_KEYFRAME = "push_home"
PUSH_BOX_BODY_NAME = "push_box_body"
PUSH_BOX_JOINT_NAME = "push_box_joint"
PUSH_BOX_GEOM_NAME = "push_box"
PUSH_PAD_GEOM_NAMES = (
    "push_pad_stator_left",
    "push_pad_stator_right",
    "push_pad_mover_left",
    "push_pad_mover_right",
)
END_EFFECTOR_SITE_NAME = "z1_ee"
PUSH_CONTACT_SITE_NAME = "push_contact_site"
PREPUSH_SITE_NAME = "prepush_site"
GOAL_SITE_NAME = "goal_site"
DEFAULT_APPROACH_STOP_DISTANCE = 0.2
DEFAULT_ALIGN_DURATION = 6.0
DEFAULT_ALIGN_DISTANCE_THRESHOLD = 0.08
DEFAULT_PUSH_COMMAND_X = 0.025
DEFAULT_GOAL_THRESHOLD = 0.08
DEFAULT_SUCCESS_HOLD_STEPS = 100
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
    _TASK_REWARD_NAMES = (
        "approach_progress",
        "align_progress",
        "push_progress",
        "hold",
        "success_bonus",
        "object_speed",
        "object_height",
    )

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_PUSH_XML,
        approach_stop_distance: float = DEFAULT_APPROACH_STOP_DISTANCE,
        align_duration: float = DEFAULT_ALIGN_DURATION,
        align_distance_threshold: float = DEFAULT_ALIGN_DISTANCE_THRESHOLD,
        push_command_x: float = DEFAULT_PUSH_COMMAND_X,
        push_command_ramp_duration: float = 0.0,
        object_speed_governor_start: float | None = None,
        object_speed_governor_stop: float | None = None,
        hold_entry_command_decay_duration: float = 0.0,
        align_gait_phase_sync: bool = False,
        align_gait_phase_fraction: float = 0.0,
        align_entry_gait_phase_fraction: float | None = None,
        goal_threshold: float = DEFAULT_GOAL_THRESHOLD,
        success_hold_steps: int = DEFAULT_SUCCESS_HOLD_STEPS,
        approach_progress_scale: float = 10.0,
        align_progress_scale: float = 5.0,
        push_progress_scale: float = 20.0,
        hold_scale: float = 1.0,
        success_bonus_scale: float = 10.0,
        object_speed_limit: float = 0.5,
        object_speed_cost_scale: float = 20.0,
        object_height_tolerance: float = 0.02,
        object_height_cost_scale: float = 5.0,
        object_position_x_offset_range: Sequence[float] = (0.0, 0.0),
        object_position_y_offset_range: Sequence[float] = (0.0, 0.0),
        push_box_solref_timeconst: float | None = None,
        push_pad_solref_timeconst: float | None = None,
        push_arm_residual_enabled: bool = False,
        push_arm_residual_mode: str = "joint",
        push_arm_ee_x_range: float = 0.005,
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
        if push_command_ramp_duration < 0.0:
            raise ValueError("push command ramp duration must be non-negative")
        if (object_speed_governor_start is None) != (
            object_speed_governor_stop is None
        ):
            raise ValueError("object speed governor thresholds must be set together")
        if object_speed_governor_start is not None:
            assert object_speed_governor_stop is not None
            if object_speed_governor_start < 0.0:
                raise ValueError("object speed governor start must be non-negative")
            if object_speed_governor_stop <= object_speed_governor_start:
                raise ValueError("object speed governor stop must exceed start")
            if object_speed_governor_stop > object_speed_limit:
                raise ValueError(
                    "object speed governor stop must not exceed the speed limit"
                )
        if hold_entry_command_decay_duration < 0.0:
            raise ValueError(
                "hold entry command decay duration must be non-negative"
            )
        if not 0.0 <= align_gait_phase_fraction < 1.0:
            raise ValueError("align gait phase fraction must be in [0, 1)")
        if align_entry_gait_phase_fraction is not None and not (
            0.0 <= align_entry_gait_phase_fraction < 1.0
        ):
            raise ValueError("align entry gait phase fraction must be in [0, 1)")
        if align_gait_phase_sync and align_entry_gait_phase_fraction is not None:
            raise ValueError("continuous and entry-only gait phase sync conflict")
        if goal_threshold <= 0.0:
            raise ValueError("goal threshold must be positive")
        if success_hold_steps <= 0:
            raise ValueError("success hold steps must be positive")
        if (
            min(
                approach_progress_scale,
                align_progress_scale,
                push_progress_scale,
                hold_scale,
                success_bonus_scale,
            )
            < 0.0
        ):
            raise ValueError("task reward scales must be non-negative")
        if object_speed_limit <= 0.0:
            raise ValueError("object speed limit must be positive")
        if object_height_tolerance <= 0.0:
            raise ValueError("object height tolerance must be positive")
        if object_speed_cost_scale < 0.0 or object_height_cost_scale < 0.0:
            raise ValueError("task cost scales must be non-negative")
        if push_box_solref_timeconst is not None and push_box_solref_timeconst <= 0.0:
            raise ValueError("push box solref time constant must be positive")
        if push_pad_solref_timeconst is not None and push_pad_solref_timeconst <= 0.0:
            raise ValueError("push pad solref time constant must be positive")
        if push_arm_residual_mode not in ("joint", "ee_x"):
            raise ValueError("push arm residual mode must be joint or ee_x")
        if push_arm_ee_x_range <= 0.0:
            raise ValueError("push arm EE x range must be positive")
        if push_arm_residual_mode != "joint" and not push_arm_residual_enabled:
            raise ValueError("push arm residual mode requires arm residuals")
        for axis, offset_range in (
            ("x", object_position_x_offset_range),
            ("y", object_position_y_offset_range),
        ):
            if len(offset_range) != 2:
                raise ValueError(
                    f"object position {axis} offset range must contain [min, max]"
                )
            if offset_range[0] > offset_range[1]:
                raise ValueError(
                    f"object position {axis} offset range must be ordered"
                )
        self._approach_stop_distance = float(approach_stop_distance)
        self._align_duration = float(align_duration)
        self._align_distance_threshold = float(align_distance_threshold)
        self._push_command = jnp.asarray(
            [push_command_x, 0.0, 0.0],
            dtype=jnp.float32,
        )
        self._push_command_ramp_duration = float(push_command_ramp_duration)
        self._object_speed_governor_start = (
            None
            if object_speed_governor_start is None
            else float(object_speed_governor_start)
        )
        self._object_speed_governor_stop = (
            None
            if object_speed_governor_stop is None
            else float(object_speed_governor_stop)
        )
        self._hold_entry_command_decay_duration = float(
            hold_entry_command_decay_duration
        )
        self._align_gait_phase_sync = bool(align_gait_phase_sync)
        self._align_gait_phase = float(2.0 * jnp.pi * align_gait_phase_fraction)
        self._align_entry_gait_phase = (
            None
            if align_entry_gait_phase_fraction is None
            else float(2.0 * jnp.pi * align_entry_gait_phase_fraction)
        )
        self._goal_threshold = float(goal_threshold)
        self._success_hold_steps = int(success_hold_steps)
        self._object_speed_limit = float(object_speed_limit)
        self._object_height_tolerance = float(object_height_tolerance)
        self._object_position_x_offset_range = tuple(
            float(value) for value in object_position_x_offset_range
        )
        self._object_position_y_offset_range = tuple(
            float(value) for value in object_position_y_offset_range
        )
        self._randomize_object_position = any(
            value != 0.0
            for value in (
                *self._object_position_x_offset_range,
                *self._object_position_y_offset_range,
            )
        )
        self._push_arm_residual_enabled = bool(push_arm_residual_enabled)
        self._push_arm_residual_mode = push_arm_residual_mode
        self._push_arm_ee_x_range = float(push_arm_ee_x_range)
        self._task_reward_scales = {
            "approach_progress": float(approach_progress_scale),
            "align_progress": float(align_progress_scale),
            "push_progress": float(push_progress_scale),
            "hold": float(hold_scale),
            "success_bonus": float(success_bonus_scale),
            "object_speed": -float(object_speed_cost_scale),
            "object_height": -float(object_height_cost_scale),
        }
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
        if self._push_arm_residual_enabled and self._mask_arm:
            raise ValueError("push arm residual requires mask_arm=False")

        self._box_body_id = self._required_id(
            mujoco.mjtObj.mjOBJ_BODY, PUSH_BOX_BODY_NAME
        )
        self._box_joint_id = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, PUSH_BOX_JOINT_NAME
        )
        self._box_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, PUSH_BOX_GEOM_NAME
        )
        self._push_pad_geom_ids = tuple(
            self._required_id(mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in PUSH_PAD_GEOM_NAMES
        )
        contact_model_changed = False
        if push_box_solref_timeconst is not None:
            self.mj_model.geom_solref[self._box_geom_id, 0] = float(
                push_box_solref_timeconst
            )
            contact_model_changed = True
        if push_pad_solref_timeconst is not None:
            for geom_id in self._push_pad_geom_ids:
                self.mj_model.geom_solref[geom_id, 0] = float(
                    push_pad_solref_timeconst
                )
            contact_model_changed = True
        if contact_model_changed:
            self._mjx_model = mjx.put_model(self.mj_model, impl="jax")
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
        self._push_arm_linear_jacobian = None
        self._push_arm_joint_delta = None
        self._push_arm_policy_direction = None
        if self._push_arm_residual_mode == "ee_x":
            self._initialize_push_arm_ee_x_projection()
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
        if self._randomize_object_position:
            object_x_key, object_y_key = jax.random.split(state.info["rng"])
            object_offset = jnp.asarray(
                [
                    jax.random.uniform(
                        object_x_key,
                        (),
                        minval=self._object_position_x_offset_range[0],
                        maxval=self._object_position_x_offset_range[1],
                    ),
                    jax.random.uniform(
                        object_y_key,
                        (),
                        minval=self._object_position_y_offset_range[0],
                        maxval=self._object_position_y_offset_range[1],
                    ),
                ]
            )
            object_qpos = state.data.qpos[self._box_qpos_slice]
            object_qpos = object_qpos.at[:2].set(
                self._initial_box_qpos[:2] + object_offset
            )
            data = state.data.replace(
                qpos=state.data.qpos.at[self._box_qpos_slice].set(object_qpos)
            )
            state = state.replace(data=mjx.forward(self.mjx_model, data))
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
        in_goal = state.metrics["object_to_goal_distance"] <= self._goal_threshold
        phase = jnp.where(
            (phase == int(TaskPhase.PUSH)) & in_goal,
            int(TaskPhase.HOLD),
            phase,
        )
        phase = jnp.where(
            (phase == int(TaskPhase.HOLD)) & ~in_goal,
            int(TaskPhase.PUSH),
            phase,
        )
        command = jnp.where(
            phase == int(TaskPhase.APPROACH),
            state.info["command"],
            jnp.where(
                phase == int(TaskPhase.PUSH),
                self._push_command_for_state(state),
                jnp.where(
                    phase == int(TaskPhase.HOLD),
                    self._hold_entry_command_for_state(state),
                    jnp.zeros_like(state.info["command"]),
                ),
            ),
        )
        align_step_limit = round(self._align_duration / self.dt)
        align_steps = jnp.where(
            phase >= int(TaskPhase.ALIGN),
            jnp.minimum(state.info["align_steps"] + 1, align_step_limit),
            0,
        )
        push_steps = jnp.where(
            phase >= int(TaskPhase.PUSH),
            state.info["push_steps"] + (phase == int(TaskPhase.PUSH)).astype(jnp.int32),
            0,
        )
        success_count = jnp.where(
            phase == int(TaskPhase.HOLD),
            state.info["success_count"] + 1,
            0,
        )
        staged_info = {
            **state.info,
            "phase": phase,
            "command": command,
            "align_steps": align_steps,
            "push_steps": push_steps,
            "success_count": success_count,
        }
        if self._align_entry_gait_phase is not None and "gait_phase" in state.info:
            staged_info["gait_phase"] = jnp.where(
                enter_align,
                jnp.full_like(
                    state.info["gait_phase"],
                    self._align_entry_gait_phase,
                ),
                state.info["gait_phase"],
            )
        if self._align_gait_phase_sync and "gait_phase" in state.info:
            staged_info["gait_phase"] = jnp.where(
                phase == int(TaskPhase.ALIGN),
                jnp.full_like(state.info["gait_phase"], self._align_gait_phase),
                state.info["gait_phase"],
            )
        staged = state.replace(info=staged_info)
        stepped = self._with_task_state(super().step(staged, action))
        raw_task_rewards = self._task_reward_components(state, stepped)
        scaled_task_rewards = {
            name: jnp.nan_to_num(
                raw_task_rewards[name] * self._task_reward_scales[name],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for name in self._TASK_REWARD_NAMES
        }
        task_reward = jnp.nan_to_num(
            sum(scaled_task_rewards.values()) * self.dt,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        metrics = {
            **stepped.metrics,
            **{
                f"reward/task_{name}": value * self.dt
                for name, value in scaled_task_rewards.items()
            },
            "task_reward": task_reward,
        }
        return stepped.replace(
            reward=jnp.clip(stepped.reward + task_reward, 0.0, 10000.0),
            done=jnp.maximum(stepped.done, stepped.metrics["success"]),
            metrics=metrics,
        )

    def _push_command_for_state(self, state) -> jax.Array:
        ramp_scale = jnp.asarray(1.0, dtype=self._push_command.dtype)
        if self._push_command_ramp_duration > 0.0:
            progress = jnp.clip(
                state.info["push_steps"]
                * self.dt
                / self._push_command_ramp_duration,
                0.0,
                1.0,
            )
            ramp_scale = progress * progress * (3.0 - 2.0 * progress)
        return self._push_command * ramp_scale * self._object_speed_governor_scale(
            state
        )

    def _object_speed_governor_scale(self, state) -> jax.Array:
        if self._object_speed_governor_start is None:
            return jnp.asarray(1.0, dtype=self._push_command.dtype)
        assert self._object_speed_governor_stop is not None
        object_speed = jnp.linalg.norm(state.info["object_qvel"][:2])
        progress = jnp.clip(
            (object_speed - self._object_speed_governor_start)
            / (
                self._object_speed_governor_stop
                - self._object_speed_governor_start
            ),
            0.0,
            1.0,
        )
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return 1.0 - smooth_progress

    def _hold_entry_command_for_state(self, state) -> jax.Array:
        if self._hold_entry_command_decay_duration == 0.0:
            return jnp.zeros_like(self._push_command)
        progress = jnp.clip(
            state.info["success_count"]
            * self.dt
            / self._hold_entry_command_decay_duration,
            0.0,
            1.0,
        )
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return self._push_command * (1.0 - smooth_progress)

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

    def _task_policy_action_mask(self, state) -> jax.Array:
        """Expose policy residuals only while physically pushing the box."""
        pushing = state.info["phase"] == int(TaskPhase.PUSH)
        if self._push_arm_residual_enabled:
            residual_mask = jnp.ones(self.action_size, dtype=jnp.float32).at[
                ACTION_LAYOUT.gripper
            ].set(0.0)
        else:
            residual_mask = jnp.ones(self.action_size, dtype=jnp.float32)
        return residual_mask * pushing.astype(jnp.float32)

    def _initialize_push_arm_ee_x_projection(self) -> None:
        """Build a local arm projection for pure world-x EE displacement."""
        data = mujoco.MjData(self.mj_model)
        data.qpos[:] = self.mj_model.key_qpos[self._home_keyframe_id]
        arm_qpos_indices = np.asarray(
            self._joint_qpos_indices[ACTION_LAYOUT.arm],
            dtype=np.int32,
        )
        arm_dof_indices = np.asarray(
            self._joint_dof_indices[ACTION_LAYOUT.arm],
            dtype=np.int32,
        )
        data.qpos[arm_qpos_indices] = np.asarray(self._align_arm_joint_target)
        mujoco.mj_forward(self.mj_model, data)

        jacobian = np.zeros((3, self.mj_model.nv))
        angular_jacobian = np.zeros((3, self.mj_model.nv))
        mujoco.mj_jacSite(
            self.mj_model,
            data,
            jacobian,
            angular_jacobian,
            self._end_effector_site_id,
        )
        arm_jacobian = jacobian[:, arm_dof_indices]
        target_displacement = np.asarray(
            [self._push_arm_ee_x_range, 0.0, 0.0]
        )
        joint_delta = np.linalg.pinv(arm_jacobian, rcond=1e-6) @ target_displacement
        predicted_displacement = arm_jacobian @ joint_delta
        if not np.allclose(
            predicted_displacement,
            target_displacement,
            atol=1e-8,
            rtol=1e-6,
        ):
            raise ValueError("push arm EE x projection is ill-conditioned")
        direction_norm = np.linalg.norm(joint_delta)
        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError("push arm EE x projection is non-finite")
        self._push_arm_linear_jacobian = jnp.asarray(arm_jacobian)
        self._push_arm_joint_delta = jnp.asarray(joint_delta)
        self._push_arm_policy_direction = jnp.asarray(
            joint_delta / direction_norm
        )

    def _policy_ctrl_residual(self, action: jax.Array) -> jax.Array:
        residual = super()._policy_ctrl_residual(action)
        if (
            self._push_arm_residual_enabled
            and self._push_arm_residual_mode == "ee_x"
        ):
            arm_action = action[ACTION_LAYOUT.arm]
            scalar = jnp.clip(
                jnp.dot(arm_action, self._push_arm_policy_direction),
                -1.0,
                1.0,
            )
            residual = residual.at[ACTION_LAYOUT.arm].set(
                scalar * self._push_arm_joint_delta
            )
            residual = residual.at[ACTION_LAYOUT.gripper].set(0.0)
        return residual

    def _task_reward_components(self, previous, current):
        phase = current.info["phase"]
        approach = phase == int(TaskPhase.APPROACH)
        align = phase == int(TaskPhase.ALIGN)
        push = phase == int(TaskPhase.PUSH)
        hold = phase == int(TaskPhase.HOLD)
        approach_progress = jnp.clip(
            (
                previous.metrics["base_to_prepush_distance"]
                - current.metrics["base_to_prepush_distance"]
            )
            / self.dt,
            -0.5,
            0.5,
        )
        align_progress = jnp.clip(
            (
                previous.metrics["end_effector_to_push_distance"]
                - current.metrics["end_effector_to_push_distance"]
            )
            / self.dt,
            -0.5,
            0.5,
        )
        push_progress = jnp.clip(
            (
                previous.metrics["object_to_goal_distance"]
                - current.metrics["object_to_goal_distance"]
            )
            / self.dt,
            -0.5,
            0.5,
        )
        object_speed = jnp.linalg.norm(current.info["object_qvel"][:2])
        speed_excess = (
            jnp.maximum(object_speed - self._object_speed_limit, 0.0)
            / self._object_speed_limit
        )
        object_height_error = (
            jnp.maximum(
                jnp.abs(current.metrics["object_height"] - 0.1)
                - self._object_height_tolerance,
                0.0,
            )
            / self._object_height_tolerance
        )
        first_success = (current.metrics["success"] > 0.0) & (
            previous.metrics["success"] <= 0.0
        )
        manipulating = push | hold
        return {
            "approach_progress": approach_progress * approach,
            "align_progress": align_progress * align,
            "push_progress": push_progress * push,
            "hold": (
                hold
                & (current.metrics["object_to_goal_distance"] <= self._goal_threshold)
            ).astype(jnp.float32),
            "success_bonus": first_success.astype(jnp.float32),
            "object_speed": jnp.square(speed_excess) * manipulating,
            "object_height": jnp.square(object_height_error) * manipulating,
        }

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
        if command is not None:
            command = self._policy_command_for_observation(phase, command)
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

    def _policy_command_for_observation(
        self,
        phase: jax.Array,
        applied_command: jax.Array,
    ) -> jax.Array:
        """Keep the requested Push setpoint separate from its safety filter."""
        if self._object_speed_governor_start is None:
            return applied_command
        return jnp.where(
            phase == int(TaskPhase.PUSH),
            self._push_command,
            applied_command,
        )

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
        if self._randomize_object_position:
            object_x_key, object_y_key = jax.random.split(state.info["rng"])
            initial_object_xy = self._initial_box_qpos[:2] + jnp.asarray(
                [
                    jax.random.uniform(
                        object_x_key,
                        (),
                        minval=self._object_position_x_offset_range[0],
                        maxval=self._object_position_x_offset_range[1],
                    ),
                    jax.random.uniform(
                        object_y_key,
                        (),
                        minval=self._object_position_y_offset_range[0],
                        maxval=self._object_position_y_offset_range[1],
                    ),
                ]
            )
        else:
            initial_object_xy = self._initial_box_qpos[:2]
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
                object_qpos[:2] - initial_object_xy
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
            "success_hold_count": info["success_count"].astype(jnp.float32),
            "success": (
                (info["phase"] == int(TaskPhase.HOLD))
                & (info["success_count"] >= self._success_hold_steps)
            ).astype(jnp.float32),
            "goal_threshold": jnp.asarray(self._goal_threshold),
        }
        metrics.update(
            {f"reward/task_{name}": jnp.zeros(()) for name in self._TASK_REWARD_NAMES}
        )
        metrics["task_reward"] = jnp.zeros(())
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
