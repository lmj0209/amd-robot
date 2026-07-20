"""Command-conditioned flat-ground locomotion for the 19-DoF Go2+Z1 robot.

This is the first Phase-2 curriculum slice. The policy still emits the frozen
19-dimensional action, while the arm and gripper residuals are masked so their
home PD controllers keep the carry pose. Training commands are deliberately
limited to zero or low forward velocity before lateral/yaw commands and the
push task are introduced.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from amd_robo.contracts import ACTION_LAYOUT
from amd_robo.envs.go2_kinematics import go2_foot_space_crawl_reference
from amd_robo.envs.go2_z1 import _LEG_MASK, FOOT_GEOM_NAMES, Go2Z1Env, _rotmat

FOOT_SITE_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
_CRAWL_SEQUENCE = jnp.asarray([0, 3, 1, 2], dtype=jnp.int32)
_CRAWL_PHASE_OFFSETS = jnp.asarray([0.0, 0.5, 0.75, 0.25])
_CRAWL_FORE_AFT_SIGNS = jnp.asarray([1.0, 1.0, -1.0, -1.0])
_CRAWL_LEFT_RIGHT_SIGNS = jnp.asarray([1.0, -1.0, 1.0, -1.0])


class Go2Z1LocomotionEnv(Go2Z1Env):
    """Track a low-speed planar command while keeping the Z1 arm tucked."""

    _BASE_REWARD_NAMES = (
        "tracking_linear_velocity",
        "tracking_angular_velocity",
        "pose",
        "vertical_velocity",
        "angular_velocity_xy",
        "orientation",
        "stand_still",
        "torques",
        "action_rate",
        "action_magnitude",
        "arm_action_magnitude",
        "feet_slip",
        "feet_clearance",
        "feet_height",
        "feet_air_time",
        "termination",
        "illegal_contact",
    )

    def __init__(
        self,
        *args,
        command_x_range: Sequence[float] = (0.2, 0.6),
        zero_command_probability: float = 0.2,
        command_override: Sequence[float] | None = None,
        randomize_reset: bool = True,
        tracking_sigma: float = 0.25,
        tracking_linear_velocity_scale: float = 1.0,
        tracking_angular_velocity_scale: float = 0.5,
        pose_scale: float = 0.5,
        moving_pose_multiplier: float = 1.0,
        vertical_velocity_cost_scale: float = 0.5,
        angular_velocity_xy_cost_scale: float = 0.05,
        orientation_cost_scale: float = 2.0,
        stand_still_cost_scale: float = 0.5,
        torque_cost_scale: float = 0.0001,
        action_rate_cost_scale: float = 0.01,
        action_magnitude_cost_scale: float = 0.001,
        arm_action_magnitude_cost_scale: float = 0.01,
        feet_slip_cost_scale: float = 0.1,
        feet_clearance_cost_scale: float = 2.0,
        feet_height_cost_scale: float = 0.2,
        feet_air_time_scale: float = 0.1,
        max_foot_height: float = 0.1,
        gait_cycle_time: float | None = None,
        trot_contact_scale: float = 0.0,
        trot_swing_height_cost_scale: float = 0.0,
        trot_timing_scale: float = 0.0,
        trot_timing_std: float = 0.1,
        trot_timing_max_error: float = 0.2,
        trot_timing_min_air_time: float = 0.0,
        crawl_reference_enabled: bool = False,
        crawl_stride: float = 0.08,
        crawl_shift: float = 0.06,
        crawl_lift: float = 0.45,
        crawl_shift_end_fraction: float = 0.3,
        crawl_lift_start_fraction: float = 0.3,
        crawl_lift_end_fraction: float = 0.8,
        crawl_foot_space_enabled: bool = False,
        crawl_foot_step_length: float = 0.08,
        crawl_foot_clearance: float = 0.04,
        crawl_body_shift_x: float = 0.02,
        crawl_body_shift_y: float = 0.02,
        crawl_min_air_time: float = 0.07,
        crawl_pose_reference_enabled: bool = False,
        termination_cost_scale: float = 2.0,
        illegal_contact_cost_scale: float = 2.0,
        workspace_limit: float = 5.0,
        **kwargs,
    ) -> None:
        if len(command_x_range) != 2 or command_x_range[0] > command_x_range[1]:
            raise ValueError("command_x_range must contain ordered [min, max] values")
        if not 0.0 <= zero_command_probability <= 1.0:
            raise ValueError("zero_command_probability must be in [0, 1]")
        if command_override is not None and len(command_override) != 3:
            raise ValueError("command_override must contain [vx, vy, yaw_rate]")
        if tracking_sigma <= 0.0 or max_foot_height <= 0.0:
            raise ValueError("tracking_sigma and max_foot_height must be positive")
        if not 0.0 <= moving_pose_multiplier <= 1.0:
            raise ValueError("moving_pose_multiplier must be in [0, 1]")
        if gait_cycle_time is not None and gait_cycle_time <= 0.0:
            raise ValueError("gait_cycle_time must be positive when enabled")
        if (
            trot_contact_scale < 0.0
            or trot_swing_height_cost_scale < 0.0
            or trot_timing_scale < 0.0
        ):
            raise ValueError("trot reward scales must be non-negative")
        if trot_timing_std <= 0.0 or trot_timing_max_error <= 0.0:
            raise ValueError("trot timing std and max error must be positive")
        if trot_timing_min_air_time < 0.0:
            raise ValueError("trot timing minimum air time must be non-negative")
        if (
            crawl_stride < 0.0
            or crawl_shift <= 0.0
            or crawl_lift <= 0.0
            or crawl_min_air_time <= 0.0
        ):
            raise ValueError(
                "crawl stride must be non-negative; crawl shift and lift "
                "and minimum air time must be positive"
            )
        if not (
            0.0 < crawl_shift_end_fraction
            <= crawl_lift_start_fraction
            < crawl_lift_end_fraction
            < 1.0
        ):
            raise ValueError(
                "crawl timing must satisfy 0 < shift end <= lift start "
                "< lift end < 1"
            )
        if (
            crawl_foot_step_length <= 0.0
            or crawl_foot_clearance <= 0.0
            or crawl_body_shift_x < 0.0
            or crawl_body_shift_y < 0.0
        ):
            raise ValueError(
                "crawl foot step and clearance must be positive; body shifts "
                "must be non-negative"
            )
        if gait_cycle_time is None and (
            trot_contact_scale > 0.0
            or trot_swing_height_cost_scale > 0.0
            or trot_timing_scale > 0.0
        ):
            raise ValueError("trot reward scales require gait_cycle_time")
        if gait_cycle_time is None and crawl_reference_enabled:
            raise ValueError("crawl reference requires gait_cycle_time")
        if crawl_foot_space_enabled and not crawl_reference_enabled:
            raise ValueError("foot-space crawl requires crawl reference")
        if crawl_pose_reference_enabled and not crawl_reference_enabled:
            raise ValueError("crawl pose reference requires crawl reference")
        if crawl_reference_enabled and (
            trot_contact_scale > 0.0
            or trot_swing_height_cost_scale > 0.0
            or trot_timing_scale > 0.0
        ):
            raise ValueError("crawl reference cannot be combined with trot rewards")
        if workspace_limit <= 0.0:
            raise ValueError("workspace_limit must be positive")

        self._command_x_range = tuple(float(v) for v in command_x_range)
        self._zero_command_probability = float(zero_command_probability)
        self._command_override = (
            None
            if command_override is None
            else jnp.asarray(command_override, dtype=jnp.float32)
        )
        self._randomize_reset = bool(randomize_reset)
        self._tracking_sigma = float(tracking_sigma)
        self._moving_pose_multiplier = float(moving_pose_multiplier)
        self._gait_cycle_time = (
            None if gait_cycle_time is None else float(gait_cycle_time)
        )
        self._trot_timing_enabled = trot_timing_scale > 0.0
        self._trot_timing_std = float(trot_timing_std)
        self._trot_timing_max_error = float(trot_timing_max_error)
        self._trot_timing_min_air_time = float(trot_timing_min_air_time)
        self._crawl_reference_enabled = bool(crawl_reference_enabled)
        self._crawl_stride = float(crawl_stride)
        self._crawl_shift = float(crawl_shift)
        self._crawl_lift = float(crawl_lift)
        self._crawl_shift_end_fraction = float(crawl_shift_end_fraction)
        self._crawl_lift_start_fraction = float(crawl_lift_start_fraction)
        self._crawl_lift_end_fraction = float(crawl_lift_end_fraction)
        self._crawl_foot_space_enabled = bool(crawl_foot_space_enabled)
        self._crawl_foot_step_length = float(crawl_foot_step_length)
        self._crawl_foot_clearance = float(crawl_foot_clearance)
        self._crawl_body_shift_x = float(crawl_body_shift_x)
        self._crawl_body_shift_y = float(crawl_body_shift_y)
        self._crawl_min_air_time = float(crawl_min_air_time)
        self._crawl_pose_reference_enabled = bool(crawl_pose_reference_enabled)
        self._reward_names = self._BASE_REWARD_NAMES
        if self._gait_cycle_time is not None:
            self._reward_names += ("trot_contact", "trot_swing_height")
        if self._trot_timing_enabled:
            self._reward_names += ("trot_timing",)
        self._reward_scales = {
            "tracking_linear_velocity": float(tracking_linear_velocity_scale),
            "tracking_angular_velocity": float(tracking_angular_velocity_scale),
            "pose": float(pose_scale),
            "vertical_velocity": -float(vertical_velocity_cost_scale),
            "angular_velocity_xy": -float(angular_velocity_xy_cost_scale),
            "orientation": -float(orientation_cost_scale),
            "stand_still": -float(stand_still_cost_scale),
            "torques": -float(torque_cost_scale),
            "action_rate": -float(action_rate_cost_scale),
            "action_magnitude": -float(action_magnitude_cost_scale),
            "arm_action_magnitude": -float(arm_action_magnitude_cost_scale),
            "feet_slip": -float(feet_slip_cost_scale),
            "feet_clearance": -float(feet_clearance_cost_scale),
            "feet_height": -float(feet_height_cost_scale),
            "feet_air_time": float(feet_air_time_scale),
            "termination": -float(termination_cost_scale),
            "illegal_contact": -float(illegal_contact_cost_scale),
        }
        if self._gait_cycle_time is not None:
            self._reward_scales.update(
                {
                    "trot_contact": float(trot_contact_scale),
                    "trot_swing_height": -float(trot_swing_height_cost_scale),
                }
            )
        if self._trot_timing_enabled:
            self._reward_scales["trot_timing"] = float(trot_timing_scale)
        self._max_foot_height = float(max_foot_height)
        self._workspace_limit = float(workspace_limit)

        kwargs.setdefault("ctrl_dt", 0.01)
        kwargs.setdefault("foot_condim", 6)
        kwargs.setdefault("mask_arm", True)
        super().__init__(*args, **kwargs)

        self._floor_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        if self._floor_geom_id < 0:
            raise ValueError("locomotion scene is missing the floor geom")
        self._foot_geom_ids = jnp.asarray(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                for geom_name in FOOT_GEOM_NAMES
            ],
            dtype=jnp.int32,
        )
        self._allowed_floor_geom_ids = self._foot_geom_ids
        self._foot_site_ids = jnp.asarray(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                for site_name in FOOT_SITE_NAMES
            ],
            dtype=jnp.int32,
        )
        if bool(jnp.any(self._foot_site_ids < 0)):
            raise ValueError("locomotion model is missing one or more foot sites")
        self._ctrl_min = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_max = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 1])
        self._global_linvel_slice = self._sensor_slice("global_linvel")
        self._gyro_slice = self._sensor_slice("gyro")
        self._foot_linvel_sensor_indices = jnp.asarray(
            [
                list(
                    range(
                        self._sensor_slice(f"{site_name}_linvel").start,
                        self._sensor_slice(f"{site_name}_linvel").stop,
                    )
                )
                for site_name in FOOT_SITE_NAMES
            ],
            dtype=jnp.int32,
        )

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0:
            raise ValueError(f"required sensor not found: {name}")
        start = int(self.mj_model.sensor_adr[sensor_id])
        return slice(start, start + int(self.mj_model.sensor_dim[sensor_id]))

    @property
    def observation_size(self) -> int:
        return 73 + (2 if self._gait_cycle_time is not None else 0)

    def reset(self, rng: jax.Array):
        if self._gait_cycle_time is None:
            rng, command_key, zero_key, joint_key, velocity_key = jax.random.split(
                rng, 5
            )
            gait_phase = None
        else:
            (
                rng,
                command_key,
                zero_key,
                joint_key,
                velocity_key,
                gait_key,
            ) = jax.random.split(rng, 6)
            gait_phase = jnp.where(
                self._randomize_reset,
                jax.random.uniform(gait_key, (), minval=0.0, maxval=2.0 * jnp.pi),
                0.0,
            )
        state = super().reset(rng)
        command = self._sample_command(command_key, zero_key)

        data = state.data
        if self._randomize_reset:
            leg_mask = jnp.concatenate([jnp.ones(12), jnp.zeros(7)])
            joint_delta = jax.random.uniform(
                joint_key,
                (ACTION_LAYOUT.size,),
                minval=-0.03,
                maxval=0.03,
            )
            joint_velocity = jax.random.uniform(
                velocity_key,
                (ACTION_LAYOUT.size,),
                minval=-0.05,
                maxval=0.05,
            )
            data = data.replace(
                qpos=data.qpos.at[self._joint_qpos_indices].add(
                    joint_delta * leg_mask
                ),
                qvel=data.qvel.at[self._joint_dof_indices].set(
                    joint_velocity * leg_mask
                ),
            )
            data = mjx.forward(self.mjx_model, data)

        crawl_reference = None
        crawl_foot_targets = None
        crawl_ik_reachable = None
        if self._crawl_reference_enabled:
            crawl_reference = self._crawl_joint_reference(gait_phase, command)
            if self._crawl_foot_space_enabled:
                _, crawl_foot_targets, crawl_ik_reachable = (
                    self._crawl_foot_space_solution(gait_phase)
                )
            data = data.replace(
                qpos=data.qpos.at[self._joint_qpos_indices[:12]].add(
                    crawl_reference
                ),
                ctrl=data.ctrl.at[:12].add(crawl_reference),
            )
            data = mjx.forward(self.mjx_model, data)

        info = {
            **state.info,
            "rng": rng,
            "command": command,
            "last_last_action": jnp.zeros(ACTION_LAYOUT.size),
            "feet_air_time": jnp.zeros(len(FOOT_SITE_NAMES)),
            "last_contact": self._foot_floor_contacts(data),
            "swing_peak": jnp.zeros(len(FOOT_SITE_NAMES)),
        }
        if gait_phase is not None:
            info["gait_phase"] = gait_phase
        if crawl_reference is not None:
            active_leg, swing_window = self._crawl_schedule(
                gait_phase,
                self._crawl_lift_start_fraction,
                self._crawl_lift_end_fraction,
            )
            info.update(
                {
                    "crawl_reference": crawl_reference,
                    "crawl_active_leg": active_leg,
                    "crawl_swing_window": swing_window,
                    "crawl_sustained_touchdown": jnp.zeros(
                        len(FOOT_SITE_NAMES), dtype=bool
                    ),
                }
            )
            if crawl_foot_targets is not None:
                info["crawl_foot_targets"] = crawl_foot_targets
                info["crawl_ik_reachable"] = crawl_ik_reachable
        if self._trot_timing_enabled:
            info["feet_contact_time"] = jnp.zeros(len(FOOT_SITE_NAMES))
        metrics = {
            **state.metrics,
            "command_x": command[0],
            "base_forward_velocity": jnp.zeros(()),
            "tracking_linear_error": jnp.abs(command[0]),
            "illegal_contact": jnp.zeros(()),
            "workspace_bounds": jnp.zeros(()),
            "nonfinite_state": jnp.zeros(()),
            "swing_peak": jnp.zeros(()),
        }
        metrics.update(
            {f"reward/{name}": jnp.zeros(()) for name in self._reward_names}
        )
        if crawl_reference is not None:
            metrics["crawl_reference_rms"] = jnp.sqrt(
                jnp.mean(jnp.square(crawl_reference))
            )
            if crawl_ik_reachable is not None:
                metrics["crawl_ik_reachable_fraction"] = jnp.mean(
                    crawl_ik_reachable.astype(jnp.float32)
                )
            if self._crawl_pose_reference_enabled:
                reference_error = (
                    data.qpos[self._joint_qpos_indices[:12]]
                    - self._home_qpos[self._joint_qpos_indices[:12]]
                    - crawl_reference
                )
                metrics["crawl_reference_error_rms"] = jnp.sqrt(
                    jnp.mean(jnp.square(reference_error))
                )
        return state.replace(
            data=data,
            obs=self._observation(
                data,
                state.info["last_action"],
                state.info["phase"],
                command,
                gait_phase,
            ),
            metrics=metrics,
            info=info,
        )

    def _sample_command(self, command_key: jax.Array, zero_key: jax.Array) -> jax.Array:
        if self._command_override is not None:
            return self._command_override
        forward = jax.random.uniform(
            command_key,
            (),
            minval=self._command_x_range[0],
            maxval=self._command_x_range[1],
        )
        use_zero = jax.random.bernoulli(zero_key, self._zero_command_probability)
        return jnp.asarray([jnp.where(use_zero, 0.0, forward), 0.0, 0.0])

    def step(self, state, action: jax.Array):
        policy_action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
        previous_action = state.info["last_action"]
        crawl_reference = None
        crawl_foot_targets = None
        crawl_ik_reachable = None
        if self._crawl_reference_enabled:
            applied_action = policy_action * self._task_policy_action_mask(state)
            if self._mask_arm:
                applied_action = applied_action * _LEG_MASK
            crawl_reference = self._crawl_joint_reference(
                state.info["gait_phase"],
                state.info["command"],
            )
            if self._crawl_foot_space_enabled:
                _, crawl_foot_targets, crawl_ik_reachable = (
                    self._crawl_foot_space_solution(state.info["gait_phase"])
                )
            ctrl = (
                self._home_ctrl
                + self._action_scale_vector * applied_action
                + self._task_actuator_reference(state)
            )
            ctrl = ctrl.at[:12].add(crawl_reference)
            ctrl = jnp.clip(ctrl, self._ctrl_min, self._ctrl_max)
            stepped = self._step_with_ctrl(
                state,
                applied_action,
                ctrl,
            )
        else:
            stepped = super().step(state, policy_action)

        foot_contact = self._foot_floor_contacts(stepped.data)
        contact_filt = foot_contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) & contact_filt
        crawl_sustained_touchdown = None
        if self._crawl_reference_enabled:
            crawl_sustained_touchdown = (
                state.info["feet_air_time"] + self.dt
                >= self._crawl_min_air_time
            ) & contact_filt
            first_contact = crawl_sustained_touchdown
        feet_air_time = state.info["feet_air_time"] + self.dt
        current_air_time = feet_air_time * ~foot_contact
        current_contact_time = None
        if self._trot_timing_enabled:
            current_contact_time = (
                state.info["feet_contact_time"] + self.dt
            ) * foot_contact
        foot_height = stepped.data.site_xpos[self._foot_site_ids, -1]
        swing_peak = jnp.maximum(state.info["swing_peak"], foot_height)
        illegal_contact = self._has_illegal_floor_contact(stepped.data)
        nonfinite_state = ~(
            jnp.all(jnp.isfinite(stepped.data.qpos))
            & jnp.all(jnp.isfinite(stepped.data.qvel))
            & jnp.all(jnp.isfinite(stepped.data.qacc))
            & jnp.all(jnp.isfinite(stepped.data.sensordata))
            & jnp.all(jnp.isfinite(stepped.data.actuator_force))
        )
        workspace_bounds = jnp.any(
            jnp.abs(stepped.data.qpos[:2]) > self._workspace_limit
        )
        done = jnp.maximum(
            stepped.done,
            (illegal_contact | workspace_bounds | nonfinite_state).astype(jnp.float32),
        )
        reward_components = self._reward_components(
            stepped.data,
            stepped.info["command"],
            policy_action,
            previous_action,
            done,
            illegal_contact,
            foot_contact,
            first_contact,
            feet_air_time,
            swing_peak,
            state.info.get("gait_phase"),
            current_air_time,
            current_contact_time,
            crawl_reference,
        )
        scaled_components = {
            name: jnp.nan_to_num(
                reward_components[name] * self._reward_scales[name],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for name in self._reward_names
        }
        reward = jnp.clip(
            jnp.nan_to_num(
                sum(scaled_components.values()), nan=0.0, posinf=0.0, neginf=0.0
            ),
            0.0,
            10000.0,
        ) * self.dt

        local_linvel = self._local_linear_velocity(stepped.data)
        next_gait_phase = None
        if self._gait_cycle_time is not None:
            next_gait_phase = jnp.mod(
                state.info["gait_phase"]
                + 2.0 * jnp.pi * self.dt / self._gait_cycle_time,
                2.0 * jnp.pi,
            )
        info = {
            **stepped.info,
            # Preserve the raw 19-D policy action for observation/reward. The
            # base environment has already masked [12:19] before applying ctrl.
            "last_action": policy_action,
            "last_last_action": previous_action,
            "feet_air_time": current_air_time,
            "last_contact": foot_contact,
            "swing_peak": swing_peak * ~foot_contact,
        }
        if next_gait_phase is not None:
            info["gait_phase"] = next_gait_phase
        if crawl_reference is not None:
            active_leg, swing_window = self._crawl_schedule(
                state.info["gait_phase"],
                self._crawl_lift_start_fraction,
                self._crawl_lift_end_fraction,
            )
            info.update(
                {
                    "crawl_reference": crawl_reference,
                    "crawl_active_leg": active_leg,
                    "crawl_swing_window": swing_window,
                    "crawl_sustained_touchdown": crawl_sustained_touchdown,
                }
            )
            if crawl_foot_targets is not None:
                info["crawl_foot_targets"] = crawl_foot_targets
                info["crawl_ik_reachable"] = crawl_ik_reachable
        if current_contact_time is not None:
            info["feet_contact_time"] = current_contact_time
        metrics = {
            **stepped.metrics,
            "command_x": info["command"][0],
            "base_forward_velocity": local_linvel[0],
            "tracking_linear_error": jnp.linalg.norm(
                info["command"][:2] - local_linvel[:2]
            ),
            "illegal_contact": illegal_contact.astype(jnp.float32),
            "workspace_bounds": workspace_bounds.astype(jnp.float32),
            "nonfinite_state": nonfinite_state.astype(jnp.float32),
            "swing_peak": jnp.mean(swing_peak),
        }
        metrics.update(
            {f"reward/{name}": value for name, value in scaled_components.items()}
        )
        if crawl_reference is not None:
            metrics["crawl_reference_rms"] = jnp.sqrt(
                jnp.mean(jnp.square(crawl_reference))
            )
            if crawl_ik_reachable is not None:
                metrics["crawl_ik_reachable_fraction"] = jnp.mean(
                    crawl_ik_reachable.astype(jnp.float32)
                )
            if self._crawl_pose_reference_enabled:
                reference_error = (
                    stepped.data.qpos[self._joint_qpos_indices[:12]]
                    - self._home_qpos[self._joint_qpos_indices[:12]]
                    - crawl_reference
                )
                metrics["crawl_reference_error_rms"] = jnp.sqrt(
                    jnp.mean(jnp.square(reference_error))
                )
        return stepped.replace(
            obs=self._observation(
                stepped.data,
                info["last_action"],
                info["phase"],
                info["command"],
                next_gait_phase,
            ),
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _task_actuator_reference(self, state) -> jax.Array:
        """Return a task-specific actuator offset without changing action size."""
        del state
        return jnp.zeros_like(self._home_ctrl)

    def _task_policy_action_mask(self, state) -> jax.Array:
        """Return task-specific residual gates without changing action size."""
        del state
        return jnp.ones(ACTION_LAYOUT.size, dtype=jnp.float32)

    def _observation(
        self,
        data,
        last_action,
        phase,
        command: jax.Array | None = None,
        gait_phase: jax.Array | None = None,
    ) -> jax.Array:
        base = super()._observation(data, last_action, phase)
        if command is None:
            command = jnp.zeros(3)
        # Match the explicit scaling used by the official Playground Go1 task
        # instead of relying on a changing running-statistics transform. Field
        # layout is inherited from Go2Z1Env._observation.
        fields = [
            base[0:3],  # projected gravity
            base[3:6] * 2.0,  # local linear velocity
            base[6:9] * 0.25,  # local angular velocity
            base[9:28],  # joint position error
            base[28:47] * 0.05,  # joint velocity
            base[47:66],  # previous raw policy action
            base[66:70],  # task phase
            command,
        ]
        if gait_phase is not None:
            fields.append(jnp.asarray([jnp.sin(gait_phase), jnp.cos(gait_phase)]))
        scaled = jnp.concatenate(fields)
        return jnp.clip(scaled, -10.0, 10.0).astype(jnp.float32)

    def _local_linear_velocity(self, data) -> jax.Array:
        global_linvel = data.sensordata[self._global_linvel_slice]
        return _rotmat(data.qpos[3:7]).T @ global_linvel

    def _local_angular_velocity(self, data) -> jax.Array:
        return data.sensordata[self._gyro_slice]

    def _has_illegal_floor_contact(self, data) -> jax.Array:
        contact = data._impl.contact
        geom1, geom2 = contact.geom[:, 0], contact.geom[:, 1]
        floor_contact = (geom1 == self._floor_geom_id) | (geom2 == self._floor_geom_id)
        other_geom = jnp.where(geom1 == self._floor_geom_id, geom2, geom1)
        allowed_floor_geom = jnp.any(
            other_geom[:, None] == self._allowed_floor_geom_ids[None, :],
            axis=1,
        )
        active = contact.dist < 0.0
        return jnp.any(active & floor_contact & ~allowed_floor_geom)

    def _foot_floor_contacts(self, data) -> jax.Array:
        contact = data._impl.contact
        geom1, geom2 = contact.geom[:, 0], contact.geom[:, 1]
        floor_contact = (geom1 == self._floor_geom_id) | (geom2 == self._floor_geom_id)
        other_geom = jnp.where(geom1 == self._floor_geom_id, geom2, geom1)
        active = contact.dist < 0.0
        return jnp.any(
            active[:, None]
            & floor_contact[:, None]
            & (other_geom[:, None] == self._foot_geom_ids[None, :]),
            axis=0,
        )

    def _reward_components(
        self,
        data,
        command,
        action,
        previous_action,
        done,
        illegal_contact,
        foot_contact,
        first_contact,
        feet_air_time,
        swing_peak,
        gait_phase,
        current_air_time,
        current_contact_time,
        crawl_reference,
    ) -> dict[str, jax.Array]:
        local_linvel = self._local_linear_velocity(data)
        local_angvel = self._local_angular_velocity(data)
        linear_error = jnp.sum(jnp.square(command[:2] - local_linvel[:2]))
        angular_error = jnp.square(command[2] - local_angvel[2])
        projected_gravity = _rotmat(data.qpos[3:7]).T @ jnp.asarray([0.0, 0.0, -1.0])
        pose_weight = jnp.asarray([1.0, 1.0, 0.1] * 4)
        command_norm = jnp.linalg.norm(command)
        moving = command_norm > 0.01
        leg_qpos_indices = self._joint_qpos_indices[:12]
        home_position_error = (
            data.qpos[leg_qpos_indices] - self._home_qpos[leg_qpos_indices]
        )
        leg_position_error = home_position_error
        if self._crawl_pose_reference_enabled:
            reference_position_error = home_position_error - crawl_reference
            leg_position_error = jnp.where(
                moving,
                reference_position_error,
                home_position_error,
            )
        leg_torque = data.actuator_force[:12]
        foot_linvel = data.sensordata[self._foot_linvel_sensor_indices]
        foot_vel_xy = foot_linvel[:, :2]
        foot_vel_xy_norm_sq = jnp.sum(jnp.square(foot_vel_xy), axis=-1)
        foot_height = data.site_xpos[self._foot_site_ids, -1]
        swing_height_error = swing_peak / self._max_foot_height - 1.0
        pose = jnp.exp(-jnp.sum(jnp.square(leg_position_error) * pose_weight))
        pose_multiplier = jnp.where(
            moving,
            self._moving_pose_multiplier,
            1.0,
        )
        components = {
            "tracking_linear_velocity": jnp.exp(-linear_error / self._tracking_sigma),
            "tracking_angular_velocity": jnp.exp(-angular_error / self._tracking_sigma),
            "pose": pose * pose_multiplier,
            "vertical_velocity": jnp.square(local_linvel[2]),
            "angular_velocity_xy": jnp.sum(jnp.square(local_angvel[:2])),
            "orientation": jnp.sum(jnp.square(projected_gravity[:2])),
            "stand_still": jnp.sum(jnp.abs(leg_position_error)) * (command_norm < 0.01),
            "torques": jnp.sum(jnp.square(leg_torque)),
            "action_rate": jnp.sum(jnp.square(action - previous_action)),
            "action_magnitude": jnp.sum(jnp.square(action)),
            "arm_action_magnitude": jnp.sum(jnp.square(action[12:])),
            "feet_slip": jnp.sum(foot_vel_xy_norm_sq * foot_contact) * moving,
            "feet_clearance": jnp.sum(
                jnp.abs(foot_height - self._max_foot_height)
                * jnp.sqrt(jnp.linalg.norm(foot_vel_xy, axis=-1))
            ),
            "feet_height": (
                jnp.sum(jnp.square(swing_height_error) * first_contact) * moving
            ),
            "feet_air_time": (
                jnp.sum((feet_air_time - 0.1) * first_contact) * moving
            ),
            "termination": done,
            "illegal_contact": illegal_contact.astype(jnp.float32),
        }
        if gait_phase is not None:
            desired_contact = self._desired_trot_contact(gait_phase)
            desired_swing = ~desired_contact
            phase_swing_height_error = foot_height / self._max_foot_height - 1.0
            components.update(
                {
                    "trot_contact": (
                        jnp.mean(foot_contact == desired_contact) * moving
                    ),
                    "trot_swing_height": (
                        jnp.mean(
                            jnp.square(phase_swing_height_error)
                            * desired_swing.astype(jnp.float32)
                        )
                        * moving
                    ),
                }
            )
        if current_contact_time is not None:
            upright = jnp.clip(-projected_gravity[2], 0.0, 0.7) / 0.7
            components["trot_timing"] = (
                self._trot_timing_score(
                    current_air_time,
                    current_contact_time,
                    self._trot_timing_std,
                    self._trot_timing_max_error,
                    self._trot_timing_min_air_time,
                )
                * moving
                * upright
            )
        return components

    @staticmethod
    def _smoothstep(value: jax.Array) -> jax.Array:
        value = jnp.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _crawl_schedule(
        gait_phase: jax.Array,
        lift_start_fraction: float = 0.3,
        lift_end_fraction: float = 0.8,
    ) -> tuple[jax.Array, jax.Array]:
        cycle_position = jnp.mod(gait_phase, 2.0 * jnp.pi) / (2.0 * jnp.pi)
        quarter_position = 4.0 * cycle_position
        slot = jnp.floor(quarter_position).astype(jnp.int32)
        quarter_phase = quarter_position - jnp.floor(quarter_position)
        active_leg = _CRAWL_SEQUENCE[slot]
        swing_window = (quarter_phase >= lift_start_fraction) & (
            quarter_phase < lift_end_fraction
        )
        return active_leg, swing_window

    def _crawl_foot_space_solution(
        self,
        gait_phase: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return go2_foot_space_crawl_reference(
            gait_phase,
            step_length=self._crawl_foot_step_length,
            foot_clearance=self._crawl_foot_clearance,
            body_shift_x=self._crawl_body_shift_x,
            body_shift_y=self._crawl_body_shift_y,
            shift_end_fraction=self._crawl_shift_end_fraction,
            lift_start_fraction=self._crawl_lift_start_fraction,
            lift_end_fraction=self._crawl_lift_end_fraction,
        )

    def _crawl_joint_reference(
        self,
        gait_phase: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        """Four-beat FL-RR-FR-RL crawl with a three-foot support target."""

        if self._crawl_foot_space_enabled:
            reference, _, _ = self._crawl_foot_space_solution(gait_phase)
            moving = (jnp.linalg.norm(command) > 0.01).astype(jnp.float32)
            return (reference * moving).astype(jnp.float32)

        cycle_position = jnp.mod(gait_phase, 2.0 * jnp.pi) / (2.0 * jnp.pi)
        quarter_position = 4.0 * cycle_position
        slot = jnp.floor(quarter_position).astype(jnp.int32)
        quarter_phase = quarter_position - jnp.floor(quarter_position)
        active_leg = _CRAWL_SEQUENCE[slot]
        previous_leg = _CRAWL_SEQUENCE[jnp.mod(slot - 1, 4)]

        shift_blend = self._smoothstep(
            quarter_phase / self._crawl_shift_end_fraction
        )
        previous_pitch = -_CRAWL_FORE_AFT_SIGNS[previous_leg] * self._crawl_shift
        active_pitch = -_CRAWL_FORE_AFT_SIGNS[active_leg] * self._crawl_shift
        previous_hip = _CRAWL_LEFT_RIGHT_SIGNS[previous_leg] * self._crawl_shift
        active_hip = _CRAWL_LEFT_RIGHT_SIGNS[active_leg] * self._crawl_shift
        common_pitch = (
            (1.0 - shift_blend) * previous_pitch + shift_blend * active_pitch
        )
        common_hip = (1.0 - shift_blend) * previous_hip + shift_blend * active_hip

        leg_phase = jnp.mod(cycle_position - _CRAWL_PHASE_OFFSETS, 1.0)
        swing_progress = self._smoothstep(leg_phase / 0.25)
        stance_progress = self._smoothstep((leg_phase - 0.25) / 0.75)
        stride_pitch = jnp.where(
            leg_phase < 0.25,
            self._crawl_stride * (1.0 - 2.0 * swing_progress),
            self._crawl_stride * (-1.0 + 2.0 * stance_progress),
        )
        thigh = common_pitch + stride_pitch

        lift_duration = (
            self._crawl_lift_end_fraction - self._crawl_lift_start_fraction
        )
        lift_progress = jnp.clip(
            (quarter_phase - self._crawl_lift_start_fraction) / lift_duration,
            0.0,
            1.0,
        )
        lift_window = (
            quarter_phase >= self._crawl_lift_start_fraction
        ) & (quarter_phase < self._crawl_lift_end_fraction)
        knee_lift = (
            self._crawl_lift
            * jnp.square(jnp.sin(jnp.pi * lift_progress))
            * lift_window.astype(jnp.float32)
            * (jnp.arange(4) == active_leg).astype(jnp.float32)
        )
        calf = (
            0.65 * common_hip * _CRAWL_LEFT_RIGHT_SIGNS
            + 0.8 * jnp.square(thigh)
            - knee_lift
        )
        reference = jnp.stack(
            [jnp.full(4, common_hip), thigh, calf],
            axis=-1,
        ).reshape(12)
        moving = (jnp.linalg.norm(command) > 0.01).astype(jnp.float32)
        return (reference * moving).astype(jnp.float32)

    @staticmethod
    def _desired_trot_contact(gait_phase: jax.Array) -> jax.Array:
        diagonal_a_stance = jnp.cos(gait_phase) >= 0.0
        return jnp.asarray(
            [
                diagonal_a_stance,
                ~diagonal_a_stance,
                ~diagonal_a_stance,
                diagonal_a_stance,
            ]
        )

    @staticmethod
    def _trot_timing_score(
        air_time: jax.Array,
        contact_time: jax.Array,
        std: float,
        max_error: float,
        min_air_time: float = 0.0,
    ) -> jax.Array:
        """Reward diagonal-pair timing agreement and pair opposition."""

        max_squared_error = max_error**2

        def squared_error(first, second):
            return jnp.minimum(jnp.square(first - second), max_squared_error)

        def sync_reward(first: int, second: int):
            error = squared_error(
                air_time[first], air_time[second]
            ) + squared_error(contact_time[first], contact_time[second])
            return jnp.exp(-error / std)

        def async_reward(first: int, second: int):
            error = squared_error(
                air_time[first], contact_time[second]
            ) + squared_error(contact_time[first], air_time[second])
            return jnp.exp(-error / std)

        # FL+RR and FR+RL are internally synchronized. Every cross-pair
        # combination is expected to be in the opposite contact mode.
        timing_score = (
            sync_reward(0, 3)
            * sync_reward(1, 2)
            * async_reward(0, 1)
            * async_reward(3, 2)
            * async_reward(0, 2)
            * async_reward(1, 3)
        )
        if min_air_time <= 0.0:
            return timing_score

        diagonal_air_time = jnp.maximum(
            jnp.minimum(air_time[0], air_time[3]),
            jnp.minimum(air_time[1], air_time[2]),
        )
        dwell_gate = jnp.clip(diagonal_air_time / min_air_time, 0.0, 1.0)
        return timing_score * dwell_gate
