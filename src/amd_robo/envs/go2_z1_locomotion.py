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
from amd_robo.envs.go2_z1 import FOOT_GEOM_NAMES, Go2Z1Env, _rotmat

FOOT_SITE_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


class Go2Z1LocomotionEnv(Go2Z1Env):
    """Track a low-speed planar command while keeping the Z1 arm tucked."""

    _REWARD_NAMES = (
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
        self._foot_site_ids = jnp.asarray(
            [
                mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                for site_name in FOOT_SITE_NAMES
            ],
            dtype=jnp.int32,
        )
        if bool(jnp.any(self._foot_site_ids < 0)):
            raise ValueError("locomotion model is missing one or more foot sites")
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

    def reset(self, rng: jax.Array):
        rng, command_key, zero_key, joint_key, velocity_key = jax.random.split(rng, 5)
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
                qpos=data.qpos.at[7:].add(joint_delta * leg_mask),
                qvel=data.qvel.at[6:].set(joint_velocity * leg_mask),
            )
            data = mjx.forward(self.mjx_model, data)

        info = {
            **state.info,
            "rng": rng,
            "command": command,
            "last_last_action": jnp.zeros(ACTION_LAYOUT.size),
            "feet_air_time": jnp.zeros(len(FOOT_SITE_NAMES)),
            "last_contact": jnp.zeros(len(FOOT_SITE_NAMES), dtype=bool),
            "swing_peak": jnp.zeros(len(FOOT_SITE_NAMES)),
        }
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
        metrics.update({f"reward/{name}": jnp.zeros(()) for name in self._REWARD_NAMES})
        return state.replace(
            data=data,
            obs=self._observation(
                data, state.info["last_action"], state.info["phase"], command
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
        stepped = super().step(state, policy_action)

        foot_contact = self._foot_floor_contacts(stepped.data)
        contact_filt = foot_contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) & contact_filt
        feet_air_time = state.info["feet_air_time"] + self.dt
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
        )
        scaled_components = {
            name: jnp.nan_to_num(
                reward_components[name] * self._reward_scales[name],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for name in self._REWARD_NAMES
        }
        reward = jnp.clip(
            jnp.nan_to_num(
                sum(scaled_components.values()), nan=0.0, posinf=0.0, neginf=0.0
            ),
            0.0,
            10000.0,
        ) * self.dt

        local_linvel = self._local_linear_velocity(stepped.data)
        info = {
            **stepped.info,
            # Preserve the raw 19-D policy action for observation/reward. The
            # base environment has already masked [12:19] before applying ctrl.
            "last_action": policy_action,
            "last_last_action": previous_action,
            "feet_air_time": feet_air_time * ~foot_contact,
            "last_contact": foot_contact,
            "swing_peak": swing_peak * ~foot_contact,
        }
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
        return stepped.replace(
            obs=self._observation(
                stepped.data,
                info["last_action"],
                info["phase"],
                info["command"],
            ),
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _observation(
        self,
        data,
        last_action,
        phase,
        command: jax.Array | None = None,
    ) -> jax.Array:
        base = super()._observation(data, last_action, phase)
        if command is None:
            command = jnp.zeros(3)
        # Match the explicit scaling used by the official Playground Go1 task
        # instead of relying on a changing running-statistics transform. Field
        # layout is inherited from Go2Z1Env._observation.
        scaled = jnp.concatenate(
            [
                base[0:3],  # projected gravity
                base[3:6] * 2.0,  # local linear velocity
                base[6:9] * 0.25,  # local angular velocity
                base[9:28],  # joint position error
                base[28:47] * 0.05,  # joint velocity
                base[47:66],  # previous raw policy action
                base[66:70],  # task phase
                command,
            ]
        )
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
        allowed_foot = jnp.any(
            other_geom[:, None] == self._foot_geom_ids[None, :], axis=1
        )
        active = contact.dist < 0.0
        return jnp.any(active & floor_contact & ~allowed_foot)

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
    ) -> dict[str, jax.Array]:
        local_linvel = self._local_linear_velocity(data)
        local_angvel = self._local_angular_velocity(data)
        linear_error = jnp.sum(jnp.square(command[:2] - local_linvel[:2]))
        angular_error = jnp.square(command[2] - local_angvel[2])
        projected_gravity = _rotmat(data.qpos[3:7]).T @ jnp.asarray([0.0, 0.0, -1.0])
        leg_position_error = data.qpos[7:19] - self._home_qpos[7:19]
        pose_weight = jnp.asarray([1.0, 1.0, 0.1] * 4)
        command_norm = jnp.linalg.norm(command)
        leg_torque = data.actuator_force[:12]
        foot_linvel = data.sensordata[self._foot_linvel_sensor_indices]
        foot_vel_xy = foot_linvel[:, :2]
        foot_vel_xy_norm_sq = jnp.sum(jnp.square(foot_vel_xy), axis=-1)
        foot_height = data.site_xpos[self._foot_site_ids, -1]
        moving = command_norm > 0.01
        swing_height_error = swing_peak / self._max_foot_height - 1.0
        pose = jnp.exp(-jnp.sum(jnp.square(leg_position_error) * pose_weight))
        pose_multiplier = jnp.where(
            moving,
            self._moving_pose_multiplier,
            1.0,
        )
        return {
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
