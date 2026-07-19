"""Go2+Z1 (19-DoF) MuJoCo Playground-style environment -- Slice 1 (env-side G1).

Slice 1 wraps the assembled ``go2_z1`` MJCF in a Playground-style ``MjxEnv`` whose
only goal is the environment-side Gate G1: ``reset``/``step`` run under JIT and
VMAP and stay finite. The arm and gripper (action indices ``12:19``) are masked
to their home values so the slice exercises locomotion only; the 19-DoF action
interface is unchanged (project rule 7). Reward is zero, termination is
non-finite/tilt only, and the observation is the locomotion subset of
``POLICY_OBSERVATION_FIELDS``.

Deferred to later slices: the push box + goal zone, the
object/goal/end-effector observation fields, reward shaping, and the remaining
termination signals (illegal_contact, workspace_bounds, success_hold).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src.mjx_env import MjxEnv, State

from amd_robo.contracts import ACTION_LAYOUT, TaskPhase

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_mjx.xml"
FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")

# Leg/arm/gripper layout mirrors ACTION_LAYOUT. Slice 1 zeroes [12:19] so only
# the 12 leg actuators respond; the arm and gripper are held at their home ctrl.
_LEG_MASK = jnp.concatenate(
    [
        jnp.ones(ACTION_LAYOUT.leg.stop - ACTION_LAYOUT.leg.start),  # 12 legs
        jnp.zeros(ACTION_LAYOUT.size - ACTION_LAYOUT.leg.stop),  # 7 arm+gripper
    ]
)


def _rotmat(quat: jax.Array) -> jax.Array:
    """Body->world rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class Go2Z1Env(MjxEnv):
    """Slice-1 Go2+Z1 flat-ground environment, finite under JIT and VMAP."""

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        ctrl_dt: float = 0.01,
        action_scale: float = 0.25,
        leg_kp: float | None = None,
        leg_kd: float | None = None,
        home_keyframe: str | None = "home",
        tilt_limit_deg: float = 60.0,
        mask_arm: bool = True,
        foot_condim: int | None = None,
        bound_observations: bool = True,
    ) -> None:
        self._xml_path = str(xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        if leg_kp is not None:
            if leg_kp <= 0.0:
                raise ValueError("leg_kp must be positive")
            self._mj_model.actuator_gainprm[:12, 0] = leg_kp
            self._mj_model.actuator_biasprm[:12, 1] = -leg_kp
        if leg_kd is not None:
            if leg_kd < 0.0:
                raise ValueError("leg_kd must be non-negative")
            self._mj_model.actuator_biasprm[:12, 2] = -leg_kd
        if foot_condim is not None:
            if foot_condim not in (1, 3, 4, 6):
                raise ValueError(
                    f"foot_condim must be one of 1, 3, 4, or 6; got {foot_condim}"
                )
            for name in FOOT_GEOM_NAMES:
                geom_id = mujoco.mj_name2id(
                    self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                if geom_id < 0:
                    raise ValueError(f"foot geom not found: {name}")
                self._mj_model.geom_condim[geom_id] = foot_condim
        if home_keyframe is None:
            home_keyframe_id = -1
        else:
            home_keyframe_id = mujoco.mj_name2id(
                self._mj_model,
                mujoco.mjtObj.mjOBJ_KEY,
                home_keyframe,
            )
            if home_keyframe_id < 0:
                raise ValueError(f"home keyframe not found: {home_keyframe}")
        self._home_keyframe = home_keyframe
        self._home_keyframe_id = home_keyframe_id
        self._home_qpos = jnp.asarray(
            self._mj_model.key_qpos[home_keyframe_id]
            if home_keyframe_id >= 0
            else self._mj_model.qpos0
        )
        self._home_ctrl = (
            jnp.asarray(self._mj_model.key_ctrl[home_keyframe_id])
            if home_keyframe_id >= 0 and self._mj_model.key_ctrl.shape[1] > 0
            else jnp.zeros(self._mj_model.nu)
        )
        if any(
            transmission != mujoco.mjtTrn.mjTRN_JOINT
            for transmission in self._mj_model.actuator_trntype
        ):
            raise ValueError("all policy actuators must use joint transmission")
        actuator_joint_ids = tuple(
            int(joint_id) for joint_id in self._mj_model.actuator_trnid[:, 0]
        )
        if len(set(actuator_joint_ids)) != ACTION_LAYOUT.size:
            raise ValueError("policy actuators must map to 19 unique joints")
        if any(
            self._mj_model.jnt_type[joint_id]
            not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            for joint_id in actuator_joint_ids
        ):
            raise ValueError("policy actuator joints must have one DoF")
        self._joint_qpos_indices = jnp.asarray(
            self._mj_model.jnt_qposadr[list(actuator_joint_ids)],
            dtype=jnp.int32,
        )
        self._joint_dof_indices = jnp.asarray(
            self._mj_model.jnt_dofadr[list(actuator_joint_ids)],
            dtype=jnp.int32,
        )
        self._mjx_model = mjx.put_model(self._mj_model, impl="jax")
        self._action_scale = float(action_scale)
        self._leg_kp = (
            float(self._mj_model.actuator_gainprm[0, 0])
            if leg_kp is None
            else float(leg_kp)
        )
        self._leg_kd = (
            -float(self._mj_model.actuator_biasprm[0, 2])
            if leg_kd is None
            else float(leg_kd)
        )
        self._tilt_limit = jnp.radians(float(tilt_limit_deg))
        self._mask_arm = bool(mask_arm)
        self._bound_observations = bool(bound_observations)
        config = config_dict.ConfigDict(
            {"ctrl_dt": float(ctrl_dt), "sim_dt": float(self._mj_model.opt.timestep)}
        )
        super().__init__(config=config)

    # --- ProjectMjxEnv / MjxEnv interface ---
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return ACTION_LAYOUT.size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    # --- core ---
    def reset(self, rng: jax.Array) -> State:
        data = mjx.make_data(self._mj_model, impl="jax")
        data = data.replace(qpos=self._home_qpos, ctrl=self._home_ctrl)
        data = mjx.forward(self._mjx_model, data)
        action = jnp.zeros(ACTION_LAYOUT.size)
        phase = jnp.asarray(int(TaskPhase.APPROACH))
        info = {
            "rng": rng,
            "phase": phase,
            "goal_pos": jnp.zeros(3),
            "last_action": action,
            "success_count": jnp.asarray(0, dtype=jnp.int32),
        }
        return State(
            data=data,
            obs=self._observation(data, action, phase),
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            metrics={"tilt_deg": jnp.zeros(()), "base_height": data.qpos[2]},
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
        if self._mask_arm:
            action = action * _LEG_MASK
        ctrl = self._home_ctrl + self._action_scale * action
        return self._step_with_ctrl(state, action, ctrl)

    def _step_with_ctrl(
        self,
        state: State,
        action: jax.Array,
        ctrl: jax.Array,
    ) -> State:
        """Advance one control step using an explicit actuator target.

        ``action`` remains the policy-side value used by observations and
        diagnostics, while ``ctrl`` may include an environment-side reference
        motion.  Keeping those two quantities separate lets later curriculum
        slices add a nominal controller without changing the frozen action
        interface or charging the reference motion to the policy residual.
        """

        # Unroll the five physics substeps as a fixed Python loop (traced as
        # sequential mjx.step ops) instead of lax.scan. On gfx1100, vmap of a
        # lax.scan over mjx.step on this complex model segfaults in XLA/ROCm.
        # Ten Python-unrolled substeps also fail during CompileAndLoad, while
        # five pass together with a Brax compiled training scan of two.
        data = state.data
        for _ in range(self.n_substeps):
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self._mjx_model, data)

        tilt = self._tilt_rad(data)
        # done is float32 (0.0/1.0): Brax's EpisodeWrapper derives info
        # episode_done/truncation from state.done via *_like(state.done), so a
        # bool done makes those fields bool/int32 in step while reset inits them
        # float32, breaking the scan carry type check.
        done = (
            ~jnp.all(jnp.isfinite(data.qpos))
            | ~jnp.all(jnp.isfinite(data.qvel))
            | (tilt > self._tilt_limit)
        ).astype(jnp.float32)
        info = {**state.info, "last_action": action}
        metrics = {
            **state.metrics,
            "tilt_deg": jnp.degrees(tilt),
            "base_height": data.qpos[2],
        }
        return state.replace(
            data=data,
            obs=self._observation(data, action, state.info["phase"]),
            reward=jnp.zeros(()),
            done=done,
            metrics=metrics,
            info=info,
        )

    def _observation(self, data, last_action, phase) -> jax.Array:
        qpos = data.qpos
        qvel = data.qvel
        # Freejoint occupies qpos[0:7] (3 pos + 4 quat) and qvel[0:6].
        r_inv = _rotmat(qpos[3:7]).T  # world -> body
        projected_gravity = r_inv @ jnp.asarray([0.0, 0.0, -1.0])
        base_lin_vel = r_inv @ qvel[0:3]
        base_ang_vel = r_inv @ qvel[3:6]
        joint_pos = qpos[self._joint_qpos_indices]
        joint_vel = qvel[self._joint_dof_indices]
        joint_pos_error = joint_pos - self._home_qpos[self._joint_qpos_indices]
        phase_onehot = jax.nn.one_hot(phase, len(TaskPhase))
        obs = jnp.concatenate(
            [
                projected_gravity,
                base_lin_vel,
                base_ang_vel,
                joint_pos_error,
                joint_vel,
                last_action,
                phase_onehot,
            ]
        )
        if self._bound_observations:
            # Contact solvers can produce a very large but still finite velocity
            # on the transition immediately before termination.  Keep that
            # transition from poisoning Brax's running observation statistics.
            # Playground's locomotion environments use the same +/-100 bound.
            obs = jnp.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
            obs = jnp.clip(obs, -100.0, 100.0)
        return obs.astype(jnp.float32)

    @staticmethod
    def _tilt_rad(data) -> jax.Array:
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        # z-component of the base z-axis expressed in the world frame: cos(tilt).
        rz = w * w - x * x - y * y + z * z
        return jnp.arccos(jnp.clip(rz, -1.0, 1.0))
