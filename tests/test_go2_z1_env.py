"""Slice-1 smoke: Go2Z1Env reset/step stay finite under JIT and VMAP."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import pytest

from amd_robo.envs.go2_z1 import FOOT_GEOM_NAMES, Go2Z1Env
from amd_robo.envs.protocol import ProjectMjxEnv
from amd_robo.platform.smoke import _block_tree, _tree_is_finite


def test_env_matches_contract() -> None:
    env = Go2Z1Env()
    assert env.action_size == 19
    assert env.dt == 0.01
    assert env.n_substeps == 5
    assert isinstance(env, ProjectMjxEnv)
    assert getattr(env.mjx_model.impl, "value", None) == "jax"
    assert not jnp.allclose(env.mj_model.qpos0[7:], env.mj_model.key_qpos[0, 7:])


def test_leg_pd_override_preserves_the_actuator_contract() -> None:
    env = Go2Z1Env(leg_kp=40.0, leg_kd=8.0)

    assert env._leg_kp == 40.0
    assert env._leg_kd == 8.0
    assert jnp.allclose(env.mj_model.actuator_gainprm[:12, 0], 40.0)
    assert jnp.allclose(env.mj_model.actuator_biasprm[:12, 1], -40.0)
    assert jnp.allclose(env.mj_model.actuator_biasprm[:12, 2], -8.0)
    assert env.mj_model.actuator_gainprm[12, 0] == 1000.0


def test_arm_pd_override_preserves_legs_and_gripper() -> None:
    env = Go2Z1Env(arm_kp=300.0, arm_kd=30.0)

    assert jnp.allclose(env.mj_model.actuator_gainprm[:12, 0], 50.0)
    assert jnp.allclose(env.mj_model.actuator_gainprm[12:18, 0], 300.0)
    assert jnp.allclose(env.mj_model.actuator_biasprm[12:18, 1], -300.0)
    assert jnp.allclose(env.mj_model.actuator_biasprm[12:18, 2], -30.0)
    assert env.mj_model.actuator_gainprm[18, 0] == 1000.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"arm_kp": 0.0}, "arm_kp must be positive"),
        ({"arm_kd": -1.0}, "arm_kd must be non-negative"),
    ],
)
def test_arm_pd_override_rejects_invalid_values(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Go2Z1Env(**kwargs)


def test_solver_iterations_override() -> None:
    env = Go2Z1Env(solver_iterations=8)

    assert env.mj_model.opt.iterations == 8


def test_foot_condim_override() -> None:
    env = Go2Z1Env(foot_condim=1)
    for name in FOOT_GEOM_NAMES:
        assert env.mj_model.geom(name).condim == 1


def test_default_scene_has_supporting_ground() -> None:
    env = Go2Z1Env()
    floor_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    assert env.mj_model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE

    data = mujoco.MjData(env.mj_model)
    data.qpos[:] = env.mj_model.key_qpos[0]
    data.ctrl[:] = env.mj_model.key_ctrl[0]
    mujoco.mj_forward(env.mj_model, data)
    assert all(
        floor_id in (contact.geom1, contact.geom2)
        for contact in data.contact[: data.ncon]
    )
    for _ in range(500):
        mujoco.mj_step(env.mj_model, data)

    assert data.qpos[2] > 0.1
    assert data.ncon > 0


def test_reset_step_finite_single_env() -> None:
    env = Go2Z1Env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert jnp.allclose(state.data.qpos, env.mj_model.key_qpos[0])
    assert _tree_is_finite(state.data)
    nxt = jax.jit(env.step)(state, jnp.zeros(env.action_size))
    _block_tree(nxt)
    assert _tree_is_finite(nxt.data)
    assert nxt.obs.shape[-1] > 0
    # all REQUIRED_INFO_KEYS present
    from amd_robo.contracts import REQUIRED_INFO_KEYS

    assert REQUIRED_INFO_KEYS <= set(nxt.info.keys())


def test_reset_step_finite_under_vmap() -> None:
    """Env-side G1: 256 envs x N control steps stay finite under jit+vmap.

    Driven by a sequential Python loop reusing one compiled vmap kernel, NOT a
    fused lax.scan: a long fused scan over env.step (which itself scans over
    n_substeps) trips the known gfx1100 long-fused-scan XLA-ROCm segfault (see
    src/amd_robo/platform/rollout_probe.py). Brax PPO also needs the measured
    five-substep control kernel and a compiled training scan no larger than two.
    """
    env = Go2Z1Env()
    n_envs, n_control_steps = 256, 50
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    state = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.zeros((n_envs, env.action_size))
    step_fn = jax.jit(jax.vmap(env.step))

    for i in range(n_control_steps):
        state = step_fn(state, actions)
        if i % 10 == 9:
            assert _tree_is_finite(state.data), f"non-finite at control step {i + 1}"
    _block_tree(state)
    assert _tree_is_finite(state.data), "Go2Z1Env went non-finite under vmap rollout"
