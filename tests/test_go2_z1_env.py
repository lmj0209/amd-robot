"""Slice-1 smoke: Go2Z1Env reset/step stay finite under JIT and VMAP."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp

from amd_robo.envs.go2_z1 import FOOT_GEOM_NAMES, Go2Z1Env
from amd_robo.envs.protocol import ProjectMjxEnv
from amd_robo.platform.smoke import _block_tree, _tree_is_finite


def test_env_matches_contract() -> None:
    env = Go2Z1Env()
    assert env.action_size == 19
    assert isinstance(env, ProjectMjxEnv)
    assert getattr(env.mjx_model.impl, "value", None) == "jax"


def test_foot_condim_override() -> None:
    env = Go2Z1Env(foot_condim=1)
    for name in FOOT_GEOM_NAMES:
        assert env.mj_model.geom(name).condim == 1


def test_reset_step_finite_single_env() -> None:
    env = Go2Z1Env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
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
    src/amd_robo/platform/rollout_probe.py). Brax PPO is unaffected because it
    scans over a small unroll_length, not a whole episode.
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
