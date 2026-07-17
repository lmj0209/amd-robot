#!/usr/bin/env python3
"""Diagnostic: does the standing-reward PPO compile for a given MJCF on gfx1100?

Same structure that segfaults on go2_z1 (home reset, Python-unrolled substeps,
exp-free standing reward in env.step, float32 done), but model-agnostic so it
can point at any MJCF. Run on go2_mjx.xml (12 DoF, no arm) vs go2_z1.xml (19 DoF)
to isolate whether the ARM / arm-contacts push go2_z1's PPO compile over the
gfx1100 XLA limit.

    python scripts/standing_diag.py --xml assets/menagerie/unitree_go2/go2_mjx.xml

Prints TRAINING_DONE on success; a process exit with no further output = the
known compile segfault.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src.mjx_env import MjxEnv, State


def _rotmat(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class StandingEnv(MjxEnv):
    """Minimal model-agnostic standing env: home reset, reward in step."""

    def __init__(self, xml_path, ctrl_dt=0.02, action_scale=0.25, tilt_limit_deg=60.0):
        self._xml_path = str(xml_path)
        self._mj = mujoco.MjModel.from_xml_path(self._xml_path)
        if self._mj.nkey > 0:
            self._mj.qpos0[:] = self._mj.key_qpos[0]
        self._mjx = mjx.put_model(self._mj, impl="jax")
        self._home_qpos = jnp.asarray(self._mj.qpos0)
        self._home_ctrl = (
            jnp.asarray(self._mj.key_ctrl[0])
            if self._mj.nkey > 0 and self._mj.key_ctrl.shape[1] > 0
            else jnp.zeros(self._mj.nu)
        )
        self._action_scale = float(action_scale)
        self._tilt_limit = jnp.radians(float(tilt_limit_deg))
        super().__init__(config=config_dict.ConfigDict(
            {"ctrl_dt": float(ctrl_dt), "sim_dt": float(self._mj.opt.timestep)}))

    @property
    def xml_path(self): return self._xml_path
    @property
    def action_size(self): return self._mj.nu
    @property
    def mj_model(self): return self._mj
    @property
    def mjx_model(self): return self._mjx

    def reset(self, rng):
        data = mjx.make_data(self._mj, impl="jax")
        data = data.replace(qpos=self._home_qpos, ctrl=self._home_ctrl)
        data = mjx.forward(self._mjx, data)
        action = jnp.zeros(self._mj.nu)
        return State(
            data=data, obs=self._obs(data, action), reward=jnp.zeros(()),
            done=jnp.zeros(()), metrics={}, info={"rng": rng, "last_action": action},
        )

    def step(self, state, action):
        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
        ctrl = self._home_ctrl + self._action_scale * action
        # Python-unrolled substeps (vmap x lax.scan x this model segfaults on gfx1100).
        data = state.data
        for _ in range(self.n_substeps):
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self._mjx, data)
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        rz = w * w - x * x - y * y + z * z
        upright = jnp.maximum(0.0, rz)
        dz = data.qpos[2] - self._home_qpos[2]
        height = jnp.maximum(0.0, 1.0 - (dz * dz) / 0.02)
        reward = upright + height - 0.001 * jnp.sum(action * action)
        done = (
            ~jnp.all(jnp.isfinite(data.qpos))
            | ~jnp.all(jnp.isfinite(data.qvel))
            | (jnp.arccos(jnp.clip(rz, -1.0, 1.0)) > self._tilt_limit)
        ).astype(jnp.float32)
        info = {**state.info, "last_action": action}
        return state.replace(data=data, obs=self._obs(data, action), reward=reward, done=done, info=info)

    def _obs(self, data, last_action):
        qpos, qvel = data.qpos, data.qvel
        r_inv = _rotmat(qpos[3:7]).T
        return jnp.concatenate([
            r_inv @ jnp.asarray([0.0, 0.0, -1.0]),
            r_inv @ qvel[0:3], r_inv @ qvel[3:6],
            qpos[7:] - self._home_qpos[7:], qvel[6:], last_action,
        ]).astype(jnp.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=str(REPO_ROOT / "assets/menagerie/unitree_go2/go2_mjx.xml"))
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--unroll", type=int, default=4)
    args = p.parse_args()

    from amd_robo.platform import _compat
    _compat.apply_brax_compat()
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    env = StandingEnv(args.xml)
    bs = args.num_envs // 4
    nmb = args.num_envs // bs
    print(f"diag: xml={args.xml} nq={env._mj.nq} nu={env._mj.nu} num_envs={args.num_envs} unroll={args.unroll} batch={bs} mb={nmb}", flush=True)
    _, params, metrics = ppo.train(
        environment=env, num_timesteps=5120, max_devices_per_host=1,
        num_envs=args.num_envs, episode_length=64, action_repeat=1,
        learning_rate=3e-4, entropy_cost=1e-3, discounting=0.97,
        unroll_length=args.unroll, batch_size=bs, num_minibatches=nmb,
        num_updates_per_batch=2, normalize_observations=True,
        num_evals=1, num_eval_envs=4, run_evals=False, seed=0,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    jax.tree_util.tree_map(lambda v: v.block_until_ready(), params)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
