#!/usr/bin/env python3
"""Standing-learning smoke: does PPO optimize a real reward on go2_z1 (gfx1100)?

Runs PPO on ``Go2Z1StandingEnv`` (dense standing reward + light init
randomization) with ``run_evals=False`` -- Brax's default eval rollout segfaults
on go2_z1/gfx1100 (long fused scan over a whole episode), so evaluation is done
manually with a sequential Python loop reusing one compiled vmap kernel (the
known-stable path; see src/amd_robo/platform/rollout_probe.py). Prints the mean
standing reward of the trained policy vs a zero-action baseline on the SAME
randomized initial states.

Run on RGC::

    # gfx1100 host-loop training: compile a two-step epoch once, then reuse it.
    /workspace/.venv/bin/python scripts/standing_learn_smoke.py \
      --foot-condim 1 --num-timesteps 5120

    # CPU learning oracle.
    JAX_PLATFORMS=cpu /workspace/.venv/bin/python \
      scripts/standing_learn_smoke.py --foot-condim 6 \
      --num-timesteps 5120 --learning-rate-schedule ADAPTIVE_KL \
      --max-grad-norm 1

The smoke maps Brax's ``num_evals`` iterations to a host loop while disabling
evaluation. The complete Brax ``TrainingState`` remains live across calls, and
each compiled ``training_epoch`` scan is bounded by
``--training-steps-per-host-call``. ``--params-in``/``--params-out`` still only
save inference parameters and are not resumable optimizer checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mujoco import mjx  # noqa: E402

from amd_robo.envs.go2_z1 import Go2Z1Env  # noqa: E402
from amd_robo.training.host_loop import plan_brax_host_loop  # noqa: E402


class Go2Z1StandingEnv(Go2Z1Env):
    """Go2Z1Env with a dense standing reward and light init randomization."""

    def __init__(
        self,
        *args,
        sanitize_reward=True,
        randomize_reset=True,
        **kwargs,
    ):
        self._sanitize_reward = bool(sanitize_reward)
        self._randomize_reset = bool(randomize_reset)
        super().__init__(*args, **kwargs)

    def reset(self, rng):
        if not self._randomize_reset:
            return super().reset(rng)
        rng, joint_key, velocity_key = jax.random.split(rng, 3)
        state = super().reset(rng)
        leg_mask = jnp.concatenate([jnp.ones(12), jnp.zeros(7)])
        joint_delta = jax.random.uniform(
            joint_key,
            (self.action_size,),
            minval=-0.03,
            maxval=0.03,
        )
        joint_velocity = jax.random.uniform(
            velocity_key,
            (self.action_size,),
            minval=-0.05,
            maxval=0.05,
        )
        data = state.data.replace(
            qpos=state.data.qpos.at[7:].add(joint_delta * leg_mask),
            qvel=state.data.qvel.at[6:].set(joint_velocity * leg_mask),
        )
        data = mjx.forward(self.mjx_model, data)
        info = {**state.info, "rng": rng}
        obs = self._observation(data, state.info["last_action"], state.info["phase"])
        return state.replace(data=data, obs=obs, info=info)

    def step(self, state, action):
        state = super().step(state, action)
        reward = self._standing_reward(state.data, state.info["last_action"])
        return state.replace(reward=reward)

    def _standing_reward(self, data, last_action) -> jax.Array:
        w, x, y, z = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        rz = w * w - x * x - y * y + z * z
        upright = (
            jnp.clip(rz, 0.0, 1.0) if self._sanitize_reward else jnp.maximum(0.0, rz)
        )
        dz = data.qpos[2] - self._home_qpos[2]
        height = jnp.maximum(0.0, 1.0 - (dz * dz) / 0.02)
        reward = upright + height + 0.1 - 0.001 * jnp.sum(last_action * last_action)
        if self._sanitize_reward:
            # AutoReset replaces terminal data and observations, but preserves
            # terminal reward.  Do not leak a contact-divergence NaN into PPO.
            reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        return reward


def _sequential_eval(env, action_fns, n_envs, n_steps, reset_key):
    """Evaluate named policies with one reusable sequential step kernel.

    No fused lax.scan: one compiled vmap(env.step) kernel is reused per step,
    which is the gfx1100-stable path. Returns mean per-step rewards.
    """
    from mujoco_playground import wrapper

    wrapped = wrapper.wrap_for_brax_training(env, episode_length=64, action_repeat=1)
    reset_fn = jax.jit(wrapped.reset)
    step_fn = jax.jit(wrapped.step)
    reset_keys = jax.random.split(reset_key, n_envs)
    results = {}
    for name, act_fn in action_fns.items():
        state = reset_fn(reset_keys)
        total = jnp.zeros(())
        for _ in range(n_steps):
            actions = act_fn(state.obs)
            state = step_fn(state, actions)
            total = total + jnp.mean(state.reward)
        results[name] = float(total / n_steps)
    return results


def _random_rollout_preflight(env, n_envs, n_steps, reset_key):
    """Check the exact wrapped transition values consumed by Brax PPO."""
    from mujoco_playground import wrapper

    wrapped = wrapper.wrap_for_brax_training(env, episode_length=64, action_repeat=1)
    state = jax.jit(wrapped.reset)(jax.random.split(reset_key, n_envs))
    step_fn = jax.jit(wrapped.step)
    action_key = reset_key
    nonfinite_rewards = jnp.zeros((), dtype=jnp.int32)
    nonfinite_observations = jnp.zeros((), dtype=jnp.int32)
    done_count = jnp.zeros((), dtype=jnp.int32)
    max_abs_observation = jnp.zeros(())

    for _ in range(n_steps):
        action_key, sample_key = jax.random.split(action_key)
        actions = jax.random.uniform(
            sample_key,
            (n_envs, env.action_size),
            minval=-1.0,
            maxval=1.0,
        )
        state = step_fn(state, actions)
        nonfinite_rewards += jnp.sum(~jnp.isfinite(state.reward))
        nonfinite_observations += jnp.sum(~jnp.isfinite(state.obs))
        done_count += jnp.sum(state.done.astype(jnp.int32))
        finite_obs = jnp.nan_to_num(state.obs, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs_observation = jnp.maximum(
            max_abs_observation, jnp.max(jnp.abs(finite_obs))
        )

    result = {
        "transitions": n_envs * n_steps,
        "done_count": int(done_count),
        "nonfinite_rewards": int(nonfinite_rewards),
        "nonfinite_observations": int(nonfinite_observations),
        "max_abs_observation": float(max_abs_observation),
    }
    print(
        "PREFLIGHT " + " ".join(f"{key}={value}" for key, value in result.items()),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-timesteps", type=int, default=512)
    parser.add_argument(
        "--training-steps-per-host-call",
        type=int,
        default=2,
        help="Maximum Brax training steps in each compiled host-loop call.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--foot-condim", type=int, choices=(1, 3, 4, 6))
    parser.add_argument("--unsafe-reward", action="store_true")
    parser.add_argument("--unbounded-observations", action="store_true")
    parser.add_argument("--no-reset-randomization", action="store_true")
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("ADAPTIVE_KL", "NONE"),
        default="NONE",
    )
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--preflight-steps", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--params-in")
    parser.add_argument("--params-out")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    from amd_robo.platform import _compat

    _compat.apply_brax_compat()
    from brax.io import model as brax_model
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    env = Go2Z1StandingEnv(
        foot_condim=args.foot_condim,
        sanitize_reward=not args.unsafe_reward,
        bound_observations=not args.unbounded_observations,
        randomize_reset=not args.no_reset_randomization,
    )
    if args.preflight_steps:
        preflight = _random_rollout_preflight(
            env,
            n_envs=64,
            n_steps=args.preflight_steps,
            reset_key=jax.random.PRNGKey(20260717),
        )
        if args.preflight_only:
            return int(
                preflight["nonfinite_rewards"] > 0
                or preflight["nonfinite_observations"] > 0
            )

    num_envs, batch_size, num_minibatches = 64, 16, 4
    unroll_length = 4
    num_timesteps = args.num_timesteps
    env_steps_per_training_step = batch_size * unroll_length * num_minibatches
    host_loop = plan_brax_host_loop(
        num_timesteps=num_timesteps,
        env_steps_per_training_step=env_steps_per_training_step,
        max_training_steps_per_call=args.training_steps_per_host_call,
    )
    max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0.0 else None
    print(
        f"standing smoke: num_envs={num_envs} unroll={unroll_length} "
        f"batch={batch_size} "
        f"mb={num_minibatches} num_timesteps={num_timesteps} "
        f"actual_timesteps={host_loop.actual_timesteps} "
        f"host_calls={host_loop.host_calls} "
        f"training_scan={host_loop.training_steps_per_call} "
        f"foot_condim={args.foot_condim or 6} learning_rate={args.learning_rate} "
        f"max_grad_norm={max_grad_norm} "
        f"safe_reward={not args.unsafe_reward} "
        f"bounded_obs={not args.unbounded_observations} "
        f"randomized_reset={not args.no_reset_randomization} "
        f"lr_schedule={args.learning_rate_schedule} desired_kl={args.desired_kl} "
        "run_evals=False",
        flush=True,
    )

    def progress(step, training_metrics):
        fields = []
        for key in (
            "training/walltime",
            "training/sps",
            "training/kl_mean",
            "training/learning_rate",
        ):
            if key in training_metrics:
                fields.append(f"{key}={training_metrics[key]}")
        print(
            f"TRAINING_PROGRESS step={step} {' '.join(fields)}",
            flush=True,
        )

    restore_params = brax_model.load_params(args.params_in) if args.params_in else None
    if args.params_in:
        print(f"PARAMS_LOADED path={args.params_in}", flush=True)
    metrics = {}
    print(f"TRAINING_START timesteps={num_timesteps}", flush=True)
    make_policy, params, metrics = ppo.train(
        environment=env,
        num_timesteps=num_timesteps,
        max_devices_per_host=1,
        num_envs=num_envs,
        episode_length=64,
        action_repeat=1,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        learning_rate_schedule_min_lr=1e-5,
        learning_rate_schedule_max_lr=args.learning_rate,
        desired_kl=args.desired_kl,
        entropy_cost=1e-3,
        discounting=0.97,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=2,
        max_grad_norm=max_grad_norm,
        normalize_observations=True,
        # With run_evals=False, Brax uses these iterations as a Python host
        # loop and carries the full TrainingState between compiled epochs.
        num_evals=host_loop.brax_num_evals,
        num_eval_envs=4,
        run_evals=False,
        progress_fn=progress,
        seed=0,
        restore_params=restore_params,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print("TRAINING_DONE", flush=True)
    for key in sorted(metrics):
        print(f"METRIC {key}: {metrics[key]}", flush=True)
    if args.params_out:
        brax_model.save_params(args.params_out, params)
        print(f"PARAMS_SAVED path={args.params_out}", flush=True)
    if args.skip_eval:
        print("STANDING_SMOKE_DONE eval=skipped", flush=True)
        return 0

    # Manual sequential eval (gfx1100-safe): trained policy vs zero-action
    # baseline, same randomized initial states and autoreset behavior.
    policy = make_policy(params, deterministic=True)
    trained_act = jax.jit(lambda obs: policy(obs, jax.random.PRNGKey(0))[0])

    def zero_act(obs):
        return jnp.zeros((obs.shape[0], env.action_size))

    eval_key = jax.random.PRNGKey(777)
    n_eval_envs, n_eval_steps = 64, 50
    print("evaluating (sequential, no scan)...", flush=True)
    eval_results = _sequential_eval(
        env,
        {"baseline": zero_act, "trained": trained_act},
        n_eval_envs,
        n_eval_steps,
        eval_key,
    )
    baseline = eval_results["baseline"]
    trained = eval_results["trained"]
    print(
        f"EVAL mean standing reward over {n_eval_steps} steps "
        f"({n_eval_envs} envs, same randomized initial states): "
        f"baseline(zero)={baseline:.4f} trained={trained:.4f} "
        f"delta={trained - baseline:+.4f}",
        flush=True,
    )
    print("STANDING_SMOKE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
