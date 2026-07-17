"""Probe MJX model stability over many control steps on a single ROCm GPU.

The Gate G1 model-stability check: load a robot MJCF, batch it across many
environments, step it for many control steps from the home keyframe, and check
that the state stays finite. This is the safe, reusable variant of the Go2
rollout probe.

Known platform limitation (observed 2026-07-16 on RGC W7900 / gfx1100,
ROCm 7.2.1, jax 0.10.2 + jax-rocm7-plugin/pjrt 0.10.2, mujoco-mjx 3.10.0):

    Fusing many steps into ONE jax.lax.scan over mjx.step segfaults on gfx1100
    once the scan is long enough (~1000 steps) AND the mjx.step body is complex
    (e.g. the 12-DoF Unitree Go2 with contacts). A trivial box model scans fine
    for the same length -- see platform.smoke._check_mjx -- so the crash is the
    combination of (long scan) x (complex mjx.step body) x (vmap) on this GPU.

    The same 1000 steps run as SEQUENTIAL single-step calls reusing one compiled
    vmap kernel -- what _probe_sequential below does -- stay finite, so the model
    and physics are sound. The crash is an XLA/ROCm limitation on long fused
    scans, not a model bug.

    Practical impact: PPO training is unaffected because Brax rollouts scan over
    a small unroll_length (tens of steps), not a whole episode. Keep
    unroll_length modest. Use --repro-scan-segfault to run the crashing fused-scan
    variant captured as upstream bug-report evidence (it may dump core).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

# Source layout: <repo>/src/amd_robo/platform/rollout_probe.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = REPO_ROOT / "assets" / "menagerie" / "unitree_go2" / "go2_mjx.xml"

# Reuse the G0 helpers so the finite/throughput logic stays identical to the
# accepted smoke test rather than diverging.
from amd_robo.platform.smoke import (  # noqa: E402
    _block_tree,
    _distribution_versions,
    _tree_is_finite,
)


def _load_batched(xml_path: Path, n_envs: int) -> tuple[Any, Any]:
    """Load the MJCF at the home keyframe and broadcast its data to n_envs."""
    import jax  # noqa: F401
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    model_cpu = mujoco.MjModel.from_xml_path(str(xml_path))
    if model_cpu.nkey > 0:
        model_cpu.qpos0[:] = model_cpu.key_qpos[0]

    model = mjx.put_model(model_cpu, impl="jax")
    data = mjx.make_data(model_cpu, impl="jax")
    # Hold the home pose: default ctrl=0 leaves Go2 torque-free and it collapses.
    if model_cpu.nkey > 0 and model_cpu.key_ctrl.shape[1] > 0:
        data = data.replace(ctrl=jnp.asarray(model_cpu.key_ctrl[0]))

    batched = jax.tree_util.tree_map(
        lambda value: jnp.array(
            jnp.broadcast_to(value, (n_envs,) + value.shape), copy=True
        ),
        data,
    )
    return model, batched


def _probe_sequential(
    model: Any, batched: Any, n_envs: int, n_steps: int, check_every: int
) -> dict[str, Any]:
    """Step n_steps times, reusing ONE compiled vmap kernel from a Python loop.

    This path is known-stable on gfx1100 and is what every new model should be
    checked with. Contrast with _repro_scan_segfault.
    """
    import jax
    from mujoco import mjx

    step_fn = jax.jit(jax.vmap(lambda one: mjx.step(model, one)))

    # First call compiles the kernel; every later call reuses it.
    start = time.perf_counter()
    current = step_fn(batched)
    _block_tree(current)
    compile_seconds = time.perf_counter() - start

    first_bad: int | None = None
    start = time.perf_counter()
    for step in range(1, n_steps):
        current = step_fn(current)
        if step == n_steps - 1 or step % check_every == 0:
            if not _tree_is_finite(current):
                first_bad = step
                break
    steady_seconds = time.perf_counter() - start
    steady_steps = n_steps - 1  # step 0 was counted under compile

    return {
        "mode": "sequential",
        "n_envs": n_envs,
        "n_steps": n_steps,
        "finite": first_bad is None,
        "first_nonfinite_step": first_bad,
        "compile_plus_first_step_seconds": round(compile_seconds, 3),
        "steady_state_seconds": round(steady_seconds, 3),
        # Diagnostic only: periodic finite checks add sync overhead and there is
        # no JIT-warmup/steady split here. Not formal benchmark evidence.
        "diagnostic_env_steps_per_second": round(n_envs * steady_steps / steady_seconds)
        if steady_seconds
        else None,
        "performance_evidence": False,
    }


def _repro_scan_segfault(
    model: Any, batched: Any, n_envs: int, n_steps: int
) -> dict[str, Any]:
    """Fuse all steps into one jax.lax.scan. Known to segfault on gfx1100.

    Kept as the minimal reproducer for the upstream XLA/ROCm bug report. On
    gfx1100 with the Go2 model this dumps core before returning; a simple box
    model returns finite for the same scan length.
    """
    import jax
    from mujoco import mjx

    @jax.jit
    def fused_rollout(init: Any) -> Any:
        def body(carry: Any, _):
            return jax.vmap(lambda one: mjx.step(model, one))(carry), None

        return jax.lax.scan(body, init, None, length=n_steps)[0]

    out = fused_rollout(batched)
    _block_tree(out)
    return {
        "mode": "fused_scan",
        "n_envs": n_envs,
        "n_steps": n_steps,
        "finite": _tree_is_finite(out),
        "note": "Reached here only if this GPU/build does not reproduce the segfault.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument(
        "--repro-scan-segfault",
        action="store_true",
        help="Also run the known-crashing fused-scan variant (may dump core).",
    )
    args = parser.parse_args(argv)

    if args.n_envs < 1 or args.n_steps < 1:
        parser.error("--n-envs and --n-steps must be positive")
    if not args.xml.exists():
        parser.error(
            f"model XML not found: {args.xml} "
            "(run scripts/fetch_menagerie.sh first)"
        )

    import jax

    report: dict[str, Any] = {
        "versions": _distribution_versions(),
        "default_backend": jax.default_backend(),
        "xml": str(args.xml),
        "checks": {},
    }

    try:
        model, batched = _load_batched(args.xml, args.n_envs)
        report["checks"]["sequential"] = _probe_sequential(
            model, batched, args.n_envs, args.n_steps, args.check_every
        )
        if args.repro_scan_segfault:
            import sys

            print(
                "WARNING: running fused-scan repro; expected to segfault on gfx1100.",
                file=sys.stderr,
                flush=True,
            )
            report["checks"]["fused_scan"] = _repro_scan_segfault(
                model, batched, args.n_envs, args.n_steps
            )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "FAILED"
        print(json.dumps(report, indent=2, default=str))
        return 1

    finite = bool(report["checks"]["sequential"]["finite"])
    report["status"] = "PASSED" if finite else "FAILED"
    print(json.dumps(report, indent=2, default=str))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
