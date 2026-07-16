#!/usr/bin/env python3
"""Collect a reproducibility fingerprint: backend, devices, versions, GPU info.

Writes JSON to benchmarks/system_info_<date>.json and prints it. Use it to attach
the exact hardware/software context to every benchmark and evaluation run (required
by the benchmark discipline in CLAUDE.md).

Run:  python scripts/system_info.py

Phase 0 TODO: this runs defensively. After RGC bring-up, confirm the amd-smi /
rocm-smi parsing matches the instance and hard-pin versions.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:  # noqa: BLE001
        return f"NOT INSTALLED ({type(e).__name__})"


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
        return (out.stdout or out.stderr or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"<failed: {type(e).__name__}>"


def collect() -> dict:
    info: dict = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env_ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "env_HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "versions": {
            "mujoco": _version("mujoco"),
            "brax": _version("brax"),
            "jax": _version("jax"),
            "jaxlib": _version("jaxlib"),
        },
        "amd-smi": _run(["amd-smi", "monitor", "--json"]) or _run(["amd-smi"]),
        "rocm-smi": _run(["rocm-smi", "--showproductname", "--json"]),
        "rocminfo(gpus)": _run(["rocminfo"]),
    }
    try:
        import jax

        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as e:  # noqa: BLE001
        info["jax"] = f"<failed: {type(e).__name__}: {e}>"
    return info


def main() -> int:
    info = collect()
    print(json.dumps(info, indent=2))
    out_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = os.path.abspath(os.path.join(out_dir, f"system_info_{date}.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("\nsaved:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
