#!/usr/bin/env python3
"""Run the Gate G1 Go2 MJX model-stability probe from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amd_robo.platform.rollout_probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
