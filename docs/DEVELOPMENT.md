# Development workflow

## One task at a time

Each task must state its input, output, files in scope, and acceptance command.
Before implementation, read `PROJECT_RULES.md`, `ARCHITECTURE.md`, and the
relevant committed config. After implementation, run focused tests and then the
full CPU-safe suite.

## Required checks

```bash
python -m pytest
python -m ruff check .
```

ROCm tests are separate because local CPU success is not RGC evidence:

```bash
python scripts/system_info.py --config configs/smoke.yaml \
  --image-name TODO_FROM_RGC --image-digest TODO_FROM_RGC
python scripts/smoke_test.py
```

## Gate workflow

For each gate, commit code and configs, record the commit in the evidence, run
the gate checks, update the report, merge to `main`, and create the matching tag.
Do not mark a gate complete from documentation or partial diagnostics alone.

GitHub Actions runs only lint and CPU-safe contract tests. A green CI job is not
ROCm evidence and cannot replace the RGC Gate G0 smoke test.
