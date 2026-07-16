# Experiment records

Each formal experiment uses `YYYY-MM-DD-task-name/` and commits only:

- `config.yaml`: the exact run input;
- `manifest.json`: commit, seed, system-info file, command, and artifacts;
- `result.md`: one concise outcome and the decision it supports.

Raw logs, caches, checkpoints, videos, and tracking directories remain ignored.
Formal benchmark CSV belongs in `benchmarks/raw/`, not here.
