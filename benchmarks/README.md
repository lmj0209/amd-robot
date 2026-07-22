# Benchmark evidence

- `raw/`: immutable CSV/JSON evidence, commands, config hashes, and system info.
- `figures/`: versioned evaluator-facing plots derived from `raw/`.

`raw/push_chunked_cross_validation_2026-07-21.json` records the two-seed
sequential-versus-chunked implementation gate. It is explicitly
cross-validation evidence, not the held-out 100-episode qualification result.

Warmup and steady-state timing are separate. Every timed JAX result calls
`block_until_ready()`. Report every attempted batch size, including OOM and NaN.
