# Benchmark evidence

- `raw/`: immutable CSV/JSON evidence, commands, config hashes, and system info.
- `figures/`: versioned evaluator-facing plots derived from `raw/`.

Warmup and steady-state timing are separate. Every timed JAX result calls
`block_until_ready()`. Report every attempted batch size, including OOM and NaN.
