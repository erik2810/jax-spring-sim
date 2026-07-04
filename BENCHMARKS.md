# Benchmarks

Measured with [`benchmarks/profile_engine.py`](benchmarks/profile_engine.py) on 2026-07-03 22:33.
Environment: Darwin arm64, Python 3.13.9, JAX 0.10.2, devices: CPU. Peak process RSS 1142 MB.

Timing is the median of repeated runs after a warm-up call; JAX results use `block_until_ready()` so device work is actually finished when the clock stops. Eager rows run the identical `step` function under `jax.disable_jit()` in a Python loop. Compile time is the first call to the jitted rollout, including XLA compilation.

## Rollout: jit vs. eager, by system size

| Grid | Nodes | Springs | Device | Arrays (MB) | Compile (s) | jit (us/step) | Eager (us/step) | Speedup |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 32x32 | 1,024 | 3,906 | CPU | 0.1 | 0.16 | 336.9 | 16071 | 48x |
| 100x100 | 10,000 | 39,402 | CPU | 1.0 | 0.33 | 816.9 | 16384 | 20x |
| 224x224 | 50,176 | 199,362 | CPU | 4.8 | 1.05 | 2686.1 | 17651 | 7x |

## Gradient through the rollout (100x100 grid, 10,000 nodes)

`jax.value_and_grad` of a terminal-state loss w.r.t. every spring rest length, backpropagated through all integration steps.

| Device | Horizon (steps) | Forward (ms) | Forward+backward (ms) | Ratio |
|---|---:|---:|---:|---:|
| CPU | 25 | 12.86 | 30.50 | 2.4x |
| CPU | 50 | 26.15 | 60.30 | 2.3x |
| CPU | 100 | 51.10 | 120.88 | 2.4x |
| CPU | 200 | 102.72 | 243.38 | 2.4x |

## What the numbers say

Most of the eager cost is dispatch, not physics. On this machine the eager loop pays 16071 us per step at 1,024 nodes, almost all of it Python-level dispatch of individual XLA ops. The jitted `lax.scan` rollout runs the same physics at 336.9 us per step. Across the sizes measured here the compiled version is 7x to 48x faster per step.

The dispatch overhead is roughly constant per step, so the eager gap is widest on small systems, where the arithmetic is cheap. As the system grows the compute itself starts to dominate: the jitted per-step cost rises from 336.9 us at 1,024 nodes to 2686.1 us at 50,176 nodes, an empirical scaling of about n^0.53 over this range.

Reverse-mode differentiation through the whole rollout costs a small constant multiple of the forward pass, 2.3x to 2.4x here, and the multiple stays flat as the horizon grows from 25 to 200 steps. That is the expected behaviour of backpropagating through `lax.scan`, and it is what makes gradient-based inverse problems through this simulator practical: doubling the horizon roughly doubles both passes instead of blowing up the backward one.

Array memory grows linearly with node count (exact figures in the table above). One caveat for the backward pass: `lax.scan` stores per-step residuals, so gradient memory also grows linearly with the horizon. `simulate_final` keeps only the final state to hold that down.
