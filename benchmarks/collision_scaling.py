"""Before/after scaling of the collision force: naive O(N^2) vs the hash grid.

Both methods compute the same thing: the gradient of the short-range collision
energy (the collision force) for a particle cloud held at *constant density*, so
the box grows with N and each particle keeps O(1) neighbours. That is the regime
where the spatial grid is genuinely linear and the all-pairs method is quadratic.

For each system size we time ``jax.grad`` of the collision energy, jitted and
with ``block_until_ready`` so the clock stops when the device is done, then fit
the empirical exponent ``alpha`` in ``t ~ N^alpha`` between the smallest and
largest size. Expect the naive method near ``alpha = 2`` and the hashed method
near ``alpha = 1``.

Run::

    uv run python benchmarks/collision_scaling.py            # full sweep
    uv run python benchmarks/collision_scaling.py --quick    # smaller, faster
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable

import jax

from jax_spring_sim import collision_energy, collision_energy_naive, make_cloth

# Constant-density cloud: box side grows as N^(1/3) at fixed spacing, so the
# expected neighbour count per particle is independent of N.
SPACING = 1.0
CUTOFF = 1.3 * SPACING  # r_c: a handful of neighbours per particle

SIZES_FULL = (512, 2048, 8192, 32768)
SIZES_QUICK = (256, 1024, 4096)
# Naive builds an (N, N, D) tensor; skip it past this size to avoid an OOM.
NAIVE_MAX = 8192


def _median_time(fn: Callable[[], object], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _cloud(n: int) -> jax.Array:
    """A random constant-density cloud, shape ``(n, 3)``."""
    side = SPACING * round(n ** (1 / 3)) + SPACING
    return jax.random.uniform(jax.random.PRNGKey(n), (n, 3)) * side


def _time_grad(fn: Callable, pos: jax.Array, system: object) -> float | None:
    """Median wall time of one jitted gradient evaluation, or None on OOM."""
    try:
        grad = jax.jit(jax.grad(fn))
        grad(pos, system).block_until_ready()  # compile + first run
        return _median_time(lambda: grad(pos, system).block_until_ready(), repeat=5)
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        print(f"    (skipped: {type(exc).__name__})")
        return None


def _alpha(sizes: list[int], times: list[float]) -> float:
    """Empirical exponent in t ~ N^alpha, from the first and last measured size."""
    return math.log(times[-1] / times[0]) / math.log(sizes[-1] / sizes[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller, faster sweep")
    args = parser.parse_args()
    sizes = SIZES_QUICK if args.quick else SIZES_FULL

    # collision_energy only reads the two collision fields off the system.
    _, system = make_cloth(2, 2, collision_stiffness=1.0, collision_radius=CUTOFF)

    print(f"device: {jax.devices()[0].platform.upper()}, cutoff r_c = {CUTOFF}, constant density\n")
    rows: list[tuple[int, float | None, float, float | None]] = []
    naive_sizes: list[int] = []
    naive_times: list[float] = []
    hash_sizes: list[int] = []
    hash_times: list[float] = []

    for n in sizes:
        pos = _cloud(n)
        print(f"N = {n}:")
        t_naive = _time_grad(collision_energy_naive, pos, system) if n <= NAIVE_MAX else None
        t_hash = _time_grad(collision_energy, pos, system)
        speedup = (t_naive / t_hash) if (t_naive and t_hash) else None
        rows.append((n, t_naive, t_hash, speedup))
        if t_naive is not None:
            naive_sizes.append(n)
            naive_times.append(t_naive)
        if t_hash is not None:
            hash_sizes.append(n)
            hash_times.append(t_hash)
        naive_str = f"{t_naive * 1e3:.2f} ms" if t_naive is not None else "-  (OOM)"
        print(f"    naive O(N^2): {naive_str}    hashed grid: {t_hash * 1e3:.2f} ms\n")

    print("=" * 68)
    print("Before / after: gradient of the collision energy (collision force)")
    print("=" * 68)
    print(f"{'N':>8} | {'naive O(N^2)':>14} | {'hashed grid':>14} | {'speedup':>8}")
    print(f"{'-' * 8}-+-{'-' * 14}-+-{'-' * 14}-+-{'-' * 8}")
    for n, t_naive, t_hash, speedup in rows:
        naive_c = f"{t_naive * 1e3:.2f} ms" if t_naive is not None else "-"
        hash_c = f"{t_hash * 1e3:.2f} ms" if t_hash is not None else "-"
        speed_c = f"{speedup:.1f}x" if speedup is not None else "-"
        print(f"{n:>8} | {naive_c:>14} | {hash_c:>14} | {speed_c:>8}")

    print()
    if len(naive_times) >= 2:
        print(f"naive  scaling: t ~ N^{_alpha(naive_sizes, naive_times):.2f}  (expected ~2.0)")
    if len(hash_times) >= 2:
        print(f"hashed scaling: t ~ N^{_alpha(hash_sizes, hash_times):.2f}  (expected ~1.0)")


if __name__ == "__main__":
    main()
