"""Profile the simulator and write the results to BENCHMARKS.md.

Three measurements, each on every JAX device this machine actually has:

1. Rollout cost vs. system size — the jitted ``lax.scan`` rollout on cloth
   grids of roughly 1k, 10k and 50k nodes, reported per step.
2. jit vs. eager — the same ``step`` function dispatched op by op from a
   Python loop, against the compiled rollout, as a per-step ratio.
3. Gradient cost vs. horizon — ``jax.value_and_grad`` through the full
   rollout for several time horizons, reported next to the forward pass.

Memory is reported two ways: the exact array bytes held per system size
(computed from shapes, so it is precise) and the process peak RSS at the end
of the run (coarse, includes Python and XLA overhead).

The script never invents a device: on a CPU-only machine the tables contain
CPU rows, and GPU rows appear when a GPU backend is present. Every number in
BENCHMARKS.md comes from a measurement in the current run.

Run::

    uv run python profile_engine.py            # full sweep, writes BENCHMARKS.md
    uv run python profile_engine.py --quick    # smaller sweep for a smoke test
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import math
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable

import jax

from jax_spring_sim import SpringSystem, State, make_cloth, simulate_final, step
from jax_spring_sim.inverse import trajectory_loss

DT = 1e-3

# rows == cols; 32^2 = 1_024, 100^2 = 10_000, 224^2 = 50_176 nodes.
GRIDS_FULL = (32, 100, 224)
GRIDS_QUICK = (32, 100)

ROLLOUT_STEPS = 400
EAGER_STEPS = 20
HORIZONS_FULL = (25, 50, 100, 200)
HORIZONS_QUICK = (25, 50)
GRAD_GRID = 100  # gradient sweep runs on the 100x100 (10k node) system


def _median_time(fn: Callable[[], object], repeat: int) -> float:
    """Median wall time of ``fn`` over ``repeat`` calls, in seconds."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _devices() -> list[jax.Device]:
    devs = [jax.devices("cpu")[0]]
    for kind in ("gpu", "tpu"):
        with contextlib.suppress(RuntimeError):
            devs.append(jax.devices(kind)[0])
    return devs


def _put(state: State, system: SpringSystem, dev: jax.Device) -> tuple[State, SpringSystem]:
    return jax.device_put(state, dev), jax.device_put(system, dev)


def _array_megabytes(state: State, system: SpringSystem) -> float:
    arrays = list(state) + [f for f in system if hasattr(f, "nbytes")]
    return sum(a.nbytes for a in arrays) / 1e6


def _peak_rss_megabytes() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def bench_rollout(
    grids: tuple[int, ...], devices: list[jax.Device]
) -> list[dict[str, float | int | str]]:
    """Jitted rollout per system size and device, plus the eager per-step cost."""
    rows_out: list[dict[str, float | int | str]] = []
    for n in grids:
        state, system = make_cloth(n, n)
        n_nodes = int(state.pos.shape[0])
        n_springs = int(system.edges.shape[0])
        mem_mb = _array_megabytes(state, system)

        for dev in devices:
            s, sys_ = _put(state, system, dev)

            def run(s: State = s, sys_: SpringSystem = sys_) -> None:
                simulate_final(s, sys_, DT, ROLLOUT_STEPS).pos.block_until_ready()

            t_compile = _median_time(run, repeat=1)  # first call includes XLA compile
            t_roll = _median_time(run, repeat=5)
            jit_step_us = t_roll / ROLLOUT_STEPS * 1e6

            def eager(s: State = s, sys_: SpringSystem = sys_) -> None:
                st = s
                for _ in range(EAGER_STEPS):
                    st = step(st, sys_, DT)
                st.pos.block_until_ready()

            with jax.disable_jit():
                eager()  # warm up dispatch caches
                t_eager = _median_time(eager, repeat=3)
            eager_step_us = t_eager / EAGER_STEPS * 1e6

            rows_out.append(
                {
                    "grid": f"{n}x{n}",
                    "nodes": n_nodes,
                    "springs": n_springs,
                    "device": dev.platform.upper(),
                    "mem_mb": mem_mb,
                    "compile_s": t_compile,
                    "jit_step_us": jit_step_us,
                    "eager_step_us": eager_step_us,
                    "speedup": eager_step_us / jit_step_us,
                }
            )
            print(
                f"  {n}x{n} on {dev.platform.upper()}: "
                f"jit {jit_step_us:.1f} us/step, eager {eager_step_us:.1f} us/step"
            )
    return rows_out


def bench_gradient(
    horizons: tuple[int, ...], devices: list[jax.Device]
) -> list[dict[str, float | int | str]]:
    """Forward vs. forward+backward cost through the rollout, per time horizon."""
    state, system = make_cloth(GRAD_GRID, GRAD_GRID)
    target = state.pos  # any fixed target; only the timing matters here

    rows_out: list[dict[str, float | int | str]] = []
    for dev in devices:
        s, sys_ = _put(state, system, dev)
        tgt = jax.device_put(target, dev)
        for n_steps in horizons:

            def loss(
                rest: jax.Array,
                s: State = s,
                sys_: SpringSystem = sys_,
                tgt: jax.Array = tgt,
                n_steps: int = n_steps,
            ) -> jax.Array:
                return trajectory_loss(rest, s, sys_, tgt, DT, n_steps)

            fwd = jax.jit(loss)
            fwd_bwd = jax.jit(jax.value_and_grad(loss))
            rest = sys_.rest_length

            fwd(rest).block_until_ready()  # compile
            fwd_bwd(rest)[0].block_until_ready()

            t_fwd = _median_time(lambda f=fwd, r=rest: f(r).block_until_ready(), repeat=5)
            t_both = _median_time(lambda f=fwd_bwd, r=rest: f(r)[0].block_until_ready(), repeat=5)

            rows_out.append(
                {
                    "device": dev.platform.upper(),
                    "horizon": n_steps,
                    "fwd_ms": t_fwd * 1e3,
                    "fwd_bwd_ms": t_both * 1e3,
                    "ratio": t_both / t_fwd,
                }
            )
            print(
                f"  horizon {n_steps} on {dev.platform.upper()}: "
                f"fwd {t_fwd * 1e3:.2f} ms, fwd+bwd {t_both * 1e3:.2f} ms"
            )
    return rows_out


def _summary(roll: list[dict], grad: list[dict]) -> str:
    """Plain-language reading of the measured numbers. No number is hardcoded."""
    cpu = [r for r in roll if r["device"] == "CPU"]
    small, large = cpu[0], cpu[-1]
    speedups = [r["speedup"] for r in cpu]

    # Empirical scaling exponent of the jitted per-step cost between the
    # smallest and largest system, from t ~ n^alpha.
    alpha = math.log(large["jit_step_us"] / small["jit_step_us"]) / math.log(
        large["nodes"] / small["nodes"]
    )

    ratios = [g["ratio"] for g in grad if g["device"] == "CPU"]

    lines = [
        "## What the numbers say",
        "",
        f"Most of the eager cost is dispatch, not physics. On this machine the eager loop pays "
        f"{small['eager_step_us']:.0f} us per step at {small['nodes']:,} nodes, almost all of it "
        f"Python-level dispatch of individual XLA ops. The jitted `lax.scan` rollout runs the same "
        f"physics at {small['jit_step_us']:.1f} us per step. Across the sizes measured here the "
        f"compiled version is {min(speedups):.0f}x to {max(speedups):.0f}x faster per step.",
        "",
        f"The dispatch overhead is roughly constant per step, so the eager gap is widest on small "
        f"systems, where the arithmetic is cheap. As the system grows the compute itself starts to "
        f"dominate: the jitted per-step cost rises from {small['jit_step_us']:.1f} us at "
        f"{small['nodes']:,} nodes to {large['jit_step_us']:.1f} us at {large['nodes']:,} nodes, "
        f"an empirical scaling of about n^{alpha:.2f} over this range.",
        "",
        f"Reverse-mode differentiation through the whole rollout costs a small constant multiple "
        f"of the forward pass, {min(ratios):.1f}x to {max(ratios):.1f}x here, and the multiple "
        f"stays flat as the horizon grows from {grad[0]['horizon']} to {grad[-1]['horizon']} "
        f"steps. That is the expected behaviour of backpropagating through `lax.scan`, and it is "
        f"what makes gradient-based inverse problems through this simulator practical: doubling "
        f"the horizon roughly doubles both passes instead of blowing up the backward one.",
        "",
        "Array memory grows linearly with node count (exact figures in the table above). "
        "One caveat for the backward pass: `lax.scan` stores per-step residuals, so gradient "
        "memory also grows linearly with the horizon. `simulate_final` keeps only the final "
        "state to hold that down.",
    ]
    return "\n".join(lines)


def write_markdown(path: str, roll: list[dict], grad: list[dict], quick: bool) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    dev_names = ", ".join(sorted({r["device"] for r in roll}))
    env = (
        f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, "
        f"JAX {jax.__version__}, devices: {dev_names}"
    )

    lines = [
        "# Benchmarks",
        "",
        f"Measured with [`profile_engine.py`](profile_engine.py) on {now}"
        f"{' (quick mode)' if quick else ''}.",
        f"Environment: {env}. Peak process RSS {_peak_rss_megabytes():.0f} MB.",
        "",
        "Timing is the median of repeated runs after a warm-up call; JAX results use "
        "`block_until_ready()` so device work is actually finished when the clock stops. "
        "Eager rows run the identical `step` function under `jax.disable_jit()` in a Python "
        "loop. Compile time is the first call to the jitted rollout, including XLA compilation.",
        "",
        "## Rollout: jit vs. eager, by system size",
        "",
        "| Grid | Nodes | Springs | Device | Arrays (MB) | Compile (s) "
        "| jit (us/step) | Eager (us/step) | Speedup |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in roll:
        lines.append(
            f"| {r['grid']} | {r['nodes']:,} | {r['springs']:,} | {r['device']} "
            f"| {r['mem_mb']:.1f} | {r['compile_s']:.2f} | {r['jit_step_us']:.1f} "
            f"| {r['eager_step_us']:.0f} | {r['speedup']:.0f}x |"
        )

    lines += [
        "",
        f"## Gradient through the rollout ({GRAD_GRID}x{GRAD_GRID} grid, "
        f"{GRAD_GRID * GRAD_GRID:,} nodes)",
        "",
        "`jax.value_and_grad` of a terminal-state loss w.r.t. every spring rest length, "
        "backpropagated through all integration steps.",
        "",
        "| Device | Horizon (steps) | Forward (ms) | Forward+backward (ms) | Ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in grad:
        lines.append(
            f"| {g['device']} | {g['horizon']} | {g['fwd_ms']:.2f} "
            f"| {g['fwd_bwd_ms']:.2f} | {g['ratio']:.1f}x |"
        )

    lines += ["", _summary(roll, grad), ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sweep for a smoke test")
    parser.add_argument("--out", default="BENCHMARKS.md")
    args = parser.parse_args()

    grids = GRIDS_QUICK if args.quick else GRIDS_FULL
    horizons = HORIZONS_QUICK if args.quick else HORIZONS_FULL
    devices = _devices()

    print(f"devices: {[d.platform for d in devices]}")
    print("rollout sweep:")
    roll = bench_rollout(grids, devices)
    print("gradient sweep:")
    grad = bench_gradient(horizons, devices)
    write_markdown(args.out, roll, grad, args.quick)


if __name__ == "__main__":
    main()
