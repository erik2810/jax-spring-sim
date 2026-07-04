"""Performance benchmark: the case for JAX over a hand-written physics loop.

Compares four implementations of the *same* hanging-chain rollout:

1. Naive Python      — nested ``for`` loops over steps and edges (the textbook
   "iterative physics loop").
2. NumPy vectorised  — vectorised over edges/particles, but the time loop and
   kernel dispatch still run in the Python interpreter.
3. JAX (jit)         — the whole ``lax.scan`` rollout fused into one XLA kernel.
4. JAX (vmap)        — an ensemble of independent initial conditions advanced in
   a single batched kernel, reported as time-per-trajectory.

Run::

    uv run python benchmarks/benchmark.py
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import jax
import numpy as np

from jax_spring_sim import make_chain, perturb_initial, simulate_ensemble, simulate_final
from jax_spring_sim.system import State

N = 60
N_STEPS = 2000
DT = 5e-3
ENSEMBLE = 512


def naive_python(
    pos: list[list[float]],
    vel: list[list[float]],
    edges: list[tuple[int, int]],
    rest: list[float],
    k: list[float],
    mass: list[float],
    fixed: list[float],
    gravity: list[float],
    damping: float,
    dt: float,
    n_steps: int,
) -> list[list[float]]:
    n, d = len(pos), len(gravity)
    pos = [row[:] for row in pos]
    vel = [row[:] for row in vel]
    for _ in range(n_steps):
        force = [[mass[i] * gravity[c] for c in range(d)] for i in range(n)]
        for e, (i, j) in enumerate(edges):
            diff = [pos[i][c] - pos[j][c] for c in range(d)]
            length = math.sqrt(sum(c * c for c in diff)) + 1e-12
            mag = k[e] * (length - rest[e])
            for c in range(d):
                fc = mag * diff[c] / length
                force[i][c] -= fc
                force[j][c] += fc
        for i in range(n):
            if fixed[i]:
                vel[i] = [0.0] * d
                continue
            for c in range(d):
                vel[i][c] = (vel[i][c] + dt * force[i][c] / mass[i]) * damping
                pos[i][c] += dt * vel[i][c]
    return pos


def numpy_vectorised(
    pos: np.ndarray,
    vel: np.ndarray,
    edges: np.ndarray,
    rest: np.ndarray,
    k: np.ndarray,
    mass: np.ndarray,
    fixed: np.ndarray,
    gravity: np.ndarray,
    damping: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    pos, vel = pos.copy(), vel.copy()
    free = (1.0 - fixed)[:, None]
    inv_m = (1.0 / mass)[:, None]
    i, j = edges[:, 0], edges[:, 1]
    for _ in range(n_steps):
        force = mass[:, None] * gravity[None, :]
        diff = pos[i] - pos[j]
        length = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-12
        fc = (k[:, None] * (length - rest[:, None])) * diff / length
        np.add.at(force, i, -fc)
        np.add.at(force, j, fc)
        vel = (vel + dt * force * inv_m) * damping * free
        pos = pos + dt * vel
    return pos


def _time(fn: Callable[[], object], repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    state, system = make_chain(N)

    # Plain-Python inputs.
    pos_l = state.pos.tolist()
    vel_l = state.vel.tolist()
    edges_l = [tuple(e) for e in system.edges.tolist()]
    rest_l = system.rest_length.tolist()
    k_l = system.stiffness.tolist()
    mass_l = system.mass.tolist()
    fixed_l = system.fixed.tolist()
    grav_l = system.gravity.tolist()
    damp = float(system.damping)

    # NumPy inputs.
    np_args = (
        np.asarray(pos_l),
        np.asarray(vel_l),
        np.asarray(edges_l),
        np.asarray(rest_l),
        np.asarray(k_l),
        np.asarray(mass_l),
        np.asarray(fixed_l),
        np.asarray(grav_l),
        damp,
        DT,
        N_STEPS,
    )

    print(f"chain: N={N} particles, steps={N_STEPS}, dt={DT}, ensemble={ENSEMBLE}\n")

    # --- Naive Python --------------------------------------------------------
    t_py = _time(
        lambda: naive_python(
            pos_l, vel_l, edges_l, rest_l, k_l, mass_l, fixed_l, grav_l, damp, DT, N_STEPS
        ),
        repeat=1,
    )

    # --- NumPy ---------------------------------------------------------------
    t_np = _time(lambda: numpy_vectorised(*np_args), repeat=3)

    # --- JAX jit -------------------------------------------------------------
    def jit_run() -> None:
        simulate_final(state, system, DT, N_STEPS).pos.block_until_ready()

    t_compile = _time(jit_run, repeat=1)  # first call includes XLA compilation
    t_jax = _time(jit_run, repeat=5)

    # --- JAX vmap ensemble ---------------------------------------------------
    states0 = perturb_initial(state, jax.random.PRNGKey(0), ENSEMBLE, scale=0.05)

    def ens_run() -> None:
        simulate_ensemble(states0, system, DT, N_STEPS).pos.block_until_ready()

    ens_run()  # warm up compilation
    t_ens_total = _time(ens_run, repeat=3)
    t_ens_each = t_ens_total / ENSEMBLE

    # Serial JAX baseline for the ensemble (loop of single jitted rollouts).
    def serial_ensemble() -> None:
        for b in range(ENSEMBLE):
            simulate_final(
                State(states0.pos[b], states0.vel[b]), system, DT, N_STEPS
            ).pos.block_until_ready()

    t_serial = _time(serial_ensemble, repeat=1)

    # --- Report --------------------------------------------------------------
    rows = [
        ("naive Python loop", t_py, t_py / t_jax),
        ("NumPy vectorised", t_np, t_np / t_jax),
        ("JAX jit (1st call, w/ compile)", t_compile, t_compile / t_jax),
        ("JAX jit (steady state)", t_jax, 1.0),
    ]
    print(f"{'implementation':<34}{'time / rollout':>16}{'speedup':>12}")
    print("-" * 62)
    for name, t, speed in rows:
        print(f"{name:<34}{t * 1e3:>13.2f} ms{speed:>11.1f}x")

    print()
    print(f"{'ensemble of ' + str(ENSEMBLE) + ' trajectories':<34}{'time':>16}{'per-traj':>12}")
    print("-" * 62)
    print(
        f"{'JAX serial (Python loop of jit)':<34}{t_serial * 1e3:>13.2f} ms"
        f"{t_serial / ENSEMBLE * 1e3:>9.3f} ms"
    )
    print(
        f"{'JAX vmap (single batched kernel)':<34}{t_ens_total * 1e3:>13.2f} ms"
        f"{t_ens_each * 1e3:>9.3f} ms"
    )
    print(f"\nvmap speedup over serial JAX: {t_serial / t_ens_total:.1f}x")

    # Correctness cross-check (NumPy vs JAX final positions).
    ref = numpy_vectorised(*np_args)
    got = np.asarray(simulate_final(state, system, DT, N_STEPS).pos)
    print(f"max |NumPy - JAX| final position: {np.abs(ref - got).max():.2e}")


if __name__ == "__main__":
    main()
