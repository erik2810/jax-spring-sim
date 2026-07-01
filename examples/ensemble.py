"""Ensemble simulation: variance propagation over initial conditions via vmap.

A single ``jax.vmap`` turns a Monte-Carlo sweep over initial conditions into one
batched, JIT-compiled kernel. We pin both ends of an undamped cable, perturb the
starting positions, and visualise the band of configurations the ensemble
occupies — an uncertainty envelope around the mean shape.

Run::

    uv run --extra viz python examples/ensemble.py
"""

from __future__ import annotations

import jax
import matplotlib.pyplot as plt
import numpy as np
from _common import output_path

from jax_spring_sim import make_chain, perturb_initial, simulate_ensemble

N = 25
N_STEPS = 500
DT = 4e-3
BATCH = 256


def main() -> None:
    # Both ends pinned, undamped: perturbations persist so the ensemble keeps a
    # visible spread instead of collapsing onto one settled shape.
    state, system = make_chain(N, stiffness=400.0, damping=1.0)
    system = system._replace(fixed=system.fixed.at[-1].set(1.0))

    states0 = perturb_initial(state, jax.random.PRNGKey(0), BATCH, scale=0.5)
    finals = simulate_ensemble(states0, system, DT, N_STEPS)
    pos = np.asarray(finals.pos)  # (BATCH, N, 2)
    mean = pos.mean(axis=0)
    std = pos.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for b in range(BATCH):
        ax.plot(pos[b, :, 0], pos[b, :, 1], "-", color="C0", alpha=0.04)
    ax.plot(mean[:, 0], mean[:, 1], "-o", color="C3", ms=4, label="ensemble mean")
    ax.fill_between(
        mean[:, 0],
        mean[:, 1] - std[:, 1],
        mean[:, 1] + std[:, 1],
        color="C3",
        alpha=0.2,
        label="±1σ (y)",
    )
    ax.plot(
        [state.pos[0, 0], state.pos[-1, 0]],
        [state.pos[0, 1], state.pos[-1, 1]],
        "ks",
        ms=10,
        label="pinned ends",
    )

    ax.set_aspect("equal")
    ax.set_title(f"{BATCH} perturbed initial conditions, batched with vmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower center")

    path = output_path("ensemble.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
