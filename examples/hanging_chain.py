"""Forward simulation: a cable pinned at both ends settling into a catenary.

The chain starts as a straight horizontal line and sags under gravity until it
reaches the classic catenary equilibrium. Snapshots are coloured by time to
show the jit-compiled ``lax.scan`` rollout relaxing toward steady state.

Run::

    uv run --extra viz python examples/hanging_chain.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import output_path

from jax_spring_sim import make_chain, simulate

N = 25
N_STEPS = 2500
DT = 4e-3
N_FRAMES = 9


def main() -> None:
    # Pin both ends (left from make_chain, right added here) so gravity pulls
    # the cable into a catenary instead of a free-falling pendulum.
    state, system = make_chain(N, stiffness=400.0, damping=0.99)
    system = system._replace(fixed=system.fixed.at[-1].set(1.0))

    _, traj = simulate(state, system, DT, N_STEPS, save_every=N_STEPS // N_FRAMES)
    frames = np.asarray(traj.pos)  # (N_FRAMES, N, 2)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    cmap = plt.cm.viridis
    for f in range(frames.shape[0]):
        c = cmap(f / (frames.shape[0] - 1))
        ax.plot(frames[f, :, 0], frames[f, :, 1], "-o", ms=3, color=c, alpha=0.85)
    ax.plot(
        [state.pos[0, 0], state.pos[-1, 0]],
        [state.pos[0, 1], state.pos[-1, 1]],
        "ks",
        ms=10,
        label="pinned ends",
    )

    ax.set_aspect("equal")
    ax.set_title(f"Cable settling into a catenary ({N} particles, {N_STEPS} steps)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper center")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, label="time →", fraction=0.046)

    path = output_path("hanging_chain.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
