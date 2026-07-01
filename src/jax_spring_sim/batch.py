r"""Batched simulation via :func:`jax.vmap`.

A common scientific workflow is an *ensemble*: run the same simulator over many
initial conditions (perturbations, Monte-Carlo samples, a parameter sweep) and
collect statistics. The naive approach is a Python loop over conditions; here we
``vmap`` the single-trajectory function, and XLA turns the batch dimension into
vectorised hardware work — one compiled kernel, no per-sample dispatch.

``vmap`` maps over pytree leaves along a chosen axis, so a *batch of states* is
simply a :class:`~jax_spring_sim.system.State` whose ``pos`` / ``vel`` carry a
leading axis of size ``B``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .dynamics import simulate_final
from .system import SpringSystem, State


@partial(jax.jit, static_argnames=("n_steps",))
def simulate_ensemble(
    states0: State,
    system: SpringSystem,
    dt: float,
    n_steps: int,
) -> State:
    """Roll a batch of ``B`` initial states forward under a shared ``system``.

    Args:
        states0: Batched initial state; ``pos`` / ``vel`` have shape ``(B, N, D)``.
        system: A single (unbatched) parameter set, broadcast across the batch.
        dt: Time step.
        n_steps: Number of integration steps.

    Returns:
        Batched final state with leading axis ``B``.
    """
    rollout = partial(simulate_final, system=system, dt=dt, n_steps=n_steps)
    return jax.vmap(rollout)(states0)


def perturb_initial(
    state0: State,
    key: jax.Array,
    batch: int,
    scale: float = 0.1,
) -> State:
    """Create ``batch`` jittered copies of ``state0`` for ensemble runs.

    Gaussian noise of standard deviation ``scale`` is added to the positions;
    velocities are replicated unchanged.
    """
    noise = scale * jax.random.normal(key, (batch, *state0.pos.shape))
    pos = state0.pos[None] + noise
    vel = jnp.broadcast_to(state0.vel[None], pos.shape)
    return State(pos=pos, vel=vel)
