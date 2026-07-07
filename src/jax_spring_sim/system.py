"""Core state and parameter containers for the particle-spring system.

Everything is a :class:`typing.NamedTuple`, which JAX registers as a *pytree*
automatically. That gives us two things for free:

1. Functional updates via ``state._replace(pos=...)`` — no in-place mutation,
   so the whole pipeline stays compatible with ``jax.jit`` / ``jax.grad``.
2. Transparent flattening, so ``jax.vmap`` / ``jax.tree_util`` can map over a
   batch of states or differentiate w.r.t. any leaf without bespoke plumbing.

Convention: ``pos`` / ``vel`` have shape ``(N, D)`` where ``N`` is the particle
count and ``D`` the spatial dimension (2 or 3). Edge arrays have shape
``(E, 2)`` with integer indices into the particle arrays.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class State(NamedTuple):
    """Time-varying state of the system.

    Attributes:
        pos: Particle positions, shape ``(N, D)``.
        vel: Particle velocities, shape ``(N, D)``.
    """

    pos: jax.Array
    vel: jax.Array


class SpringSystem(NamedTuple):
    """Static (per-rollout) parameters of a Hookean spring network.

    These are the quantities an inverse-design problem typically optimises
    over (``rest_length``, ``stiffness``, ...). Keeping them in a flat pytree
    means ``jax.grad(loss)(system)`` differentiates the simulation w.r.t. every
    physical parameter at once.

    Attributes:
        edges: Integer index pairs, shape ``(E, 2)``.
        rest_length: Natural length $L_0$ of each spring, shape ``(E,)``.
        stiffness: Hooke constant $k$ of each spring, shape ``(E,)``.
        mass: Per-particle mass $m$, shape ``(N,)``.
        fixed: Pin mask in ``{0.0, 1.0}``, shape ``(N,)``. A value of ``1.0``
            clamps that particle (Dirichlet boundary).
        gravity: Uniform acceleration vector $\\mathbf{g}$, shape ``(D,)``.
        damping: Per-step multiplicative velocity damping in ``(0, 1]``.
        collision_stiffness: Scalar penalty stiffness $k_\\text{col}$ for the
            short-range repulsion in :mod:`.spatial`. Defaults to ``0.0``, which
            disables collision and leaves every existing rollout unchanged.
        collision_radius: Scalar interaction cutoff $r_c$ (also the grid cell
            side). Only relevant when ``collision_stiffness > 0``.
    """

    edges: jax.Array
    rest_length: jax.Array
    stiffness: jax.Array
    mass: jax.Array
    fixed: jax.Array
    gravity: jax.Array
    damping: jax.Array
    collision_stiffness: jax.Array = jnp.asarray(0.0)
    collision_radius: jax.Array = jnp.asarray(1.0)

    @property
    def n_particles(self) -> int:
        return self.mass.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def dim(self) -> int:
        return self.gravity.shape[0]
