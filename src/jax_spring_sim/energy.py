r"""Potential energy and forces.

The defining idea of this package: **we never hand-derive forces.** We write
down a single scalar potential energy $U(\mathbf{x})$ and obtain the force field
by automatic differentiation,

$$\mathbf{F} = -\nabla_{\mathbf{x}}\, U(\mathbf{x}).$$

In a classical engine you would differentiate the spring term by hand, get the
$\hat{\mathbf{r}}\,(\lVert\mathbf{r}\rVert - L_0)$ expression, and re-derive it
every time you add a new energy term. Here :func:`jax.grad` does it exactly
(to machine precision, not a finite-difference approximation), so adding a new
potential is a one-line change with forces "for free".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .system import SpringSystem


def spring_energy(
    pos: jax.Array,
    edges: jax.Array,
    rest_length: jax.Array,
    stiffness: jax.Array,
) -> jax.Array:
    r"""Hookean spring energy $\tfrac12 \sum_e k_e (\lVert \mathbf{r}_e\rVert - L_e)^2$.

    Args:
        pos: Positions, shape ``(N, D)``.
        edges: Index pairs, shape ``(E, 2)``.
        rest_length: Rest lengths, shape ``(E,)``.
        stiffness: Spring constants, shape ``(E,)``.

    Returns:
        Scalar total spring energy.
    """
    delta = pos[edges[:, 0]] - pos[edges[:, 1]]
    length = jnp.linalg.norm(delta, axis=-1)
    return 0.5 * jnp.sum(stiffness * (length - rest_length) ** 2)


def gravity_energy(pos: jax.Array, mass: jax.Array, gravity: jax.Array) -> jax.Array:
    r"""Gravitational potential $-\sum_i m_i\, \mathbf{g}\cdot\mathbf{x}_i$.

    The sign is chosen so that $-\nabla U = m\,\mathbf{g}$, i.e. a downward
    ``gravity`` vector produces a downward force.
    """
    return -jnp.sum(mass * (pos @ gravity))


def total_energy(pos: jax.Array, system: SpringSystem) -> jax.Array:
    """Sum of all potential energy terms for ``system`` at configuration ``pos``."""
    return spring_energy(pos, system.edges, system.rest_length, system.stiffness) + gravity_energy(
        pos, system.mass, system.gravity
    )


# Gradient of the energy w.r.t. positions, evaluated once and reused. The force
# is the *negative* gradient; ``jax.grad`` differentiates argument 0 (``pos``).
_energy_grad = jax.grad(total_energy, argnums=0)


def compute_force(pos: jax.Array, system: SpringSystem) -> jax.Array:
    r"""Force on every particle, $\mathbf{F} = -\nabla_{\mathbf{x}} U$.

    Args:
        pos: Positions, shape ``(N, D)``.
        system: Parameters defining the potential.

    Returns:
        Force array, shape ``(N, D)``.
    """
    return -_energy_grad(pos, system)
