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

from .system import Obstacles, SpringSystem


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


def obstacle_energy(pos: jax.Array, obstacles: Obstacles) -> jax.Array:
    r"""Penalty potential of rigid boundary obstacles (the differentiable contact model).

    Each constraint violation is penalised quadratically,
    $U = \tfrac12 k \sum \text{pen}^2$ with penetration depth
    $\text{pen} = \max(0, \cdot)$, so the potential is $C^1$: the reaction force
    $-\nabla U = k\,\text{pen}\,\hat n$ is exact, points along the contact
    normal, and ramps smoothly from zero at first touch. This is the classic
    penalty method for inequality constraints, expressed as one more energy term
    so autodiff produces the contact forces like every other force here.

    Args:
        pos: Positions, shape ``(N, D)``.
        obstacles: Half-space and sphere obstacles; see
            :class:`~jax_spring_sim.system.Obstacles`.

    Returns:
        Scalar obstacle energy (``0.0`` when there are no obstacles).
    """
    e = jnp.asarray(0.0, dtype=pos.dtype)
    # Obstacle counts are static shapes under jit, so empty terms trace to nothing.
    if obstacles.n_planes > 0:
        # Signed height above each plane: (N, P); negative values penetrate.
        s = pos @ obstacles.plane_normal.T - obstacles.plane_offset[None, :]
        pen = jnp.maximum(0.0, -s)
        e = e + 0.5 * obstacles.stiffness * jnp.sum(pen**2)
    if obstacles.n_spheres > 0:
        diff = pos[:, None, :] - obstacles.sphere_center[None, :, :]  # (N, S, D)
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)  # (N, S), safe at centre
        pen = jnp.maximum(0.0, obstacles.sphere_radius[None, :] - dist)
        e = e + 0.5 * obstacles.stiffness * jnp.sum(pen**2)
    return e


def total_energy(pos: jax.Array, system: SpringSystem, collide: bool = False) -> jax.Array:
    r"""Sum of all potential energy terms for ``system`` at configuration ``pos``.

    Args:
        pos: Positions, shape ``(N, D)``.
        system: Parameters defining the potential.
        collide: When ``True``, add the short-range collision potential from
            :mod:`.spatial` (O(N) via the hash grid). ``collide`` is a static
            Python flag, so when ``False`` the grid is never traced and the
            result is identical to the collision-free engine.

    Returns:
        Scalar total potential energy.
    """
    e = spring_energy(pos, system.edges, system.rest_length, system.stiffness) + gravity_energy(
        pos, system.mass, system.gravity
    )
    e = e + obstacle_energy(pos, system.obstacles)
    if collide:
        # Imported lazily to avoid a module import cycle (spatial imports system).
        from .spatial import collision_energy

        e = e + collision_energy(pos, system)
    return e


def compute_force(pos: jax.Array, system: SpringSystem, collide: bool = False) -> jax.Array:
    r"""Force on every particle, $\mathbf{F} = -\nabla_{\mathbf{x}} U$.

    Args:
        pos: Positions, shape ``(N, D)``.
        system: Parameters defining the potential.
        collide: Include the collision term (see :func:`total_energy`).

    Returns:
        Force array, shape ``(N, D)``.
    """
    return -jax.grad(total_energy, argnums=0)(pos, system, collide)
