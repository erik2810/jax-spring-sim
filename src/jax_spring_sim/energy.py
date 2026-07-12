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


def obstacle_friction_force(pos: jax.Array, vel: jax.Array, obstacles: Obstacles) -> jax.Array:
    r"""Regularised Coulomb friction at obstacle contacts (a dissipative force).

    Friction removes energy, so unlike every other force in this engine it cannot
    be the gradient of a potential; it enters the integrator beside damping, as a
    force that depends on the slip velocity. The law is the standard smooth
    regularisation of Coulomb's,

    $$F_t = -\mu\, k\,\text{pen}\; \frac{v_t}{\sqrt{\lVert v_t\rVert^2 + \epsilon^2}},$$

    where $k\,\text{pen}$ is the normal force magnitude (identical to the penalty
    energy's gradient, so normal and tangential contact stay consistent) and
    $v_t$ is the velocity component tangent to the contact. Well above the slip
    scale $\epsilon$ the magnitude saturates at the Coulomb limit
    $\mu \lVert F_n \rVert$; below it the law is stiff viscous drag, the smooth
    stand-in for static friction (bodies on a shallow incline creep at a tiny
    residual rate instead of locking exactly, the honest price of
    differentiability). Smooth in both ``pos`` and ``vel`` wherever there is
    contact, so ``jax.grad`` can differentiate through stick-slip trajectories,
    including w.r.t. $\mu$ itself.

    Time-step guidance: under explicit integration the near-zero-slip regime is
    a viscous term of coefficient $\mu k \,\text{pen} / \epsilon$, so keep
    $\Delta t \lesssim \epsilon\, m / (\mu k\, \text{pen})$, which is
    $\epsilon / (\mu g)$ for a body resting under gravity. Larger steps do not
    blow up (the force saturates at the Coulomb limit) but leave a residual slip
    jitter that grows with $\Delta t$: measured on the stopping-block test, the
    final speed rises from exactly 0 at ``dt=1e-3`` to about 2e-2 at ``dt=8e-3``.

    Args:
        pos: Positions, shape ``(N, D)``.
        vel: Velocities, shape ``(N, D)``.
        obstacles: Contact geometry, stiffness, and friction parameters.

    Returns:
        Tangential friction force per particle, shape ``(N, D)``.
    """
    mu = obstacles.friction
    k = obstacles.stiffness
    eps2 = obstacles.friction_smoothing**2
    force = jnp.zeros_like(pos)

    if obstacles.n_planes > 0:
        n = obstacles.plane_normal  # (P, D)
        pen = jnp.maximum(0.0, obstacles.plane_offset[None, :] - pos @ n.T)  # (N, P)
        v_n = vel @ n.T  # (N, P) normal slip component
        v_t = vel[:, None, :] - v_n[:, :, None] * n[None, :, :]  # (N, P, D)
        speed = jnp.sqrt(jnp.sum(v_t * v_t, axis=-1) + eps2)  # (N, P)
        force = force - jnp.sum((mu * k * pen / speed)[:, :, None] * v_t, axis=1)

    if obstacles.n_spheres > 0:
        diff = pos[:, None, :] - obstacles.sphere_center[None, :, :]  # (N, S, D)
        dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + 1e-12)  # (N, S)
        n_hat = diff / dist[:, :, None]  # (N, S, D) outward contact normals
        pen = jnp.maximum(0.0, obstacles.sphere_radius[None, :] - dist)  # (N, S)
        v_n = jnp.sum(vel[:, None, :] * n_hat, axis=-1)  # (N, S)
        v_t = vel[:, None, :] - v_n[:, :, None] * n_hat  # (N, S, D)
        speed = jnp.sqrt(jnp.sum(v_t * v_t, axis=-1) + eps2)  # (N, S)
        force = force - jnp.sum((mu * k * pen / speed)[:, :, None] * v_t, axis=1)

    return force


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
