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


class Obstacles(NamedTuple):
    """Rigid collision obstacles, enforced by a differentiable penalty energy.

    Two primitive kinds compose into engineering environments: half-spaces
    (ground planes, rigid walls; the allowed region is ``normal . x >= offset``)
    and keep-out spheres (custom bounding volumes). Both enter the dynamics as a
    C1 penalty potential ``0.5 * stiffness * penetration**2`` in
    :func:`.energy.obstacle_energy`, so the normal reaction force
    ``stiffness * penetration * normal`` is produced by ``jax.grad`` exactly like
    every other force, and it ramps smoothly from zero at first contact.

    Being a pytree of arrays, obstacles are differentiable parameters too:
    ``jax.grad`` can optimise a wall position or a sphere radius through the
    rollout, the same way it optimises rest lengths.

    Attributes:
        plane_normal: Unit inward normals, shape ``(P, D)``.
        plane_offset: Plane offsets, shape ``(P,)``; allowed side is
            ``normal . x >= offset``.
        sphere_center: Keep-out sphere centres, shape ``(S, D)``.
        sphere_radius: Keep-out sphere radii, shape ``(S,)``.
        stiffness: Scalar penalty stiffness of the contact.
    """

    plane_normal: jax.Array
    plane_offset: jax.Array
    sphere_center: jax.Array
    sphere_radius: jax.Array
    stiffness: jax.Array

    @classmethod
    def none(cls, dim: int = 3) -> Obstacles:
        """No obstacles; adds exactly zero energy and zero cost."""
        return cls(
            plane_normal=jnp.zeros((0, dim)),
            plane_offset=jnp.zeros((0,)),
            sphere_center=jnp.zeros((0, dim)),
            sphere_radius=jnp.zeros((0,)),
            stiffness=jnp.asarray(0.0),
        )

    @classmethod
    def build(
        cls,
        *,
        planes: list[tuple[tuple[float, ...], float]] | None = None,
        spheres: list[tuple[tuple[float, ...], float]] | None = None,
        stiffness: float = 1_000.0,
        dim: int = 3,
    ) -> Obstacles:
        """Assemble obstacles from plain Python lists.

        Args:
            planes: ``(normal, offset)`` pairs; normals are normalised here and
                point into the allowed half-space (``normal . x >= offset``).
            spheres: ``(center, radius)`` keep-out pairs.
            stiffness: Penalty stiffness shared by all obstacles.
            dim: Spatial dimension, used when a list is empty.

        Returns:
            An :class:`Obstacles` pytree ready to attach to a system.
        """
        if planes:
            normal = jnp.asarray([n for n, _ in planes], dtype=jnp.float32) * 1.0
            normal = normal / jnp.linalg.norm(normal, axis=1, keepdims=True)
            offset = jnp.asarray([o for _, o in planes], dtype=normal.dtype)
        else:
            normal = jnp.zeros((0, dim))
            offset = jnp.zeros((0,))
        if spheres:
            center = jnp.asarray([c for c, _ in spheres], dtype=jnp.float32) * 1.0
            radius = jnp.asarray([r for _, r in spheres], dtype=center.dtype)
        else:
            center = jnp.zeros((0, dim))
            radius = jnp.zeros((0,))
        return cls(
            plane_normal=normal,
            plane_offset=offset,
            sphere_center=center,
            sphere_radius=radius,
            stiffness=jnp.asarray(stiffness),
        )

    @classmethod
    def ground(cls, height: float = 0.0, *, dim: int = 3, stiffness: float = 1_000.0) -> Obstacles:
        """A horizontal ground plane: particles stay at or above ``height``."""
        normal = tuple(0.0 for _ in range(dim - 1)) + (1.0,)
        return cls.build(planes=[(normal, height)], stiffness=stiffness, dim=dim)

    @property
    def n_planes(self) -> int:
        return self.plane_normal.shape[0]

    @property
    def n_spheres(self) -> int:
        return self.sphere_center.shape[0]


# Shared empty default so systems without obstacles trace identically to before.
_NO_OBSTACLES = Obstacles.none()


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
        obstacles: Rigid boundary obstacles (ground planes, walls, keep-out
            spheres), enforced as a differentiable penalty potential. Defaults to
            no obstacles, which adds zero energy and leaves rollouts unchanged.
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
    obstacles: Obstacles = _NO_OBSTACLES

    @property
    def n_particles(self) -> int:
        return self.mass.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def dim(self) -> int:
        return self.gravity.shape[0]
