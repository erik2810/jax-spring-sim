"""Constructors for common spring-network topologies.

These return a ``(State, SpringSystem)`` pair ready to feed to
:func:`~jax_spring_sim.dynamics.simulate`. Rest lengths default to the initial
edge lengths, so each network starts at rest (modulo gravity).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .system import Obstacles, SpringSystem, State


def _rest_lengths(pos: jax.Array, edges: jax.Array) -> jax.Array:
    delta = pos[edges[:, 0]] - pos[edges[:, 1]]
    return jnp.linalg.norm(delta, axis=-1)


def _pin_mask(n: int, base: jax.Array, fixed_nodes: Sequence[int] | None) -> jax.Array:
    """Combine a builder's default pins with a user-supplied list of anchor nodes."""
    if fixed_nodes is None:
        return base
    return base.at[jnp.asarray(list(fixed_nodes), dtype=jnp.int32)].set(1.0)


def make_chain(
    n: int,
    *,
    spacing: float = 1.0,
    stiffness: float = 50.0,
    mass: float = 1.0,
    gravity: tuple[float, ...] = (0.0, -9.81),
    damping: float = 0.999,
    pin_first: bool = True,
    fixed_nodes: Sequence[int] | None = None,
    obstacles: Obstacles | None = None,
    collision_stiffness: float = 0.0,
    collision_radius: float | None = None,
) -> tuple[State, SpringSystem]:
    """A horizontal chain of ``n`` particles linked by ``n-1`` springs.

    With ``pin_first`` the leftmost particle is clamped, giving a pendulum /
    hanging-rope that swings down under gravity — a clean test bed for both
    forward dynamics and inverse design. ``fixed_nodes`` anchors any extra node
    indices (exact Dirichlet pins, on top of ``pin_first``), and ``obstacles``
    attaches rigid boundaries (see :class:`~jax_spring_sim.system.Obstacles`).
    Set ``collision_stiffness > 0`` to give the chain self-repulsion (see
    :mod:`.spatial`); ``collision_radius`` defaults to ``spacing`` so bonded
    neighbours sit at the cutoff.
    """
    dim = len(gravity)
    pos = jnp.zeros((n, dim)).at[:, 0].set(jnp.arange(n) * spacing)
    vel = jnp.zeros((n, dim))
    edges = jnp.stack([jnp.arange(n - 1), jnp.arange(1, n)], axis=1)

    fixed = jnp.zeros(n).at[0].set(1.0) if pin_first else jnp.zeros(n)
    fixed = _pin_mask(n, fixed, fixed_nodes)
    system = SpringSystem(
        edges=edges,
        rest_length=_rest_lengths(pos, edges),
        stiffness=jnp.full(n - 1, stiffness),
        mass=jnp.full(n, mass),
        fixed=fixed,
        gravity=jnp.asarray(gravity),
        damping=jnp.asarray(damping),
        collision_stiffness=jnp.asarray(collision_stiffness),
        collision_radius=jnp.asarray(spacing if collision_radius is None else collision_radius),
        obstacles=obstacles if obstacles is not None else Obstacles.none(dim),
    )
    return State(pos=pos, vel=vel), system


def make_cloth(
    rows: int,
    cols: int,
    *,
    spacing: float = 1.0,
    stiffness: float = 80.0,
    mass: float = 1.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    damping: float = 0.995,
    pin_top: bool = True,
    shear: bool = True,
    fixed_nodes: Sequence[int] | None = None,
    obstacles: Obstacles | None = None,
    collision_stiffness: float = 0.0,
    collision_radius: float | None = None,
) -> tuple[State, SpringSystem]:
    """A ``rows x cols`` grid of particles with structural (and shear) springs.

    The grid lies in the $xy$-plane and falls along $z$. With ``pin_top`` the
    top row is clamped, producing a hanging curtain. ``shear`` adds the two
    diagonal springs per cell that keep the sheet from collapsing.
    ``fixed_nodes`` anchors any extra node indices (exact Dirichlet pins), and
    ``obstacles`` attaches rigid boundaries such as a ground plane (see
    :class:`~jax_spring_sim.system.Obstacles`). Set ``collision_stiffness > 0``
    for self-collision (see :mod:`.spatial`); ``collision_radius`` defaults to
    ``0.9 * spacing`` so the sheet does not self-repel at rest but resists
    folding through itself.
    """
    xs, ys = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows), indexing="xy")
    flat = (ys * cols + xs).reshape(-1)  # noqa: F841 (documents index layout)
    pos = jnp.stack(
        [xs.reshape(-1) * spacing, ys.reshape(-1) * spacing, jnp.zeros(rows * cols)],
        axis=1,
    )
    vel = jnp.zeros_like(pos)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    edge_list: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edge_list.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edge_list.append((idx(r, c), idx(r + 1, c)))
            if shear and r + 1 < rows and c + 1 < cols:
                edge_list.append((idx(r, c), idx(r + 1, c + 1)))
                edge_list.append((idx(r, c + 1), idx(r + 1, c)))
    edges = jnp.asarray(edge_list)

    fixed = jnp.zeros(rows * cols)
    if pin_top:
        fixed = fixed.at[jnp.arange(cols)].set(1.0)
    fixed = _pin_mask(rows * cols, fixed, fixed_nodes)

    system = SpringSystem(
        edges=edges,
        rest_length=_rest_lengths(pos, edges),
        stiffness=jnp.full(edges.shape[0], stiffness),
        mass=jnp.full(rows * cols, mass),
        fixed=fixed,
        gravity=jnp.asarray(gravity),
        damping=jnp.asarray(damping),
        collision_stiffness=jnp.asarray(collision_stiffness),
        collision_radius=jnp.asarray(
            0.9 * spacing if collision_radius is None else collision_radius
        ),
        obstacles=obstacles if obstacles is not None else Obstacles.none(3),
    )
    return State(pos=pos, vel=vel), system
