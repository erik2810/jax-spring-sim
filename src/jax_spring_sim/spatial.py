r"""Spatial hash grid for O(N) non-bonded (collision) interactions.

The spring term in :mod:`.energy` runs over an explicit edge list, so it is
already O(E) ~ O(N). Collision / self-avoidance is different: it is a
short-range interaction between *any* two particles that come close, which costs
O(N^2) if you test every pair. This module drops that to O(N) pair work (plus an
O(N log N) index sort) with a fixed-capacity hashed cell list, and keeps the
whole thing differentiable so the collision force is produced by :func:`jax.grad`
exactly like every other force in the engine.

Design (JAX-MD style, static shapes so it composes with ``jit`` / ``vmap`` /
``grad``):

* Each particle is hashed into one of ``n_buckets`` cells of side ``r_c`` (the
  interaction cutoff). Because the cell side equals the cutoff, any two particles
  closer than ``r_c`` land in the same cell or in adjacent cells, so scanning the
  ``3**D`` cell neighbourhood of a particle never misses a real interaction.
* Particle indices are scattered into a ``(n_buckets, CELL_CAPACITY)`` table.
* Each particle gathers the ``3**D`` cells around it as candidate neighbours and
  keeps those inside the cutoff.

Only the *selection* of neighbours uses integer ops (hashing, sorting, gathering
indices), which carry no gradient, and correctly so: they choose *which* pairs
interact, like a branch. The energy is a smooth function of the continuous
positions of the selected pairs, so gradients flow through
``pos -> displacement -> distance -> energy`` to machine precision. Hash
collisions never change the result, only the runtime: a colliding cell adds a few
extra candidates that the ``r < r_c`` cutoff then discards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .system import SpringSystem

# Maximum particles stored per hash bucket. Each particle then sees up to
# ``3**D * CELL_CAPACITY`` candidates. Buckets that overflow drop their extra
# occupants; for near-uniform density this never happens, and it can only ever
# drop a pair, never invent one. Keep it comfortably above the local density.
CELL_CAPACITY = 64

# Odd multipliers for the spatial hash (Teschner et al. 2003, "Optimized Spatial
# Hashing for Collision Detection of Deformable Objects"). One per axis; D <= 4.
_PRIMES = (73856093, 19349663, 83492791, 39916801)

# Small regulariser so ``sqrt`` (and its gradient) stay finite even when two
# particles coincide, where the repulsion direction is undefined. Shared by the
# hashed and naive paths so their energies remain identical.
_EPS = 1e-12


def _offsets(dim: int) -> jax.Array:
    r"""The ``3**dim x dim`` integer offsets of a cell's neighbourhood.

    Returns:
        Offsets, shape ``(3**dim, dim)``, each row in ``{-1, 0, 1}**dim``.
    """
    axes = np.meshgrid(*([(-1, 0, 1)] * dim), indexing="ij")
    return jnp.asarray(np.stack([a.ravel() for a in axes], axis=1), dtype=jnp.int32)


def _hash(cells: jax.Array, n_buckets: int) -> jax.Array:
    r"""Hash integer cell coordinates into ``[0, n_buckets)``.

    Args:
        cells: Integer cell coordinates, shape ``(..., D)``.
        n_buckets: Number of buckets (static).

    Returns:
        Bucket index per input cell, shape ``(...,)``, in ``[0, n_buckets)``.
    """
    dim = cells.shape[-1]
    h = jnp.zeros(cells.shape[:-1], dtype=jnp.int32)
    for d in range(dim):
        h = jnp.bitwise_xor(h, cells[..., d].astype(jnp.int32) * jnp.int32(_PRIMES[d]))
    return jnp.mod(h, jnp.int32(n_buckets))


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def build_cell_list(pos: jax.Array, cell_size: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Bin particles into a fixed-capacity hashed cell list.

    Args:
        pos: Particle positions, shape ``(N, D)``.
        cell_size: Scalar cell side (the interaction cutoff ``r_c``).

    Returns:
        ``(bins, cell_coords)`` where ``bins`` has shape
        ``(n_buckets, CELL_CAPACITY)`` holding particle indices (sentinel ``N``
        marks empty slots), and ``cell_coords`` has shape ``(N, D)`` giving each
        particle's integer cell (used to enumerate neighbourhoods).
    """
    n, dim = pos.shape
    n_buckets = _next_pow2(2 * n)  # static: N is concrete under jit tracing

    origin = jnp.min(pos, axis=0)  # non-negative cell coords; integer, no grad
    cell_coords = jnp.floor((pos - origin) / cell_size).astype(jnp.int32)  # (N, D)
    h = _hash(cell_coords, n_buckets)  # (N,)

    order = jnp.argsort(h).astype(jnp.int32)  # (N,) particles grouped by bucket
    h_sorted = h[order]  # (N,)

    # Rank of each particle within its bucket, in sorted order: the running max
    # of the segment-start index gives the first slot of the current bucket.
    idx = jnp.arange(n, dtype=jnp.int32)
    is_new = jnp.concatenate([jnp.array([True]), h_sorted[1:] != h_sorted[:-1]])
    seg_start = jax.lax.cummax(jnp.where(is_new, idx, jnp.int32(0)))
    rank = idx - seg_start  # (N,) in [0, bucket_count); >= CELL_CAPACITY overflows

    # Scatter with an out-of-range column (rank >= CELL_CAPACITY) dropped.
    bins = jnp.full((n_buckets, CELL_CAPACITY), n, dtype=jnp.int32)
    bins = bins.at[h_sorted, rank].set(order, mode="drop", unique_indices=True)
    return bins, cell_coords


def _neighbor_candidates(pos: jax.Array, cell_size: jax.Array) -> tuple[jax.Array, jax.Array]:
    r"""Candidate neighbours for every particle, with hash-collisions filtered out.

    For each particle we hash the ``3**D`` cells around it and gather their
    buckets. Two distinct cells can hash to the same bucket, which would gather
    (and double count) a genuine neighbour, so each candidate is verified against
    the *exact* cell it was queried for: a candidate is kept only when its true
    cell coordinate equals the neighbourhood cell that produced it. This makes the
    result independent of hash collisions.

    Args:
        pos: Positions, shape ``(N, D)``.
        cell_size: Scalar cell side / cutoff.

    Returns:
        ``(cand, cell_ok)`` each of shape ``(N, 3**D * CELL_CAPACITY)``. ``cand``
        holds candidate particle indices (sentinel ``N`` for empty slots) and
        ``cell_ok`` is True exactly where the candidate really lives in the queried
        cell (and hence is a non-empty, non-collision neighbour).
    """
    n, dim = pos.shape
    n_buckets = _next_pow2(2 * n)
    bins, cell_coords = build_cell_list(pos, cell_size)  # (B, C), (N, D)

    offsets = _offsets(dim)  # (K, D), K = 3**D
    queried = cell_coords[:, None, :] + offsets[None, :, :]  # (N, K, D) cells around each particle
    cand = bins[_hash(queried, n_buckets)]  # (N, K, C) candidate indices

    # Verify each candidate's true cell equals the cell it was queried for. Empty
    # slots (index N) map to a sentinel cell that matches no real cell.
    cell_pad = jnp.concatenate([cell_coords, jnp.full((1, dim), -(1 << 20), jnp.int32)], axis=0)
    cell_ok = jnp.all(cell_pad[cand] == queried[:, :, None, :], axis=-1)  # (N, K, C)
    return cand.reshape(n, -1), cell_ok.reshape(n, -1)


def _soft_repulsion(r2: jax.Array, active: jax.Array, k: jax.Array, r_c: jax.Array) -> jax.Array:
    r"""Summed penalty ``0.25 * k * sum (r_c - r)^2`` over active ordered pairs.

    The ``0.25`` is ``0.5`` (Hooke) times ``0.5`` (each unordered pair is counted
    twice, once from each endpoint). ``sqrt(r2 + eps)`` keeps the gradient finite
    at coincident particles.

    Args:
        r2: Squared pair distances, any shape.
        active: Boolean mask, same shape, True where the pair interacts.
        k: Scalar collision stiffness.
        r_c: Scalar cutoff radius.

    Returns:
        Scalar energy.
    """
    r = jnp.sqrt(r2 + _EPS)
    pen = jnp.where(active & (r < r_c), (r_c - r) ** 2, 0.0)
    return 0.25 * k * jnp.sum(pen)


def collision_energy(pos: jax.Array, system: SpringSystem) -> jax.Array:
    r"""Short-range repulsion via the spatial hash grid, O(N) pair work.

    Energy ``U = 0.5 * k * sum_{i<j, r_ij < r_c} (r_c - r_ij)^2`` over non-bonded
    pairs, evaluated by querying only the ``3**D`` cells around each particle.
    Differentiable in ``pos``; the collision force is ``-grad`` of this term.

    Args:
        pos: Particle positions, shape ``(N, D)``.
        system: Provides ``collision_stiffness`` (k) and ``collision_radius`` (r_c).

    Returns:
        Scalar collision energy.
    """
    k = system.collision_stiffness
    r_c = system.collision_radius
    n = pos.shape[0]

    cand, cell_ok = _neighbor_candidates(pos, r_c)  # (N, M) each
    pos_pad = jnp.concatenate([pos, jnp.zeros((1, pos.shape[1]), pos.dtype)], axis=0)  # (N+1, D)
    disp = pos[:, None, :] - pos_pad[cand]  # (N, M, D)
    r2 = jnp.sum(disp * disp, axis=-1)  # (N, M)

    not_self = cand != jnp.arange(n, dtype=cand.dtype)[:, None]  # exclude i == j
    return _soft_repulsion(r2, cell_ok & not_self, k, r_c)


def collision_energy_naive(pos: jax.Array, system: SpringSystem) -> jax.Array:
    r"""All-pairs O(N^2) reference for :func:`collision_energy`.

    Identical energy and gradient to the hashed version (same cutoff, same
    ``sqrt(r2 + eps)``), used as the correctness oracle and the "before" timing.

    Args:
        pos: Particle positions, shape ``(N, D)``.
        system: Provides ``collision_stiffness`` and ``collision_radius``.

    Returns:
        Scalar collision energy.
    """
    k = system.collision_stiffness
    r_c = system.collision_radius
    n = pos.shape[0]

    disp = pos[:, None, :] - pos[None, :, :]  # (N, N, D)
    r2 = jnp.sum(disp * disp, axis=-1)  # (N, N)
    not_self = ~jnp.eye(n, dtype=bool)  # exclude the diagonal
    return _soft_repulsion(r2, not_self, k, r_c)
