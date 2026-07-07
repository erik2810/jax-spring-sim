r"""A learned, E(n)-equivariant one-step surrogate for the spring dynamics.

The rest of this package *integrates* the known spring physics. This module
learns a neural surrogate that predicts the next state directly, and the point is
the symmetry. A plain network that ate raw coordinates would give physically
wrong answers on a rotated input; this surrogate is E(n)-equivariant by
construction (Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML
2021), so it obeys the same SE(3) symmetry as the physics it imitates: rotate and
translate the input mesh and every predicted position and velocity transforms the
same way, to machine precision.

It is written in the package's style, no neural-network framework, just parameter
pytrees and ``jax.numpy``, so it composes with ``jit`` / ``grad`` / ``vmap`` and
trains with the package's own :func:`~jax_spring_sim.inverse.adam`.

Following the engine's own semi-implicit Euler integrator, the surrogate predicts
the next *velocity* (the hard part, the spring force and damping), and the
position follows the known rule :math:`x_\text{next} = x + \Delta t\, v_\text{next}`.
Messages run over the spring edge list (O(E)) and see coordinates only through the
invariant squared distance :math:`\lVert x_i - x_j \rVert^2`; velocity is the
second equivariant input (a rotation rotates it, a translation leaves it
unchanged):

    m_ij      = phi_e(h_i, h_j, ||x_i - x_j||^2, a_ij)            [Edges, message]
    h_i'      = h_i + phi_h(h_i, sum_j m_ij)                      [Nodes, hidden]
    v_i_next  = gv(h_i) v_i + (1/deg_i) sum_j (x_i - x_j) fv(m_ij)     [Nodes, 3]
    x_i_next  = x_i + dt * v_i_next        (done in predict_step)      [Nodes, 3]

where ``a_ij`` are per-edge invariant attributes (spring stiffness and rest
length) and ``gv, fv`` are learned scalar functions. Every coordinate-dependent
quantity is either the invariant squared distance or an equivariant relative
vector times an invariant scalar, so the velocity update is exactly equivariant,
and the integrated position is too. Nothing takes ``sqrt`` of a possibly-zero
distance, so the gradient is finite everywhere.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .system import SpringSystem, State

# An MLP is a list of (weight, bias) layers; a plain pytree of arrays.
MLP = list[tuple[jax.Array, jax.Array]]


class EGNNLayerParams(NamedTuple):
    """Parameters of one equivariant message-passing block."""

    edge: MLP  # phi_e: (h_i, h_j, ||x_i - x_j||^2, a_ij) -> message
    node: MLP  # phi_h: (h_i, aggregated message)         -> feature update
    v_gate: MLP  # gv: h_i    -> scalar velocity gate
    v_coord: MLP  # fv: m_ij  -> scalar edge weight in the velocity update


class EGNNParams(NamedTuple):
    """All parameters of the surrogate: an input embedding plus the blocks."""

    embed: MLP
    layers: tuple[EGNNLayerParams, ...]


def _init_mlp(
    key: jax.Array,
    sizes: list[int],
    *,
    final_zero: bool = False,
    final_bias: float = 0.0,
) -> MLP:
    """Initialise an MLP with LeCun-scaled weights, one (W, b) per layer.

    Args:
        key: PRNG key.
        sizes: Layer widths ``[in, hidden, ..., out]``.
        final_zero: Zero the last weight matrix (identity-like start).
        final_bias: Constant value for the last layer's bias.

    Returns:
        List of ``(W, b)`` with ``W`` shape ``(sizes[i], sizes[i+1])``.
    """
    params: MLP = []
    for i in range(len(sizes) - 1):
        key, sub = jax.random.split(key)
        last = i == len(sizes) - 2
        scale = 0.0 if (last and final_zero) else (1.0 / sizes[i]) ** 0.5
        w = jax.random.normal(sub, (sizes[i], sizes[i + 1])) * scale
        b = jnp.full((sizes[i + 1],), final_bias if last else 0.0)
        params.append((w, b))
    return params


def _apply_mlp(params: MLP, x: jax.Array) -> jax.Array:
    """Forward an MLP with SiLU activations between layers (none on the output)."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.silu(x)
    return x


def init_params(
    key: jax.Array,
    node_feat_dim: int,
    edge_attr_dim: int,
    *,
    hidden: int = 64,
    message_dim: int = 64,
    n_layers: int = 4,
) -> EGNNParams:
    """Initialise the surrogate parameters.

    Args:
        key: PRNG key.
        node_feat_dim: Width of the per-node invariant input features.
        edge_attr_dim: Width of the per-edge invariant attributes.
        hidden: Hidden feature width carried between blocks.
        message_dim: Message width inside each block.
        n_layers: Number of equivariant message-passing blocks.

    Returns:
        An :class:`EGNNParams` pytree.
    """
    key, sub = jax.random.split(key)
    embed = _init_mlp(sub, [node_feat_dim, hidden, hidden])

    layers = []
    for _ in range(n_layers):
        key, ke, kn, kvg, kvc = jax.random.split(key, 5)
        edge = _init_mlp(ke, [2 * hidden + 1 + edge_attr_dim, hidden, message_dim])
        node = _init_mlp(kn, [hidden + message_dim, hidden, hidden])
        # Neutral start: the edge readout is zero and the velocity gate is 1, so the
        # untrained surrogate leaves the velocity unchanged and learns the update.
        v_gate = _init_mlp(kvg, [hidden, hidden, 1], final_zero=True, final_bias=1.0)
        v_coord = _init_mlp(kvc, [message_dim, hidden, 1], final_zero=True)
        layers.append(EGNNLayerParams(edge=edge, node=node, v_gate=v_gate, v_coord=v_coord))

    return EGNNParams(embed=embed, layers=tuple(layers))


def apply(
    params: EGNNParams,
    node_feat: jax.Array,
    x: jax.Array,
    v: jax.Array,
    edges: jax.Array,
    edge_attr: jax.Array,
) -> jax.Array:
    r"""Predict the next velocity of every node.

    Args:
        params: Surrogate parameters.
        node_feat: (N, node_feat_dim) invariant per-node features.
        x: (N, 3) positions.
        v: (N, 3) velocities.
        edges: (E, 2) undirected spring edges (integer node indices).
        edge_attr: (E, edge_attr_dim) invariant per-edge attributes.

    Returns:
        ``v_next`` of shape ``(N, 3)``. Under ``x -> R x + t`` and ``v -> R v`` it
        transforms as ``v_next -> R v_next`` (translation leaves it unchanged).
    """
    n = x.shape[0]

    # Undirected springs become directed edges both ways so each endpoint receives
    # from the other. For a directed edge k: receiver i = recv[k], sender j = send[k].
    send = jnp.concatenate([edges[:, 0], edges[:, 1]])  # (2E,) j
    recv = jnp.concatenate([edges[:, 1], edges[:, 0]])  # (2E,) i
    attr = jnp.concatenate([edge_attr, edge_attr], axis=0)  # (2E, A)
    deg = jax.ops.segment_sum(jnp.ones_like(recv, dtype=x.dtype), recv, n)  # (N,)
    inv_deg = (1.0 / jnp.clip(deg, 1.0))[:, None]  # (N, 1)

    # Geometry is fixed through the blocks; only the features and velocity refine.
    rel = x[recv] - x[send]  # (2E, 3) x_i - x_j (equivariant)
    dist2 = jnp.sum(rel * rel, axis=-1, keepdims=True)  # (2E, 1) invariant

    h = _apply_mlp(params.embed, node_feat)  # (N, hidden)

    for layer in params.layers:
        edge_in = jnp.concatenate([h[recv], h[send], dist2, attr], axis=-1)
        m = _apply_mlp(layer.edge, edge_in)  # (2E, message) invariant messages

        # Equivariant velocity update: gated velocity plus edge-weighted relatives.
        v_upd = inv_deg * jax.ops.segment_sum(rel * _apply_mlp(layer.v_coord, m), recv, n)
        v = _apply_mlp(layer.v_gate, h) * v + v_upd  # (N, 3) equivariant

        # Invariant feature update (aggregate invariant messages).
        h = h + _apply_mlp(layer.node, jnp.concatenate([h, jax.ops.segment_sum(m, recv, n)], -1))

    return v


def _node_features(system: SpringSystem) -> jax.Array:
    """Invariant per-node features for the surrogate: mass and pin flag, (N, 2)."""
    return jnp.stack([system.mass, system.fixed], axis=-1)


def _edge_attributes(system: SpringSystem) -> jax.Array:
    """Invariant per-edge attributes: stiffness and rest length, (E, 2)."""
    return jnp.stack([system.stiffness, system.rest_length], axis=-1)


def predict_step(params: EGNNParams, state: State, system: SpringSystem, dt: float) -> State:
    """Run the surrogate on a :class:`State`, returning the predicted next state.

    A drop-in learned stand-in for one :func:`~jax_spring_sim.dynamics.step`,
    equivariant to rigid motions of ``state``. The surrogate predicts the next
    velocity; the position follows the engine's semi-implicit rule
    ``x_next = x + dt * v_next``.
    """
    v_next = apply(
        params,
        _node_features(system),
        state.pos,
        state.vel,
        system.edges,
        _edge_attributes(system),
    )
    return State(pos=state.pos + dt * v_next, vel=v_next)
