"""Scene definitions: turn a JAX simulation into a streamable frame sequence.

Each scene builder returns a :class:`Scene` holding the static topology (edges,
pin mask, an optional target shape) plus a ``(F, N, 3)`` position trajectory and
a ``(F, N)`` per-particle scalar used for colouring on the GPU. Coordinates are
emitted y-up so the frontend can render them directly.

This module is *demo orchestration*: it composes the core library
(:mod:`jax_spring_sim`) and keeps that library free of presentation concerns.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..builders import make_chain, make_cloth
from ..dynamics import simulate, simulate_final
from ..inverse import trajectory_loss
from ..system import SpringSystem, State


@dataclass
class Scene:
    """A fully computed scene ready to stream."""

    name: str
    positions: np.ndarray  # (F, N, 3) float32, y-up
    value: np.ndarray  # (F, N) float32, colour channel
    edges: np.ndarray  # (E, 2) int
    fixed: np.ndarray  # (N,) float
    value_label: str
    target: np.ndarray | None = None  # (N, 3) optional ghost shape
    meta: dict = field(default_factory=dict)

    def as_meta(self) -> dict:
        vmin = float(self.value.min())
        vmax = float(self.value.max())
        return {
            "type": "meta",
            "scene": self.name,
            "n": int(self.positions.shape[1]),
            "frames": int(self.positions.shape[0]),
            "edges": self.edges.astype(int).tolist(),
            "fixed": self.fixed.astype(float).tolist(),
            "valueLabel": self.value_label,
            "valueRange": [vmin, vmax],
            "target": None if self.target is None else self.target.tolist(),
            **self.meta,
        }


def _pad2d(pos2d: np.ndarray) -> np.ndarray:
    """(F, N, 2) -> (F, N, 3) with z = 0."""
    f, n, _ = pos2d.shape
    out = np.zeros((f, n, 3), dtype=np.float32)
    out[..., :2] = pos2d
    return out


def chain_scene(
    n: int = 48,
    stiffness: float = 400.0,
    steps: int = 2400,
    dt: float = 4e-3,
    frames: int = 200,
) -> Scene:
    """A cable pinned at both ends settling into a catenary."""
    n = int(np.clip(n, 4, 120))
    state, system = make_chain(n, stiffness=stiffness, damping=0.99)
    system = system._replace(fixed=system.fixed.at[-1].set(1.0))

    save_every = max(1, steps // frames)
    _, traj = simulate(state, system, dt, steps, save_every=save_every)
    pos = _pad2d(np.asarray(traj.pos))
    speed = np.linalg.norm(np.asarray(traj.vel), axis=-1).astype(np.float32)

    return Scene(
        name="chain",
        positions=pos,
        value=speed,
        edges=np.asarray(system.edges),
        fixed=np.asarray(system.fixed),
        value_label="speed",
    )


def cloth_scene(
    rows: int = 16,
    cols: int = 16,
    stiffness: float = 220.0,
    steps: int = 1800,
    dt: float = 4e-3,
    frames: int = 200,
) -> Scene:
    """A pinned cloth grid draping under gravity (genuinely 3D)."""
    rows = int(np.clip(rows, 4, 28))
    cols = int(np.clip(cols, 4, 28))
    state, system = make_cloth(rows, cols, stiffness=stiffness, damping=0.992)

    save_every = max(1, steps // frames)
    _, traj = simulate(state, system, dt, steps, save_every=save_every)
    # Sim is x,y in-plane, gravity along -z. Remap to y-up: (x, y, z) -> (x, z, y).
    raw = np.asarray(traj.pos)
    pos = np.stack([raw[..., 0], raw[..., 2], raw[..., 1]], axis=-1).astype(np.float32)
    speed = np.linalg.norm(np.asarray(traj.vel), axis=-1).astype(np.float32)

    return Scene(
        name="cloth",
        positions=pos,
        value=speed,
        edges=np.asarray(system.edges),
        fixed=np.asarray(system.fixed),
        value_label="speed",
    )


def _adam_with_history(
    state0: State,
    system: SpringSystem,
    target: jax.Array,
    dt: float,
    n_steps: int,
    opt_steps: int,
    lr: float,
) -> jax.Array:
    """Run Adam on the rest lengths, returning the *history* of rest lengths.

    Captures one rest-length vector per optimisation step so we can replay the
    optimiser sculpting the chain toward the target.
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    grad = jax.value_and_grad(trajectory_loss)
    params = system.rest_length
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)

    def body(carry, t):
        params, m, v = carry
        _, g = grad(params, state0, system, target, dt, n_steps)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        params = params - lr * (m / (1 - b1 ** (t + 1))) / (jnp.sqrt(v / (1 - b2 ** (t + 1))) + eps)
        return (params, m, v), params

    _, history = jax.lax.scan(body, (params, m, v), jnp.arange(opt_steps))
    return history  # (opt_steps, E)


def inverse_scene(
    n: int = 28,
    stiffness: float = 200.0,
    steps: int = 200,
    dt: float = 1e-2,
    opt_steps: int = 220,
    lr: float = 0.05,
) -> Scene:
    """System identification: replay Adam recovering hidden rest lengths.

    Each frame is the settled chain for the rest lengths at that optimisation
    step, morphing from the uniform guess toward the observed target shape. The
    settled shapes for the whole optimiser history are produced in one batched
    ``vmap`` over the parameter trajectory.
    """
    n = int(np.clip(n, 6, 80))
    state, system = make_chain(n, stiffness=stiffness, damping=0.99)

    edge_x = jnp.linspace(0.0, 1.0, n - 1)
    true_rest = system.rest_length * (1.0 + 0.4 * jnp.sin(2.0 * jnp.pi * edge_x))
    target_state = simulate_final(state, system._replace(rest_length=true_rest), dt, steps)
    target = target_state.pos  # (N, 2)

    history = _adam_with_history(state, system, target, dt, steps, opt_steps, lr)

    # Settled shape for every rest-length snapshot, batched over the history.
    def settle(rest: jax.Array) -> jax.Array:
        return simulate_final(state, system._replace(rest_length=rest), dt, steps).pos

    shapes = jax.vmap(settle)(history)  # (opt_steps, N, 2)
    shapes_np = np.asarray(shapes)
    target_np = np.asarray(target)

    pos = _pad2d(shapes_np)
    # Colour channel: per-particle distance to the target (shrinks to ~0).
    value = np.linalg.norm(shapes_np - target_np[None], axis=-1).astype(np.float32)

    target3d = np.zeros((n, 3), dtype=np.float32)
    target3d[:, :2] = target_np

    return Scene(
        name="inverse",
        positions=pos,
        value=value,
        edges=np.asarray(system.edges),
        fixed=np.asarray(system.fixed),
        value_label="distance to target",
        target=target3d,
    )


# Registry of scene name -> builder (each clamps its own parameters).
SCENE_BUILDERS: dict[str, Callable[..., Scene]] = {
    "chain": chain_scene,
    "cloth": cloth_scene,
    "inverse": inverse_scene,
}


def build_scene(name: str, params: dict | None = None) -> Scene:
    """Dispatch to a scene builder by name, passing through known parameters."""
    if name not in SCENE_BUILDERS:
        raise ValueError(f"unknown scene: {name!r}")
    builder = SCENE_BUILDERS[name]
    params = params or {}
    # Only forward kwargs the builder actually accepts.
    allowed = set(inspect.signature(builder).parameters)
    kwargs = {k: v for k, v in params.items() if k in allowed and v is not None}
    return builder(**kwargs)
