r"""Inverse design: differentiate *through* the simulation.

Forward simulation maps parameters $\theta$ (rest lengths, stiffness, initial
state, ...) to an outcome. Inverse design asks the reverse question: *which
$\theta$ produces a desired outcome?* Because every operation in the rollout is
differentiable, we can answer it with gradient descent on a loss

$$\mathcal{L}(\theta) = \big\lVert \mathbf{x}_{\text{final}}(\theta)
- \mathbf{x}_{\text{target}} \big\rVert^2,$$

where the gradient $\nabla_\theta \mathcal{L}$ is obtained by reverse-mode
autodiff through the entire ``lax.scan`` rollout — exact, and at a cost
independent of $\dim\theta$ (unlike finite differences, which scale linearly in
the number of parameters).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from .dynamics import simulate_final
from .system import SpringSystem, State

Params = TypeVar("Params")


def trajectory_loss(
    rest_length: jax.Array,
    state0: State,
    system: SpringSystem,
    target: jax.Array,
    dt: float,
    n_steps: int,
) -> jax.Array:
    """Mean squared distance of the final configuration from ``target``.

    Differentiating this w.r.t. ``rest_length`` back-propagates through every
    integration step.
    """
    tuned = system._replace(rest_length=rest_length)
    final = simulate_final(state0, tuned, dt, n_steps)
    return jnp.mean(jnp.sum((final.pos - target) ** 2, axis=-1))


def adam(
    loss_and_grad: Callable[[Params], tuple[jax.Array, Params]],
    params: Params,
    n_steps: int,
    lr: float = 1e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Params, jax.Array]:
    """Minimal Adam optimiser that runs entirely inside ``lax.scan``.

    Works on an arbitrary pytree of parameters via :func:`jax.tree_util.tree_map`,
    so it optimises a single array or a whole :class:`SpringSystem` unchanged.

    Args:
        loss_and_grad: A ``jax.value_and_grad``-style callable.
        params: Initial parameter pytree.
        n_steps: Number of optimisation iterations.
        lr: Learning rate.
        b1, b2, eps: Standard Adam hyper-parameters.

    Returns:
        ``(optimised_params, loss_history)`` with ``loss_history`` of length
        ``n_steps``.
    """
    tree_map = jax.tree_util.tree_map
    m0 = tree_map(jnp.zeros_like, params)
    v0 = tree_map(jnp.zeros_like, params)

    def body(carry: Any, t: jax.Array) -> tuple[Any, jax.Array]:
        params, m, v = carry
        loss, grads = loss_and_grad(params)
        m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
        v = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, v, grads)
        bias1 = 1 - b1 ** (t + 1)
        bias2 = 1 - b2 ** (t + 1)
        params = tree_map(
            lambda p, m, v: p - lr * (m / bias1) / (jnp.sqrt(v / bias2) + eps),
            params,
            m,
            v,
        )
        return (params, m, v), loss

    (params, _, _), losses = jax.lax.scan(body, (params, m0, v0), jnp.arange(n_steps))
    return params, losses


@partial(jax.jit, static_argnames=("n_steps", "opt_steps"))
def fit_rest_lengths(
    state0: State,
    system: SpringSystem,
    target: jax.Array,
    dt: float,
    n_steps: int,
    opt_steps: int,
    lr: float = 5e-2,
) -> tuple[jax.Array, jax.Array]:
    """Optimise spring rest lengths so the system settles onto ``target``.

    The full pipeline — ``opt_steps`` of Adam, each requiring a differentiated
    ``n_steps`` rollout — compiles into a single XLA program.

    Returns:
        ``(rest_length, loss_history)``.
    """
    loss_and_grad = jax.value_and_grad(trajectory_loss)

    def lg(rest_length: jax.Array) -> tuple[jax.Array, jax.Array]:
        return loss_and_grad(rest_length, state0, system, target, dt, n_steps)

    return adam(lg, system.rest_length, opt_steps, lr)
