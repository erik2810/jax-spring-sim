"""The learned surrogate: SE(3) equivariance, gradient sanity, and that it learns.

The surrogate predicts one step of the spring dynamics. Two things must hold. It
must be equivariant to rigid motions (a rotated, translated mesh gives rotated,
translated predictions), which is a property of the architecture and holds for any
parameters. And it must actually fit the physics, which we check by training it on
ground-truth one-step data from the real integrator and beating the do-nothing
baseline. Gravity is switched off in the training data so the true one-step map is
itself SE(3)-equivariant and an equivariant model can represent it.

float64 is on globally (see ``conftest.py``) so the equivariance and gradient
checks have headroom.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from jax_spring_sim import egnn, make_cloth
from jax_spring_sim.dynamics import step
from jax_spring_sim.inverse import adam
from jax_spring_sim.system import State

TOL = 1e-5


def _random_rotation(seed: int) -> jax.Array:
    """A uniformly random proper rotation in SO(3), shape (3, 3)."""
    a = jax.random.normal(jax.random.PRNGKey(seed), (3, 3))
    q, r = jnp.linalg.qr(a)
    q = q * jnp.sign(jnp.diag(r))
    return jnp.where(jnp.linalg.det(q) < 0, q.at[:, 0].set(-q[:, 0]), q)


def _perturb(params: egnn.EGNNParams, seed: int) -> egnn.EGNNParams:
    """Nudge every parameter so the surrogate is a non-trivial (non-identity) map."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    noised = [p + 0.3 * jax.random.normal(k, p.shape) for p, k in zip(leaves, keys, strict=True)]
    return jax.tree_util.tree_unflatten(treedef, noised)


def _setup(seed: int = 0):
    """A gravity-free cloth, random velocities, and perturbed surrogate params."""
    state, system = make_cloth(4, 4, gravity=(0.0, 0.0, 0.0))
    n = state.pos.shape[0]
    vel = 0.3 * jax.random.normal(jax.random.PRNGKey(seed), (n, 3))
    state = State(pos=state.pos, vel=vel)
    params = _perturb(
        egnn.init_params(jax.random.PRNGKey(seed + 1), 2, 2, hidden=16, message_dim=16, n_layers=3),
        seed + 2,
    )
    return params, state, system


def test_surrogate_se3_equivariance() -> None:
    params, state, system = _setup()
    r = _random_rotation(3)
    t = jax.random.normal(jax.random.PRNGKey(4), (3,))

    pred = egnn.predict_step(params, state, system, dt=5e-3)
    moved = State(pos=state.pos @ r.T + t, vel=state.vel @ r.T)
    pred_moved = egnn.predict_step(params, moved, system, dt=5e-3)

    # Positions are equivariant, velocities are equivariant (translation-free).
    assert jnp.allclose(pred.pos @ r.T + t, pred_moved.pos, atol=TOL)
    assert jnp.allclose(pred.vel @ r.T, pred_moved.vel, atol=TOL)
    # The map is non-trivial: it genuinely changes the velocity.
    assert not jnp.allclose(pred.vel, state.vel, atol=1e-3)


def test_surrogate_gradient_is_finite_and_correct() -> None:
    # Tiny model + graph so the finite-difference check over the whole parameter
    # pytree stays cheap; catches any non-differentiable op in the surrogate.
    state, system = make_cloth(3, 3, gravity=(0.0, 0.0, 0.0))
    params = egnn.init_params(jax.random.PRNGKey(0), 2, 2, hidden=8, message_dim=8, n_layers=2)
    target = step(state, system, 5e-3)

    def loss(p: egnn.EGNNParams) -> jax.Array:
        pred = egnn.predict_step(p, state, system, dt=5e-3)
        return jnp.mean((pred.pos - target.pos) ** 2) + jnp.mean((pred.vel - target.vel) ** 2)

    check_grads(loss, (params,), order=1, modes=["rev", "fwd"])


def test_surrogate_learns_one_step_physics() -> None:
    dt = 5e-3
    state0, system = make_cloth(4, 4, gravity=(0.0, 0.0, 0.0))
    n = state0.pos.shape[0]

    # A batch of perturbed states and their true next state from the integrator.
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    states = State(
        pos=state0.pos[None] + 0.15 * jax.random.normal(k1, (32, n, 3)),
        vel=0.3 * jax.random.normal(k2, (32, n, 3)),
    )
    truth = jax.vmap(lambda s: step(s, system, dt))(states)

    params = egnn.init_params(jax.random.PRNGKey(1), 2, 2, hidden=32, message_dim=32, n_layers=3)

    def loss(p: egnn.EGNNParams) -> jax.Array:
        pred = jax.vmap(lambda s: egnn.predict_step(p, s, system, dt))(states)
        return jnp.mean((pred.pos - truth.pos) ** 2) + jnp.mean((pred.vel - truth.vel) ** 2)

    trained, losses = jax.jit(lambda p: adam(jax.value_and_grad(loss), p, 400, lr=5e-3))(params)

    # Training clearly reduces the one-step loss. The untrained model already
    # equals the do-nothing baseline (it predicts the velocity unchanged), so any
    # solid reduction is a real improvement over it. The factor here is loose on
    # purpose: the exact convergence rate is not portable across XLA backends, but
    # the beats-baseline checks below are.
    assert losses[-1] < 0.6 * losses[0]

    # The trained surrogate beats the do-nothing baseline on both position and
    # velocity, against a fixed reference, so it has actually learned the dynamics.
    pred = jax.vmap(lambda s: egnn.predict_step(trained, s, system, dt))(states)
    pos_rmse = jnp.sqrt(jnp.mean((pred.pos - truth.pos) ** 2))
    vel_rmse = jnp.sqrt(jnp.mean((pred.vel - truth.vel) ** 2))
    base_pos = jnp.sqrt(jnp.mean((states.pos - truth.pos) ** 2))
    base_vel = jnp.sqrt(jnp.mean((states.vel - truth.vel) ** 2))
    assert pos_rmse < base_pos
    assert vel_rmse < base_vel
