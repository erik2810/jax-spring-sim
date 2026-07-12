"""Regularised Coulomb friction: force law, classic mechanics, and system ID.

Three levels of validation. The force law itself (direction, Coulomb limit,
zero outside contact, smooth gradients). Classic mechanics: stick versus slip on
an incline decided by ``mu`` against ``tan(theta)``, and the flat-ground
stopping distance ``v0^2 / (2 mu g)``. And the differentiable-physics payoff:
recovering an unknown friction coefficient from an observed trajectory by
gradient descent through the rollout. float64 is on globally (``conftest.py``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from jax_spring_sim import (
    Obstacles,
    adam,
    make_chain,
    obstacle_friction_force,
    simulate,
)
from jax_spring_sim.system import SpringSystem, State

G = 9.81


def _block_on_ground(mu: float, v0: float, k: float = 5000.0) -> tuple[State, SpringSystem]:
    """A single free particle resting on the ground at balance depth, sliding at v0."""
    obs = Obstacles.ground(0.0, dim=3, stiffness=k, friction=mu)
    state, system = make_chain(
        1, pin_first=False, damping=1.0, gravity=(0.0, 0.0, -G), obstacles=obs
    )
    state = State(pos=state.pos.at[0, 2].set(-G / k), vel=jnp.array([[v0, 0.0, 0.0]]))
    return state, system


def test_friction_opposes_slip_at_the_coulomb_limit() -> None:
    # Sliding fast (|v_t| >> eps): the force is -mu * k * pen along the slip direction.
    obs = Obstacles.ground(0.0, stiffness=100.0, friction=0.5)
    pos = jnp.array([[0.0, 0.0, -0.2]])  # pen 0.2 -> normal force 20
    vel = jnp.array([[3.0, 0.0, 0.0]])
    f = obstacle_friction_force(pos, vel, obs)
    assert jnp.allclose(f[0], jnp.array([-0.5 * 20.0, 0.0, 0.0]), rtol=1e-4)


def test_no_friction_without_contact_or_mu() -> None:
    obs = Obstacles.ground(0.0, stiffness=100.0, friction=0.5)
    above = obstacle_friction_force(jnp.array([[0.0, 0.0, 1.0]]), jnp.ones((1, 3)), obs)
    assert jnp.allclose(above, 0.0, atol=1e-12)
    frictionless = Obstacles.ground(0.0, stiffness=100.0, friction=0.0)
    touching = obstacle_friction_force(
        jnp.array([[0.0, 0.0, -0.1]]), jnp.ones((1, 3)), frictionless
    )
    assert jnp.allclose(touching, 0.0, atol=1e-12)


def test_friction_force_is_smooth_in_pos_and_vel() -> None:
    # Smooth in both arguments at a contact, including near zero slip where the
    # unregularised Coulomb law would be singular.
    obs = Obstacles.build(
        planes=[((0.0, 0.0, 1.0), 0.0)],
        spheres=[((1.0, 0.0, 0.3), 0.5)],
        stiffness=200.0,
        friction=0.4,
    )
    pos = jnp.array([[0.1, 0.0, -0.2], [1.2, 0.1, 0.25]])
    vel = jnp.array([[0.5, -0.2, 0.0], [1e-3, 2e-3, 0.0]])  # second: near-zero slip
    # The regularised law has curvature ~1/eps^2 near zero slip, so the
    # finite-difference step must be well below the smoothing scale.
    check_grads(
        lambda p, v: jnp.sum(obstacle_friction_force(p, v, obs) ** 2),
        (pos, vel),
        order=1,
        modes=["rev", "fwd"],
        eps=1e-6,
    )


def test_incline_stick_versus_slip() -> None:
    # The Coulomb criterion: mu > tan(theta) sticks, mu < tan(theta) slides.
    theta = math.radians(20.0)  # tan = 0.364
    normal = (-math.sin(theta), math.cos(theta))

    def slide_distance(mu: float) -> float:
        obs = Obstacles.build(planes=[(normal, 0.0)], stiffness=5000.0, friction=mu, dim=2)
        state, system = make_chain(
            1, pin_first=False, damping=1.0, gravity=(0.0, -G), obstacles=obs
        )
        state = State(pos=state.pos - jnp.asarray(normal) * (G / 5000.0), vel=state.vel)
        final, _ = simulate(state, system, dt=1e-3, n_steps=3000, save_every=3000)
        return float(jnp.linalg.norm(final.pos[0] - state.pos[0]))

    creep = slide_distance(0.6)  # above tan(theta): sticks (small regularisation creep)
    slide = slide_distance(0.1)  # below tan(theta): slides freely
    assert creep < 0.05
    assert slide > 5.0


def test_stopping_distance_matches_coulomb_theory() -> None:
    # A block sliding at v0 on flat ground stops after v0^2 / (2 mu g).
    mu, v0 = 0.5, 2.0
    state, system = _block_on_ground(mu, v0)
    final, _ = simulate(state, system, dt=1e-3, n_steps=2000, save_every=2000)
    theory = v0**2 / (2 * mu * G)
    assert abs(float(final.pos[0, 0]) - theory) < 0.02 * theory  # within 2 percent
    assert float(jnp.linalg.norm(final.vel)) < 1e-6  # actually stopped

    # And with mu = 0 (only dissipation would be friction), the block never slows.
    state0, system0 = _block_on_ground(0.0, v0)
    final0, _ = simulate(state0, system0, dt=1e-3, n_steps=2000, save_every=2000)
    assert abs(float(final0.vel[0, 0]) - v0) < 1e-9


def test_friction_coefficient_recovered_by_gradient_descent() -> None:
    # System identification: observe where a block stops, recover mu by
    # differentiating the rollout w.r.t. the friction coefficient.
    mu_true = 0.4
    state, system = _block_on_ground(mu_true, 2.0)
    observed = simulate(state, system, dt=1e-3, n_steps=1500, save_every=1500)[0].pos

    def loss(mu: jax.Array) -> jax.Array:
        tuned = system._replace(obstacles=system.obstacles._replace(friction=mu))
        final, _ = simulate(state, tuned, dt=1e-3, n_steps=1500, save_every=1500)
        return jnp.sum((final.pos - observed) ** 2)

    mu_fit, losses = jax.jit(lambda m: adam(jax.value_and_grad(loss), m, n_steps=150, lr=8e-3))(
        jnp.asarray(0.2)
    )

    assert abs(float(mu_fit) - mu_true) < 0.02
    assert float(losses[-1]) < 1e-4 * float(losses[0])
