"""Rigid boundary obstacles: reaction forces, gradients, and settling behaviour.

The contact model is a C1 penalty potential, so three things must hold. The
reaction force must be exactly ``stiffness * penetration * normal`` (the penalty
method's defining property), it must come out of ``jax.grad`` smoothly (verified
against finite differences), and a body dropped on the ground must settle at the
analytic force-balance depth ``m g / k`` instead of falling through or bouncing
forever. float64 is on globally (see ``conftest.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from jax_spring_sim import (
    Obstacles,
    make_chain,
    make_cloth,
    obstacle_energy,
    simulate,
    total_energy,
)
from jax_spring_sim.system import State


def test_plane_reaction_force_is_exact() -> None:
    # A node 0.2 below a ground plane of stiffness 100 feels exactly k * pen * n.
    obs = Obstacles.ground(0.0, stiffness=100.0)
    pos = jnp.array([[0.3, 0.7, -0.2], [1.0, 2.0, 0.5]])
    force = -jax.grad(obstacle_energy)(pos, obs)
    assert jnp.allclose(force[0], jnp.array([0.0, 0.0, 100.0 * 0.2]), atol=1e-12)
    assert jnp.allclose(force[1], jnp.zeros(3), atol=1e-12)  # above the plane: no force


def test_no_force_outside_contact() -> None:
    obs = Obstacles.build(
        planes=[((0.0, 0.0, 1.0), 0.0)],
        spheres=[((5.0, 5.0, 5.0), 1.0)],
        stiffness=500.0,
    )
    pos = jnp.array([[0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])  # clear of plane and sphere
    assert float(obstacle_energy(pos, obs)) == 0.0
    assert jnp.allclose(jax.grad(obstacle_energy)(pos, obs), 0.0, atol=1e-12)


def test_sphere_pushes_particles_out() -> None:
    # A particle inside a keep-out sphere is pushed radially outward.
    obs = Obstacles.build(spheres=[((0.0, 0.0, 0.0), 1.0)], stiffness=50.0)
    pos = jnp.array([[0.5, 0.0, 0.0]])  # 0.5 deep inside the unit sphere
    force = -jax.grad(obstacle_energy)(pos, obs)
    assert float(force[0, 0]) > 0.0  # outward along +x
    assert jnp.allclose(force[0, 1:], 0.0, atol=1e-12)
    assert jnp.allclose(jnp.abs(force[0, 0]), 50.0 * 0.5, atol=1e-9)


def test_obstacle_gradient_matches_finite_difference() -> None:
    # Smooth reaction forces via AD, checked at a penetrating configuration away
    # from the (measure-zero) contact kink.
    obs = Obstacles.build(
        planes=[((0.0, 0.0, 1.0), 0.0)],
        spheres=[((1.0, 1.0, 0.5), 0.8)],
        stiffness=200.0,
    )
    pos = jnp.array([[0.2, 0.1, -0.3], [1.2, 0.9, 0.4], [3.0, 3.0, 3.0]])
    check_grads(lambda p: obstacle_energy(p, obs), (pos,), order=2, modes=["rev", "fwd"])


def test_dropped_cloth_settles_on_ground_plane() -> None:
    # Requirement check: a free-falling cloth converges onto the ground at the
    # analytic penalty balance depth m g / k, with velocities damped to zero.
    k = 5000.0
    state, system = make_cloth(
        6, 8, pin_top=False, damping=0.98, obstacles=Obstacles.ground(0.0, stiffness=k)
    )
    state = state._replace(pos=state.pos.at[:, 2].add(1.0))  # drop from z = 1
    final, _ = simulate(state, system, dt=2e-3, n_steps=4000, save_every=4000)

    balance = 9.81 / k  # mass 1 per node
    z_min = float(final.pos[:, 2].min())
    assert jnp.all(jnp.isfinite(final.pos))
    assert z_min > -2.0 * balance  # never falls through
    assert z_min < 0.0  # genuinely in contact, not floating
    assert abs(-z_min - balance) < 0.3 * balance  # settles at the balance depth
    assert float(jnp.abs(final.vel).max()) < 1e-8  # converged, not bouncing


def test_fixed_nodes_stay_anchored() -> None:
    # The fixed_nodes parameter is an exact Dirichlet anchor through a rollout.
    anchors = [0, 7]
    state, system = make_chain(8, pin_first=False, fixed_nodes=anchors)
    final, _ = simulate(state, system, dt=1e-3, n_steps=500, save_every=500)
    for i in anchors:
        assert jnp.allclose(final.pos[i], state.pos[i], atol=1e-12)
    # An unanchored interior node sags under gravity.
    assert float(jnp.abs(final.pos[3] - state.pos[3]).max()) > 1e-3


def test_empty_obstacles_change_nothing() -> None:
    # The default system carries Obstacles.none(); energies match a system built
    # with an explicit empty set, and the term contributes exactly zero.
    state, system = make_cloth(4, 4)
    state2, system2 = make_cloth(4, 4, obstacles=Obstacles.none(3))
    pos = state.pos + 0.1
    assert float(obstacle_energy(pos, system.obstacles)) == 0.0
    assert jnp.allclose(total_energy(pos, system), total_energy(pos, system2))


def test_build_rejects_bad_inputs() -> None:
    # A zero-length normal or a zero smoothing scale would propagate NaN through
    # the contact forces; both must fail loudly at construction instead. Negative
    # stiffness (attractive wall) and negative friction (energy injection) are
    # never intended and fail the same way.
    with pytest.raises(ValueError, match="nonzero length"):
        Obstacles.build(planes=[((0.0, 0.0, 0.0), 0.0)])
    with pytest.raises(ValueError, match="friction_smoothing"):
        Obstacles.build(friction_smoothing=0.0)
    with pytest.raises(ValueError, match="stiffness"):
        Obstacles.build(stiffness=-1.0)
    with pytest.raises(ValueError, match="friction"):
        Obstacles.build(friction=-0.5)


def test_build_respects_default_float_dtype() -> None:
    # Geometry follows the configured default float width (float64 under the
    # x64 flag), and integer literals do not produce integer geometry.
    default = jnp.zeros(()).dtype
    obs = Obstacles.build(planes=[((0, 0, 1), 0)], spheres=[((0, 0, 0), 1)])
    assert obs.plane_normal.dtype == default
    assert obs.sphere_radius.dtype == default
    assert float(obs.sphere_radius[0]) == 1.0


def test_fixed_nodes_out_of_range_raises() -> None:
    # JAX scatter would silently drop a bad index; the builder must not.
    with pytest.raises(ValueError, match="out of range"):
        make_cloth(4, 4, fixed_nodes=[999])


def test_obstacles_are_differentiable_parameters() -> None:
    # The wall itself is a differentiable parameter: the gradient of a rollout
    # loss w.r.t. the plane offset is finite and nonzero when contact happens.
    def loss(offset: jax.Array) -> jax.Array:
        obs = Obstacles(
            plane_normal=jnp.array([[0.0, 0.0, 1.0]]),
            plane_offset=offset,
            sphere_center=jnp.zeros((0, 3)),
            sphere_radius=jnp.zeros((0,)),
            stiffness=jnp.asarray(2000.0),
        )
        state, system = make_cloth(3, 3, pin_top=False, damping=0.98, obstacles=obs)
        state = State(pos=state.pos.at[:, 2].add(0.5), vel=state.vel)
        final, _ = simulate(state, system, dt=2e-3, n_steps=400, save_every=400)
        return jnp.mean(final.pos[:, 2])

    g = jax.grad(loss)(jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.abs(g).max()) > 1e-3  # the resting height tracks the wall
