"""Spatial hash grid and collision energy correctness.

The grid is an optimisation: it must reproduce the naive all-pairs collision
energy *and its gradient* exactly (up to the shared regulariser), for any
particle configuration, and it must not introduce a non-differentiable op that
silently breaks the force. These tests pin all of that down, plus the physical
sanity of the repulsion and that the collision-enabled rollout runs.

float64 is on globally (see ``conftest.py``) so the finite-difference gradient
checks have headroom.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from jax_spring_sim import (
    collision_energy,
    collision_energy_naive,
    compute_force,
    make_chain,
    make_cloth,
    simulate,
    total_energy,
)


def _cloud(seed: int, n: int, scale: float, dim: int = 3) -> jax.Array:
    """A random particle cloud, ``(n, dim)``, spread over ``[0, scale)``."""
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, dim)) * scale


@pytest.mark.parametrize("seed,scale", [(0, 3.0), (1, 2.0), (2, 5.0)])
def test_hashed_energy_matches_naive(seed: int, scale: float) -> None:
    # The grid must return the same energy as testing every pair.
    _, system = make_cloth(8, 8, collision_stiffness=5.0)
    pos = _cloud(seed, 80, scale)
    assert jnp.allclose(
        collision_energy(pos, system), collision_energy_naive(pos, system), atol=1e-9
    )


@pytest.mark.parametrize("seed,scale", [(0, 3.0), (1, 2.0), (2, 5.0)])
def test_hashed_force_matches_naive(seed: int, scale: float) -> None:
    # The whole point of requirement 2: gradients survive the spatial sort.
    _, system = make_cloth(8, 8, collision_stiffness=5.0)
    pos = _cloud(seed, 80, scale)
    g_hash = jax.grad(collision_energy)(pos, system)
    g_naive = jax.grad(collision_energy_naive)(pos, system)
    assert jnp.allclose(g_hash, g_naive, atol=1e-9)
    assert jnp.all(jnp.isfinite(g_hash))


def test_collision_gradient_matches_finite_difference() -> None:
    # Catches a non-differentiable op or the sqrt-at-zero NaN sneaking in.
    _, system = make_cloth(6, 6, collision_stiffness=3.0)
    pos = _cloud(7, 48, 2.5)
    check_grads(lambda p: collision_energy(p, system), (pos,), order=2, modes=["rev", "fwd"])


def test_force_is_repulsive_for_overlapping_pair() -> None:
    # Two particles inside the cutoff must be pushed apart along their axis.
    _, system = make_chain(
        2, gravity=(0.0, 0.0, 0.0), collision_stiffness=10.0, collision_radius=1.0
    )
    pos = jnp.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])  # 0.4 < r_c = 1.0
    force = -jax.grad(collision_energy)(pos, system)
    assert force[0, 0] < 0.0  # left particle pushed left
    assert force[1, 0] > 0.0  # right particle pushed right


def test_zero_energy_beyond_cutoff() -> None:
    # Particles farther apart than r_c must not interact at all.
    _, system = make_chain(
        2, gravity=(0.0, 0.0, 0.0), collision_stiffness=10.0, collision_radius=1.0
    )
    pos = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    assert jnp.allclose(collision_energy(pos, system), 0.0, atol=1e-12)
    assert jnp.allclose(jax.grad(collision_energy)(pos, system), 0.0, atol=1e-12)


def test_dense_cluster_within_capacity_matches_naive() -> None:
    # A tight cluster forces many particles into few cells (hash collisions and
    # near-capacity buckets); the grid must still match the all-pairs result.
    _, system = make_cloth(8, 8, collision_stiffness=4.0, collision_radius=1.0)
    pos = _cloud(3, 64, 1.5)  # ~64 particles in a ~1.5^3 box, several per cell
    assert jnp.allclose(
        collision_energy(pos, system), collision_energy_naive(pos, system), atol=1e-9
    )
    g_hash = jax.grad(collision_energy)(pos, system)
    g_naive = jax.grad(collision_energy_naive)(pos, system)
    assert jnp.allclose(g_hash, g_naive, atol=1e-9)


def test_collision_off_matches_baseline() -> None:
    # collide=False must reproduce the collision-free energy exactly (no regression).
    # Use r_c > spacing so structural neighbours sit inside the cutoff and the
    # collision term is genuinely active at rest.
    state, system = make_cloth(6, 6, collision_stiffness=5.0, collision_radius=1.5)
    perturbed = state.pos.at[10].add(jnp.array([0.1, 0.1, 0.2]))
    assert jnp.allclose(
        total_energy(perturbed, system, collide=False),
        total_energy(perturbed, system),  # default is collide=False
    )
    # And the collision term genuinely changes the total when enabled.
    assert not jnp.allclose(
        total_energy(perturbed, system, collide=True),
        total_energy(perturbed, system, collide=False),
    )


def test_rollout_with_collision_runs_and_stays_finite() -> None:
    # End-to-end: the collision force integrated through the jitted lax.scan
    # rollout produces finite, correctly-shaped state.
    state, system = make_cloth(6, 6, collision_stiffness=2.0, collision_radius=0.9)
    final, traj = simulate(state, system, dt=1e-3, n_steps=40, save_every=10, collide=True)
    assert final.pos.shape == state.pos.shape
    assert traj.pos.shape == (4, *state.pos.shape)
    assert jnp.all(jnp.isfinite(final.pos))


def test_compute_force_collide_flag_adds_collision() -> None:
    # compute_force must route the flag through to the collision term.
    state, system = make_cloth(6, 6, collision_stiffness=8.0, collision_radius=1.2)
    pos = _cloud(5, state.pos.shape[0], 2.0)
    f_off = compute_force(pos, system, collide=False)
    f_on = compute_force(pos, system, collide=True)
    assert not jnp.allclose(f_on, f_off)
    assert jnp.all(jnp.isfinite(f_on))
