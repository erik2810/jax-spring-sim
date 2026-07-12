"""Energy and force correctness."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_spring_sim import compute_force, make_chain, spring_energy, total_energy


def test_spring_energy_zero_at_rest() -> None:
    state, system = make_chain(4, gravity=(0.0, 0.0))
    e = spring_energy(state.pos, system.edges, system.rest_length, system.stiffness)
    assert jnp.allclose(e, 0.0, atol=1e-9)


def test_spring_energy_positive_when_stretched() -> None:
    state, system = make_chain(4, gravity=(0.0, 0.0))
    stretched = state.pos.at[3, 0].add(0.5)
    e = spring_energy(stretched, system.edges, system.rest_length, system.stiffness)
    assert e > 0.0


def test_force_at_rest_equals_gravity() -> None:
    # At rest every spring is at its natural length, so the only force is gravity.
    state, system = make_chain(5)
    force = compute_force(state.pos, system)
    expected = system.mass[:, None] * system.gravity[None, :]
    assert jnp.allclose(force, expected, atol=1e-6)


def test_force_is_negative_energy_gradient() -> None:
    state, system = make_chain(5)
    perturbed = state.pos.at[2, 1].add(0.3)
    force = compute_force(perturbed, system)
    grad = jax.grad(total_energy)(perturbed, system)
    assert jnp.allclose(force, -grad, atol=1e-9)


def test_force_is_finite_at_coincident_connected_nodes() -> None:
    # A bare norm has a NaN gradient at zero separation, which would silently
    # poison a rollout the first time two connected particles collapse onto each
    # other. The regularised length keeps the force finite (and zero, since the
    # direction is genuinely undefined there).
    state, system = make_chain(3, pin_first=False)
    pos = state.pos.at[1].set(state.pos[0])
    force = compute_force(pos, system)
    assert jnp.all(jnp.isfinite(force))
