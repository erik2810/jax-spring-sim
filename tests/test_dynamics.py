"""Integrator behaviour: pinning, shapes, symplectic energy conservation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_spring_sim import kinetic_energy, make_chain, simulate, total_energy


def test_pinned_particle_stays_put() -> None:
    state, system = make_chain(5)
    final, _ = simulate(state, system, 0.01, 300)
    assert jnp.allclose(final.pos[0], state.pos[0], atol=1e-9)


def test_trajectory_shape_with_subsampling() -> None:
    state, system = make_chain(5)
    _, traj = simulate(state, system, 0.01, 100, save_every=10)
    assert traj.pos.shape == (10, 5, 2)


def test_symplectic_energy_is_bounded() -> None:
    # No gravity, no damping: a perturbed pinned chain should oscillate with
    # total energy bounded (the hallmark of a symplectic integrator).
    state, system = make_chain(6, gravity=(0.0, 0.0), damping=1.0)
    state = state._replace(pos=state.pos.at[3, 1].add(0.4))

    e0 = total_energy(state.pos, system) + kinetic_energy(state, system)
    _, traj = simulate(state, system, 1e-3, 4000, save_every=1)

    energies = jax.vmap(
        lambda p, v: total_energy(p, system) + 0.5 * jnp.sum(system.mass * jnp.sum(v**2, axis=-1))
    )(traj.pos, traj.vel)

    drift = (energies.max() - energies.min()) / e0
    assert drift < 0.05
