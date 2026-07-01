"""Inverse design must actually drive the loss down (and recover parameters)."""

from __future__ import annotations

import jax.numpy as jnp

from jax_spring_sim import fit_rest_lengths, make_chain, simulate_final


def test_fit_reduces_loss() -> None:
    state, system = make_chain(6)
    # Build a reachable target by simulating a system with *different* rest
    # lengths; the optimiser starts from the original lengths, so loss > 0.
    true_rl = system.rest_length * 1.25
    target = simulate_final(state, system._replace(rest_length=true_rl), 0.01, 120).pos

    _, losses = fit_rest_lengths(state, system, target, 0.01, 120, opt_steps=120, lr=0.05)

    assert losses[-1] < 0.1 * losses[0]


def test_fit_recovers_target_shape() -> None:
    state, system = make_chain(6)
    true_rl = system.rest_length * 1.25
    target = simulate_final(state, system._replace(rest_length=true_rl), 0.01, 120).pos

    rl, _ = fit_rest_lengths(state, system, target, 0.01, 120, opt_steps=200, lr=0.05)
    achieved = simulate_final(state, system._replace(rest_length=rl), 0.01, 120).pos

    assert jnp.mean(jnp.sum((achieved - target) ** 2, axis=-1)) < 1e-3
