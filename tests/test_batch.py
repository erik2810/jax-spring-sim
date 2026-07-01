"""vmap ensemble must agree with an explicit per-sample loop."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_spring_sim import make_chain, perturb_initial, simulate_ensemble, simulate_final
from jax_spring_sim.system import State


def test_ensemble_matches_serial_loop() -> None:
    key = jax.random.PRNGKey(0)
    state, system = make_chain(5)
    states0 = perturb_initial(state, key, batch=4, scale=0.05)

    batched = simulate_ensemble(states0, system, 0.01, 60)

    for i in range(4):
        single = simulate_final(State(pos=states0.pos[i], vel=states0.vel[i]), system, 0.01, 60)
        assert jnp.allclose(batched.pos[i], single.pos, atol=1e-6)


def test_ensemble_leading_axis_shape() -> None:
    key = jax.random.PRNGKey(1)
    state, system = make_chain(7)
    states0 = perturb_initial(state, key, batch=16, scale=0.1)
    batched = simulate_ensemble(states0, system, 0.01, 30)
    assert batched.pos.shape == (16, 7, 2)
