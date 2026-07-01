"""Gradient correctness via finite-difference checks.

The JAX analogue of ``torch.autograd.gradcheck``. A silently-broken gradient
(e.g. a non-differentiable op slipping into the rollout) is the most common and
most costly bug in differentiable-physics code, so we pin it down explicitly.
"""

from __future__ import annotations

from jax.test_util import check_grads

from jax_spring_sim import make_chain, total_energy, trajectory_loss


def test_energy_gradient_matches_finite_difference() -> None:
    state, system = make_chain(4)
    perturbed = state.pos.at[2, 1].add(0.3)
    check_grads(
        lambda pos: total_energy(pos, system),
        (perturbed,),
        order=2,
        modes=["rev", "fwd"],
    )


def test_loss_gradient_through_rollout() -> None:
    # Differentiating w.r.t. rest_length back-propagates through every step of
    # the lax.scan rollout; verify it against finite differences.
    state, system = make_chain(4)
    target = state.pos + 0.1
    check_grads(
        lambda rl: trajectory_loss(rl, state, system, target, 0.01, 20),
        (system.rest_length,),
        order=1,
        modes=["rev"],
        atol=2e-3,
        rtol=2e-3,
    )
