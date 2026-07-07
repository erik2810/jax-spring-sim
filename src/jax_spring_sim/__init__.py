"""jax-spring-sim — a differentiable, JIT-compiled particle-spring simulator.

A compact demonstration of the four JAX transforms that define modern
Scientific Machine Learning:

* ``jax.numpy``        — array/mesh manipulation (:mod:`.energy`, :mod:`.builders`)
* ``jax.grad``         — forces as the gradient of an energy (:mod:`.energy`)
* ``jax.jit``          — XLA-fused rollouts via ``lax.scan`` (:mod:`.dynamics`)
* ``jax.vmap``         — ensembles over initial conditions (:mod:`.batch`)
* ``jax.value_and_grad`` — inverse design through the simulator (:mod:`.inverse`)
"""

from __future__ import annotations

from .batch import perturb_initial, simulate_ensemble
from .builders import make_chain, make_cloth
from . import egnn
from .dynamics import kinetic_energy, simulate, simulate_final, step
from .energy import compute_force, gravity_energy, spring_energy, total_energy
from .inverse import adam, fit_rest_lengths, trajectory_loss
from .spatial import build_cell_list, collision_energy, collision_energy_naive
from .system import SpringSystem, State

__version__ = "0.1.0"

__all__ = [
    "SpringSystem",
    "State",
    "adam",
    "build_cell_list",
    "collision_energy",
    "collision_energy_naive",
    "compute_force",
    "egnn",
    "fit_rest_lengths",
    "gravity_energy",
    "kinetic_energy",
    "make_chain",
    "make_cloth",
    "perturb_initial",
    "simulate",
    "simulate_ensemble",
    "simulate_final",
    "spring_energy",
    "step",
    "total_energy",
    "trajectory_loss",
]
