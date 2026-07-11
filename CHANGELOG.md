# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Optional short-range collision force with a differentiable spatial hash grid
  (`spatial.py`): O(N) non-bonded contact instead of O(N^2) all-pairs, with the
  collision force produced by `jax.grad` like every other force. Enabled per
  rollout with `collide=True` and the new `collision_stiffness` /
  `collision_radius` fields on `SpringSystem` (off by default, so existing
  rollouts are unchanged).
- `tests/test_spatial.py`: hashed-vs-naive energy and gradient parity,
  `check_grads` on the collision energy, and a collision-enabled rollout.
- `benchmarks/collision_scaling.py`: naive O(N^2) vs hashed scaling comparison.
- `egnn.py`: a learned, E(n)-equivariant one-step surrogate for the spring
  dynamics (Satorras et al. 2021), written in the package's from-scratch pytree
  style. Messages depend on coordinates only through the invariant squared
  distance and the predicted velocity is an equivariant combination of the
  relative vectors, so `predict_step` is SE(3)-equivariant by construction and
  trains with the package's own `adam`.
- `tests/test_egnn.py`: SE(3) equivariance to 1e-5, a `check_grads` gradient
  check, and a training run that fits the true one-step map and beats the
  do-nothing baseline.
- Rigid boundary obstacles via the differentiable penalty method: an `Obstacles`
  pytree of half-space planes (ground, walls) and keep-out spheres on
  `SpringSystem`, entering the dynamics as a C1 penalty energy so `jax.grad`
  produces the exact contact reaction force. Builders gain `fixed_nodes` (exact
  Dirichlet anchors by index) and `obstacles` parameters; defaults change
  nothing.
- `tests/test_obstacles.py`: exact reaction force, finite-difference gradient
  check, settling at the analytic m g / k balance depth, anchored nodes, and
  differentiability w.r.t. obstacle parameters.
- Regularised Coulomb friction at obstacle contacts (`friction=mu` on
  `Obstacles`): a dissipative tangential force beside damping, saturating at the
  Coulomb limit when sliding and acting as stiff viscous drag near zero slip, so
  it stays smooth for autodiff. Defaults to frictionless.
- `tests/test_friction.py`: the force law, incline stick versus slip at the
  tan(theta) threshold, stopping distance within 2 percent of v0^2/(2 mu g),
  and gradient-descent recovery of an unknown mu from an observed trajectory.

## [0.1.0] - 2026-06-27

### Added

- Particle-spring system as JAX pytrees (`State`, `SpringSystem`).
- Energy-based dynamics: forces via `jax.grad` of a scalar potential energy.
- Symplectic-Euler integrator with `jax.lax.scan` rollout under `jax.jit`.
- Ensemble simulation over initial conditions via `jax.vmap`.
- Inverse design (`fit_rest_lengths`) differentiating through the rollout with
  `jax.value_and_grad`, plus a self-contained Adam optimiser.
- Builders for hanging chains and pinned cloth grids.
- Test suite: energy/force correctness, integrator behaviour, `check_grads`
  gradient verification, vmap-vs-loop equivalence, inverse-design convergence.
- Examples: forward chain, inverse design, ensemble, and a performance
  benchmark (naive Python vs NumPy vs JAX).
- Interactive WebGPU viewer: a FastAPI WebSocket backend
  (`jax_spring_sim.server`) streaming JAX trajectories with a binary frame
  protocol, and a Vite + React + three.js (WebGPU/TSL) frontend with catenary,
  cloth, and inverse-design scenes. Backend covered by `test_server.py`.
- Static GitHub Pages build: `scripts/bake_scenes.py` pre-computes scenes to
  binary bundles the frontend loads without a backend, deployed by the
  `deploy-pages` workflow.
