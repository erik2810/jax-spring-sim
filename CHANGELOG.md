# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
