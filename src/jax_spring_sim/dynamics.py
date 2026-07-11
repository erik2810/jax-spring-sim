r"""Time integration.

We use **semi-implicit (symplectic) Euler**, which updates velocity before
position:

$$\mathbf{v}_{t+1} = \big(\mathbf{v}_t + \Delta t\, \mathbf{a}_t\big)\,\gamma,
\qquad \mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t\, \mathbf{v}_{t+1}.$$

Symplectic Euler conserves energy far better than explicit Euler for
oscillatory systems, at the same cost. The whole rollout is expressed with
:func:`jax.lax.scan` rather than a Python ``for`` loop: this is what lets XLA
compile the *entire* trajectory into one fused kernel (see ``README``), and it
is what makes the rollout efficiently reverse-mode differentiable.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .energy import compute_force, obstacle_friction_force
from .system import SpringSystem, State


def step(state: State, system: SpringSystem, dt: float, collide: bool = False) -> State:
    """Advance the system by one symplectic-Euler step (pure function).

    Pinned particles (``system.fixed == 1``) have their velocity zeroed, which
    holds them in place without special-casing positions. ``collide`` toggles the
    O(N) collision force (see :func:`.energy.total_energy`). Obstacle friction,
    being dissipative, enters here beside damping rather than through the
    potential (see :func:`.energy.obstacle_friction_force`); the obstacle count
    is a static shape, so systems without obstacles trace exactly as before.
    """
    force = compute_force(state.pos, system, collide)
    if system.obstacles.n_planes > 0 or system.obstacles.n_spheres > 0:
        force = force + obstacle_friction_force(state.pos, state.vel, system.obstacles)
    acc = force / system.mass[:, None]
    free = 1.0 - system.fixed[:, None]

    vel = (state.vel + dt * acc) * system.damping
    vel = vel * free  # clamp pinned particles
    pos = state.pos + dt * vel
    return State(pos=pos, vel=vel)


@partial(jax.jit, static_argnames=("n_steps", "save_every", "collide"))
def simulate(
    state0: State,
    system: SpringSystem,
    dt: float,
    n_steps: int,
    save_every: int = 1,
    collide: bool = False,
) -> tuple[State, State]:
    """Roll the system forward for ``n_steps`` and record the trajectory.

    The loop is a single :func:`jax.lax.scan`, so under ``jit`` it compiles to
    one kernel regardless of ``n_steps`` — no Python-level per-step dispatch.

    Args:
        state0: Initial state.
        system: System parameters (held constant over the rollout).
        dt: Time step.
        n_steps: Number of integration steps.
        save_every: Subsample factor for the returned trajectory.

    Returns:
        ``(final_state, trajectory)`` where ``trajectory`` is a :class:`State`
        whose ``pos`` / ``vel`` have a leading time axis of length
        ``n_steps // save_every``.
    """

    def body(carry: State, _: None) -> tuple[State, State | None]:
        new = step(carry, system, dt, collide)
        return new, new

    # Run the inner steps without materialising every frame, then subsample.
    def chunk(carry: State, _: None) -> tuple[State, State]:
        carry, _ = jax.lax.scan(body, carry, None, length=save_every)
        return carry, carry

    n_frames = n_steps // save_every
    final, traj = jax.lax.scan(chunk, state0, None, length=n_frames)
    return final, traj


def simulate_final(
    state0: State,
    system: SpringSystem,
    dt: float,
    n_steps: int,
    collide: bool = False,
) -> State:
    """Convenience wrapper returning only the final state (no trajectory stored).

    Preferred inside loss functions: the trajectory is dead weight for gradient
    computation and skipping it reduces peak memory of the reverse pass.
    """

    def body(carry: State, _: None) -> tuple[State, None]:
        return step(carry, system, dt, collide), None

    final, _ = jax.lax.scan(body, state0, None, length=n_steps)
    return final


def kinetic_energy(state: State, system: SpringSystem) -> jax.Array:
    r"""Total kinetic energy $\tfrac12 \sum_i m_i \lVert\mathbf{v}_i\rVert^2$."""
    return 0.5 * jnp.sum(system.mass * jnp.sum(state.vel**2, axis=-1))
