"""Test configuration.

Enable 64-bit precision globally so finite-difference gradient checks
(:func:`jax.test_util.check_grads`) have the headroom for tight tolerances.
JAX defaults to float32, which is too coarse for second-order checks.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
