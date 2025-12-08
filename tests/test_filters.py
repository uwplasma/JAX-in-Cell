# tests/test_filters.py

import numpy as np
import jax.numpy as jnp
from jax import jit, grad

from jaxincell._filters import (
    _repeat_filter,
    filter_scalar_field,
    filter_vector_field,
)


def test_repeat_filter_noop_for_nonpositive_passes():
    """
    _repeat_filter(y, stride, passes<=0, alpha) must return y unchanged,
    and must be safe under jit.
    """
    x = jnp.linspace(0.0, 1.0, 8)

    # Direct calls (passes as Python ints)
    y0 = _repeat_filter(x, stride=1, passes=0, alpha=0.5)
    y_neg = _repeat_filter(x, stride=1, passes=-1, alpha=0.5)

    np.testing.assert_allclose(np.array(y0), np.array(x))
    np.testing.assert_allclose(np.array(y_neg), np.array(x))

    # JAX-traced passes: passes is now a tracer inside _repeat_filter.
    @jit
    def apply_with_traced_passes(passes):
        return _repeat_filter(x, stride=1, passes=passes, alpha=0.5)

    y_traced = apply_with_traced_passes(jnp.array(0))
    np.testing.assert_allclose(np.array(y_traced), np.array(x))


def test_filter_scalar_field_smooths_for_positive_passes():
    """
    For passes > 0, filter_scalar_field should smooth the data,
    i.e., reduce its variance.
    """
    # Make something oscillatory so smoothing is visible
    x = jnp.linspace(0.0, 2 * jnp.pi, 64)
    phi = jnp.sin(4.0 * x) + 0.1 * jnp.cos(9.0 * x)

    var_before = jnp.var(phi)

    # A few passes and multiple strides
    phi_filtered = filter_scalar_field(phi, passes=3, alpha=0.5, strides=(1, 2))

    var_after = jnp.var(phi_filtered)

    # Should be strictly smoother (smaller variance), but allow tiny numerical wiggle
    assert float(var_after) < float(var_before)


def test_filter_vector_field_jit_and_grad_safe():
    """
    filter_vector_field(F, passes, alpha) should be safe under jit and grad.
    We treat alpha as a differentiable parameter.
    """
    G = 16
    # Simple 2-component "vector field" on the grid
    x = jnp.linspace(0.0, 1.0, G)
    F = jnp.stack([jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)], axis=1)

    def energy(alpha):
        # passes is a Python int here; alpha will be a tracer under grad/jit
        F_filt = filter_vector_field(F, passes=2, alpha=alpha, strides=(1, 2))
        return jnp.sum(F_filt**2)

    # JIT-compiled gradient w.r.t. alpha
    grad_energy = jit(grad(energy))

    # Just ensure it runs and returns a finite scalar
    g = grad_energy(0.5)
    assert jnp.isfinite(g), "Gradient of filtered field energy should be finite."
