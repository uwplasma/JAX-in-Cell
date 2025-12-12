# tests/test_filters.py

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, random

from jaxincell._filters import (
    binomial_filter_3point,
    _repeat_filter,
    filter_scalar_field,
    filter_vector_field,
    _MAX_FILTER_PASSES,
)


def test_binomial_filter_3point_alpha_one_is_identity():
    """
    For alpha=1, the 3-point filter should reduce to x^f_j = x_j,
    regardless of stride (since the neighbor term is multiplied by 0).
    """
    x = jnp.linspace(0.0, 1.0, 10)
    y = binomial_filter_3point(x, alpha=1.0, stride=1)
    np.testing.assert_allclose(np.array(y), np.array(x), rtol=0, atol=1e-12)

    # Try a different stride to be sure periodic rolling doesn't matter here
    y2 = binomial_filter_3point(x, alpha=1.0, stride=2)
    np.testing.assert_allclose(np.array(y2), np.array(x), rtol=0, atol=1e-12)


def test_binomial_filter_3point_matches_manual_formula():
    """
    Check binomial_filter_3point against a direct manual implementation for a small vector.
    """
    x = jnp.arange(5.0)  # [0,1,2,3,4]
    alpha = 0.5
    stride = 1

    # Manual periodic neighbors
    x_plus = jnp.roll(x, stride)
    x_minus = jnp.roll(x, -stride)
    manual = alpha * x + (1 - alpha) * 0.5 * (x_plus + x_minus)

    y = binomial_filter_3point(x, alpha=alpha, stride=stride)

    np.testing.assert_allclose(np.array(y), np.array(manual), rtol=0, atol=1e-12)


def _naive_repeat_filter(y, stride, passes, alpha):
    """
    Reference implementation of _repeat_filter in vanilla Python/JAX ops,
    without any JAX control flow tricks.

    This mirrors the logic in _repeat_filter:
      - passes <= 0 -> return y unchanged
      - otherwise: (passes - 1) regular passes with alpha,
                   followed by 1 compensation pass with comp_alpha.
    """
    if passes <= 0:
        return y

    num_regular = passes - 1
    for _ in range(num_regular):
        y = binomial_filter_3point(y, alpha=alpha, stride=stride)

    comp_alpha = passes - alpha * (passes - 1)
    y = binomial_filter_3point(y, alpha=comp_alpha, stride=stride)
    return y


def test_repeat_filter_zero_and_one_passes_behavior():
    """
    _repeat_filter should:
      - return the input unchanged for passes <= 0,
      - for passes=1, reduce to a single "compensation" pass which is the identity
        (comp_alpha=1) and thus also returns the input.
    """
    x = jnp.linspace(-1.0, 1.0, 11)

    # passes = 0 -> unchanged
    y0 = _repeat_filter(x, stride=1, passes=0, alpha=0.3)
    np.testing.assert_allclose(np.array(y0), np.array(x), rtol=0, atol=1e-12)

    # passes < 0 -> treated as <= 0, also unchanged
    y_neg = _repeat_filter(x, stride=1, passes=-3, alpha=0.3)
    np.testing.assert_allclose(np.array(y_neg), np.array(x), rtol=0, atol=1e-12)

    # passes = 1 -> no regular passes, comp_alpha = 1 => identity binomial filter
    y1 = _repeat_filter(x, stride=1, passes=1, alpha=0.3)
    np.testing.assert_allclose(np.array(y1), np.array(x), rtol=0, atol=1e-12)


def test_repeat_filter_matches_naive_for_reasonable_passes():
    """
    For passes in [2, 3, 5] (all <= _MAX_FILTER_PASSES), _repeat_filter should
    match the naive reference implementation to numerical precision.
    """
    x = jnp.cos(jnp.linspace(0.0, 2.0 * jnp.pi, 17))
    alpha = 0.4
    stride = 1

    for passes in [2, 3, 5]:
        y_ref = _naive_repeat_filter(x, stride=stride, passes=passes, alpha=alpha)
        y = _repeat_filter(x, stride=stride, passes=passes, alpha=alpha)
        np.testing.assert_allclose(
            np.array(y), np.array(y_ref), rtol=1e-10, atol=1e-12,
            err_msg=f"Mismatch for passes={passes}"
        )


def test_repeat_filter_handles_large_passes_gracefully():
    """
    If passes > _MAX_FILTER_PASSES, the implementation uses a static scan of length
    _MAX_FILTER_PASSES and masks extra iterations. This test just checks that it
    runs and returns a finite array of the right shape.
    """
    x = jnp.sin(jnp.linspace(0.0, 4.0 * jnp.pi, 33))
    alpha = 0.5
    stride = 1

    # Choose passes greater than _MAX_FILTER_PASSES to exercise masking logic
    passes = _MAX_FILTER_PASSES + 5

    y = _repeat_filter(x, stride=stride, passes=passes, alpha=alpha)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_filter_scalar_field_identity_and_smoothing():
    """
    filter_scalar_field should:
      - act as identity for passes <= 0,
      - reduce high-frequency content when passes > 0.
    """
    # High-frequency signal
    G = 64
    x_grid = jnp.linspace(0.0, 2.0 * jnp.pi, G, endpoint=False)
    scalar = jnp.sin(8.0 * x_grid)  # quite oscillatory

    # passes = 0 -> unchanged
    y0 = filter_scalar_field(scalar, passes=0, alpha=0.5, strides=(1, 2, 4))
    np.testing.assert_allclose(np.array(y0), np.array(scalar), rtol=0, atol=1e-12)

    # passes > 0 -> filtered; variance (or L2 norm) should decrease
    y = filter_scalar_field(scalar, passes=5, alpha=0.5, strides=(1, 2, 4))

    orig_norm2 = float(jnp.sum(scalar**2))
    filt_norm2 = float(jnp.sum(y**2))
    assert filt_norm2 <= orig_norm2 + 1e-9  # allow tiny FP drift
    assert y.shape == scalar.shape


def test_filter_scalar_field_accepts_various_stride_containers():
    """
    filter_scalar_field should accept strides as tuple, list, or JAX array
    and produce consistent results.
    """
    x = jnp.linspace(-1.0, 1.0, 21)

    strides_tuple = (1, 2, 4)
    strides_list = [1, 2, 4]
    strides_array = jnp.array([1, 2, 4])

    y_tuple = filter_scalar_field(x, passes=3, alpha=0.5, strides=strides_tuple)
    y_list = filter_scalar_field(x, passes=3, alpha=0.5, strides=strides_list)
    y_array = filter_scalar_field(x, passes=3, alpha=0.5, strides=strides_array)

    np.testing.assert_allclose(np.array(y_tuple), np.array(y_list), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.array(y_tuple), np.array(y_array), rtol=0, atol=1e-12)


def test_filter_vector_field_shapes_and_identity():
    """
    filter_vector_field should:
      - preserve shape (G, C),
      - act as identity for passes <= 0.
    """
    G = 16
    C = 3
    # Build a simple vector field with different patterns on each component
    grid = jnp.linspace(0.0, 2.0 * jnp.pi, G, endpoint=False)
    F = jnp.stack(
        [
            jnp.sin(grid),
            jnp.cos(2.0 * grid),
            jnp.sin(3.0 * grid),
        ],
        axis=1,
    )  # shape (G, 3)

    # passes = 0 -> unchanged
    F0 = filter_vector_field(F, passes=0, alpha=0.5, strides=(1, 2))
    np.testing.assert_allclose(np.array(F0), np.array(F), rtol=0, atol=1e-12)
    assert F0.shape == (G, C)

    # passes > 0 -> still shape (G,C), but smoothed (at least one component)
    Ff = filter_vector_field(F, passes=4, alpha=0.5, strides=(1, 2))
    assert Ff.shape == (G, C)

    # Norm should not explode; typically it should decrease or stay similar
    orig_norm = float(jnp.linalg.norm(F))
    filt_norm = float(jnp.linalg.norm(Ff))
    assert filt_norm <= orig_norm + 1e-6


def test_filter_scalar_field_jit_and_grad_compatible():
    """
    Verify that filter_scalar_field is both jit- and grad-compatible by
    differentiating a simple loss that depends on the filtered field.
    """
    key = random.PRNGKey(0)
    x0 = random.normal(key, shape=(32,))

    def loss_fn(x):
        y = filter_scalar_field(
            x,
            passes=3,
            alpha=0.4,
            strides=(1, 2),
            bc_left=1,
            bc_right=2,
        )
        return jnp.sum(y**2)

    loss_jit = jit(loss_fn)
    grad_loss = jit(grad(loss_fn))

    # Just ensure they run and produce finite results
    loss_val = loss_jit(x0)
    grad_val = grad_loss(x0)

    # JAX "scalars" are 0-d arrays; this is stable across versions.
    assert loss_val.shape == () or getattr(loss_val, "ndim", None) == 0

    assert jnp.all(jnp.isfinite(grad_val))
    assert grad_val.shape == x0.shape


def test_filter_vector_field_jit_and_grad_compatible():
    """
    Same as above, but for vector fields: check jit+grad through the filter
    by flattening the vector field into a 1D parameter.
    """
    key = random.PRNGKey(1)
    G = 20
    C = 3
    flat0 = random.normal(key, shape=(G * C,))

    def loss_fn(flat):
        F = flat.reshape(G, C)
        Ff = filter_vector_field(
            F,
            passes=4,
            alpha=0.3,
            strides=(1, 2, 4),
            bc_left=1,
            bc_right=2,
        )
        return jnp.sum(Ff**2)

    loss_jit = jit(loss_fn)
    grad_loss = jit(grad(loss_fn))

    loss_val = loss_jit(flat0)
    grad_val = grad_loss(flat0)

    assert loss_val.shape == () or getattr(loss_val, "ndim", None) == 0

    assert jnp.all(jnp.isfinite(grad_val))
    assert grad_val.shape == flat0.shape


def test_binomial_filter_3point_reflective_bc_clamps_edges():
    """
    For reflective BCs (bc_left=bc_right=1), neighbors beyond the edge
    are clamped to the boundary values.
    """
    x = jnp.arange(5.0)  # [0,1,2,3,4]
    alpha = 0.5
    stride = 1

    # Manual reflective implementation
    def manual_reflective(x):
        n = x.shape[0]
        out = []
        for j in range(n):
            jm = max(j - stride, 0)
            jp = min(j + stride, n - 1)
            val = alpha * x[j] + (1 - alpha) * 0.5 * (x[jm] + x[jp])
            out.append(val)
        return jnp.array(out)

    y_ref = manual_reflective(x)
    y = binomial_filter_3point(
        x,
        alpha=alpha,
        stride=stride,
        bc_left=1,
        bc_right=1,
    )

    np.testing.assert_allclose(np.array(y), np.array(y_ref), rtol=0, atol=1e-12)

def test_binomial_filter_3point_absorbing_bc_zeros_outside():
    """
    For absorbing BCs (bc_left=bc_right=2), neighbors beyond the domain
    are treated as zero, not wrapped or clamped.
    """
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    alpha = 0.5
    stride = 1

    # Manual absorbing implementation
    def manual_absorbing(x):
        n = x.shape[0]
        out = []
        for j in range(n):
            jm = j - stride
            jp = j + stride
            left = x[jm] if jm >= 0 else 0.0
            right = x[jp] if jp < n else 0.0
            val = alpha * x[j] + (1 - alpha) * 0.5 * (left + right)
            out.append(val)
        return jnp.array(out)

    y_ref = manual_absorbing(x)
    y = binomial_filter_3point(
        x,
        alpha=alpha,
        stride=stride,
        bc_left=2,
        bc_right=2,
    )

    np.testing.assert_allclose(np.array(y), np.array(y_ref), rtol=0, atol=1e-12)

def test_filter_scalar_field_nonperiodic_bcs_dont_wrap():
    """
    For absorbing BCs, high-frequency content at one edge should not
    wrap around to the other edge, unlike periodic filtering.

    We use passes >= 2 so that the filter actually touches neighbors
    (passes=1 is intentionally an identity in _repeat_filter).
    """
    G = 16
    scalar = jnp.zeros(G).at[0].set(1.0)  # spike at left boundary

    # Periodic filtering can wrap influence to the right edge
    y_periodic = filter_scalar_field(
        scalar,
        passes=2, 
        alpha=0.0,
        strides=(1,),
        bc_left=0,
        bc_right=0,
    )

    # Absorbing filtering should not wrap; the last cell sees no contribution
    y_absorbing = filter_scalar_field(
        scalar,
        passes=2,  
        alpha=0.0,
        strides=(1,),
        bc_left=2,
        bc_right=2,
    )

    # Last cell should differ between periodic and absorbing
    assert not np.allclose(
        np.array(y_periodic[-1]),
        np.array(y_absorbing[-1]),
    )

    # With absorbing BCs, last cell should be ~0 for this pattern
    assert abs(float(y_absorbing[-1])) < 1e-12


def test_filter_vector_field_with_nonperiodic_bcs():
    """
    Basic shape and finiteness check for vector field filtering
    with non-periodic boundary conditions.
    """
    G, C = 10, 3
    grid = jnp.linspace(0.0, 1.0, G)
    F = jnp.stack(
        [jnp.sin(2.0 * jnp.pi * grid),
         jnp.cos(4.0 * jnp.pi * grid),
         jnp.linspace(-1.0, 1.0, G)],
        axis=1,
    )

    Ff_reflective = filter_vector_field(
        F,
        passes=3,
        alpha=0.5,
        strides=(1, 2),
        bc_left=1,
        bc_right=1,
    )
    Ff_absorbing = filter_vector_field(
        F,
        passes=3,
        alpha=0.5,
        strides=(1, 2),
        bc_left=2,
        bc_right=2,
    )

    assert Ff_reflective.shape == F.shape
    assert Ff_absorbing.shape == F.shape
    assert jnp.all(jnp.isfinite(Ff_reflective))
    assert jnp.all(jnp.isfinite(Ff_absorbing))
