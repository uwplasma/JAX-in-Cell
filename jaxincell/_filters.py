import jax.numpy as jnp
from jax import jit, lax

# Maximum number of "regular" filter passes we support.
# This keeps the scan length static so grad() and jit() are happy,
# while still covering all reasonable use cases (default is 5).
_MAX_FILTER_PASSES = 16

@jit
def _shift_with_bc_1d(x, shift, bc_left, bc_right):
    """
    Shift x by 'shift' cells along axis=0 with boundary conditions:

    bc = 0: periodic
    bc = 1: reflective (clamp to boundary cell)
    bc = 2: absorbing (outside domain -> 0)

    Works for arrays with shape (G, ...) – only axis 0 is shifted.
    """
    n = x.shape[0]

    def _periodic(x):
        return jnp.roll(x, shift, axis=0)

    def _nonperiodic(x):
        idx = jnp.arange(n) + shift

        # Clip to domain [0, n-1] for reflective-type behavior
        idx_clipped = jnp.clip(idx, 0, n - 1)
        x_shifted = x[idx_clipped]          # shape (n, ...)

        # Absorbing: if we step outside, set to 0 instead of boundary value
        mask_left  = (idx < 0) & (bc_left  == 2)
        mask_right = (idx >= n) & (bc_right == 2)
        mask_out = mask_left | mask_right   # shape (n,)

        # Broadcast mask_out along remaining axes to match x_shifted
        # (n,) -> (n, 1, 1, ..., 1) with same ndim as x_shifted
        extra_dims = x_shifted.ndim - mask_out.ndim
        mask_out_b = mask_out.reshape(mask_out.shape + (1,) * extra_dims)

        x_shifted = jnp.where(mask_out_b, 0.0, x_shifted)
        return x_shifted

    return lax.cond(
        (bc_left == 0) & (bc_right == 0),
        _periodic,
        _nonperiodic,
        x,
    )

@jit
def binomial_filter_3point(x, alpha=0.5, stride=1, bc_left=0, bc_right=0):
    """
    3-point digital filter along axis 0 with BCs:

      x^f_j = α x_j + (1-α)/2 [ x_{j-stride} + x_{j+stride} ]

    bc_left / bc_right:
        0: periodic
        1: reflective (clamp)
        2: absorbing (outside -> 0)
    """
    left  = _shift_with_bc_1d(x, -stride, bc_left, bc_right)
    right = _shift_with_bc_1d(x, +stride, bc_left, bc_right)
    return alpha * x + (1 - alpha) * 0.5 * (left + right)

@jit
def _repeat_filter(y, stride, passes, alpha, bc_left=0, bc_right=0):
    """
    JAX-safe version of the 3-point digital filter with compensation:
    - If passes <= 0: return y unchanged.
    - If passes > 0:
        * apply (passes - 1) regular binomial_filter_3point passes with alpha
          (up to a static maximum _MAX_FILTER_PASSES),
        * then a final compensation pass with comp_alpha = passes - alpha*(passes - 1).
    This is fully jit- and grad-safe even when `passes` is a traced value.

    Note: The number of regular filter passes is internally capped at _MAX_FILTER_PASSES (16).
    If passes > _MAX_FILTER_PASSES + 1, the function will only apply _MAX_FILTER_PASSES regular
    passes plus one compensation pass, which may not match the expected filtering behavior.
    For typical use cases (default is 5), this limit should not be reached.
    """

    passes_f = jnp.asarray(passes)
    passes_clamped = jnp.minimum(passes_f, _MAX_FILTER_PASSES + 1)
    num_regular = jnp.maximum(passes_f - 1, 0)

    # This makes the capping behavior explicit in the code
    num_regular = jnp.minimum(num_regular, _MAX_FILTER_PASSES)

    def do_filter(y0):
        # Static-length scan; we mask out iterations beyond num_regular.
        def body(y_curr, i):
            apply = i < num_regular  # bool scalar, can be traced
            y_candidate = binomial_filter_3point(
                y_curr, alpha, stride=stride,
                bc_left=bc_left, bc_right=bc_right
            )
            # If apply is False, keep y_curr; otherwise use y_candidate.
            y_next = jnp.where(apply, y_candidate, y_curr)
            return y_next, None

        # i runs from 0 .. _MAX_FILTER_PASSES-1 (static)
        indices = jnp.arange(_MAX_FILTER_PASSES, dtype=jnp.int32)
        y1, _ = lax.scan(body, y0, indices)

        # compensation pass - sharp cutoff in k-space
        comp_alpha = passes_clamped - alpha * (passes_clamped - 1)
        y2 = binomial_filter_3point(
            y1, comp_alpha, stride=stride,
            bc_left=bc_left, bc_right=bc_right
        )
        return y2

    # Use lax.cond so passes can be a tracer and we still branch safely
    return lax.cond(
        passes_f <= 0,
        lambda y0: y0,  # no filtering
        do_filter,      # full filter pipeline
        y,
    )


@jit
def filter_scalar_field(scalar_field, passes=5, alpha=0.5, strides=(1, 2, 4),
                        bc_left=0, bc_right=0):
    """
    Apply a multi-pass 3-point binomial digital filter to a scalar field.
    
    Args:
        scalar_field: Input scalar field array to be filtered.
        passes: Number of filter passes (default: 5). Note: internally capped at 17 
                (16 regular passes + 1 compensation pass).
        alpha: Filter strength parameter (default: 0.5).
        strides: Tuple/list of stride values for filtering (default: (1, 2, 4)).
        bc_left: Boundary condition for the left side (default: 0).
                 0: periodic, 1: reflective, 2: absorbing.
        bc_right: Boundary condition for the right side (default: 0).
                  0: periodic, 1: reflective, 2: absorbing.
    
    Returns:
        Filtered scalar field array.
    """
    # Accept strides as tuple/list/array; convert to JAX array so it's OK as a dynamic arg.
    s = jnp.asarray(strides)

    def body(y, stride):
        return _repeat_filter(y, stride, passes, alpha,
                              bc_left=bc_left, bc_right=bc_right), None

    y, _ = lax.scan(body, scalar_field, s)
    return y


@jit
def filter_vector_field(F, passes=5, alpha=0.5, strides=(1, 2, 4),
                        bc_left=0, bc_right=0):
    """
    Apply digital filter along the grid axis (axis=0) for each component.
    F has shape (G, C), typically (grid_points, 3) for a vector field.

    Args:
        F: Input vector field array, shape (G, C), typically (grid_points, 3).
        passes: Number of filter passes (default: 5). Note: internally capped at 17 
                (16 regular passes + 1 compensation pass).
        alpha: Filter strength parameter (default: 0.5).
        strides: Tuple/list of stride values for filtering (default: (1, 2, 4)).
        bc_left: Boundary condition for the left side (default: 0).
                 0: periodic, 1: reflective, 2: absorbing.
        bc_right: Boundary condition for the right side (default: 0).
                  0: periodic, 1: reflective, 2: absorbing.
    
    Returns:
        Filtered vector field array with the same shape as input.
    """
    s = jnp.asarray(strides)

    def body(y, stride):
        return _repeat_filter(y, stride, passes, alpha,
                              bc_left=bc_left, bc_right=bc_right), None

    y, _ = lax.scan(body, F, s)
    return y
