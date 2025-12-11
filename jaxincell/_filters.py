import jax.numpy as jnp
from jax import jit, lax

@jit
def binomial_filter_3point(x, alpha=0.5, stride=1):
    # 3-point: x^f_j = α x_j + (1-α)(x_{j-1}+x_{j+1})/2
    # periodic ends assumed at call site (we’ll pad/roll in callers)
    return alpha * x + (1 - alpha) * 0.5 * (
        jnp.roll(x, stride, axis=0) + jnp.roll(x, -stride, axis=0)
    )

# Maximum number of "regular" filter passes we support.
# This keeps the scan length static so grad() and jit() are happy,
# while still covering all reasonable use cases (default is 5).
_MAX_FILTER_PASSES = 16

@jit
def _repeat_filter(y, stride, passes, alpha):
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
    num_regular = jnp.maximum(passes_f - 1, 0)

    # This makes the capping behavior explicit in the code
    num_regular = jnp.minimum(num_regular, _MAX_FILTER_PASSES)

    def do_filter(y0):
        # Static-length scan; we mask out iterations beyond num_regular.
        def body(y_curr, i):
            apply = i < num_regular  # bool scalar, can be traced
            y_candidate = binomial_filter_3point(y_curr, alpha, stride=stride)
            # If apply is False, keep y_curr; otherwise use y_candidate.
            y_next = jnp.where(apply, y_candidate, y_curr)
            return y_next, None

        # i runs from 0 .. _MAX_FILTER_PASSES-1 (static)
        indices = jnp.arange(_MAX_FILTER_PASSES, dtype=jnp.int32)
        y1, _ = lax.scan(body, y0, indices)

        # compensation pass - sharp cutoff in k-space
        comp_alpha = passes_f - alpha * (passes_f - 1)
        y2 = binomial_filter_3point(y1, comp_alpha, stride=stride)
        return y2

    # Use lax.cond so passes can be a tracer and we still branch safely
    return lax.cond(
        passes_f <= 0,
        lambda y0: y0,  # no filtering
        do_filter,      # full filter pipeline
        y,
    )


@jit
def filter_scalar_field(scalar_field, passes=5, alpha=0.5, strides=(1, 2, 4)):
    """
    Apply a multi-pass 3-point binomial digital filter to a scalar field.
    
    Args:
        phi: Input scalar field array to be filtered.
        passes: Number of filter passes (default: 5). Note: internally capped at 17 
                (16 regular passes + 1 compensation pass).
        alpha: Filter strength parameter (default: 0.5).
        strides: Tuple/list of stride values for filtering (default: (1, 2, 4)).
    
    Returns:
        Filtered scalar field array.
    """
    # Accept strides as tuple/list/array; convert to JAX array so it's OK as a dynamic arg.
    s = jnp.asarray(strides)

    def body(y, stride):
        return _repeat_filter(y, stride, passes, alpha), None

    y, _ = lax.scan(body, scalar_field, s)
    return y


@jit
def filter_vector_field(F, passes=5, alpha=0.5, strides=(1, 2, 4)):
    """
    Apply digital filter along the grid axis (axis=0) for each component.
    F has shape (G, C), typically (grid_points, 3) for a vector field.

    Args:
        F: Input vector field array, shape (G, C), typically (grid_points, 3).
        passes: Number of filter passes (default: 5). Note: internally capped at 17 
                (16 regular passes + 1 compensation pass).
        alpha: Filter strength parameter (default: 0.5).
        strides: Tuple/list of stride values for filtering (default: (1, 2, 4)).
    
    Returns:
        Filtered vector field array with the same shape as input.
    """
    s = jnp.asarray(strides)

    def body(y, stride):
        return _repeat_filter(y, stride, passes, alpha), None

    y, _ = lax.scan(body, F, s)
    return y