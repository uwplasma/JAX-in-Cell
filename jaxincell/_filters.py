import jax.numpy as jnp
from jax import jit, lax

@jit
def binomial_filter_3point(x, alpha=0.5, stride=1):
    # 3-point: x^f_j = α x_j + (1-α)(x_{j-1}+x_{j+1})/2
    # periodic ends assumed at call site (we’ll pad/roll in callers)
    return alpha * x + (1 - alpha) * 0.5 * (
        jnp.roll(x, stride, axis=0) + jnp.roll(x, -stride, axis=0)
    )

@jit
def _repeat_filter(y, stride, passes, alpha):
    # (passes-1) regular passes
    def body(_, val):
        return binomial_filter_3point(val, alpha, stride=stride)
    y = lax.fori_loop(0, jnp.maximum(passes - 1, 0), body, y)
    # compensation pass - sharp cutoff in k-space (see https://warpx.readthedocs.io/en/latest/theory/pic.html)
    comp_alpha = passes - alpha * (passes - 1)
    y = binomial_filter_3point(y, comp_alpha, stride=stride)
    return y

@jit
def filter_scalar_field(phi, passes=5, alpha=0.5, strides=(1, 2, 4)):
    # Accept strides as tuple/list/array; convert to JAX array so it's OK as a dynamic arg.
    s = jnp.asarray(strides)
    def body(y, stride):
        return _repeat_filter(y, stride, passes, alpha), None
    y, _ = lax.scan(body, phi, s)
    return y

@jit
def filter_vector_field(F, passes=5, alpha=0.5, strides=(1, 2, 4)):
    s = jnp.asarray(strides)
    def body(y, stride):
        return _repeat_filter(y, stride, passes, alpha), None
    y, _ = lax.scan(body, F, s)
    return y