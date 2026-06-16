import jax.numpy as jnp

__all__ = ["make_tuple", "make_tuple_values_floats", "as_float_parameter"]

def make_tuple(thing_to_make_a_tuple):
    if not isinstance(thing_to_make_a_tuple, (list, tuple)):
        thing_to_make_a_tuple = (thing_to_make_a_tuple,)
    else:
        thing_to_make_a_tuple = tuple(thing_to_make_a_tuple)
    return thing_to_make_a_tuple

def make_tuple_values_floats(thing_to_make_tuple_values_floats):
    thing_to_make_tuple_values_floats = list(thing_to_make_tuple_values_floats)
    for i in range(len(thing_to_make_tuple_values_floats)):
        thing_to_make_tuple_values_floats[i] = float(thing_to_make_tuple_values_floats[i])
    return tuple(thing_to_make_tuple_values_floats)

def as_float_parameter(value):
    return jnp.asarray(value, dtype=float)
