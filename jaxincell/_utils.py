import jax.numpy as jnp

__all__ = ["make_tuple", "make_tuple_values_floats", "as_float_parameter", "make_differentiable_type"]

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

def make_differentiable_type(value):
        """
        Convert a parameter value to a type that can be used as a differentiable input to the simulation function.
        For example, if the value is a list or tuple, convert it to a jax array. If it's a single value, convert it to a
        scalar jax array.
        """
        return as_float_parameter(value)
