def test_make_tuple_scalar_list_and_tuple_cases():
    """Test jaxincell._utils.make_tuple.

    Cases to implement:
    - scalar input is wrapped in a one-item tuple.
    - list input is converted to a tuple with the same values.
    - tuple input is returned as an equivalent tuple.
    """


def test_make_tuple_values_floats_converts_all_entries():
    """Test jaxincell._utils.make_tuple_values_floats.

    Cases to implement:
    - integer and string numeric entries are converted to floats.
    - tuple and list inputs produce tuple outputs.
    - invalid nonnumeric values raise the underlying float-conversion error.
    """


def test_as_float_parameter_returns_jax_float_array():
    """Test jaxincell._utils.as_float_parameter.

    Cases to implement:
    - Python floats, integers, and NumPy scalar values become JAX arrays with float dtype.
    - the returned value works in JAX differentiation-sensitive code paths.
    - shape is preserved for array-like input.
    """
