import pytest

from jaxincell._parameters._solver_parameters import (
    DEFAULT_SOLVER_PARAMETERS,
    clean_and_initialize_solver_parameters,
    build_solver_hash,
)

def test_clean_and_initialize_solver_parameters_defaults_and_tuple_coercion():
    """Test jaxincell._parameters._solver_parameters.clean_and_initialize_solver_parameters.

    Cases to implement:
    - defaults are populated when no solver parameters are supplied.
    - input_parameters override explicit solver_parameters for matching keys.
    - list filter_strides are converted to tuples.
    - tolerance_Picard_iterations_implicit_CN is converted to float.
    """
    default_applied = clean_and_initialize_solver_parameters({})
    assert default_applied == DEFAULT_SOLVER_PARAMETERS

    input_parameters = {"filter_strides": [1, 3, 5], "tolerance_Picard_iterations_implicit_CN": "1e-2"}
    overridden = clean_and_initialize_solver_parameters(input_parameters)
    assert overridden["filter_strides"] == (1, 3, 5)
    assert overridden["tolerance_Picard_iterations_implicit_CN"] == 1e-2

    overridden_input = clean_and_initialize_solver_parameters({}, input_parameters)
    assert overridden_input["filter_strides"] == (1, 3, 5)
    assert overridden_input["tolerance_Picard_iterations_implicit_CN"] == 1e-2


def test_clean_and_initialize_solver_parameters_rejects_invalid_values():
    """Test jaxincell._parameters._solver_parameters.clean_and_initialize_solver_parameters.

    Cases to implement:
    - field_solver and time_evolution_algorithm reject unsupported values.
    - Picard iteration counts and substeps must be positive integers.
    - filter_passes must be a nonnegative integer.
    - filter_alpha must be strictly between zero and one.
    - filter_strides must contain positive integers.
    """
    input_parameters = {"field_solver": 2}
    with pytest.raises(AssertionError, match="Invalid field solver. Must be 0"):
        clean_and_initialize_solver_parameters({}, input_parameters)
    
    input_parameters = {"time_evolution_algorithm": 2}
    with pytest.raises(AssertionError, match="Invalid time evolution algorithm. Must be 0"):
        clean_and_initialize_solver_parameters({}, input_parameters)

    input_parameters = {"max_number_of_Picard_iterations_implicit_CN": -1}
    with pytest.raises(AssertionError, match="Maximum number of Picard iterations for implicit Crank-Nicholson method must be a positive integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"max_number_of_Picard_iterations_implicit_CN": 0}
    with pytest.raises(AssertionError, match="Maximum number of Picard iterations for implicit Crank-Nicholson method must be a positive integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"max_number_of_Picard_iterations_implicit_CN": 1.5}
    with pytest.raises(AssertionError, match="Maximum number of Picard iterations for implicit Crank-Nicholson method must be a positive integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)

    input_parameters = {"number_of_particle_substeps_implicit_CN": 0}
    with pytest.raises(AssertionError, match="Number of particle substeps per field step for implicit Crank-Nicholson method must be a positive integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"number_of_particle_substeps_implicit_CN": -1}
    with pytest.raises(AssertionError, match="Number of particle substeps per field step for implicit Crank-Nicholson method must be a positive integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"number_of_particle_substeps_implicit_CN": 1.5}
    with pytest.raises(AssertionError, match="Number of particle substeps per field step for implicit Crank-Nicholson method must be a positive integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)

    input_parameters = {"tolerance_Picard_iterations_implicit_CN": -1}
    with pytest.raises(AssertionError, match="Tolerance for Picard iterations in implicit Crank-Nicholson method must be positive."):
        clean_and_initialize_solver_parameters({}, input_parameters)

    input_parameters = {"filter_passes": -1}
    with pytest.raises(AssertionError, match="Number of passes of the digital filter must be a non-negative integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"filter_passes": 1.5}
    with pytest.raises(AssertionError, match="Number of passes of the digital filter must be a non-negative integer."):
        clean_and_initialize_solver_parameters({}, input_parameters)

    input_parameters = {"filter_alpha": 0}
    with pytest.raises(AssertionError, match="Filter strength must be a float between 0 and 1."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"filter_alpha": 1}
    with pytest.raises(AssertionError, match="Filter strength must be a float between 0 and 1."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"filter_alpha": -0.5}
    with pytest.raises(AssertionError, match="Filter strength must be a float between 0 and 1."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"filter_alpha": 1.5}
    with pytest.raises(AssertionError, match="Filter strength must be a float between 0 and 1."):
        clean_and_initialize_solver_parameters({}, input_parameters)

    input_parameters = {"filter_strides": (1, -2, 3)}
    with pytest.raises(AssertionError, match="Filter strides must be a tuple of positive integers."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"filter_strides": (1, 0, 3)}
    with pytest.raises(AssertionError, match="Filter strides must be a tuple of positive integers."):
        clean_and_initialize_solver_parameters({}, input_parameters)
    input_parameters = {"filter_strides": (1, 2.5, 3)}
    with pytest.raises(AssertionError, match="Filter strides must be a tuple of positive integers."):
        clean_and_initialize_solver_parameters({}, input_parameters)


def test_build_solver_hash_is_stable_and_sensitive_to_values():
    """Test jaxincell._parameters._solver_parameters.build_solver_hash.

    Cases to implement:
    - identical cleaned solver parameters produce identical hashes.
    - changing filter_alpha changes the hash.
    - changing an algorithm selection changes the hash.
    """
    default_parameters = clean_and_initialize_solver_parameters({})
    default_hash = build_solver_hash(default_parameters)
    from_default_parameters = clean_and_initialize_solver_parameters({}, default_parameters)
    from_default_hash = build_solver_hash(from_default_parameters)
    assert default_hash == from_default_hash

    alpha_changed_parameters = clean_and_initialize_solver_parameters({"filter_alpha": 0.6})
    alpha_changed_hash = build_solver_hash(alpha_changed_parameters)
    assert default_hash != alpha_changed_hash

    field_solver_changed_parameters = clean_and_initialize_solver_parameters({"field_solver": 1})
    field_solver_changed_hash = build_solver_hash(field_solver_changed_parameters)
    assert default_hash != field_solver_changed_hash

    time_evolution_algorithm_changed_parameters = clean_and_initialize_solver_parameters({"time_evolution_algorithm": 1})
    time_evolution_algorithm_changed_hash = build_solver_hash(time_evolution_algorithm_changed_parameters)
    assert default_hash != time_evolution_algorithm_changed_hash
