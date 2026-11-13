import pytest

from jaxincell._parameters._domain_parameters import (
    clean_and_initialize_domain_parameters,
    DEFAULT_DOMAIN_PARAMETERS,
    build_domain_hash,
)

def test_clean_and_initialize_domain_parameters_defaults_and_input_precedence():
    """Test jaxincell._parameters._domain_parameters.clean_and_initialize_domain_parameters.

    Cases:
    - defaults are populated when no domain parameters are supplied.
    - input_parameters override explicit domain_parameters for matching keys.
    - length, length_y, and length_z are converted to JAX float arrays.
    """
    default_applied = clean_and_initialize_domain_parameters({})
    assert default_applied == DEFAULT_DOMAIN_PARAMETERS

    input_parameters = {"length": 2.0, "length_y": 3.0, "length_z": 4.0}
    overridden = clean_and_initialize_domain_parameters(input_parameters)
    assert overridden["length"] == 2.0
    assert overridden["length_y"] == 3.0
    assert overridden["length_z"] == 4.0

    overridden_input = clean_and_initialize_domain_parameters({}, input_parameters)
    assert overridden_input["length"] == 2.0
    assert overridden_input["length_y"] == 3.0
    assert overridden_input["length_z"] == 4.0

    input_parameters = {"length": "2.0", "length_y": 3, "length_z": "4.0"}
    converted = clean_and_initialize_domain_parameters({}, input_parameters)
    assert converted["length"] == 2.0
    assert converted["length_y"] == 3.0
    assert converted["length_z"] == 4.0


def test_clean_and_initialize_domain_parameters_rejects_invalid_values():
    """Test jaxincell._parameters._domain_parameters.clean_and_initialize_domain_parameters.

    Cases:
    - total_steps must be a positive integer.
    - length must be positive and transverse lengths must be nonnegative.
    - particle and field boundary conditions must be one of 0, 1, or 2.
    """
    input_parameters = {"total_steps": -1}
    with pytest.raises(AssertionError, match="Total number of time steps must be an integer."):
        clean_and_initialize_domain_parameters({}, input_parameters)

    input_parameters = {"length": -1.0}
    with pytest.raises(AssertionError, match="Length of the simulation box must be positive."):
        clean_and_initialize_domain_parameters({}, input_parameters)
    input_parameters = {"length": 0}
    with pytest.raises(AssertionError, match="Length of the simulation box must be positive."):
        clean_and_initialize_domain_parameters({}, input_parameters)
    input_parameters = {"length_y": -1.0}
    with pytest.raises(AssertionError, match="Length of the simulation box in y must be positive."):
        clean_and_initialize_domain_parameters({}, input_parameters)
    input_parameters = {"length_z": -1.0}
    with pytest.raises(AssertionError, match="Length of the simulation box in z must be positive."):
        clean_and_initialize_domain_parameters({}, input_parameters)
    
    for bc_key in ["particle_BC_left", "particle_BC_right", "field_BC_left", "field_BC_right"]:
        input_parameters = {bc_key: -1}
        with pytest.raises(AssertionError, match="Invalid .* boundary condition .* Must be 0 \\(periodic\\), 1 \\(reflecting\\), or 2 \\(absorbing\\)."):
            clean_and_initialize_domain_parameters({}, input_parameters)
        input_parameters = {bc_key: 3}
        with pytest.raises(AssertionError, match="Invalid .* boundary condition .* Must be 0 \\(periodic\\), 1 \\(reflecting\\), or 2 \\(absorbing\\)."):
            clean_and_initialize_domain_parameters({}, input_parameters)
        input_parameters = {bc_key: 1.5}
        with pytest.raises(AssertionError, match="Invalid .* boundary condition .* Must be 0 \\(periodic\\), 1 \\(reflecting\\), or 2 \\(absorbing\\)."):
            clean_and_initialize_domain_parameters({}, input_parameters)


def test_build_domain_hash_is_stable_and_sensitive_to_values():
    """Test jaxincell._parameters._domain_parameters.build_domain_hash.

    Cases:
    - identical cleaned domain parameters produce identical hashes.
    - changing a differentiable domain parameter changes the hash.
    - changing a non-differentiable domain parameter changes the hash.
    """
    default_parameters = clean_and_initialize_domain_parameters({})
    default_hash = build_domain_hash(default_parameters)
    from_default_parameters = clean_and_initialize_domain_parameters(DEFAULT_DOMAIN_PARAMETERS)
    from_default_hash = build_domain_hash(from_default_parameters)
    assert default_hash == from_default_hash

    length_changed_parameters = clean_and_initialize_domain_parameters({"length": 2.0})
    length_changed_hash = build_domain_hash(length_changed_parameters)
    assert default_hash != length_changed_hash

    total_steps_changed_parameters = clean_and_initialize_domain_parameters({"total_steps": 400})
    total_steps_changed_hash = build_domain_hash(total_steps_changed_parameters)
    assert default_hash != total_steps_changed_hash


def test_optional_transverse_cells_none_selects_one_dimensional_geometry():
    parameters = clean_and_initialize_domain_parameters({"number_grid_points_y": None,
                                                         "number_grid_points_z": None})
    assert parameters["number_grid_points_y"] == parameters["number_grid_points_z"] == 0


@pytest.mark.parametrize("kind", ["particle", "field"])
@pytest.mark.parametrize("left,right", [(0, 1), (0, 2), (1, 0), (2, 0)])
def test_periodic_boundaries_require_a_partner(kind, left, right):
    with pytest.raises(AssertionError, match=f"Periodic {kind} boundaries must be paired"):
        clean_and_initialize_domain_parameters({f"{kind}_BC_left": left, f"{kind}_BC_right": right})


def test_nonperiodic_wall_types_can_differ_between_sides():
    clean_and_initialize_domain_parameters({"particle_BC_left": 1, "particle_BC_right": 2,
                                          "field_BC_left": 2, "field_BC_right": 1})


@pytest.mark.parametrize("grid_points", [0, 1, -2, 2.5, True])
def test_grid_requires_two_integer_cells(grid_points):
    with pytest.raises(AssertionError, match="grid points must be an integer of at least two"):
        clean_and_initialize_domain_parameters({"number_grid_points": grid_points})


@pytest.mark.parametrize("ratio", [0., -1., float("nan"), float("inf")])
def test_time_step_ratio_must_be_finite_and_positive(ratio):
    with pytest.raises(AssertionError, match="Time step ratio must be finite and positive"):
        clean_and_initialize_domain_parameters({"timestep_over_spatialstep_times_c": ratio})
