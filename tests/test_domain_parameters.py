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
    - particle boundary codes are 0 through 4; field codes are 0 through 2.
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
        with pytest.raises(AssertionError, match="Invalid .* boundary condition"):
            clean_and_initialize_domain_parameters({}, input_parameters)
        input_parameters = {bc_key: 5}
        with pytest.raises(AssertionError, match="Invalid .* boundary condition"):
            clean_and_initialize_domain_parameters({}, input_parameters)
        input_parameters = {bc_key: 1.5}
        with pytest.raises(AssertionError, match="Invalid .* boundary condition"):
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


@pytest.mark.parametrize("parameter", ["mixed_BC_weight", "COR_left", "COR_right"])
@pytest.mark.parametrize("value", [-.1, 1.1, float("nan")])
def test_wall_return_and_restitution_fractions_reject_invalid_values(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        clean_and_initialize_domain_parameters({parameter: value})


@pytest.mark.parametrize("value", [0., -1., float("nan")])
def test_velocity_dependent_return_requires_a_positive_scale(value):
    with pytest.raises(ValueError, match="mixed_BC_velocity_scale"):
        clean_and_initialize_domain_parameters({"mixed_BC_velocity_scale": value})


@pytest.mark.parametrize("kind", ["particle", "field"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_one_periodic_face_cannot_be_combined_with_a_wall(kind, side):
    with pytest.raises(ValueError, match=f"periodic {kind} boundaries must be paired"):
        clean_and_initialize_domain_parameters({f"{kind}_BC_{side}": 1})
