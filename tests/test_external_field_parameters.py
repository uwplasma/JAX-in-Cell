from jaxincell._parameters._external_field_parameters import (
    clean_and_initialize_external_field_parameters,
    DEFAULT_EXTERNAL_FIELD_PARAMETERS,
    build_external_field_hash,
)

def test_clean_and_initialize_external_field_parameters_defaults_and_float_conversion():
    """Test jaxincell._parameters._external_field_parameters.clean_and_initialize_external_field_parameters.

    Cases to implement:
    - defaults are populated when no external field parameters are supplied.
    - input_parameters override explicit external_field_parameters for matching keys.
    - electric and magnetic amplitudes and wavenumbers are converted to floats.
    - callable external field functions are preserved without being called.
    """
    default_applied = clean_and_initialize_external_field_parameters({})
    assert default_applied == DEFAULT_EXTERNAL_FIELD_PARAMETERS

    
    input_parameters = {"external_electric_field_amplitude": "1.0", "external_magnetic_field_amplitude": 2, "external_electric_field_wavenumber": "3.0", "external_magnetic_field_wavenumber": 4}
    overridden = clean_and_initialize_external_field_parameters(input_parameters)
    assert overridden["external_electric_field_amplitude"] == 1.0
    assert overridden["external_magnetic_field_amplitude"] == 2.0
    assert overridden["external_electric_field_wavenumber"] == 3.0
    assert overridden["external_magnetic_field_wavenumber"] == 4.0

    overridden_input = clean_and_initialize_external_field_parameters({}, input_parameters)
    assert overridden_input["external_electric_field_amplitude"] == 1.0
    assert overridden_input["external_magnetic_field_amplitude"] == 2.0
    assert overridden_input["external_electric_field_wavenumber"] == 3.0
    assert overridden_input["external_magnetic_field_wavenumber"] == 4.0

    def example_external_field_function(x):
        return x**2
    input_parameters = {"external_electric_field_function": example_external_field_function}
    preserved = clean_and_initialize_external_field_parameters({}, input_parameters)
    assert preserved["external_electric_field_function"] is example_external_field_function


def test_build_external_field_hash_is_stable_and_sensitive_to_values():
    """Test jaxincell._parameters._external_field_parameters.build_external_field_hash.

    Cases to implement:
    - identical cleaned external field parameters produce identical hashes.
    - changing electric or magnetic amplitude changes the hash.
    - changing electric or magnetic wavenumber changes the hash.
    """
    default_parameters = clean_and_initialize_external_field_parameters({})
    default_hash = build_external_field_hash(default_parameters)
    from_default_parameters = clean_and_initialize_external_field_parameters(DEFAULT_EXTERNAL_FIELD_PARAMETERS)
    from_default_hash = build_external_field_hash(from_default_parameters)
    assert default_hash == from_default_hash

    amplitude_modified_parameters = clean_and_initialize_external_field_parameters({"external_electric_field_amplitude": 1.0})
    amplitude_modified_hash = build_external_field_hash(amplitude_modified_parameters)
    assert default_hash != amplitude_modified_hash

    wavenumber_modified_parameters = clean_and_initialize_external_field_parameters({"external_magnetic_field_wavenumber": 1.0})
    wavenumber_modified_hash = build_external_field_hash(wavenumber_modified_parameters)
    assert default_hash != wavenumber_modified_hash
