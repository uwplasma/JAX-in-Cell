from copy import deepcopy

import pytest

from jaxincell._parameters._source_parameters import (
    DEFAULT_SOURCE_PARAMETERS,
    build_source_hash,
    clean_and_initialize_source_parameters,
    make_tuple,
    make_tuple_values_floats,
)


NORMALIZED_DEFAULT_SOURCE_PARAMETERS = {
    **DEFAULT_SOURCE_PARAMETERS,
    "source_species": (1,),
    "how_often_source_should_produce_quasiparticles": (20,),
    "source_particles_per_second": (1e16,),
    "location_of_source": (0,),
    "width_of_source": (1,),
    "injection_speed_x": (1e7,),
    "injection_speed_y": (0.0,),
    "injection_speed_z": (0.0,),
}


def test_make_tuple_scalar_list_and_tuple_cases():
    """Test jaxincell._parameters._source_parameters.make_tuple.

    Cases covered:
    - scalar input is wrapped in a one-item tuple.
    - list input is converted to a tuple with the same values.
    - tuple input is returned as an equivalent tuple.
    - string input is treated as a scalar, not an iterable.
    """
    assert make_tuple(3) == (3,)
    assert make_tuple(3.5) == (3.5,)
    assert make_tuple("abc") == ("abc",)
    assert make_tuple([1, 2, 3]) == (1, 2, 3)
    assert make_tuple((1, 2, 3)) == (1, 2, 3)


def test_make_tuple_values_floats_converts_all_entries():
    """Test jaxincell._parameters._source_parameters.make_tuple_values_floats.

    Cases covered:
    - integer and string numeric entries are converted to floats.
    - tuple and list inputs produce tuple outputs.
    - invalid nonnumeric values raise the underlying float-conversion error.
    """
    assert make_tuple_values_floats((1, 2.5, "3.5", "4e1")) == (1.0, 2.5, 3.5, 40.0)
    assert make_tuple_values_floats([1, "2"]) == (1.0, 2.0)

    with pytest.raises(ValueError):
        make_tuple_values_floats((1, "not-a-number"))


def test_clean_and_initialize_source_parameters_inactive_defaults_and_tuple_conversion():
    """Test jaxincell._parameters._source_parameters.clean_and_initialize_source_parameters.

    Cases covered:
    - inactive default source parameters clean successfully.
    - scalar source_species is converted to a tuple of integers.
    - source_particles_per_second and injection speeds are converted to tuples of floats.
    - input_parameters override explicit source_parameters for matching keys.
    - source parameter and input parameter dictionaries are not mutated.
    - inactive sources normalize tuple-like fields without broadcasting them.
    """
    default_applied = clean_and_initialize_source_parameters({})
    explicit_none_default_applied = clean_and_initialize_source_parameters(
        {},
        input_parameters=None,
    )
    assert default_applied == NORMALIZED_DEFAULT_SOURCE_PARAMETERS
    assert explicit_none_default_applied == NORMALIZED_DEFAULT_SOURCE_PARAMETERS

    input_parameters = {
        "source_species": 2,
        "source_particles_per_second": "2e16",
        "injection_speed_x": "1.5e7",
        "injection_speed_y": 3,
        "injection_speed_z": "4.0",
    }
    overridden = clean_and_initialize_source_parameters(input_parameters)
    assert overridden["source_species"] == (2,)
    assert overridden["source_particles_per_second"] == (2e16,)
    assert overridden["injection_speed_x"] == (1.5e7,)
    assert overridden["injection_speed_y"] == (3.0,)
    assert overridden["injection_speed_z"] == (4.0,)

    explicit_parameters = {
        "source_species": 0,
        "source_particles_per_second": 1e15,
        "injection_speed_x": 1e6,
    }
    input_overrides = {
        "source_species": 3,
        "source_particles_per_second": 5e15,
        "injection_speed_x": 6e6,
    }
    explicit_parameters_copy = deepcopy(explicit_parameters)
    input_overrides_copy = deepcopy(input_overrides)

    overridden_input = clean_and_initialize_source_parameters(
        explicit_parameters,
        input_overrides,
    )
    assert overridden_input["source_species"] == (3,)
    assert overridden_input["source_particles_per_second"] == (5e15,)
    assert overridden_input["injection_speed_x"] == (6e6,)
    assert explicit_parameters == explicit_parameters_copy
    assert input_overrides == input_overrides_copy

    inactive_multi_source = clean_and_initialize_source_parameters(
        {
            "source_species": [0, 1],
            "source_particles_per_second": [1e16],
            "how_often_source_should_produce_quasiparticles": [5],
            "location_of_source": [3],
            "width_of_source": [2],
            "injection_speed_x": [1e6],
            "injection_speed_y": [2e6],
            "injection_speed_z": [3e6],
        }
    )
    assert inactive_multi_source["source_species"] == (0, 1)
    assert inactive_multi_source["source_particles_per_second"] == (1e16,)
    assert inactive_multi_source["how_often_source_should_produce_quasiparticles"] == (5,)
    assert inactive_multi_source["location_of_source"] == (3,)
    assert inactive_multi_source["width_of_source"] == (2,)
    assert inactive_multi_source["injection_speed_x"] == (1e6,)
    assert inactive_multi_source["injection_speed_y"] == (2e6,)
    assert inactive_multi_source["injection_speed_z"] == (3e6,)


def test_clean_and_initialize_source_parameters_active_broadcasting_and_lengths():
    """Test jaxincell._parameters._source_parameters.clean_and_initialize_source_parameters.

    Cases covered:
    - active sources broadcast single-value tuple parameters to match source_species length.
    - multi-source tuple lengths matching source_species are preserved.
    - list values are accepted and converted to tuple values.
    - mismatched multi-value lengths raise AssertionError.
    """
    broadcasted = clean_and_initialize_source_parameters(
        {
            "source_term_active": 1,
            "source_species": (0, 1),
            "how_often_source_should_produce_quasiparticles": 5,
            "source_particles_per_second": 2e16,
            "location_of_source": 3,
            "width_of_source": 2,
            "injection_speed_x": 1e6,
            "injection_speed_y": 2e6,
            "injection_speed_z": 3e6,
        }
    )
    assert broadcasted["source_species"] == (0, 1)
    assert broadcasted["how_often_source_should_produce_quasiparticles"] == (5, 5)
    assert broadcasted["source_particles_per_second"] == (2e16, 2e16)
    assert broadcasted["location_of_source"] == (3, 3)
    assert broadcasted["width_of_source"] == (2, 2)
    assert broadcasted["injection_speed_x"] == (1e6, 1e6)
    assert broadcasted["injection_speed_y"] == (2e6, 2e6)
    assert broadcasted["injection_speed_z"] == (3e6, 3e6)

    preserved = clean_and_initialize_source_parameters(
        {
            "source_term_active": 1,
            "source_species": (0, 1),
            "how_often_source_should_produce_quasiparticles": (5, 10),
            "source_particles_per_second": (2e16, 3e16),
            "location_of_source": (1, 2),
            "width_of_source": (2, 3),
            "injection_speed_x": (1e6, 2e6),
            "injection_speed_y": (3e6, 4e6),
            "injection_speed_z": (5e6, 6e6),
        }
    )
    assert preserved["how_often_source_should_produce_quasiparticles"] == (5, 10)
    assert preserved["source_particles_per_second"] == (2e16, 3e16)
    assert preserved["location_of_source"] == (1, 2)
    assert preserved["width_of_source"] == (2, 3)
    assert preserved["injection_speed_x"] == (1e6, 2e6)
    assert preserved["injection_speed_y"] == (3e6, 4e6)
    assert preserved["injection_speed_z"] == (5e6, 6e6)

    list_values = clean_and_initialize_source_parameters(
        {
            "source_term_active": 1,
            "source_species": [0, 1],
            "how_often_source_should_produce_quasiparticles": [5, 10],
            "source_particles_per_second": [2e16, 3e16],
            "location_of_source": [1, 2],
            "width_of_source": [2, 3],
            "injection_speed_x": [1e6, 2e6],
            "injection_speed_y": [3e6, 4e6],
            "injection_speed_z": [5e6, 6e6],
        }
    )
    assert list_values["source_species"] == (0, 1)
    assert list_values["how_often_source_should_produce_quasiparticles"] == (5, 10)
    assert list_values["source_particles_per_second"] == (2e16, 3e16)
    assert list_values["location_of_source"] == (1, 2)
    assert list_values["width_of_source"] == (2, 3)
    assert list_values["injection_speed_x"] == (1e6, 2e6)
    assert list_values["injection_speed_y"] == (3e6, 4e6)
    assert list_values["injection_speed_z"] == (5e6, 6e6)

    input_parameters = {
        "source_term_active": 1,
        "source_species": (0, 1, 2),
        "source_particles_per_second": (1e16, 2e16),
    }
    with pytest.raises(AssertionError, match="Length of source_particles_per_second must match"):
        clean_and_initialize_source_parameters(input_parameters)
    input_parameters = {
        "source_term_active": 1,
        "source_species": (0, 1, 2),
        "location_of_source": (1, 2),
    }
    with pytest.raises(AssertionError, match="Length of location_of_source must match"):
        clean_and_initialize_source_parameters(input_parameters)


def test_clean_and_initialize_source_parameters_rejects_invalid_values():
    """Test jaxincell._parameters._source_parameters.clean_and_initialize_source_parameters.

    Cases covered:
    - source_term_active must be 0 or 1.
    - source_species values must be integers.
    - production cadence and source width values must be positive integers.
    - source_particles_per_second must be positive.
    - location_of_source must be one of 0, 1, 2, or 3.
    - injection speed values must be floats after conversion.
    """
    input_parameters = {"source_term_active": 2}
    with pytest.raises(AssertionError, match="Source term active must be 0"):
        clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"source_term_active": -1}
    with pytest.raises(AssertionError, match="Source term active must be 0"):
        clean_and_initialize_source_parameters({}, input_parameters)

    input_parameters = {"source_species": "ions"}
    with pytest.raises(AssertionError, match="All source species must be specified as integers"):
        clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"source_species": (0, 1.5)}
    with pytest.raises(AssertionError, match="All source species must be specified as integers"):
        clean_and_initialize_source_parameters({}, input_parameters)

    for bad_value in [0, -1, 1.5]:
        input_parameters = {"how_often_source_should_produce_quasiparticles": bad_value}
        with pytest.raises(AssertionError, match="must be positive integers"):
            clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"how_often_source_should_produce_quasiparticles": (5, 0)}
    with pytest.raises(AssertionError, match="must be positive integers"):
        clean_and_initialize_source_parameters({}, input_parameters)

    for bad_value in [0, -1.0]:
        input_parameters = {"source_particles_per_second": bad_value}
        with pytest.raises(AssertionError, match="must be positive"):
            clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"source_particles_per_second": (1e16, -1.0)}
    with pytest.raises(AssertionError, match="must be positive"):
        clean_and_initialize_source_parameters({}, input_parameters)

    input_parameters = {"location_of_source": -1}
    with pytest.raises(AssertionError, match="must be 0 \\(center\\), 1 \\(left\\), 2 \\(right\\), or 3"):
        clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"location_of_source": 4}
    with pytest.raises(AssertionError, match="must be 0 \\(center\\), 1 \\(left\\), 2 \\(right\\), or 3"):
        clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"location_of_source": 1.5}
    with pytest.raises(AssertionError, match="must be 0 \\(center\\), 1 \\(left\\), 2 \\(right\\), or 3"):
        clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"location_of_source": (0, 4)}
    with pytest.raises(AssertionError, match="must be 0 \\(center\\), 1 \\(left\\), 2 \\(right\\), or 3"):
        clean_and_initialize_source_parameters({}, input_parameters)

    for bad_value in [0, -1, 1.5]:
        input_parameters = {"width_of_source": bad_value}
        with pytest.raises(AssertionError, match="must be positive integers"):
            clean_and_initialize_source_parameters({}, input_parameters)
    input_parameters = {"width_of_source": (1, 0)}
    with pytest.raises(AssertionError, match="must be positive integers"):
        clean_and_initialize_source_parameters({}, input_parameters)

    for injection_speed_key in ("injection_speed_x", "injection_speed_y", "injection_speed_z"):
        input_parameters = {injection_speed_key: "not-a-number"}
        with pytest.raises(ValueError):
            clean_and_initialize_source_parameters({}, input_parameters)


def test_build_source_hash_is_stable_and_sensitive_to_values():
    """Test jaxincell._parameters._source_parameters.build_source_hash.

    Cases covered:
    - identical cleaned source parameters produce identical hashes.
    - changing source activity changes the hash.
    - changing a source injection parameter changes the hash.
    - changing a non-injection source parameter changes the hash.
    """
    default_parameters = clean_and_initialize_source_parameters({})
    default_hash = build_source_hash(default_parameters)
    from_default_parameters = clean_and_initialize_source_parameters(DEFAULT_SOURCE_PARAMETERS)
    from_default_hash = build_source_hash(from_default_parameters)
    assert default_hash == from_default_hash

    active_changed_parameters = clean_and_initialize_source_parameters({"source_term_active": 1})
    active_changed_hash = build_source_hash(active_changed_parameters)
    assert default_hash != active_changed_hash

    injection_changed_parameters = clean_and_initialize_source_parameters({"injection_speed_x": 2e7})
    injection_changed_hash = build_source_hash(injection_changed_parameters)
    assert default_hash != injection_changed_hash

    location_changed_parameters = clean_and_initialize_source_parameters({"location_of_source": 3})
    location_changed_hash = build_source_hash(location_changed_parameters)
    assert default_hash != location_changed_hash

    species_changed_parameters = clean_and_initialize_source_parameters({"source_species": 0})
    species_changed_hash = build_source_hash(species_changed_parameters)
    assert default_hash != species_changed_hash
