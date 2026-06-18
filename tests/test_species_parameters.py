from copy import deepcopy

from numpy.testing import assert_allclose
import pytest

from jaxincell._constants import mass_electron, mass_proton
from jaxincell._parameters._species_parameters import (
    build_species_hash,
    clean_and_initialize_species_parameters,
)
from tests.helpers import scalar


def clean_species_parameters(species_parameters=None, input_parameters=None):
    return clean_and_initialize_species_parameters(
        deepcopy(species_parameters or {}),
        deepcopy(input_parameters or {}),
    )


def valid_two_species_input():
    return {
        "ions": {
            "beam": {
                "number_pseudoparticles": 4,
                "mass_over_proton_mass": 2.0,
                "vth_over_c_x": 0.01,
                "vth_over_c_y": 0.01,
                "vth_over_c_z": 0.01,
            },
        },
        "electrons": {
            "cloud": {
                "number_pseudoparticles": 5,
                "vth_over_c_x": 0.03,
            },
        },
    }


def test_empty_species_input_builds_initial_ion_and_electron_species():
    species_parameters = clean_species_parameters()

    assert list(species_parameters["ions"]) == ["_ions0"]
    assert list(species_parameters["electrons"]) == ["_electrons0"]

    ion = species_parameters["ions"]["_ions0"]
    electron = species_parameters["electrons"]["_electrons0"]

    assert ion["user_label"] == "ions0"
    assert electron["user_label"] == "electrons0"
    assert ion["number_pseudoparticles"] == 500
    assert electron["number_pseudoparticles"] == 500
    assert scalar(ion["grid_points_per_Debye_length"]) == 2.0
    assert scalar(electron["grid_points_per_Debye_length"]) == 2.0
    assert scalar(ion["perturbation_amplitude_x"]) == 1e-7
    assert scalar(electron["perturbation_amplitude_x"]) == 1e-7
    assert scalar(electron["perturbation_wavenumber_x"]) == 8.0
    assert scalar(electron["drift_speed_x"]) == 1e8
    assert scalar(electron["vth_over_c_x"]) == 0.05
    assert electron["velocity_plus_minus_x"] is True
    assert not isinstance(ion["vth_over_c_x"], str)


def test_nested_species_input_merges_initial_defaults_into_first_species():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {
                "beam": {
                    "number_pseudoparticles": 4,
                    "mass_over_proton_mass": 2.0,
                    "drift_speed_x": 7.0,
                },
            },
            "electrons": {
                "cloud": {
                    "number_pseudoparticles": 5,
                    "vth_over_c_x": 0.03,
                    "drift_speed_x": -3.0,
                },
            },
        },
    )

    assert list(species_parameters["ions"]) == ["_ions0"]
    assert list(species_parameters["electrons"]) == ["_electrons0"]

    ion = species_parameters["ions"]["_ions0"]
    electron = species_parameters["electrons"]["_electrons0"]

    assert ion["user_label"] == "beam"
    assert electron["user_label"] == "cloud"
    assert ion["number_pseudoparticles"] == 4
    assert electron["number_pseudoparticles"] == 5
    assert scalar(ion["grid_points_per_Debye_length"]) == 2.0
    assert scalar(electron["grid_points_per_Debye_length"]) == 2.0
    assert scalar(ion["perturbation_amplitude_x"]) == 1e-7
    assert scalar(electron["perturbation_amplitude_x"]) == 1e-7
    assert scalar(ion["drift_speed_x"]) == 7.0
    assert scalar(electron["drift_speed_x"]) == -3.0


def test_loose_species_input_is_packed_before_canonical_numbering():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {
                "number_pseudoparticles": 4,
                "mass_over_proton_mass": 3.0,
                "drift_speed_x": 2.0,
            },
            "electrons": {
                "number_pseudoparticles": 5,
                "vth_over_c_x": 0.04,
                "drift_speed_x": -2.0,
            },
        },
    )

    ion = species_parameters["ions"]["_ions0"]
    electron = species_parameters["electrons"]["_electrons0"]

    assert ion["user_label"] == "ions0"
    assert electron["user_label"] == "electrons0"
    assert ion["number_pseudoparticles"] == 4
    assert electron["number_pseudoparticles"] == 5
    assert scalar(ion["mass_over_proton_mass"]) == 3.0
    assert scalar(ion["perturbation_amplitude_x"]) == 1e-7
    assert scalar(electron["perturbation_amplitude_x"]) == 1e-7
    assert scalar(ion["drift_speed_x"]) == 2.0
    assert scalar(electron["drift_speed_x"]) == -2.0


def test_additional_nested_species_use_plain_defaults():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {
                "beam0": {
                    "number_pseudoparticles": 4,
                    "mass_over_proton_mass": 2.0,
                },
                "beam1": {
                    "number_pseudoparticles": 6,
                    "mass_over_proton_mass": 3.0,
                },
            },
            "electrons": {
                "cloud0": {
                    "number_pseudoparticles": 5,
                },
                "cloud1": {
                    "number_pseudoparticles": 7,
                },
            },
        },
    )

    assert list(species_parameters["ions"]) == ["_ions0", "_ions1"]
    assert list(species_parameters["electrons"]) == ["_electrons0", "_electrons1"]

    first_ion = species_parameters["ions"]["_ions0"]
    second_ion = species_parameters["ions"]["_ions1"]
    first_electron = species_parameters["electrons"]["_electrons0"]
    second_electron = species_parameters["electrons"]["_electrons1"]

    assert first_ion["user_label"] == "beam0"
    assert second_ion["user_label"] == "beam1"
    assert first_electron["user_label"] == "cloud0"
    assert second_electron["user_label"] == "cloud1"
    assert scalar(first_ion["perturbation_amplitude_x"]) == 1e-7
    assert scalar(second_ion["perturbation_amplitude_x"]) == 0.0
    assert not isinstance(first_ion["vth_over_c_x"], str)
    assert scalar(second_ion["vth_over_c_x"]) == 0.0
    assert scalar(first_electron["drift_speed_x"]) == 1e8
    assert scalar(second_electron["drift_speed_x"]) == 0.0
    assert first_electron["velocity_plus_minus_x"] is True
    assert second_electron["velocity_plus_minus_x"] is False


def test_nested_input_parameters_override_matching_nested_species_values():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {
                "beam": {
                    "number_pseudoparticles": 4,
                    "mass_over_proton_mass": 2.0,
                    "drift_speed_x": 1.0,
                },
            },
            "electrons": {
                "cloud": {
                    "number_pseudoparticles": 5,
                    "drift_speed_x": -1.0,
                },
            },
        },
        input_parameters={
            "ions": {
                "beam": {
                    "drift_speed_x": 6.0,
                },
            },
            "electrons": {
                "cloud": {
                    "drift_speed_x": -6.0,
                },
            },
        },
    )

    ion = species_parameters["ions"]["_ions0"]
    electron = species_parameters["electrons"]["_electrons0"]

    assert scalar(ion["drift_speed_x"]) == 6.0
    assert scalar(electron["drift_speed_x"]) == -6.0


@pytest.mark.parametrize(
    ("species_type", "species_label", "key", "value", "error_type", "error_match"),
    [
        ("ions", "beam", "number_pseudoparticles", 0, AssertionError, "Number of pseudoparticles"),
        ("ions", "beam", "number_pseudoparticles", -1, AssertionError, "Number of pseudoparticles"),
        ("electrons", "cloud", "number_pseudoparticles", 1.5, AssertionError, "Number of pseudoparticles"),
        ("electrons", "cloud", "number_pseudoparticles", "5", ValueError, "referenced 5 for number_pseudoparticles"),
        ("ions", "beam", "grid_points_per_Debye_length", 0, AssertionError, "Grid points per Debye length"),
        ("electrons", "cloud", "grid_points_per_Debye_length", -1, AssertionError, "Grid points per Debye length"),
        ("ions", "beam", "weight", -0.1, AssertionError, "Weight must be non-negative"),
        ("electrons", "cloud", "weight", -0.1, AssertionError, "Weight must be non-negative"),
        ("ions", "beam", "random_positions_x", 1, AssertionError, "random_positions_x must be a boolean"),
        ("electrons", "cloud", "velocity_plus_minus_x", "true", ValueError, "referenced true for velocity_plus_minus_x"),
        ("ions", "beam", "seed_position_override", 1, AssertionError, "seed_position_override must be a boolean"),
    ],
)
def test_species_parameter_validation_rejects_invalid_particle_counts_and_weights(
    species_type,
    species_label,
    key,
    value,
    error_type,
    error_match,
):
    """Test jaxincell._parameters._species_parameters.validate_species_parameters.

    Cases covered:
    - number_pseudoparticles must be a positive integer for ions and electrons.
    - grid_points_per_Debye_length must be positive.
    - weight must be nonnegative.
    - boolean flags must be actual bool values, not integers or strings.
    """
    species_parameters = valid_two_species_input()
    species_parameters[species_type][species_label][key] = value

    with pytest.raises(error_type, match=error_match):
        clean_species_parameters(species_parameters)


@pytest.mark.parametrize(
    ("key", "value", "error_match"),
    [
        ("mass_over_proton_mass", 0, "Mass over proton mass"),
        ("mass_over_proton_mass", -1, "Mass over proton mass"),
        ("ion_temperature_over_electron_temperature_x", -0.1, "Ion temperature over electron temperature x"),
        ("ion_temperature_over_electron_temperature_y", -0.1, "Ion temperature over electron temperature y"),
        ("ion_temperature_over_electron_temperature_z", -0.1, "Ion temperature over electron temperature z"),
    ],
)
def test_species_parameter_validation_rejects_invalid_ion_specific_values(
    key,
    value,
    error_match,
):
    """Test ion-specific validation in clean_and_initialize_species_parameters.

    Cases covered:
    - mass_over_proton_mass must be positive.
    - ion_temperature_over_electron_temperature_* values must be nonnegative.
    - seed_position_override=True requires a positive integer seed_position.
    """
    species_parameters = valid_two_species_input()
    species_parameters["ions"]["beam"][key] = value

    with pytest.raises(AssertionError, match=error_match):
        clean_species_parameters(species_parameters)

    valid_seed_placeholder = valid_two_species_input()
    valid_seed_placeholder["ions"]["beam"]["seed_position_override"] = False
    valid_seed_placeholder["ions"]["beam"]["seed_position"] = None

    clean_species_parameters(valid_seed_placeholder)

    for invalid_seed_position in (None, 0, -1, 1.5):
        invalid_seed = valid_two_species_input()
        invalid_seed["ions"]["beam"]["seed_position_override"] = True
        invalid_seed["ions"]["beam"]["seed_position"] = invalid_seed_position

        with pytest.raises(AssertionError, match="Seed position"):
            clean_species_parameters(invalid_seed)


def test_species_reference_resolution_ion_to_electron_and_electron_to_ion():
    """Test jaxincell._parameters._species_parameters.resolve_species_references.

    Cases covered:
    - ion vth_over_c_* referencing an electron species applies temperature and mass scaling.
    - electron vth_over_c_* referencing an ion species applies inverse temperature and mass scaling.
    - non-vth referenced values are copied directly.
    """
    ion_reference_parameters = clean_species_parameters(
        {
            "ions": {
                "beam": {
                    "number_pseudoparticles": 4,
                    "mass_over_proton_mass": 2.0,
                    "ion_temperature_over_electron_temperature_x": 4.0,
                    "vth_over_c_x": "_electrons0",
                    "drift_speed_y": "_electrons0",
                },
            },
            "electrons": {
                "cloud": {
                    "number_pseudoparticles": 5,
                    "vth_over_c_x": 0.2,
                    "drift_speed_y": 42.0,
                },
            },
        },
    )

    ion = ion_reference_parameters["ions"]["_ions0"]
    expected_ion_vth = (
        2.0 * 0.2 * (mass_electron / (2.0 * mass_proton)) ** 0.5
    )

    assert ion["_reference_vth_over_c_x"] == ("electrons", "_electrons0")
    assert ion["_reference_drift_speed_y"] == ("electrons", "_electrons0")
    assert_allclose(scalar(ion["vth_over_c_x"]), expected_ion_vth)
    assert scalar(ion["drift_speed_y"]) == 42.0

    electron_reference_parameters = clean_species_parameters(
        {
            "ions": {
                "beam": {
                    "number_pseudoparticles": 4,
                    "mass_over_proton_mass": 2.0,
                    "ion_temperature_over_electron_temperature_x": 4.0,
                    "vth_over_c_x": 0.03,
                    "drift_speed_y": 13.0,
                },
            },
            "electrons": {
                "cloud": {
                    "number_pseudoparticles": 5,
                    "vth_over_c_x": "_ions0",
                    "drift_speed_y": "_ions0",
                },
            },
        },
    )

    electron = electron_reference_parameters["electrons"]["_electrons0"]
    expected_electron_vth = (
        0.5 * 0.03 * (2.0 * mass_proton / mass_electron) ** 0.5
    )

    assert electron["_reference_vth_over_c_x"] == ("ions", "_ions0")
    assert electron["_reference_drift_speed_y"] == ("ions", "_ions0")
    assert_allclose(scalar(electron["vth_over_c_x"]), expected_electron_vth)
    assert scalar(electron["drift_speed_y"]) == 13.0


def test_species_reference_resolution_rejects_invalid_and_chained_references():
    """Test jaxincell._parameters._species_parameters.resolve_reference_value.

    Cases covered:
    - references to missing species labels raise ValueError during resolution setup.
    - references to another unresolved reference raise AssertionError.
    - error messages identify the species and key involved.
    """
    missing_reference_parameters = valid_two_species_input()
    missing_reference_parameters["ions"]["beam"]["vth_over_c_x"] = "missing_species"

    with pytest.raises(
        ValueError,
        match="ions _ions0 referenced missing_species for vth_over_c_x",
    ):
        clean_species_parameters(missing_reference_parameters)

    chained_reference_parameters = {
        "ions": {
            "beam": {
                "number_pseudoparticles": 4,
                "mass_over_proton_mass": 2.0,
                "vth_over_c_x": "_electrons0",
            },
        },
        "electrons": {
            "cloud": {
                "number_pseudoparticles": 5,
                "vth_over_c_x": "_ions0",
            },
        },
    }

    with pytest.raises(
        AssertionError,
        match="electrons _electrons0 referencing _ions0",
    ):
        clean_species_parameters(chained_reference_parameters)


def test_build_species_hash_stability_and_non_species_metadata():
    """Test jaxincell._parameters._species_parameters.build_species_hash.

    Cases covered:
    - identical cleaned species dictionaries produce identical hashes.
    - changing a species parameter changes the hash.
    - non-species metadata keys are included in the pre-species hash component.
    """
    species_parameters = clean_species_parameters(valid_two_species_input())
    matching_species_parameters = clean_species_parameters(valid_two_species_input())

    assert build_species_hash(species_parameters) == build_species_hash(
        matching_species_parameters
    )

    changed_species_parameters = deepcopy(species_parameters)
    changed_species_parameters["electrons"]["_electrons0"]["drift_speed_x"] = 123.0

    assert build_species_hash(species_parameters) != build_species_hash(
        changed_species_parameters
    )

    with_metadata = deepcopy(species_parameters)
    with_metadata["metadata"] = "first"
    changed_metadata = deepcopy(species_parameters)
    changed_metadata["metadata"] = "second"

    assert build_species_hash(species_parameters) != build_species_hash(with_metadata)
    assert build_species_hash(with_metadata) != build_species_hash(changed_metadata)
