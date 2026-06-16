from copy import deepcopy

from jaxincell._parameters._species_parameters import (
    clean_and_initialize_species_parameters,
)
from tests.helpers import scalar


def clean_species_parameters(species_parameters=None, input_parameters=None):
    return clean_and_initialize_species_parameters(
        deepcopy(species_parameters or {}),
        deepcopy(input_parameters or {}),
    )


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


def test_species_parameter_validation_rejects_invalid_particle_counts_and_weights():
    """Test jaxincell._parameters._species_parameters.validate_species_parameters.

    Cases to implement:
    - number_pseudoparticles must be a positive integer for ions and electrons.
    - grid_points_per_Debye_length must be positive.
    - weight must be nonnegative.
    - boolean flags must be actual bool values, not integers or strings.
    """


def test_species_parameter_validation_rejects_invalid_ion_specific_values():
    """Test ion-specific validation in clean_and_initialize_species_parameters.

    Cases to implement:
    - mass_over_proton_mass must be positive.
    - ion_temperature_over_electron_temperature_* values must be nonnegative.
    - seed_position_override=True requires a positive integer seed_position.
    """


def test_species_reference_resolution_ion_to_electron_and_electron_to_ion():
    """Test jaxincell._parameters._species_parameters.resolve_species_references.

    Cases to implement:
    - ion vth_over_c_* referencing an electron species applies temperature and mass scaling.
    - electron vth_over_c_* referencing an ion species applies inverse temperature and mass scaling.
    - non-vth referenced values are copied directly.
    """


def test_species_reference_resolution_rejects_invalid_and_chained_references():
    """Test jaxincell._parameters._species_parameters.resolve_reference_value.

    Cases to implement:
    - references to missing species labels raise ValueError during resolution setup.
    - references to another unresolved reference raise AssertionError.
    - error messages identify the species and key involved.
    """


def test_build_species_hash_stability_and_non_species_metadata():
    """Test jaxincell._parameters._species_parameters.build_species_hash.

    Cases to implement:
    - identical cleaned species dictionaries produce identical hashes.
    - changing a species parameter changes the hash.
    - non-species metadata keys are included in the pre-species hash component.
    """
