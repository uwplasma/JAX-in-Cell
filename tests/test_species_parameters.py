from copy import deepcopy

import numpy as np

from jaxincell._parameters._species_parameters import (
    clean_and_initialize_species_parameters,
)


def clean_species_parameters(species_parameters=None, input_parameters=None):
    return clean_and_initialize_species_parameters(
        deepcopy(species_parameters or {}),
        deepcopy(input_parameters or {}),
    )


def scalar(value):
    return float(np.asarray(value))


def test_legacy_species_input_builds_default_ion_and_electron_species():
    species_parameters = clean_species_parameters(
        input_parameters={
            "number_pseudoelectrons": 7,
            "grid_points_per_Debye_length": 3.0,
            "electron_drift_speed_x": 12.0,
            "ion_drift_speed_x": -4.0,
            "vth_electrons_over_c_x": 0.2,
        }
    )

    assert list(species_parameters["ions"]) == ["_ions0"]
    assert list(species_parameters["electrons"]) == ["_electrons0"]

    ion = species_parameters["ions"]["_ions0"]
    electron = species_parameters["electrons"]["_electrons0"]

    assert ion["user_label"] == "_legacy_ions"
    assert electron["user_label"] == "_legacy_electrons"
    assert ion["number_pseudoparticles"] == 7
    assert electron["number_pseudoparticles"] == 7
    assert scalar(ion["grid_points_per_Debye_length"]) == 3.0
    assert scalar(electron["grid_points_per_Debye_length"]) == 3.0
    assert scalar(ion["drift_speed_x"]) == -4.0
    assert scalar(electron["drift_speed_x"]) == 12.0
    assert scalar(electron["vth_over_c_x"]) == 0.2
    assert not isinstance(ion["vth_over_c_x"], str)


def test_nested_species_input_merges_legacy_defaults_into_first_species():
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
    assert scalar(ion["drift_speed_x"]) == 2.0
    assert scalar(electron["drift_speed_x"]) == -2.0


def test_mixed_legacy_and_nested_species_creates_both_legacy_species():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {
                "beam": {
                    "number_pseudoparticles": 4,
                    "mass_over_proton_mass": 2.0,
                },
            },
            "electrons": {
                "cloud": {
                    "number_pseudoparticles": 5,
                },
            },
        },
        input_parameters={
            "electron_drift_speed_x": 99.0,
            "ion_drift_speed_x": -8.0,
        },
    )

    assert list(species_parameters["ions"]) == ["_ions0", "_ions1"]
    assert list(species_parameters["electrons"]) == ["_electrons0", "_electrons1"]

    legacy_ion = species_parameters["ions"]["_ions0"]
    nested_ion = species_parameters["ions"]["_ions1"]
    legacy_electron = species_parameters["electrons"]["_electrons0"]
    nested_electron = species_parameters["electrons"]["_electrons1"]

    assert legacy_ion["user_label"] == "_legacy_ions"
    assert nested_ion["user_label"] == "beam"
    assert legacy_electron["user_label"] == "_legacy_electrons"
    assert nested_electron["user_label"] == "cloud"
    assert scalar(legacy_ion["drift_speed_x"]) == -8.0
    assert scalar(legacy_electron["drift_speed_x"]) == 99.0
    assert scalar(nested_electron["drift_speed_x"]) == 0.0


def test_electron_only_legacy_input_creates_both_legacy_species():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {"beam": {"number_pseudoparticles": 4}},
            "electrons": {"cloud": {"number_pseudoparticles": 5}},
        },
        input_parameters={
            "electron_drift_speed_x": 42.0,
        },
    )

    assert list(species_parameters["ions"]) == ["_ions0", "_ions1"]
    assert list(species_parameters["electrons"]) == ["_electrons0", "_electrons1"]
    assert species_parameters["ions"]["_ions0"]["user_label"] == "_legacy_ions"
    assert species_parameters["electrons"]["_electrons0"]["user_label"] == "_legacy_electrons"
    assert scalar(species_parameters["electrons"]["_electrons0"]["drift_speed_x"]) == 42.0


def test_ion_only_legacy_input_creates_both_legacy_species():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {"beam": {"number_pseudoparticles": 4}},
            "electrons": {"cloud": {"number_pseudoparticles": 5}},
        },
        input_parameters={
            "ion_drift_speed_x": -12.0,
        },
    )

    assert list(species_parameters["ions"]) == ["_ions0", "_ions1"]
    assert list(species_parameters["electrons"]) == ["_electrons0", "_electrons1"]
    assert species_parameters["ions"]["_ions0"]["user_label"] == "_legacy_ions"
    assert species_parameters["electrons"]["_electrons0"]["user_label"] == "_legacy_electrons"
    assert scalar(species_parameters["ions"]["_ions0"]["drift_speed_x"]) == -12.0


def test_shared_legacy_input_creates_both_legacy_species():
    species_parameters = clean_species_parameters(
        species_parameters={
            "ions": {"beam": {"number_pseudoparticles": 4}},
            "electrons": {"cloud": {"number_pseudoparticles": 5}},
        },
        input_parameters={
            "amplitude_perturbation_x": 0.25,
        },
    )

    assert list(species_parameters["ions"]) == ["_ions0", "_ions1"]
    assert list(species_parameters["electrons"]) == ["_electrons0", "_electrons1"]
    assert scalar(species_parameters["ions"]["_ions0"]["perturbation_amplitude_x"]) == 0.25
    assert scalar(species_parameters["electrons"]["_electrons0"]["perturbation_amplitude_x"]) == 0.25


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
