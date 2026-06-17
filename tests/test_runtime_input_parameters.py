from copy import deepcopy

import pytest

from jaxincell import Simulation
from jaxincell._parameters._species_parameters import resolve_species_references
from jaxincell._routing import build_runtime_parameter_sections
from tests.helpers import base_simulation_parameters, scalar


def test_runtime_input_parameters_are_cleaned_to_canonical_section_overrides():
    sim = Simulation(base_simulation_parameters())

    cleaned_input_parameters = sim.clean_runtime_input_parameters(
        {
            "length": 0.02,
            "ions": {
                "ions0": {
                    "grid_points_per_Debye_length": 1.5,
                    "drift_speed_x": 3.0,
                    "mass_over_proton_mass": 2.0,
                },
            },
            "electrons": {
                "electrons0": {
                    "grid_points_per_Debye_length": 1.5,
                },
            },
        }
    )

    assert cleaned_input_parameters["domain_parameters"] == {"length": 0.02}
    assert cleaned_input_parameters["species_parameters"]["ions"]["_ions0"] == {
        "grid_points_per_Debye_length": 1.5,
        "drift_speed_x": 3.0,
        "mass_over_proton_mass": 2.0,
    }
    assert cleaned_input_parameters["species_parameters"]["electrons"]["_electrons0"] == {
        "grid_points_per_Debye_length": 1.5,
    }
    assert cleaned_input_parameters["solver_parameters"] == {}


def test_runtime_input_parameters_reject_removed_flat_species_values():
    sim = Simulation(base_simulation_parameters())

    with pytest.raises(ValueError, match="ion_drift_speed_x"):
        sim.clean_runtime_input_parameters({"ion_drift_speed_x": 3.0})


def test_runtime_input_parameters_reject_non_differentiable_values():
    sim = Simulation(base_simulation_parameters())

    with pytest.raises(ValueError, match="total_steps"):
        sim.clean_runtime_input_parameters({"total_steps": 2})

    with pytest.raises(ValueError, match="ions\\.ions0\\.number_pseudoparticles"):
        sim.clean_runtime_input_parameters(
            {
                "ions": {
                    "ions0": {
                        "number_pseudoparticles": 3,
                    },
                },
            }
        )


def test_initial_input_parameters_route_all_values_but_only_expose_differentiable_values():
    parameters = deepcopy(base_simulation_parameters())
    parameters["input_parameters"] = {
        "length": 0.02,
        "total_steps": 2,
        "filter_alpha": 0.25,
        "filter_passes": 0,
        "ions": {
            "ions0": {
                "mass_over_proton_mass": 2.0,
                "number_pseudoparticles": 3,
            },
        },
    }

    sim = Simulation(parameters)
    exposed_input_parameters = sim.input_parameters

    assert scalar(sim.domain_parameters["length"]) == 0.02
    assert sim.domain_parameters["total_steps"] == 2
    assert "grid_points_per_Debye_length" not in sim.domain_parameters
    assert scalar(sim.solver_parameters["filter_alpha"]) == 0.25
    assert sim.solver_parameters["filter_passes"] == 0
    assert scalar(sim.species_parameters["ions"]["_ions0"]["mass_over_proton_mass"]) == 2.0
    assert sim.species_parameters["ions"]["_ions0"]["number_pseudoparticles"] == 3

    assert scalar(exposed_input_parameters["length"]) == 0.02
    assert "total_steps" not in exposed_input_parameters
    assert scalar(exposed_input_parameters["filter_alpha"]) == 0.25
    assert "filter_passes" not in exposed_input_parameters
    assert scalar(exposed_input_parameters["ions"]["ions0"]["mass_over_proton_mass"]) == 2.0
    assert "number_pseudoparticles" not in exposed_input_parameters["ions"]["ions0"]


def test_initial_input_parameters_reject_unrouted_values():
    parameters = deepcopy(base_simulation_parameters())
    parameters["input_parameters"] = {
        "ion_drift_speed_x": 3.0,
    }

    with pytest.raises(ValueError, match="ion_drift_speed_x"):
        Simulation(parameters)


def test_runtime_species_references_recompute_after_runtime_merge():
    parameters = base_simulation_parameters()
    parameters["species_parameters"]["ions"]["ions0"]["vth_over_c_x"] = "_electrons0"

    sim = Simulation(parameters)
    base_vth = sim.species_parameters["ions"]["_ions0"]["vth_over_c_x"]
    cleaned_input_parameters = sim.clean_runtime_input_parameters(
        {
            "ions": {
                "ions0": {
                    "ion_temperature_over_electron_temperature_x": 4.0,
                },
            },
        }
    )

    base_sections = {
        "domain_parameters": sim.domain_parameters,
        "species_parameters": sim.species_parameters,
        "external_field_parameters": sim.external_field_parameters,
        "source_parameters": sim.source_parameters,
        "solver_parameters": sim.solver_parameters,
    }
    runtime_sections = build_runtime_parameter_sections(
        base_sections,
        cleaned_input_parameters,
    )
    resolve_species_references(runtime_sections["species_parameters"])

    runtime_vth = runtime_sections["species_parameters"]["ions"]["_ions0"]["vth_over_c_x"]
    assert scalar(runtime_vth) == pytest.approx(2.0 * scalar(base_vth))


@pytest.mark.skip(reason="scaffold only")
def test_runtime_input_parameters_reject_non_dictionary_input():
    """Test Simulation.clean_runtime_input_parameters with non-dict input.

    Cases to implement:
    - list, tuple, scalar, and string runtime inputs raise TypeError.
    - the error message names runtime input_parameters.
    - None remains accepted and cleans to empty section overrides.
    """


@pytest.mark.skip(reason="scaffold only")
def test_runtime_input_parameters_reject_unknown_species_label():
    """Test Simulation.clean_runtime_input_parameters species label resolution.

    Cases to implement:
    - unknown ion user labels raise ValueError.
    - unknown electron user labels raise ValueError.
    - canonical labels and user labels for known species are both accepted.
    """


@pytest.mark.skip(reason="scaffold only")
def test_runtime_input_parameters_loose_species_values_apply_to_all_species_of_type():
    """Test Simulation.clean_runtime_input_parameters loose species routing.

    Cases to implement:
    - loose ions runtime values apply to every canonical ion species.
    - loose electrons runtime values apply to every canonical electron species.
    - non-differentiable loose species keys are rejected with paths that include the species type.
    """


@pytest.mark.skip(reason="scaffold only")
def test_initial_input_parameters_nested_species_edge_cases_are_documented():
    """Test Simulation.classify_and_sort_input_parameters nested species edge cases.

    Cases to implement:
    - differentiable nested species values are exposed through Simulation.input_parameters.
    - non-differentiable nested species values are routed into species_parameters but not exposed.
    - unknown nested species keys follow the current cleaner path or a new explicit validation branch.
    """
