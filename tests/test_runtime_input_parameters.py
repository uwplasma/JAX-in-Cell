from copy import deepcopy

import jax.numpy as jnp
import pytest

from jaxincell import Simulation
from jaxincell._parameters._sections import PARAMETER_SECTIONS
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

    assert set(cleaned_input_parameters) == set(PARAMETER_SECTIONS)
    assert cleaned_input_parameters["domain_parameters"] == {"length": 0.02}
    assert cleaned_input_parameters["species_parameters"]["ions"]["_ions0"] == {
        "grid_points_per_Debye_length": 1.5,
        "drift_speed_x": 3.0,
        "mass_over_proton_mass": 2.0,
    }
    assert cleaned_input_parameters["species_parameters"]["electrons"]["_electrons0"] == {
        "grid_points_per_Debye_length": 1.5,
    }
    assert cleaned_input_parameters["external_field_parameters"] == {}
    assert cleaned_input_parameters["source_parameters"] == {}
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


def test_initial_phase_space_overrides_are_differentiable_runtime_inputs():
    """Test runtime routing for per-species initial position and velocity overrides.

    Cases covered:
    - initial_positions and initial_velocities are accepted as differentiable species inputs.
    - runtime cleaning routes them to canonical species labels.
    - initialization-time input_parameters expose defensive JAX-array copies.
    """
    sim = Simulation(base_simulation_parameters())
    positions = jnp.array([
        [-0.1, 0.0, 0.1],
        [0.2, 0.3, 0.4],
    ])
    velocities = jnp.array([
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
    ])

    cleaned_input_parameters = sim.clean_runtime_input_parameters(
        {
            "electrons": {
                "electrons0": {
                    "initial_positions": positions,
                    "initial_velocities": velocities,
                },
            },
        }
    )

    cleaned_electron = cleaned_input_parameters["species_parameters"]["electrons"]["_electrons0"]
    assert jnp.allclose(cleaned_electron["initial_positions"], positions)
    assert jnp.allclose(cleaned_electron["initial_velocities"], velocities)

    parameters = deepcopy(base_simulation_parameters())
    parameters["input_parameters"] = {
        "electrons": {
            "electrons0": {
                "initial_positions": positions,
                "initial_velocities": velocities,
            },
        },
    }
    initialized_sim = Simulation(parameters)
    exposed_input_parameters = initialized_sim.input_parameters

    assert jnp.allclose(
        exposed_input_parameters["electrons"]["electrons0"]["initial_positions"],
        positions,
    )
    assert jnp.allclose(
        exposed_input_parameters["electrons"]["electrons0"]["initial_velocities"],
        velocities,
    )
    assert jnp.allclose(initialized_sim.positions[:2], positions)
    assert jnp.allclose(initialized_sim.velocities[:2], velocities)


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


def test_initial_loose_input_parameters_route_and_expose_only_differentiable_values():
    parameters = deepcopy(base_simulation_parameters())
    parameters.pop("species_parameters")
    parameters["input_parameters"] = {
        "ions": {
            "mass_over_proton_mass": 2.0,
            "number_pseudoparticles": 3,
        },
        "electrons": {
            "drift_speed_x": -4.0,
            "number_pseudoparticles": 4,
        },
    }

    sim = Simulation(parameters)
    exposed_input_parameters = sim.input_parameters

    ion = sim.species_parameters["ions"]["_ions0"]
    electron = sim.species_parameters["electrons"]["_electrons0"]

    assert scalar(ion["mass_over_proton_mass"]) == 2.0
    assert ion["number_pseudoparticles"] == 3
    assert scalar(electron["drift_speed_x"]) == -4.0
    assert electron["number_pseudoparticles"] == 4

    assert scalar(exposed_input_parameters["ions"]["mass_over_proton_mass"]) == 2.0
    assert "number_pseudoparticles" not in exposed_input_parameters["ions"]
    assert scalar(exposed_input_parameters["electrons"]["drift_speed_x"]) == -4.0
    assert "number_pseudoparticles" not in exposed_input_parameters["electrons"]


def test_initial_loose_input_parameters_apply_to_existing_nested_species():
    parameters = deepcopy(base_simulation_parameters())
    parameters["species_parameters"]["ions"]["beam"] = {
        "number_pseudoparticles": 2,
        "mass_over_proton_mass": 3.0,
    }
    parameters["species_parameters"]["electrons"]["cloud"] = {
        "number_pseudoparticles": 2,
        "vth_over_c_x": 0.02,
    }
    parameters["input_parameters"] = {
        "ions": {
            "mass_over_proton_mass": 2.0,
            "number_pseudoparticles": 3,
        },
        "electrons": {
            "drift_speed_x": -4.0,
            "number_pseudoparticles": 4,
        },
    }

    sim = Simulation(parameters)
    exposed_input_parameters = sim.input_parameters

    assert scalar(sim.species_parameters["ions"]["_ions0"]["mass_over_proton_mass"]) == 2.0
    assert scalar(sim.species_parameters["ions"]["_ions1"]["mass_over_proton_mass"]) == 2.0
    assert sim.species_parameters["ions"]["_ions0"]["number_pseudoparticles"] == 3
    assert sim.species_parameters["ions"]["_ions1"]["number_pseudoparticles"] == 3
    assert scalar(sim.species_parameters["electrons"]["_electrons0"]["drift_speed_x"]) == -4.0
    assert scalar(sim.species_parameters["electrons"]["_electrons1"]["drift_speed_x"]) == -4.0
    assert sim.species_parameters["electrons"]["_electrons0"]["number_pseudoparticles"] == 4
    assert sim.species_parameters["electrons"]["_electrons1"]["number_pseudoparticles"] == 4

    assert scalar(exposed_input_parameters["ions"]["mass_over_proton_mass"]) == 2.0
    assert "number_pseudoparticles" not in exposed_input_parameters["ions"]
    assert scalar(exposed_input_parameters["electrons"]["drift_speed_x"]) == -4.0
    assert "number_pseudoparticles" not in exposed_input_parameters["electrons"]


def test_input_parameters_property_returns_defensive_copy():
    parameters = deepcopy(base_simulation_parameters())
    parameters["input_parameters"] = {
        "length": 0.02,
        "ions": {
            "ions0": {
                "mass_over_proton_mass": 2.0,
            },
        },
    }
    sim = Simulation(parameters)

    exposed_input_parameters = sim.input_parameters
    exposed_input_parameters["length"] = 999.0
    exposed_input_parameters["ions"]["ions0"]["mass_over_proton_mass"] = 999.0

    refreshed_input_parameters = sim.input_parameters

    assert scalar(refreshed_input_parameters["length"]) == 0.02
    assert scalar(refreshed_input_parameters["ions"]["ions0"]["mass_over_proton_mass"]) == 2.0


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


def test_runtime_input_parameters_reject_non_dictionary_input():
    """Test Simulation.clean_runtime_input_parameters with non-dict input.

    Cases covered:
    - list, tuple, scalar, and string runtime inputs raise TypeError.
    - the error message names runtime input_parameters.
    - None remains accepted and cleans to empty section overrides.
    """
    sim = Simulation(base_simulation_parameters())

    for input_parameters in ([], (), 1.0, "length"):
        with pytest.raises(TypeError, match="Runtime input_parameters must be a dictionary"):
            sim.clean_runtime_input_parameters(input_parameters)

    with pytest.raises(TypeError, match="Runtime input parameter 'ions' must be a dictionary"):
        sim.clean_runtime_input_parameters({"ions": 3.0})

    with pytest.raises(TypeError, match="Runtime input parameter 'ions\\.beam' must be a dictionary"):
        sim.clean_runtime_input_parameters(
            {
                "ions": {
                    "ions0": {"drift_speed_x": 1.0},
                    "beam": 3.0,
                },
            }
        )

    assert sim.clean_runtime_input_parameters(None) == {
        section_name: {} for section_name in PARAMETER_SECTIONS
    }


def test_runtime_input_parameters_reject_unknown_species_label():
    """Test Simulation.clean_runtime_input_parameters species label resolution.

    Cases covered:
    - unknown ion user labels raise ValueError.
    - unknown electron user labels raise ValueError.
    - canonical labels and user labels for known species are both accepted.
    """
    sim = Simulation(base_simulation_parameters())

    with pytest.raises(ValueError, match="Could not find ions species 'missing_ion'"):
        sim.clean_runtime_input_parameters(
            {"ions": {"missing_ion": {"drift_speed_x": 1.0}}}
        )

    with pytest.raises(ValueError, match="Could not find electrons species 'missing_electron'"):
        sim.clean_runtime_input_parameters(
            {"electrons": {"missing_electron": {"drift_speed_x": -1.0}}}
        )

    cleaned_canonical = sim.clean_runtime_input_parameters(
        {"ions": {"_ions0": {"drift_speed_x": 2.0}}}
    )
    cleaned_user_label = sim.clean_runtime_input_parameters(
        {"ions": {"ions0": {"drift_speed_x": 3.0}}}
    )
    cleaned_electron_canonical = sim.clean_runtime_input_parameters(
        {"electrons": {"_electrons0": {"drift_speed_x": -2.0}}}
    )
    cleaned_electron_user_label = sim.clean_runtime_input_parameters(
        {"electrons": {"electrons0": {"drift_speed_x": -3.0}}}
    )

    assert cleaned_canonical["species_parameters"]["ions"]["_ions0"] == {
        "drift_speed_x": 2.0,
    }
    assert cleaned_user_label["species_parameters"]["ions"]["_ions0"] == {
        "drift_speed_x": 3.0,
    }
    assert cleaned_electron_canonical["species_parameters"]["electrons"]["_electrons0"] == {
        "drift_speed_x": -2.0,
    }
    assert cleaned_electron_user_label["species_parameters"]["electrons"]["_electrons0"] == {
        "drift_speed_x": -3.0,
    }


def test_runtime_input_parameters_duplicate_user_labels_route_to_all_matches():
    parameters = base_simulation_parameters()
    parameters["species_parameters"]["ions"]["beam"] = {
        "number_pseudoparticles": 2,
        "mass_over_proton_mass": 2.0,
    }
    sim = Simulation(parameters)
    sim.species_parameters["ions"]["_ions0"]["user_label"] = "shared"
    sim.species_parameters["ions"]["_ions1"]["user_label"] = "shared"
    sim.reinitialize_simulation_state()

    cleaned_input_parameters = sim.clean_runtime_input_parameters(
        {"ions": {"shared": {"drift_speed_x": 4.0}}}
    )

    assert cleaned_input_parameters["species_parameters"]["ions"] == {
        "_ions0": {"drift_speed_x": 4.0},
        "_ions1": {"drift_speed_x": 4.0},
    }


def test_runtime_input_parameters_loose_species_values_apply_to_all_species_of_type():
    """Test Simulation.clean_runtime_input_parameters loose species routing.

    Cases covered:
    - loose ions runtime values apply to every canonical ion species.
    - loose electrons runtime values apply to every canonical electron species.
    - non-differentiable loose species keys are rejected with paths that include the species type.
    """
    parameters = base_simulation_parameters()
    parameters["species_parameters"]["ions"]["beam"] = {
        "number_pseudoparticles": 2,
        "mass_over_proton_mass": 2.0,
    }
    parameters["species_parameters"]["electrons"]["cloud"] = {
        "number_pseudoparticles": 2,
        "vth_over_c_x": 0.02,
    }
    sim = Simulation(parameters)

    cleaned_input_parameters = sim.clean_runtime_input_parameters(
        {
            "ions": {"drift_speed_x": 3.0},
            "electrons": {"grid_points_per_Debye_length": 1.5},
        }
    )

    assert cleaned_input_parameters["species_parameters"]["ions"] == {
        "_ions0": {"drift_speed_x": 3.0},
        "_ions1": {"drift_speed_x": 3.0},
    }
    assert cleaned_input_parameters["species_parameters"]["electrons"] == {
        "_electrons0": {"grid_points_per_Debye_length": 1.5},
        "_electrons1": {"grid_points_per_Debye_length": 1.5},
    }

    with pytest.raises(ValueError, match="ions\\.number_pseudoparticles"):
        sim.clean_runtime_input_parameters({"ions": {"number_pseudoparticles": 3}})


def test_initial_input_parameters_nested_species_edge_cases_are_documented():
    """Test Simulation.classify_and_sort_input_parameters nested species edge cases.

    Cases covered:
    - differentiable nested species values are exposed through Simulation.input_parameters.
    - non-differentiable nested species values are routed into species_parameters but not exposed.
    - unknown nested species keys follow the current cleaner path or a new explicit validation branch.
    """
    parameters = deepcopy(base_simulation_parameters())
    parameters["input_parameters"] = {
        "ions": {
            "ions0": {
                "mass_over_proton_mass": 2.0,
                "number_pseudoparticles": 3,
                "unknown_numeric_key": 7.0,
            },
        },
        "electrons": {
            "electrons0": {
                "drift_speed_x": -4.0,
                "number_pseudoparticles": 4,
            },
        },
    }

    sim = Simulation(parameters)
    exposed_input_parameters = sim.input_parameters

    ion = sim.species_parameters["ions"]["_ions0"]
    electron = sim.species_parameters["electrons"]["_electrons0"]

    assert scalar(ion["mass_over_proton_mass"]) == 2.0
    assert ion["number_pseudoparticles"] == 3
    assert scalar(ion["unknown_numeric_key"]) == 7.0
    assert scalar(electron["drift_speed_x"]) == -4.0
    assert electron["number_pseudoparticles"] == 4

    assert scalar(exposed_input_parameters["ions"]["ions0"]["mass_over_proton_mass"]) == 2.0
    assert "number_pseudoparticles" not in exposed_input_parameters["ions"]["ions0"]
    assert "unknown_numeric_key" not in exposed_input_parameters["ions"]["ions0"]
    assert scalar(exposed_input_parameters["electrons"]["electrons0"]["drift_speed_x"]) == -4.0
    assert "number_pseudoparticles" not in exposed_input_parameters["electrons"]["electrons0"]
