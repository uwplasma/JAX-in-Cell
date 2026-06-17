# tests/test_simulation.py

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell._diagnostics import diagnostics
from jaxincell._simulation import Simulation, load_parameters


def small_simulation_parameters(total_steps=10, number_grid_points=8, number_pseudoparticles=20):
    base_species = {
        "number_pseudoparticles": number_pseudoparticles,
        "grid_points_per_Debye_length": 1.0,
        "weight": 1.0,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 1.0,
        "perturbation_wavenumber_y": 1.0,
        "perturbation_wavenumber_z": 1.0,
        "random_positions_x": False,
        "random_positions_y": False,
        "random_positions_z": False,
        "vth_over_c_x": 0.01,
        "vth_over_c_y": 0.01,
        "vth_over_c_z": 0.01,
        "drift_speed_x": 1.0,
        "drift_speed_y": 0.0,
        "drift_speed_z": 0.0,
        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
    }
    return {
        "domain_parameters": {
            "total_steps": total_steps,
            "number_grid_points": number_grid_points,
            "number_grid_points_y": 3,
            "number_grid_points_z": 3,
            "length": 0.01,
            "length_y": 0.01,
            "length_z": 0.01,
        },
        "species_parameters": {
            "electrons": {
                "electrons0": {
                    **base_species,
                    "charge_over_elementary_charge": -1.0,
                },
            },
            "ions": {
                "ions0": {
                    **base_species,
                    "charge_over_elementary_charge": 1.0,
                    "mass_over_proton_mass": 1.0,
                    "ion_temperature_over_electron_temperature_x": 1.0,
                    "ion_temperature_over_electron_temperature_y": 1.0,
                    "ion_temperature_over_electron_temperature_z": 1.0,
                },
            },
        },
        "solver_parameters": {
            "field_solver": 0,
            "filter_passes": 0,
            "filter_alpha": 0.5,
            "print_info": False,
        },
    }


def test_simulation_shapes_and_basic_consistency():
    total_steps = 10
    number_grid_points = 8
    number_pseudoparticles = 20

    sim = Simulation(
        small_simulation_parameters(
            total_steps=total_steps,
            number_grid_points=number_grid_points,
            number_pseudoparticles=number_pseudoparticles,
        )
    )
    output = sim.run()

    assert "positions" in output
    assert "velocities" in output
    assert "masses" in output
    assert "charges" in output

    n_particles = output["masses"].shape[0]
    assert output["positions"].shape == (total_steps, n_particles, 3)
    assert output["velocities"].shape == (total_steps, n_particles, 3)
    assert output["charges"].shape == (n_particles, 1)
    assert output["masses"].shape == (n_particles, 1)

    assert output["electric_field"].shape == (total_steps, number_grid_points, 3)
    assert output["magnetic_field"].shape == (total_steps, number_grid_points, 3)
    assert output["current_density"].shape == (total_steps, number_grid_points, 3)
    assert output["charge_density"].shape == (total_steps, number_grid_points)

    assert output["grid"].shape == (number_grid_points,)
    assert output["time_array"].shape == (total_steps,)
    assert output["dx"] > 0
    assert output["dt"] > 0
    assert output["plasma_frequency"] > 0

    diagnostics(output)

    for key in ["positions", "velocities", "masses", "charges"]:
        assert key not in output

    for key in [
        "position_electrons",
        "position_ions",
        "velocity_electrons",
        "velocity_ions",
        "mass_electrons",
        "mass_ions",
        "species",
        "electric_field_energy",
        "magnetic_field_energy",
        "kinetic_energy",
        "total_energy",
    ]:
        assert key in output

    ke_sum = output["kinetic_energy_electrons"] + output["kinetic_energy_ions"]
    assert np.allclose(
        np.array(output["kinetic_energy"]),
        np.array(ke_sum),
        rtol=1e-10,
        atol=1e-12,
    )

    total_calc = (
        output["electric_field_energy"]
        + output["magnetic_field_energy"]
        + output["kinetic_energy"]
    )
    assert np.allclose(
        np.array(output["total_energy"]),
        np.array(total_calc),
        rtol=1e-10,
        atol=1e-12,
    )


def test_simulation_print_info_emits_initialization_summary(capsys):
    parameters = small_simulation_parameters(
        total_steps=1,
        number_grid_points=4,
        number_pseudoparticles=4,
    )
    parameters["solver_parameters"]["print_info"] = True

    output = Simulation(parameters).run()
    output["positions"].block_until_ready()
    jax.effects_barrier()

    captured = capsys.readouterr()
    printed_output = captured.out + captured.err
    assert "Length of the simulation box" in printed_output
    assert "Relativistic gamma factor" in printed_output


def test_simulation_deterministic_with_same_parameters():
    parameters = small_simulation_parameters(
        total_steps=6,
        number_grid_points=6,
        number_pseudoparticles=10,
    )
    parameters["solver_parameters"]["seed"] = 1234

    out1 = Simulation(deepcopy(parameters)).run()
    out2 = Simulation(deepcopy(parameters)).run()

    assert jnp.allclose(out1["positions"], out2["positions"])
    assert jnp.allclose(out1["velocities"], out2["velocities"])
    assert jnp.allclose(out1["electric_field"], out2["electric_field"])
    assert jnp.allclose(out1["magnetic_field"], out2["magnetic_field"])
    assert jnp.allclose(out1["charge_density"], out2["charge_density"])
    assert jnp.allclose(out1["current_density"], out2["current_density"])


def test_auto_weight_simulation_reports_nonzero_plasma_frequency():
    parameters = small_simulation_parameters(
        total_steps=2,
        number_grid_points=6,
        number_pseudoparticles=10,
    )
    parameters["species_parameters"]["electrons"]["electrons0"]["weight"] = 0
    parameters["species_parameters"]["ions"]["ions0"]["weight"] = 0

    output = Simulation(parameters).run()

    assert jnp.isfinite(output["plasma_frequency"])
    assert output["plasma_frequency"] > 0


def test_simulation_with_extra_species_and_external_fields():
    total_steps = 4
    number_grid_points = 6
    number_pseudoparticles = 8
    number_extra_particles = 4

    parameters = small_simulation_parameters(
        total_steps=total_steps,
        number_grid_points=number_grid_points,
        number_pseudoparticles=number_pseudoparticles,
    )
    parameters["species_parameters"]["ions"]["extra_ion"] = {
        "number_pseudoparticles": number_extra_particles,
        "grid_points_per_Debye_length": 1.0,
        "weight": 1.0,
        "charge_over_elementary_charge": 2.0,
        "mass_over_proton_mass": 4.0,
        "perturbation_amplitude_x": 0.0,
        "perturbation_amplitude_y": 0.0,
        "perturbation_amplitude_z": 0.0,
        "perturbation_wavenumber_x": 0.0,
        "perturbation_wavenumber_y": 0.0,
        "perturbation_wavenumber_z": 0.0,
        "random_positions_x": True,
        "random_positions_y": True,
        "random_positions_z": True,
        "vth_over_c_x": 0.01,
        "vth_over_c_y": 0.0,
        "vth_over_c_z": 0.0,
        "ion_temperature_over_electron_temperature_x": 1.0,
        "ion_temperature_over_electron_temperature_y": 1.0,
        "ion_temperature_over_electron_temperature_z": 1.0,
        "drift_speed_x": 0.0,
        "drift_speed_y": 0.0,
        "drift_speed_z": 0.0,
        "velocity_plus_minus_x": False,
        "velocity_plus_minus_y": False,
        "velocity_plus_minus_z": False,
        "seed_position_override": False,
        "seed_position": None,
    }
    parameters["external_field_parameters"] = {
        "external_electric_field": {"E": np.zeros((number_grid_points, 3), dtype=np.float32)},
        "external_magnetic_field": {"B": np.zeros((number_grid_points, 3), dtype=np.float32)},
    }

    sim = Simulation(parameters)
    assert sim.external_electric_field.shape == (number_grid_points, 3)
    assert sim.external_magnetic_field.shape == (number_grid_points, 3)

    output = sim.run()

    n_particles_expected = 2 * number_pseudoparticles + number_extra_particles
    assert output["masses"].shape == (n_particles_expected, 1)
    assert output["charges"].shape == (n_particles_expected, 1)
    assert output["positions"].shape == (total_steps, n_particles_expected, 3)
    assert output["velocities"].shape == (total_steps, n_particles_expected, 3)

    diagnostics(output)
    assert "species" in output
    assert len(output["species"]) >= 3


def test_simulation_crank_nicolson_time_evolution_algorithm():
    total_steps = 3
    number_grid_points = 4

    parameters = small_simulation_parameters(
        total_steps=total_steps,
        number_grid_points=number_grid_points,
        number_pseudoparticles=6,
    )
    parameters["solver_parameters"].update(
        {
            "time_evolution_algorithm": 1,
            "number_of_particle_substeps_implicit_CN": 1,
            "tolerance_Picard_iterations_implicit_CN": 1e-3,
            "max_number_of_Picard_iterations_implicit_CN": 2,
        }
    )

    output = Simulation(parameters).run()
    n_particles = output["masses"].shape[0]

    assert output["positions"].shape == (total_steps, n_particles, 3)
    assert output["velocities"].shape == (total_steps, n_particles, 3)
    assert output["electric_field"].shape == (total_steps, number_grid_points, 3)
    assert output["magnetic_field"].shape == (total_steps, number_grid_points, 3)
    assert output["charge_density"].shape == (total_steps, number_grid_points)


def test_load_parameters_parses_canonical_toml(tmp_path):
    toml_text = """
[domain_parameters]
number_grid_points = 8
number_grid_points_y = 3
number_grid_points_z = 3
total_steps = 5
length = 0.01
length_y = 0.01
length_z = 0.01

[solver_parameters]
field_solver = 0
print_info = false

[species_parameters.electrons.electrons0]
number_pseudoparticles = 20
grid_points_per_Debye_length = 1.0
weight = 1.0
charge_over_elementary_charge = -1.0
vth_over_c_x = 0.01
vth_over_c_y = 0.01
vth_over_c_z = 0.01

[species_parameters.ions.ions0]
number_pseudoparticles = 20
grid_points_per_Debye_length = 1.0
weight = 1.0
charge_over_elementary_charge = 1.0
mass_over_proton_mass = 1.0
vth_over_c_x = "_electrons0"
vth_over_c_y = "_electrons0"
vth_over_c_z = "_electrons0"
ion_temperature_over_electron_temperature_x = 1.0
ion_temperature_over_electron_temperature_y = 1.0
ion_temperature_over_electron_temperature_z = 1.0
"""
    parameter_file = tmp_path / "params.toml"
    parameter_file.write_text(toml_text)

    parameters = load_parameters(str(parameter_file))
    sim = Simulation(parameters)

    assert parameters["domain_parameters"]["number_grid_points"] == 8
    assert sim.domain_parameters["total_steps"] == 5
    assert sim.species_parameters["electrons"]["_electrons0"]["number_pseudoparticles"] == 20
    assert sim.species_parameters["ions"]["_ions0"]["number_pseudoparticles"] == 20


def test_load_parameters_uses_defaults_when_species_section_is_missing(tmp_path):
    toml_text = """
[domain_parameters]
number_grid_points = 6
total_steps = 3
length = 0.01

[solver_parameters]
print_info = false
"""
    parameter_file = tmp_path / "params2.toml"
    parameter_file.write_text(toml_text)

    parameters = load_parameters(str(parameter_file))
    sim = Simulation(parameters)

    assert "species_parameters" not in parameters
    assert sim.domain_parameters["number_grid_points"] == 6
    assert sim.domain_parameters["total_steps"] == 3
    assert len(sim.species_parameters["electrons"]) == 1
    assert len(sim.species_parameters["ions"]) == 1


@pytest.mark.skip(reason="scaffold only")
def test_simulation_constructor_accepts_path_and_dict_equivalently():
    """Test jaxincell._simulation.Simulation.__init__ and load_parameters.

    Cases to implement:
    - constructing Simulation from a TOML path calls load_parameters through the path branch.
    - constructing Simulation from an equivalent dict yields matching cleaned parameter sections.
    - invalid non-dict, non-path-like inputs produce the expected loading error.
    """


@pytest.mark.skip(reason="scaffold only")
def test_simulation_property_setters_reinitialize_state_and_hashes():
    """Test Simulation.set_parameter_section and section property setters.

    Cases to implement:
    - setting domain_parameters rebuilds domain state and updates domain_hash.
    - setting species_parameters rebuilds particle state and updates species_hash.
    - setting external_field_parameters, source_parameters, and solver_parameters updates the matching hash.
    - unrelated base parameter sections remain unchanged.
    """


@pytest.mark.skip(reason="scaffold only")
def test_simulation_input_parameters_setter_reclassifies_and_reinitializes():
    """Test Simulation.input_parameters setter.

    Cases to implement:
    - assigning differentiable flat values updates exposed input_parameters.
    - assigning nested species input parameters routes differentiable and non-differentiable values correctly.
    - assigning invalid input parameter keys raises the same ValueError as initialization.
    - simulation state and hashes are rebuilt after assignment.
    """


@pytest.mark.skip(reason="scaffold only")
def test_simulation_current_domain_state_matches_attributes():
    """Test Simulation.current_domain_state.

    Cases to implement:
    - returned box_size, dx, dt, and grid match the Simulation attributes.
    - returned state can be passed to initialize_particle_state and initialize_field_state.
    - mutating the returned dictionary does not mutate the Simulation attributes.
    """


@pytest.mark.skip(reason="scaffold only")
def test_simulation_simulation_method_cleans_runtime_input_and_delegates_to_jitted_core():
    """Test Simulation.simulation.

    Cases to implement:
    - None runtime input is normalized to empty section overrides.
    - runtime input is passed through clean_runtime_input_parameters before _simulation.
    - domain/species/external/source/solver hashes are forwarded to _simulation.
    - invalid runtime input fails before calling _simulation.
    """


@pytest.mark.skip(reason="scaffold only")
def test_simulation_jitted_core_orchestrates_runtime_sections_and_output_contract():
    """Test Simulation._simulation.

    Cases to implement:
    - runtime section overrides are merged before domain, particle, and field initialization.
    - species references are resolved after runtime merge.
    - Boris and Crank-Nicolson algorithm branches both return the public output keys.
    - plasma_frequency, time_array, external fields, and shape metadata match the runtime parameters.
    """


@pytest.mark.skip(reason="scaffold only")
def test_simulation_run_delegates_to_simulation():
    """Test Simulation.run.

    Cases to implement:
    - run(None) calls simulation(None).
    - run(runtime_input_parameters) forwards the exact runtime input mapping.
    - returned output object is the output from simulation.
    """


# Explicit initial position/velocity overrides are deferred until Simulation.run()
# grows a public initial-state override API again.
#
# def test_simulation_rejects_mismatched_positions_shape():
#     ...
#
# def test_simulation_rejects_mismatched_velocities_shape():
#     ...
