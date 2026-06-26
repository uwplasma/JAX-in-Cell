# tests/test_simulation.py

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell._diagnostics import diagnostics
from jaxincell._constants import mass_proton, speed_of_light
from jaxincell._parameters._sections import PARAMETER_SECTIONS
from jaxincell._simulation import Simulation, load_parameters
from jaxincell._state_initialization import initialize_field_state, initialize_particle_state
from tests.helpers import scalar


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


def assert_simulation_output_contract(
    output,
    total_steps,
    number_grid_points,
    number_particles,
):
    expected_keys = {
        "positions",
        "velocities",
        "masses",
        "charges",
        "charge_to_mass_ratios",
        "initial_positions",
        "initial_velocities",
        "weights",
        "species_integer_index",
        "charge_integer_lookup",
        "mass_integer_lookup",
        "charge_mass_integer_lookup",
        "electric_field",
        "magnetic_field",
        "current_density",
        "charge_density",
        "number_grid_points",
        "number_pseudoelectrons",
        "total_steps",
        "time_array",
        "grid",
        "dt",
        "plasma_frequency",
        "max_initial_vth_electrons",
        "vth_electrons_over_c",
        "charge_electrons",
        "dx",
        "length",
        "box_size",
        "fields",
        "external_electric_field",
        "external_magnetic_field",
    }

    assert expected_keys <= set(output)
    assert output["positions"].shape == (total_steps, number_particles, 3)
    assert output["velocities"].shape == (total_steps, number_particles, 3)
    assert output["masses"].shape == (number_particles, 1)
    assert output["charges"].shape == (number_particles, 1)
    assert output["charge_to_mass_ratios"].shape == (number_particles, 1)
    assert output["initial_positions"].shape == (number_particles, 3)
    assert output["initial_velocities"].shape == (number_particles, 3)
    assert output["weights"].shape == (number_particles, 1)
    assert output["species_integer_index"].shape == (number_particles,)
    assert output["electric_field"].shape == (total_steps, number_grid_points, 3)
    assert output["magnetic_field"].shape == (total_steps, number_grid_points, 3)
    assert output["current_density"].shape == (total_steps, number_grid_points, 3)
    assert output["charge_density"].shape == (total_steps, number_grid_points)
    assert output["grid"].shape == (number_grid_points,)
    assert output["time_array"].shape == (total_steps,)
    assert output["external_electric_field"].shape == (number_grid_points, 3)
    assert output["external_magnetic_field"].shape == (number_grid_points, 3)
    assert output["fields"][0].shape == (number_grid_points, 3)
    assert output["fields"][1].shape == (number_grid_points, 3)
    assert output["number_grid_points"] == number_grid_points
    assert output["total_steps"] == total_steps
    assert jnp.isfinite(output["plasma_frequency"])
    assert output["plasma_frequency"] > 0
    output["positions"].block_until_ready()


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
    assert set(output["parameter_sections"]) == set(PARAMETER_SECTIONS)
    assert output["domain_parameters"]["number_grid_points"] == number_grid_points
    assert output["solver_parameters"]["field_solver"] == 0
    assert output["species_parameters"]["electrons"]["_electrons0"]["user_label"] == "electrons0"

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


def test_simulation_constructor_accepts_path_and_dict_equivalently(tmp_path):
    """Test jaxincell._simulation.Simulation.__init__ and load_parameters.

    Cases:
    - constructing Simulation from a TOML path calls load_parameters through the path branch.
    - constructing Simulation from an equivalent dict yields matching cleaned parameter sections.
    - invalid non-dict, non-path-like inputs produce the expected loading error.
    """
    toml_text = """
[domain_parameters]
number_grid_points = 4
number_grid_points_y = 3
number_grid_points_z = 3
total_steps = 2
length = 0.01
length_y = 0.01
length_z = 0.01

[solver_parameters]
field_solver = 0
filter_passes = 0
filter_alpha = 0.5
print_info = false

[species_parameters.electrons.electrons0]
number_pseudoparticles = 4
grid_points_per_Debye_length = 1.0
weight = 1.0
charge_over_elementary_charge = -1.0
vth_over_c_x = 0.01
vth_over_c_y = 0.01
vth_over_c_z = 0.01

[species_parameters.ions.ions0]
number_pseudoparticles = 4
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
    parameter_file = tmp_path / "simulation_parameters.toml"
    parameter_file.write_text(toml_text)

    dict_sim = Simulation(load_parameters(str(parameter_file)))
    path_sim = Simulation(str(parameter_file))

    assert dict_sim.domain_parameters == path_sim.domain_parameters
    assert dict_sim.species_parameters == path_sim.species_parameters
    assert dict_sim.solver_parameters == path_sim.solver_parameters
    assert dict_sim.domain_hash == path_sim.domain_hash
    assert dict_sim.species_hash == path_sim.species_hash
    assert dict_sim.solver_hash == path_sim.solver_hash

    with pytest.raises(TypeError):
        Simulation(object())


def test_simulation_property_setters_reinitialize_state_and_hashes():
    """Test Simulation.set_parameter_section and section property setters.

    Cases:
    - setting domain_parameters rebuilds domain state and updates domain_hash.
    - setting species_parameters rebuilds particle state and updates species_hash.
    - setting external_field_parameters, source_parameters, and solver_parameters updates the matching hash.
    - unrelated base parameter sections remain unchanged.
    """
    parameters = small_simulation_parameters(
        total_steps=2,
        number_grid_points=4,
        number_pseudoparticles=4,
    )
    sim = Simulation(parameters)

    original_domain_hash = sim.domain_hash
    original_species_hash = sim.species_hash
    original_external_field_hash = sim.external_field_hash
    original_source_hash = sim.source_hash
    original_solver_hash = sim.solver_hash

    new_domain_parameters = deepcopy(parameters["domain_parameters"])
    new_domain_parameters["number_grid_points"] = 5
    new_domain_parameters["length"] = 0.02
    sim.domain_parameters = new_domain_parameters

    assert sim.domain_hash != original_domain_hash
    assert sim.grid.shape == (5,)
    assert scalar(sim.domain_parameters["length"]) == 0.02

    new_species_parameters = deepcopy(parameters["species_parameters"])
    new_species_parameters["electrons"]["electrons0"]["number_pseudoparticles"] = 3
    new_species_parameters["ions"]["ions0"]["number_pseudoparticles"] = 2
    sim.species_parameters = new_species_parameters

    assert sim.species_hash != original_species_hash
    assert sim.positions.shape == (5, 3)
    assert len(sim.species_index) == 5

    sim.external_field_parameters = {
        "external_electric_field_amplitude": 2.0,
        "external_electric_field_wavenumber": 1.0,
    }
    assert sim.external_field_hash != original_external_field_hash
    assert sim.external_electric_field.shape == (5, 3)
    assert sim.external_magnetic_field.shape == (5, 3)

    sim.source_parameters = {
        "source_term_active": 1,
        "source_species": 0,
        "how_often_source_should_produce_quasiparticles": 2,
        "source_particles_per_second": 1e16,
        "location_of_source": 3,
        "width_of_source": 1,
        "injection_speed_x": 1e7,
        "injection_speed_y": 0.0,
        "injection_speed_z": 0.0,
    }
    assert sim.source_hash != original_source_hash

    sim.solver_parameters = {
        "field_solver": 0,
        "filter_passes": 0,
        "filter_alpha": 0.25,
        "print_info": False,
        "seed": 123,
    }
    assert sim.solver_hash != original_solver_hash
    assert sim.domain_parameters["number_grid_points"] == 5
    assert sim.positions.shape == (5, 3)


def test_simulation_input_parameters_setter_reclassifies_and_reinitializes():
    """Test Simulation.input_parameters setter.

    Cases:
    - assigning differentiable flat values updates exposed input_parameters.
    - assigning nested species input parameters routes differentiable and non-differentiable values correctly.
    - assigning invalid input parameter keys raises the same ValueError as initialization.
    - simulation state and hashes are rebuilt after assignment.
    """
    parameters = small_simulation_parameters(
        total_steps=2,
        number_grid_points=4,
        number_pseudoparticles=4,
    )
    sim = Simulation(parameters)
    original_domain_hash = sim.domain_hash
    original_species_hash = sim.species_hash

    sim.input_parameters = {
        "length": 0.02,
        "ions": {
            "ions0": {
                "mass_over_proton_mass": 2.0,
                "number_pseudoparticles": 3,
            },
        },
    }

    exposed_input_parameters = sim.input_parameters
    assert scalar(exposed_input_parameters["length"]) == 0.02
    assert scalar(exposed_input_parameters["ions"]["ions0"]["mass_over_proton_mass"]) == 2.0
    assert "number_pseudoparticles" not in exposed_input_parameters["ions"]["ions0"]
    assert scalar(sim.domain_parameters["length"]) == 0.02
    assert scalar(sim.species_parameters["ions"]["_ions0"]["mass_over_proton_mass"]) == 2.0
    assert sim.species_parameters["ions"]["_ions0"]["number_pseudoparticles"] == 3
    assert sim.domain_hash != original_domain_hash
    assert sim.species_hash != original_species_hash
    assert sim.positions.shape == (7, 3)

    with pytest.raises(ValueError, match="ion_drift_speed_x"):
        sim.input_parameters = {"ion_drift_speed_x": 1.0}


def test_simulation_current_domain_state_matches_attributes():
    """Test Simulation.current_domain_state.

    Cases:
    - returned box_size, dx, dt, and grid match the Simulation attributes.
    - returned state can be passed to initialize_particle_state and initialize_field_state.
    - mutating the returned dictionary does not mutate the Simulation attributes.
    """
    sim = Simulation(
        small_simulation_parameters(
            total_steps=2,
            number_grid_points=4,
            number_pseudoparticles=4,
        )
    )

    domain_state = sim.current_domain_state()

    np.testing.assert_allclose(np.asarray(domain_state["box_size"]), np.asarray(sim.box_size))
    assert scalar(domain_state["dx"]) == scalar(sim.dx)
    assert scalar(domain_state["dt"]) == scalar(sim.dt)
    np.testing.assert_allclose(np.asarray(domain_state["grid"]), np.asarray(sim.grid))

    particle_state = initialize_particle_state(
        sim.species_parameters,
        sim.domain_parameters,
        sim.solver_parameters,
        domain_state,
    )
    field_state = initialize_field_state(
        sim.domain_parameters,
        sim.solver_parameters,
        sim.external_field_parameters,
        domain_state,
        particle_state,
    )
    assert particle_state["positions"].shape == sim.positions.shape
    assert field_state["fields"][0].shape == sim.fields[0].shape
    assert field_state["external_electric_field"].shape == sim.external_electric_field.shape

    domain_state["dx"] = -1.0
    domain_state["grid"] = jnp.zeros_like(domain_state["grid"])
    assert scalar(sim.dx) > 0
    assert not jnp.allclose(domain_state["grid"], sim.grid)


def test_simulation_initial_phase_space_overrides_initialize_and_run():
    """Test per-species initial phase-space overrides through Simulation.

    Cases:
    - species_parameters initial_positions and initial_velocities set Simulation particle state.
    - runtime input_parameters accepts differentiable per-species phase-space overrides.
    - a tiny run with runtime overrides completes with the expected output contract.
    """
    parameters = small_simulation_parameters(
        total_steps=1,
        number_grid_points=4,
        number_pseudoparticles=2,
    )
    electron_positions = jnp.array([
        [-0.001, 0.0, 0.001],
        [0.001, 0.0, -0.001],
    ])
    electron_velocities = jnp.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    parameters["species_parameters"]["electrons"]["electrons0"]["initial_positions"] = electron_positions
    parameters["species_parameters"]["electrons"]["electrons0"]["initial_velocities"] = electron_velocities

    sim = Simulation(parameters)

    assert jnp.allclose(sim.positions[:2], electron_positions)
    assert jnp.allclose(sim.velocities[:2], electron_velocities)

    ion_positions = jnp.array([
        [-0.002, 0.0, 0.002],
        [0.002, 0.0, -0.002],
    ])
    ion_velocities = jnp.array([
        [0.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
    ])
    output = sim.run(
        {
            "ions": {
                "ions0": {
                    "initial_positions": ion_positions,
                    "initial_velocities": ion_velocities,
                },
            },
        }
    )

    assert_simulation_output_contract(
        output,
        total_steps=1,
        number_grid_points=4,
        number_particles=4,
    )


def test_simulation_simulation_method_cleans_runtime_input_and_delegates_to_jitted_core(monkeypatch):
    """Test Simulation.simulation.

    Cases:
    - None runtime input is normalized to empty section overrides.
    - runtime input is passed through clean_runtime_input_parameters before _simulation.
    - the raw jitted output is assembled with runtime parameter metadata before returning.
    - domain/species/external/source/solver hashes are forwarded to _simulation.
    - invalid runtime input fails before calling _simulation.
    """
    sim = Simulation(
        small_simulation_parameters(
            total_steps=1,
            number_grid_points=4,
            number_pseudoparticles=2,
        )
    )
    calls = []
    expected_output = {"sentinel": object()}

    def fake_simulation(
        input_parameters=None,
        domain_hash="",
        species_hash="",
        external_field_hash="",
        source_hash="",
        solver_hash="",
    ):
        calls.append(
            {
                "input_parameters": input_parameters,
                "domain_hash": domain_hash,
                "species_hash": species_hash,
                "external_field_hash": external_field_hash,
                "source_hash": source_hash,
                "solver_hash": solver_hash,
            }
        )
        return expected_output

    monkeypatch.setattr(sim, "_simulation", fake_simulation)

    output = sim.simulation()
    assert output["sentinel"] is expected_output["sentinel"]
    assert set(output["parameter_sections"]) == set(PARAMETER_SECTIONS)
    assert scalar(output["domain_parameters"]["length"]) == scalar(sim.domain_parameters["length"])
    assert output["species_parameters"]["ions"]["_ions0"]["user_label"] == "ions0"
    assert calls[-1]["input_parameters"] == {
        section_name: {}
        for section_name in PARAMETER_SECTIONS
    }
    assert calls[-1]["domain_hash"] == sim.domain_hash
    assert calls[-1]["species_hash"] == sim.species_hash
    assert calls[-1]["external_field_hash"] == sim.external_field_hash
    assert calls[-1]["source_hash"] == sim.source_hash
    assert calls[-1]["solver_hash"] == sim.solver_hash

    runtime_input_parameters = {
        "length": 0.02,
        "filter_alpha": 0.25,
        "ions": {
            "ions0": {
                "mass_over_proton_mass": 2.0,
            },
        },
    }
    output = sim.simulation(runtime_input_parameters)
    assert output["sentinel"] is expected_output["sentinel"]
    assert scalar(output["length"]) == 0.02
    assert scalar(output["filter_alpha"]) == 0.25
    assert scalar(
        output["species_parameters"]["ions"]["_ions0"]["mass_over_proton_mass"]
    ) == 2.0
    assert calls[-1]["input_parameters"] == {
        "domain_parameters": {"length": 0.02},
        "species_parameters": {
            "ions": {
                "_ions0": {
                    "mass_over_proton_mass": 2.0,
                },
            },
        },
        "external_field_parameters": {},
        "source_parameters": {},
        "solver_parameters": {"filter_alpha": 0.25},
    }

    with pytest.raises(ValueError, match="total_steps"):
        sim.simulation({"total_steps": 2})
    assert len(calls) == 2


def test_simulation_jitted_core_orchestrates_runtime_sections_and_output_contract():
    """Test Simulation._simulation.

    Cases:
    - runtime section overrides are merged before domain, particle, and field initialization.
    - species references are resolved after runtime merge.
    - Boris and Crank-Nicolson algorithm branches both return the public output keys.
    - plasma_frequency, time_array, external fields, and shape metadata match the runtime parameters.
    """
    total_steps = 2
    number_grid_points = 4
    number_pseudoparticles = 2
    runtime_input_parameters = {
        "length": 0.02,
        "timestep_over_spatialstep_times_c": 0.5,
        "electrons": {
            "electrons0": {
                "vth_over_c_x": 0.02,
            },
        },
        "ions": {
            "ions0": {
                "mass_over_proton_mass": 2.0,
            },
        },
    }

    for time_evolution_algorithm in (0, 1):
        parameters = small_simulation_parameters(
            total_steps=total_steps,
            number_grid_points=number_grid_points,
            number_pseudoparticles=number_pseudoparticles,
        )
        parameters["solver_parameters"].update(
            {
                "seed": 123,
                "time_evolution_algorithm": time_evolution_algorithm,
                "number_of_particle_substeps_implicit_CN": 1,
                "tolerance_Picard_iterations_implicit_CN": 1e-3,
                "max_number_of_Picard_iterations_implicit_CN": 2,
            }
        )
        for axis in ("x", "y", "z"):
            parameters["species_parameters"]["ions"]["ions0"][f"vth_over_c_{axis}"] = "_electrons0"
        sim = Simulation(parameters)

        cleaned_input_parameters = sim.clean_runtime_input_parameters(runtime_input_parameters)
        runtime_external_electric_field = jnp.ones((number_grid_points, 3)) * 1e-6
        runtime_external_magnetic_field = jnp.ones((number_grid_points, 3)) * 2e-6
        cleaned_input_parameters["external_field_parameters"] = {
            "external_electric_field": {"E": runtime_external_electric_field},
            "external_magnetic_field": {"B": runtime_external_magnetic_field},
        }
        output = sim._simulation(
            cleaned_input_parameters,
            domain_hash=sim.domain_hash,
            species_hash=sim.species_hash,
            external_field_hash=sim.external_field_hash,
            source_hash=sim.source_hash,
            solver_hash=sim.solver_hash,
        )

        assert_simulation_output_contract(
            output,
            total_steps=total_steps,
            number_grid_points=number_grid_points,
            number_particles=2 * number_pseudoparticles,
        )
        assert scalar(output["length"]) == 0.02
        assert scalar(output["dx"]) == pytest.approx(0.02 / number_grid_points)
        assert scalar(output["dt"]) == pytest.approx(
            0.5 * scalar(output["dx"]) / speed_of_light
        )
        assert jnp.allclose(output["external_electric_field"], runtime_external_electric_field)
        assert jnp.allclose(output["external_magnetic_field"], runtime_external_magnetic_field)
        assert scalar(output["time_array"][-1]) == pytest.approx(total_steps * scalar(output["dt"]))
        np.testing.assert_allclose(
            np.asarray(output["masses"][number_pseudoparticles:, 0]),
            np.full(number_pseudoparticles, 2.0 * mass_proton),
            rtol=1e-12,
            atol=0.0,
        )


def test_simulation_run_delegates_to_simulation(monkeypatch):
    """Test Simulation.run.

    Cases:
    - run(None) calls simulation(None).
    - run(runtime_input_parameters) forwards the exact runtime input mapping.
    - returned output object is the output from simulation.
    """
    sim = Simulation(
        small_simulation_parameters(
            total_steps=1,
            number_grid_points=4,
            number_pseudoparticles=2,
        )
    )
    calls = []
    expected_output = {"sentinel": object()}

    def fake_simulation(input_parameters=None):
        calls.append(input_parameters)
        return expected_output

    monkeypatch.setattr(sim, "simulation", fake_simulation)

    assert sim.run() is expected_output
    assert calls[-1] is None

    runtime_input_parameters = {"length": 0.02}
    assert sim.run(runtime_input_parameters) is expected_output
    assert calls[-1] is runtime_input_parameters


# Explicit initial position/velocity overrides are deferred until Simulation.run()
# grows a public initial-state override API again.
#
# def test_simulation_rejects_mismatched_positions_shape():
#     ...
#
# def test_simulation_rejects_mismatched_velocities_shape():
#     ...
