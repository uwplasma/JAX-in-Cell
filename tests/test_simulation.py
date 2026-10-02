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
        "dx_over_Debye_length": 1.0,
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
            "number_grid_points_y": 0,
            "number_grid_points_z": 0,
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
        "padded_external_electric_field",
        "padded_external_magnetic_field",
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
    assert output["padded_external_electric_field"].shape == (number_grid_points + 3, 3)
    assert output["padded_external_magnetic_field"].shape == (number_grid_points + 3, 3)
    assert output["fields"][0].shape == (number_grid_points, 3)
    assert output["fields"][1].shape == (number_grid_points, 3)
    assert output["number_grid_points"] == number_grid_points
    assert output["total_steps"] == total_steps
    assert jnp.isfinite(output["plasma_frequency"])
    assert output["plasma_frequency"] > 0
    output["positions"].block_until_ready()


@pytest.mark.parametrize("key", ["particle_BC_left", "particle_BC_right", "field_BC_left", "field_BC_right", "relativistic"])
def test_cn_rejects_unsupported_boundary_and_relativistic_inputs(key):
    p = small_simulation_parameters(total_steps=1)
    p["solver_parameters"]["time_evolution_algorithm"] = 1
    section = "solver_parameters" if key == "relativistic" else "domain_parameters"
    p[section][key] = True if key == "relativistic" else 1
    if key != "relativistic":
        kind = key.split("_")[0]
        p[section][f"{kind}_BC_left"] = p[section][f"{kind}_BC_right"] = 1
    with pytest.raises(ValueError, match="Implicit CN supports"):
        Simulation(p)


@pytest.mark.parametrize("kind,component", [("electric", "E"), ("magnetic", "B")])
def test_cn_rejects_ignored_prescribed_grid_fields(kind, component):
    p = small_simulation_parameters(total_steps=1)
    p["solver_parameters"]["time_evolution_algorithm"] = 1
    p["external_field_parameters"] = {f"external_{kind}_field": {component: jnp.ones((8, 3))}}
    with pytest.raises(ValueError, match="does not apply prescribed grid fields"):
        Simulation(p)


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
        assert key in output

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
        "dx_over_Debye_length": 1.0,
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
dx_over_Debye_length = 1.0
weight = 1.0
charge_over_elementary_charge = -1.0
vth_over_c_x = 0.01
vth_over_c_y = 0.01
vth_over_c_z = 0.01

[species_parameters.ions.ions0]
number_pseudoparticles = 20
dx_over_Debye_length = 1.0
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
dx_over_Debye_length = 1.0
weight = 1.0
charge_over_elementary_charge = -1.0
vth_over_c_x = 0.01
vth_over_c_y = 0.01
vth_over_c_z = 0.01

[species_parameters.ions.ions0]
number_pseudoparticles = 4
dx_over_Debye_length = 1.0
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
        "source_term_active": 0,
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


def source_simulation_parameters(T=5, G=8):
    parameters = small_simulation_parameters(T, G, 1)
    parameters["domain_parameters"]["timestep_over_spatialstep_times_c"] = 0.8
    parameters["solver_parameters"]["field_solver"] = 2
    for group in parameters["species_parameters"].values():
        for population in group.values():
            population.update(initial_positions=[[0., 0., 0.]], initial_velocities=[[0., 0., 0.]])
    parameters["source_parameters"] = dict(
        source_term_active=1, source_species=(0, 1),
        how_often_source_should_produce_quasiparticles=2,
        source_particles_per_second=1e12, location_of_source=2, width_of_source=1,
        injection_speed_x=0.4 * speed_of_light, injection_speed_y=0.1 * speed_of_light,
        injection_speed_z=-0.2 * speed_of_light,
    )
    return parameters


@pytest.mark.parametrize("G,width,location,nodes,strength", [(5, 2, 0, 3, 2), (8, 1, 0, 2, 1),
                                                           (5, 3, 0, 3, 3), (8, 2, 1, 2, 2),
                                                           (8, 2, 2, 2, 2), (5, 99, 3, 5, 5)])
def test_source_reservations_include_last_batch_and_centered_weights(G, width, location, nodes, strength):
    parameters = source_simulation_parameters(G=G)
    parameters["source_parameters"].update(source_species=1, width_of_source=width, location_of_source=location,
                                           how_often_source_should_produce_quasiparticles=3)
    sim = Simulation(parameters)
    assert sim.positions.shape == (2 + 2 * nodes, 3)
    np.testing.assert_array_equal(sim.source_birth_steps[2:], np.repeat([0, 3], nodes))
    np.testing.assert_allclose(np.asarray(sim.weights[2:]).sum(), 2 * strength * 1e12 * 3 * sim.dt)
    np.testing.assert_array_equal(sim.charges[2:], 0)
    np.testing.assert_array_equal(sim.masses[2:], 0)
    if location == 0:
        np.testing.assert_allclose(sim.source_birth_positions[2:2 + nodes, 0].sum(), 0, atol=1e-17)


@pytest.mark.parametrize("wall,fraction,restitution", [(0, 1., 1.), (2, 0., 1.), (3, 0.4, 0.5)])
def test_source_ballistic_birth_loss_energy_and_momentum_budgets(wall, fraction, restitution):
    parameters = source_simulation_parameters()
    if wall:
        parameters["domain_parameters"].update(particle_BC_left=1, particle_BC_right=wall,
                                               field_BC_left=1, field_BC_right=1,
                                               mixed_BC_weight=fraction, COR_right=restitution)
    sim = Simulation(parameters)
    output = sim.run()
    weights = np.asarray(output["weights_over_time"])
    times = np.arange(5) + 1
    births = (times + 1) // 2
    per_batch = 2e12 * 2 * float(output["dt"])
    from jaxincell._constants import mass_electron
    velocity = speed_of_light * np.array([0.4, 0.1, -0.2])
    batch_mass = (mass_electron + mass_proton) * per_batch / 2
    np.testing.assert_allclose(output["injected_weight"], births * per_batch, rtol=2e-14)
    np.testing.assert_allclose(output["injected_energy"], births * batch_mass * np.dot(velocity, velocity) / 2, rtol=2e-14)
    np.testing.assert_allclose(output["injected_momentum"], births[:, None] * batch_mass * velocity, rtol=2e-14)
    np.testing.assert_allclose(weights.sum(axis=1), 2 + output["injected_weight"] - output["lost_weight"], rtol=2e-14)
    np.testing.assert_array_equal(output["alive_particles"], weights > 0)
    np.testing.assert_array_equal(output["charges_over_time"][:, ~output["alive_particles"][-1], :][-1], 0)
    live_ke = 0.5 * np.sum(np.asarray(output["masses_over_time"]) * np.asarray(output["velocities"])**2, axis=(1, 2))
    live_p = np.sum(np.asarray(output["masses_over_time"]) * np.asarray(output["velocities"]), axis=1)
    np.testing.assert_allclose(live_ke + output["wall_energy_transfer"], output["injected_energy"], rtol=2e-12, atol=1e-22)
    np.testing.assert_allclose(live_p + output["wall_momentum_transfer"], output["injected_momentum"], rtol=2e-12, atol=1e-29)
    if wall == 2:
        np.testing.assert_allclose(output["lost_weight"], [0, per_batch, per_batch, 2 * per_batch, 2 * per_batch])
        np.testing.assert_allclose(output["lost_energy"], output["wall_energy_transfer"], rtol=2e-14, atol=1e-25)
        np.testing.assert_allclose(output["lost_momentum"], output["wall_momentum_transfer"], rtol=2e-14, atol=1e-32)
    if wall == 0:
        np.testing.assert_array_equal(output["lost_weight"], 0)
        age = (times[:, None] - np.asarray(output["source_birth_steps"])[None, 2:]) * float(output["dt"])
        expected = np.asarray(sim.source_birth_positions)[None, 2:] + age[..., None] * velocity
        expected = (expected + np.asarray(output["box_size"]) / 2) % np.asarray(output["box_size"]) - np.asarray(output["box_size"]) / 2
        live = np.asarray(output["alive_particles"])[:, 2:]
        np.testing.assert_allclose(np.asarray(output["positions"])[..., 2:, :][live], expected[live], atol=1e-16)
    diagnostics(output)
    np.testing.assert_allclose(output["kinetic_energy"], live_ke, rtol=2e-14)
    assert "weights_electrons" in output and "weights_ions" in output


def test_sources_apply_each_prescribed_velocity_and_recompute_gauss():
    parameters = source_simulation_parameters()
    parameters["source_parameters"].update(
        source_species=(1, 0, 1), how_often_source_should_produce_quasiparticles=(2, 3, 7),
        injection_speed_x=(0., 1e6, -2e6), injection_speed_y=(3e6, 4e6, 5e6),
        injection_speed_z=(6e6, 7e6, 8e6), source_particles_per_second=(1e12, 2e12, 3e12))
    output = Simulation(parameters).run()
    np.testing.assert_array_equal(output["species_integer_index"][2:], [1, 1, 1, 0, 0, 1])
    np.testing.assert_allclose(output["velocities"][0, [2, 5, 7]],
                               [[0., 3e6, 6e6], [1e6, 4e6, 7e6], [-2e6, 5e6, 8e6]], rtol=1e-10, atol=1e-5)
    np.testing.assert_array_equal(output["masses_over_time"][0, [3, 4, 6]], 0)
    E, rho = np.asarray(output["electric_field"])[..., 0], np.asarray(output["charge_density"])
    from jaxincell._constants import epsilon_0
    residual = (E - np.roll(E, 1, axis=1)) / float(output["dx"]) - (rho - rho.mean(axis=1, keepdims=True)) / epsilon_0
    assert np.max(np.abs(residual)) < 1e-12 * np.max(np.abs(rho / epsilon_0))
    assert np.isfinite(output["source_field_work"]).all()


def test_source_wall_impact_energy_refines_for_constant_electric_force():
    from jaxincell._constants import elementary_charge
    errors = []
    for courant in (0.8, 0.4, 0.2, 0.1):
        T = round(4 / courant)
        parameters = source_simulation_parameters(T=T)
        L = parameters["domain_parameters"]["length"]
        dt = courant * L / (8 * speed_of_light)
        acceleration = 0.2 * speed_of_light**2 / L
        parameters["domain_parameters"].update(timestep_over_spatialstep_times_c=courant,
                                               particle_BC_left=1, particle_BC_right=2,
                                               field_BC_left=1, field_BC_right=1)
        parameters["species_parameters"]["electrons"]["electrons0"]["charge_over_elementary_charge"] = -1e-20
        parameters["source_parameters"].update(source_species=1, injection_speed_x=0.2 * speed_of_light,
                                               injection_speed_y=0., injection_speed_z=0.,
                                               how_often_source_should_produce_quasiparticles=T + 1,
                                               source_particles_per_second=1 / (dt * (T + 1)))
        E = np.zeros((8, 3))
        E[:, 0] = acceleration * mass_proton / elementary_charge
        parameters["external_field_parameters"] = {"external_electric_field": {"E": E}}
        output = Simulation(parameters).run()
        np.testing.assert_allclose(output["lost_weight"][-1], 1, rtol=2e-14)
        incoming_speed = np.sqrt(2 * float(output["lost_energy"][-1]) / mass_proton)
        exact_speed = np.sqrt((0.2 * speed_of_light)**2 + 2 * acceleration * L / 16)
        errors.append(abs(incoming_speed - exact_speed))
        assert errors[-1] < 0.55 * acceleration * dt
    assert errors[-1] < 0.05 * errors[0]


def test_source_birth_weight_gradient_matches_prescribed_rate():
    sim = Simulation(source_simulation_parameters())
    derivative = jax.grad(lambda courant: sim.run({"timestep_over_spatialstep_times_c": courant})["injected_weight"][-1])(0.8)
    np.testing.assert_allclose(derivative, 12e12 * float(sim.dx) / speed_of_light, rtol=2e-14)


def test_source_reservation_keeps_existing_particle_rng_streams():
    parameters = source_simulation_parameters(T=3)
    for group in parameters["species_parameters"].values():
        for population in group.values():
            population.update(number_pseudoparticles=11, initial_positions=None, initial_velocities=None,
                              random_positions_x=True)
    sourced = Simulation(parameters)
    parameters["source_parameters"]["source_term_active"] = 0
    baseline = Simulation(parameters)
    np.testing.assert_array_equal(sourced.positions[:22], baseline.positions)
    np.testing.assert_array_equal(sourced.velocities[:22], baseline.velocities)
    sourced.source_parameters = {"source_term_active": 0}
    assert sourced.positions.shape == baseline.positions.shape
    assert not hasattr(sourced, "source_birth_steps")


@pytest.mark.parametrize("section,key,value,match", [("solver_parameters", "field_solver", 0, "Cartesian Gauss"),
                                                     ("solver_parameters", "time_evolution_algorithm", 1, "Boris"),
                                                     ("solver_parameters", "relativistic", True, "nonrelativistic"),
                                                     ("source_parameters", "source_species", 99, "ordered"),
                                                     ("source_parameters", "width_of_source", 9, "width"),
                                                     ("source_parameters", "injection_speed_x", speed_of_light, "below c")])
def test_source_unsupported_configuration_is_rejected(section, key, value, match):
    parameters = source_simulation_parameters()
    parameters[section][key] = value
    with pytest.raises(ValueError, match=match):
        Simulation(parameters)


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


@pytest.mark.parametrize("steps", [1, 4])
def test_stored_times_are_end_of_step_times(steps):
    out = Simulation(small_simulation_parameters(total_steps=steps)).run()
    np.testing.assert_allclose(out["time_array"], np.arange(1, steps + 1) * float(out["dt"]), rtol=1e-14, atol=0)


def test_field_setter_recompiles_for_an_interior_array_change():
    p = small_simulation_parameters(total_steps=1, number_grid_points=512, number_pseudoparticles=4)
    for species in p["species_parameters"].values():
        for values in species.values():
            values.update(initial_positions=jnp.zeros((4, 3)), initial_velocities=jnp.zeros((4, 3)))
    field = jnp.zeros((512, 3))
    p["external_field_parameters"] = {"external_electric_field": {"E": field}}
    sim = Simulation(p)
    jax.block_until_ready(sim.run())
    old_hash = sim.external_field_hash
    changed = {"external_electric_field": {"E": field.at[256, 0].set(1e6)}}
    sim.external_field_parameters = changed
    actual = sim.run()["velocities"]
    p["external_field_parameters"] = changed
    expected = Simulation(p).run()["velocities"]
    assert sim.external_field_hash != old_hash
    assert float(jnp.max(jnp.abs(actual))) > 1000
    np.testing.assert_array_equal(actual, expected)
    exposed = sim.external_field_parameters
    exposed["external_electric_field"]["E"] = field
    np.testing.assert_array_equal(sim.external_field_parameters["external_electric_field"]["E"], changed["external_electric_field"]["E"])


@pytest.mark.parametrize("section", list(PARAMETER_SECTIONS))
def test_parameter_sections_return_defensive_copies(section):
    sim = Simulation(small_simulation_parameters(total_steps=1))
    exposed = getattr(sim, section)
    exposed.clear()
    assert getattr(sim, section)


@pytest.mark.parametrize("key", ["particle_BC_left", "particle_BC_right", "field_BC_left", "field_BC_right", "relativistic"])
def test_cn_rejects_unsupported_boundary_and_relativistic_inputs(key):
    p = small_simulation_parameters(total_steps=1)
    p["solver_parameters"]["time_evolution_algorithm"] = 1
    section = "solver_parameters" if key == "relativistic" else "domain_parameters"
    p[section][key] = True if key == "relativistic" else 1
    if key != "relativistic":
        kind = key.split("_")[0]
        p[section][f"{kind}_BC_left"] = p[section][f"{kind}_BC_right"] = 1
    with pytest.raises(ValueError, match="Implicit CN supports"):
        Simulation(p)


# Explicit initial position/velocity overrides are deferred until Simulation.run()

@pytest.mark.parametrize("side", [-1, 1])
def test_relativistic_absorbed_slots_remain_finite_in_following_pushes(side):
    parameters = small_simulation_parameters(total_steps=2, number_grid_points=6, number_pseudoparticles=2)
    initial = Simulation(parameters)
    speed, dt, length = .02*299792458., float(initial.dt), float(initial.box_size[0])
    for population in parameters["species_parameters"].values():
        for species in population.values():
            species.update(initial_positions=jnp.zeros((2, 3)).at[:, 0].set(side*(length/2-dt*speed/4)),
                           initial_velocities=jnp.zeros((2, 3)).at[:, 0].set(side*speed))
    parameters["domain_parameters"].update(particle_BC_left=2, particle_BC_right=2,
                                             field_BC_left=1, field_BC_right=1)
    parameters["solver_parameters"].update(relativistic=True, field_solver=2)
    output = Simulation(parameters).run()
    for key in ("positions", "velocities", "electric_field", "magnetic_field"):
        assert np.isfinite(output[key]).all()
    np.testing.assert_array_equal(output["velocities"], 0.)
    assert np.all(side*np.asarray(output["positions"])[..., 0] > length/2)
# grows a public initial-state override API again.
#
# def test_simulation_rejects_mismatched_positions_shape():
#     ...
#
# def test_simulation_rejects_mismatched_velocities_shape():
#     ...


def test_tensor_field_interior_changes_hash_and_first_snapshot_time():
    p = small_simulation_parameters(total_steps=2, number_pseudoparticles=2)
    p["domain_parameters"].update(number_grid_points_y=8, number_grid_points_z=8)
    field = np.zeros((8, 8, 8, 3))
    p["external_field_parameters"] = {"external_magnetic_field": {"B": field}}
    sim = Simulation(p)
    first_hash = sim.external_field_hash
    field[4, 4, 4, 2] = 1.
    sim.external_field_parameters = {"external_magnetic_field": {"B": field}}
    assert sim.external_field_hash != first_hash
    out = sim.run()
    np.testing.assert_array_equal(out["time_array"], np.arange(1, 3)*out["dt"])


@pytest.mark.parametrize("name, component", [("external_electric_field", "E"), ("external_magnetic_field", "B")])
def test_external_field_shape_rejected(name, component):
    p = small_simulation_parameters(total_steps=1)
    p["external_field_parameters"] = {name: {component: np.zeros((8, 2))}}
    with pytest.raises(ValueError, match="must have shape"):
        Simulation(p)


@pytest.mark.parametrize("name, component", [("external_electric_field", "E"), ("external_magnetic_field", "B")])
def test_cn_rejects_nonzero_prescribed_fields_at_construction_and_update(name, component):
    p = small_simulation_parameters(total_steps=1)
    p["solver_parameters"]["time_evolution_algorithm"] = 1
    simulation = Simulation(p)
    prescribed = {name: {component: np.ones((8, 3))}}
    with pytest.raises(ValueError, match="does not support prescribed"):
        simulation.external_field_parameters = prescribed
    p["external_field_parameters"] = prescribed
    with pytest.raises(ValueError, match="does not support prescribed"):
        Simulation(p)
