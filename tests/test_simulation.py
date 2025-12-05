# tests/test_simulation.py

import numpy as np
import pytest
import jax.numpy as jnp

from jaxincell._simulation import simulation
from jaxincell._diagnostics import diagnostics


def _run_small_simulation(total_steps=10, number_grid_points=8, number_pseudoelectrons=20):
    """
    Helper to run a very small, cheap simulation and return the raw output.
    """
    input_parameters = {
        "length": 0.01,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,  # avoid jprint noise in tests
    }

    output = simulation(
        input_parameters,
        number_grid_points=number_grid_points,
        number_pseudoelectrons=number_pseudoelectrons,
        total_steps=total_steps,
        field_solver=0,  # Curl_EB
    )
    return output


def test_simulation_shapes_and_basic_consistency():
    total_steps = 10
    number_grid_points = 8
    number_pseudoelectrons = 20

    output = _run_small_simulation(
        total_steps=total_steps,
        number_grid_points=number_grid_points,
        number_pseudoelectrons=number_pseudoelectrons,
    )

    # Positions / velocities / masses / charges
    assert "positions" in output
    assert "velocities" in output
    assert "masses" in output
    assert "charges" in output

    T = total_steps
    G = number_grid_points
    n_particles = output["masses"].shape[0]

    assert output["positions"].shape == (T, n_particles, 3)
    assert output["velocities"].shape == (T, n_particles, 3)
    assert output["charges"].shape == (n_particles, 1)
    assert output["masses"].shape == (n_particles, 1)

    # Fields and sources
    assert output["electric_field"].shape == (T, G, 3)
    assert output["magnetic_field"].shape == (T, G, 3)
    assert output["current_density"].shape == (T, G, 3)
    assert output["charge_density"].shape == (T, G)

    # Grid and time
    assert output["grid"].shape == (G,)
    assert output["time_array"].shape == (T,)
    assert output["dx"] > 0
    assert output["dt"] > 0
    assert output["plasma_frequency"] > 0

    # Now run diagnostics on the same output
    diagnostics(output)

    # After diagnostics, heavy arrays removed
    for key in ["positions", "velocities", "masses", "charges"]:
        assert key not in output

    # Species split present
    assert "position_electrons" in output
    assert "position_ions" in output
    assert "velocity_electrons" in output
    assert "velocity_ions" in output
    assert "mass_electrons" in output
    assert "mass_ions" in output
    assert "species" in output

    # Energies should have correct shape
    assert output["electric_field_energy"].shape == (T,)
    assert output["magnetic_field_energy"].shape == (T,)
    # External energies may be scalar (if zero) or time-dependent arrays
    assert output["external_electric_field_energy"].shape in [(), (T,)]
    assert output["external_magnetic_field_energy"].shape in [(), (T,)]
    assert output["kinetic_energy"].shape == (T,)
    assert output["total_energy"].shape == (T,)

    # Kinetic energy must be sum of electron + ion contributions
    ke_sum = output["kinetic_energy_electrons"] + output["kinetic_energy_ions"]
    assert np.allclose(
        np.array(output["kinetic_energy"]),
        np.array(ke_sum),
        rtol=1e-10,
        atol=1e-12,
    )

    # Total energy must match sum of components
    # (we broadcast scalars where needed)
    external_E = output["external_electric_field_energy"]
    external_B = output["external_magnetic_field_energy"]
    if external_E.shape == ():
        external_E = jnp.zeros_like(output["electric_field_energy"]) + external_E
    if external_B.shape == ():
        external_B = jnp.zeros_like(output["magnetic_field_energy"]) + external_B

    total_calc = (
        output["electric_field_energy"]
        + external_E
        + output["magnetic_field_energy"]
        + external_B
        + output["kinetic_energy"]
    )
    assert np.allclose(
        np.array(output["total_energy"]),
        np.array(total_calc),
        rtol=1e-10,
        atol=1e-12,
    )


def test_simulation_deterministic_with_same_parameters():
    """
    Running simulation with identical parameters should produce identical results.
    """
    total_steps = 6
    number_grid_points = 6
    number_pseudoelectrons = 10

    input_parameters = {
        "length": 0.01,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,
        # Explicitly fix seed to stress determinism
        "seed": 1234,
    }

    out1 = simulation(
        input_parameters,
        number_grid_points=number_grid_points,
        number_pseudoelectrons=number_pseudoelectrons,
        total_steps=total_steps,
        field_solver=0,
    )

    out2 = simulation(
        input_parameters,
        number_grid_points=number_grid_points,
        number_pseudoelectrons=number_pseudoelectrons,
        total_steps=total_steps,
        field_solver=0,
    )

    # Compare a few representative arrays
    assert jnp.allclose(out1["positions"], out2["positions"])
    assert jnp.allclose(out1["velocities"], out2["velocities"])
    assert jnp.allclose(out1["electric_field"], out2["electric_field"])
    assert jnp.allclose(out1["magnetic_field"], out2["magnetic_field"])
    assert jnp.allclose(out1["charge_density"], out2["charge_density"])
    assert jnp.allclose(out1["current_density"], out2["current_density"])


def test_simulation_with_extra_species_and_external_fields():
    """
    Exercise:
    - additional particle species plumbing (make_particles path),
    - external field dict interface for E/B.
    """
    total_steps = 4
    G = 6
    N_e = 8

    # Define one extra species with minimal required keys
    species = [
        {
            "charge_over_elementary_charge": 2.0,
            "mass_over_proton_mass": 4.0,
            "vth_over_c_x": 0.01,
            "vth_over_c_y": 0.0,
            "vth_over_c_z": 0.0,
            "random_positions_x": True,
            "random_positions_y": True,
            "random_positions_z": True,
            "amplitude_perturbation_x": 0.0,
            "amplitude_perturbation_y": 0.0,
            "amplitude_perturbation_z": 0.0,
            "wavenumber_perturbation_x": 0.0,
            "wavenumber_perturbation_y": 0.0,
            "wavenumber_perturbation_z": 0.0,
            "drift_speed_x": 0.0,
            "drift_speed_y": 0.0,
            "drift_speed_z": 0.0,
            "weight_ratio": 1.0,
            "seed_position_override": False,
            "seed_position": 0,
        }
    ]

    extE = {"E": np.zeros((G, 3), dtype=np.float32)}
    extB = {"B": np.zeros((G, 3), dtype=np.float32)}

    input_parameters = {
        "length": 0.02,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,
        "species": species,
        "external_electric_field": extE,
        "external_magnetic_field": extB,
    }

    # One extra species population of size N_extra
    N_extra = 4
    output = simulation(
        input_parameters,
        number_grid_points=G,
        number_pseudoelectrons=N_e,
        number_pseudoparticles_species=(N_extra,),  # <-- tuple, not list
        total_steps=total_steps,
        field_solver=0,
    )
    
    # We should have 2*N_e + N_extra total particles
    n_particles_expected = 2 * N_e + N_extra
    assert output["masses"].shape == (n_particles_expected, 1)
    assert output["charges"].shape == (n_particles_expected, 1)
    assert output["positions"].shape == (total_steps, n_particles_expected, 3)
    assert output["velocities"].shape == (total_steps, n_particles_expected, 3)

    # External fields should be present with correct shape
    assert output["external_electric_field"].shape == (G, 3)
    assert output["external_magnetic_field"].shape == (G, 3)

    # Diagnostics should still run and produce species split
    diagnostics(output)
    assert "species" in output
    # At least 3 species: electrons, ions, and the new species
    assert len(output["species"]) >= 3


def test_simulation_crank_nicolson_time_evolution_algorithm():
    """
    Exercise the CN_step-based time evolution path (time_evolution_algorithm=1).
    Just checks that it runs and the shapes are sensible.
    """
    total_steps = 3
    G = 4
    N_e = 6

    input_parameters = {
        "length": 0.01,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,
        # Keep iterations small for fast tests
        "tolerance_Picard_iterations_implicit_CN": 1e-3,
        "max_number_of_Picard_iterations_implicit_CN": 2,
    }

    output = simulation(
        input_parameters,
        number_grid_points=G,
        number_pseudoelectrons=N_e,
        total_steps=total_steps,
        field_solver=0,
        time_evolution_algorithm=1,  # CN_step path
    )

    n_particles = output["masses"].shape[0]

    assert output["positions"].shape == (total_steps, n_particles, 3)
    assert output["velocities"].shape == (total_steps, n_particles, 3)
    assert output["electric_field"].shape == (total_steps, G, 3)
    assert output["magnetic_field"].shape == (total_steps, G, 3)
    assert output["charge_density"].shape == (total_steps, G)


def test_simulation_rejects_mismatched_positions_shape():
    """
    Passing positions/velocities with the wrong shape into simulation(...)
    should raise a ValueError, exercising the explicit shape checks.
    """
    total_steps = 2
    G = 4
    N_e = 4

    base_params = {
        "length": 0.01,
        "grid_points_per_Debye_length": 1.0,
        "print_info": False,
    }

    # First run: just to discover the correct initial_positions shape
    base_output = simulation(
        base_params,
        number_grid_points=G,
        number_pseudoelectrons=N_e,
        total_steps=total_steps,
        field_solver=0,
    )

    good_shape = base_output["initial_positions"].shape
    # Create positions with one fewer particle
    bad_positions = jnp.zeros((good_shape[0] - 1, good_shape[1]))
    bad_velocities = jnp.zeros_like(bad_positions)

    with pytest.raises(ValueError):
        simulation(
            base_params,
            number_grid_points=G,
            number_pseudoelectrons=N_e,
            total_steps=total_steps,
            field_solver=0,
            positions=bad_positions,
            velocities=bad_velocities,
        )
