# tests/test_simulation.py

import jax.numpy as jnp
import numpy as np

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
    assert output["external_electric_field_energy"].shape in [(), (T,)]
    assert output["external_magnetic_field_energy"].shape in [(), (T,)]
    assert output["kinetic_energy"].shape == (T,)
    assert output["total_energy"].shape == (T,)

    # Kinetic energy must be sum of electron + ion contributions
    ke_sum = output["kinetic_energy_electrons"] + output["kinetic_energy_ions"]
    assert np.allclose(
        np.array(output["kinetic_energy"]),
        np.array(ke_sum),
        rtol=1e-10, atol=1e-12,
    )

    # Total energy must match sum of components
    total_calc = (
        output["electric_field_energy"]
        + output["external_electric_field_energy"]
        + output["magnetic_field_energy"]
        + output["external_magnetic_field_energy"]
        + output["kinetic_energy"]
    )
    assert np.allclose(
        np.array(output["total_energy"]),
        np.array(total_calc),
        rtol=1e-10, atol=1e-12,
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
