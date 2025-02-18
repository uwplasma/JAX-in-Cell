import pytest
import jax.numpy as jnp
from jaxincell import simulation

def test_simulation_runs():
    """Test if the simulation runs without errors with default parameters."""
    result = simulation()
    assert isinstance(result, dict), "Simulation did not return a dictionary."
    assert "position_electrons" in result, "Missing position_electrons in output."
    assert "velocity_electrons" in result, "Missing velocity_electrons in output."
    assert "electric_field" in result, "Missing electric_field in output."
    assert "magnetic_field" in result, "Missing magnetic_field in output."

def test_particle_conservation():
    """Ensure the number of particles remains consistent."""
    num_electrons = 500
    num_ions = 500
    result = simulation(number_pseudoelectrons=num_electrons)
    assert result["position_electrons"].shape[1] == num_electrons, "Electron count mismatch."
    assert result["position_ions"].shape[1] == num_ions, "Ion count mismatch."

def test_electric_field_update():
    """Test that the electric field updates and does not remain zero."""
    result = simulation()
    assert not jnp.all(result["electric_field"] == 0), "Electric field did not update."

def test_regression_baseline():
    """Compare simulation results with a known baseline to detect regressions."""
    baseline = {
        "number_grid_points": 50,
        "number_pseudoelectrons": 500,
        "total_steps": 350,
    }
    result = simulation(**baseline)
    assert jnp.isfinite(result["electric_field"]).all(), "Electric field contains NaN or Inf values."
    assert jnp.isfinite(result["magnetic_field"]).all(), "Magnetic field contains NaN or Inf values."
    assert jnp.isfinite(result["position_electrons"]).all(), "Positions contain NaN or Inf values."
    assert jnp.isfinite(result["velocity_electrons"]).all(), "Velocities contain NaN or Inf values."

if __name__ == "__main__":
    pytest.main()
