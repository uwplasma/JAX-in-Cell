# tests/test_diagnostics.py

import jax.numpy as jnp
import numpy as np
from jaxincell._diagnostics import diagnostics
from jaxincell._constants import epsilon_0, mu_0


def test_diagnostics_basic_energy_and_species():
    """
    Check that diagnostics:
      - splits electrons and ions correctly
      - builds a species list
      - computes energies with the same formulas as in _diagnostics.py
      - removes positions/velocities/masses/charges from the top level
    """
    # Small toy system: 2 time steps, 2 grid points, 2 particles (1 electron, 1 ion)
    T = 2
    G = 2
    N = 2

    grid = jnp.array([0.25, 0.75])
    dt = 0.1
    plasma_frequency = 1.0
    dx = 0.5

    # E field: only x–component nonzero, simple pattern
    electric_field = jnp.array([
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # t = 0
        [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],  # t = 1
    ])

    # All other fields zero to simplify checks
    external_electric_field = jnp.zeros_like(electric_field)
    magnetic_field = jnp.zeros_like(electric_field)
    external_magnetic_field = jnp.zeros_like(electric_field)

    # Positions are not used in energy, but needed for species splitting
    positions = jnp.zeros((T, N, 3))

    # Velocities: one electron (index 0), one ion (index 1)
    # electron: v^2 = 1, 4
    # ion:      v^2 = 1, 4
    velocities = jnp.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # t = 0
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],  # t = 1
    ])

    # charges < 0 -> electron, charges > 0 -> ion
    charges = jnp.array([[-1.0], [1.0]])
    # masses shape (N, 1), but values distinct
    masses = jnp.array([[2.0], [3.0]])

    output = {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "charges": charges,
        "electric_field": electric_field,
        "external_electric_field": external_electric_field,
        "magnetic_field": magnetic_field,
        "external_magnetic_field": external_magnetic_field,
        "grid": grid,
        "dt": dt,
        "total_steps": T,
        "dx": dx,
        "plasma_frequency": plasma_frequency,
    }

    diagnostics(output)

    # ---- Keys and basic structure ----
    for key in [
        "position_electrons", "velocity_electrons",
        "mass_electrons", "charge_electrons",
        "position_ions", "velocity_ions",
        "mass_ions", "charge_ions",
        "species",
        "electric_field_energy_density", "electric_field_energy",
        "magnetic_field_energy_density", "magnetic_field_energy",
        "external_electric_field_energy_density", "external_electric_field_energy",
        "external_magnetic_field_energy_density", "external_magnetic_field_energy",
        "kinetic_energy", "kinetic_energy_electrons", "kinetic_energy_ions",
        "dominant_frequency", "plasma_frequency", "total_energy",
    ]:
        assert key in output, f"Missing key {key} in diagnostics output."

    # Original heavy arrays should be dropped
    for key in ["positions", "velocities", "masses", "charges"]:
        assert key not in output

    # ---- Shapes ----
    assert output["electric_field_energy_density"].shape == (T, G)
    assert output["electric_field_energy"].shape == (T,)
    assert output["magnetic_field_energy_density"].shape == (T, G)
    assert output["magnetic_field_energy"].shape == (T,)
    assert output["external_electric_field_energy_density"].shape == (T, G)
    assert output["external_electric_field_energy"].shape == (T,)
    assert output["external_magnetic_field_energy_density"].shape == (T, G)
    assert output["external_magnetic_field_energy"].shape == (T,)
    assert output["kinetic_energy"].shape == (T,)
    assert output["kinetic_energy_electrons"].shape == (T,)
    assert output["kinetic_energy_ions"].shape == (T,)
    assert output["total_energy"].shape == (T,)

    # ---- Electric field energy: match _diagnostics.py formulas exactly ----
    abs_E_squared = jnp.sum(electric_field**2, axis=-1)  # (T, G)

    def integrate(y, dx_val):
        # same trapezoidal rule as in _diagnostics.py
        return 0.5 * (jnp.asarray(dx_val) * (y[..., 1:] + y[..., :-1])).sum(-1)

    expected_E_density = (epsilon_0 / 2.0) * abs_E_squared
    expected_E_energy = (epsilon_0 / 2.0) * integrate(abs_E_squared, dx)

    assert jnp.allclose(output["electric_field_energy_density"], expected_E_density)
    assert jnp.allclose(output["electric_field_energy"], expected_E_energy)

    # ---- Magnetic & external field energies are zero in this toy setup ----
    assert jnp.allclose(output["magnetic_field_energy_density"], 0.0)
    assert jnp.allclose(output["magnetic_field_energy"], 0.0)
    assert jnp.allclose(output["external_electric_field_energy_density"], 0.0)
    assert jnp.allclose(output["external_electric_field_energy"], 0.0)
    assert jnp.allclose(output["external_magnetic_field_energy_density"], 0.0)
    assert jnp.allclose(output["external_magnetic_field_energy"], 0.0)

    # ---- Kinetic energy: electrons + ions ----
    # Use post-diagnostics arrays to exercise that path
    me = output["mass_electrons"].reshape(-1)  # (Ne,)
    mi = output["mass_ions"].reshape(-1)       # (Ni,)

    v_sq_e = jnp.sum(output["velocity_electrons"]**2, axis=-1)  # (T, Ne)
    v_sq_i = jnp.sum(output["velocity_ions"]**2, axis=-1)       # (T, Ni)

    expected_ke_e = 0.5 * jnp.sum(me * v_sq_e, axis=-1)
    expected_ke_i = 0.5 * jnp.sum(mi * v_sq_i, axis=-1)

    assert jnp.allclose(output["kinetic_energy_electrons"], expected_ke_e)
    assert jnp.allclose(output["kinetic_energy_ions"], expected_ke_i)
    assert jnp.allclose(output["kinetic_energy"], expected_ke_e + expected_ke_i)

    # ---- Dominant frequency: for this simple 2-step signal, it's zero ----
    assert jnp.isclose(output["dominant_frequency"], 0.0)

    # ---- Total energy consistency ----
    total_calc = (
        output["electric_field_energy"]
        + output["external_electric_field_energy"]
        + output["magnetic_field_energy"]
        + output["external_magnetic_field_energy"]
        + output["kinetic_energy"]
    )
    assert jnp.allclose(output["total_energy"], total_calc)

    # ---- Species list sanity ----
    species = output["species"]
    assert len(species) == 2
    names = {s["name"] for s in species}
    assert {"electrons", "ions"}.issubset(names)
    for s in species:
        # Each species carries positions and velocities with correct leading dims
        assert s["positions"].shape[0] == T
        assert s["velocities"].shape[0] == T
        assert s["positions"].shape[2] == 3
        assert s["velocities"].shape[2] == 3
