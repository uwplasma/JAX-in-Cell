# tests/test_diagnostics.py

import jax.numpy as jnp
import numpy as np
import pytest
from jaxincell._diagnostics import diagnostics
from jaxincell._constants import epsilon_0, mu_0


def _minimal_diagnostic_output(
    *,
    electric_field,
    external_electric_field=None,
    magnetic_field=None,
    external_magnetic_field=None,
    positions=None,
    velocities=None,
    charges=None,
    masses=None,
    dx=0.5,
    dt=0.1,
):
    T, G, _ = electric_field.shape
    if external_electric_field is None:
        external_electric_field = jnp.zeros_like(electric_field)
    if magnetic_field is None:
        magnetic_field = jnp.zeros_like(electric_field)
    if external_magnetic_field is None:
        external_magnetic_field = jnp.zeros_like(electric_field)
    if charges is None:
        charges = jnp.array([[-1.0], [1.0]])
    if masses is None:
        masses = jnp.array([[1.0], [2.0]])
    N = charges.shape[0]
    if positions is None:
        positions = jnp.zeros((T, N, 3))
    if velocities is None:
        velocities = jnp.zeros((T, N, 3))

    return {
        "positions": positions,
        "velocities": velocities,
        "masses": masses,
        "charges": charges,
        "electric_field": electric_field,
        "external_electric_field": external_electric_field,
        "magnetic_field": magnetic_field,
        "external_magnetic_field": external_magnetic_field,
        "grid": jnp.linspace(0.0, dx * (G - 1), G),
        "dt": dt,
        "total_steps": T,
        "dx": dx,
        "plasma_frequency": 1.0,
    }


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
        return jnp.sum(y, axis=-1) * dx_val

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


def test_diagnostics_external_field_energy_nonzero_cases():
    """Test jaxincell._diagnostics.diagnostics external field energy calculations.

    Cases:
    - nonzero external_electric_field produces expected density and integrated energy.
    - nonzero external_magnetic_field produces expected density and integrated energy.
    - total_energy includes both internal and external field energies.
    """
    dx = 0.25
    electric_field = jnp.ones((2, 3, 3)).at[:, :, 1:].set(0.0)
    external_electric_field = jnp.array([
        [[1.0, 2.0, 0.0], [0.0, 1.0, 2.0], [2.0, 0.0, 1.0]],
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
    ])
    external_magnetic_field = jnp.array([
        [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        [[1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [3.0, 0.0, 3.0]],
    ])
    output = _minimal_diagnostic_output(
        electric_field=electric_field,
        external_electric_field=external_electric_field,
        external_magnetic_field=external_magnetic_field,
        dx=dx,
    )

    diagnostics(output)

    expected_external_electric_density = (
        epsilon_0 / 2 * jnp.sum(external_electric_field**2, axis=-1)
    )
    expected_external_magnetic_density = (
        1 / (2 * mu_0) * jnp.sum(external_magnetic_field**2, axis=-1)
    )

    assert jnp.allclose(
        output["external_electric_field_energy_density"],
        expected_external_electric_density,
    )
    assert jnp.allclose(
        output["external_magnetic_field_energy_density"],
        expected_external_magnetic_density,
    )
    assert jnp.allclose(
        output["external_electric_field_energy"],
        jnp.sum(expected_external_electric_density, axis=-1) * dx,
    )
    assert jnp.allclose(
        output["external_magnetic_field_energy"],
        jnp.sum(expected_external_magnetic_density, axis=-1) * dx,
    )
    total_energy_terms = (
        output["electric_field_energy"]
        + output["external_electric_field_energy"]
        + output["magnetic_field_energy"]
        + output["external_magnetic_field_energy"]
        + output["kinetic_energy"]
    )
    assert jnp.allclose(output["total_energy"], total_energy_terms)


def test_diagnostics_species_split_with_multiple_species_signs():
    """Test jaxincell._diagnostics.diagnostics species splitting.

    Cases:
    - multiple negative-charge particles are grouped into electron arrays.
    - multiple positive-charge particles are grouped into ion arrays.
    - species metadata preserves positions, velocities, masses, and charges with correct leading dimensions.
    """
    T = 3
    charges = jnp.array([[-1.0], [-2.0], [1.0], [2.0]])
    masses = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    positions = jnp.arange(T * 4 * 3, dtype=float).reshape(T, 4, 3)
    velocities = positions + 100.0
    electric_field = jnp.ones((T, 4, 3)).at[:, :, 1:].set(0.0)
    output = _minimal_diagnostic_output(
        electric_field=electric_field,
        positions=positions,
        velocities=velocities,
        charges=charges,
        masses=masses,
    )

    diagnostics(output)

    assert output["position_electrons"].shape == (T, 2, 3)
    assert output["velocity_electrons"].shape == (T, 2, 3)
    assert output["mass_electrons"].shape == (2, 1)
    assert output["charge_electrons"].shape == (2, 1)
    assert output["position_ions"].shape == (T, 2, 3)
    assert output["velocity_ions"].shape == (T, 2, 3)
    assert output["mass_ions"].shape == (2, 1)
    assert output["charge_ions"].shape == (2, 1)
    assert jnp.all(output["charge_electrons"] < 0)
    assert jnp.all(output["charge_ions"] > 0)

    species = output["species"]
    assert len(species) == 4
    assert {species_entry["charge"] for species_entry in species} == {-2.0, -1.0, 1.0, 2.0}
    assert {species_entry["mass"] for species_entry in species} == {1.0, 2.0, 3.0, 4.0}
    assert "electrons" in {species_entry["name"] for species_entry in species}
    assert "ions" in {species_entry["name"] for species_entry in species}
    for species_entry in species:
        assert species_entry["positions"].shape == (T, 1, 3)
        assert species_entry["velocities"].shape == (T, 1, 3)


def test_diagnostics_dominant_frequency_for_oscillatory_energy():
    """Test jaxincell._diagnostics.diagnostics dominant_frequency.

    Cases:
    - a known oscillatory total energy signal reports the expected nonzero dominant frequency.
    - dt and total_steps are used consistently in the frequency axis.
    - constant energy falls back to zero dominant frequency.
    """
    T = 8
    G = 4
    dt = 0.25
    time_index = jnp.arange(T)
    electric_field = jnp.zeros((T, G, 3))
    electric_field = electric_field.at[:, G // 2, 0].set(
        jnp.sin(2 * jnp.pi * time_index / T)
    )
    output = _minimal_diagnostic_output(
        electric_field=electric_field,
        dt=dt,
    )

    diagnostics(output)

    expected_angular_frequency = 2 * jnp.pi / (T * dt)
    assert jnp.isclose(output["dominant_frequency"], expected_angular_frequency)

    constant_field = jnp.ones((T, G, 3)).at[:, :, 1:].set(0.0)
    constant_output = _minimal_diagnostic_output(
        electric_field=constant_field,
        dt=dt,
    )
    diagnostics(constant_output)

    assert jnp.isclose(constant_output["dominant_frequency"], 0.0)
