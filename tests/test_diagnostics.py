# tests/test_diagnostics.py

import jax.numpy as jnp
import numpy as np
import pytest
from jaxincell._diagnostics import diagnostics
from jaxincell._constants import epsilon_0, mu_0, boltzmann_constant, speed_of_light


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


def test_diagnostics_use_changing_macroparticle_masses_for_energy_and_momentum():
    """Independent two-particle ledger: fractional inelastic returns followed by
    complete ion collection. Lost weight must leave both energy and momentum."""
    velocity = jnp.array([[[2., 1., 0.], [-1., 0., 2.]],
                          [[-1., 1., 0.], [.5, 0., 2.]],
                          [[.5, 1., 0.], [0., 0., 0.]]])
    output = _minimal_diagnostic_output(electric_field=jnp.ones((3, 4, 3)).at[..., 1:].set(0.),
                                        velocities=velocity, masses=jnp.array([[2.], [6.]]), dx=.25)
    output["masses_over_time"] = jnp.array([[[2.], [6.]], [[1.], [3.]], [[.5], [0.]]])
    diagnostics(output)
    np.testing.assert_allclose(output["kinetic_energy_electrons"], [5., 1., .3125], rtol=1e-14)
    np.testing.assert_allclose(output["kinetic_energy_ions"], [15., 6.375, 0.], rtol=1e-14)
    np.testing.assert_allclose(output["kinetic_energy"], [20., 7.375, .3125], rtol=1e-14)
    momentum = np.array([[-2., 2., 12.], [.5, 1., 6.], [.25, .5, 0.]])
    np.testing.assert_allclose(output["total_momentum"], momentum, rtol=1e-14)
    expected_change = np.linalg.norm(momentum - momentum[0], axis=1) / (8 * np.sqrt(5))
    np.testing.assert_allclose(output["momentum_error_rel"], expected_change, rtol=1e-14)
    np.testing.assert_allclose(output["total_energy"], np.array([20., 7.375, .3125]) + epsilon_0 / 2, rtol=1e-14)


def test_diagnostics_basic_energy_and_species():
    """
    Check that diagnostics:
      - splits electrons and ions correctly
      - builds a species list
      - computes energies with the same formulas as in _diagnostics.py
      - preserves positions/velocities/masses/charges at the top level
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

    # Raw arrays remain available for another diagnostic or analysis.
    for key in ["positions", "velocities", "masses", "charges"]:
        assert key in output

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

    # A two-row, nonconstant signal has its sole nonzero bin at Nyquist.
    assert jnp.isclose(output["dominant_frequency"], jnp.pi / dt)

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


def test_diagnostics_gauss_law_error_and_momentum():
    T, G, dx = 3, 8, 0.5
    rho = jnp.stack([jnp.sin(2 * jnp.pi * jnp.arange(G) / G) * (t + 1) for t in range(T)])
    Ex = jnp.cumsum(rho, axis=1) * dx / epsilon_0          # backward-difference Gauss solution
    electric_field = jnp.zeros((T, G, 3)).at[..., 0].set(Ex)
    velocities = jnp.array([
        [[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        [[2.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
    ])
    output = _minimal_diagnostic_output(electric_field=electric_field, velocities=velocities, dx=dx)
    output["charge_density"] = rho
    diagnostics(output)

    # masses are 1 (electron) and 2 (ion): P_x = 0, 0, 1 and sum m|v| at t = 0 is 2
    assert jnp.allclose(output["total_momentum"][:, 0], jnp.array([0.0, 0.0, 1.0]))
    assert jnp.allclose(output["momentum_error_rel"], jnp.array([0.0, 0.0, 0.5]))
    assert jnp.all(output["gauss_error_Linf_rel"] < 1e-12)

    # breaking Gauss's law at one node shows up in the error
    output = _minimal_diagnostic_output(electric_field=electric_field.at[1, 3, 0].add(1.0), dx=dx)
    output["charge_density"] = rho
    diagnostics(output)
    assert output["gauss_error_Linf_rel"][1] > 1e-12
    assert jnp.all(output["gauss_error_Linf_rel"][jnp.array([0, 2])] < 1e-12)


def test_diagnostics_preserves_populations_and_weighted_temperatures_on_repeat():
    """Distinct equal-q/m populations retain their names, physical moments and data."""
    weights = jnp.array([1.0, 3.0, 2.0, 2.0, 4.0])
    physical_mass = jnp.array([2.0, 2.0, 2.0, 2.0, 5.0])
    charge = jnp.array([-1.0, -1.0, -1.0, -1.0, 1.0])
    v = jnp.zeros((2, 5, 3)).at[0, :, 0].set(jnp.array([0.0, 2.0, 10.0, 12.0, 1.0]))
    v = v.at[1].set(v[0] + jnp.array([4.0, 0.0, 0.0]))
    output = _minimal_diagnostic_output(electric_field=jnp.zeros((2, 4, 3)), velocities=v,
        masses=(physical_mass * weights)[:, None], charges=(charge * weights)[:, None])
    output.update(weights=weights[:, None], species_integer_index=jnp.array([0, 0, 1, 1, 2]),
        mass_integer_lookup=jnp.array([2.0, 2.0, 5.0]), charge_integer_lookup=jnp.array([-1.0, -1.0, 1.0]),
        species_parameters={"electrons": {"_electrons0": {"user_label": "bulk"},
                                          "_electrons1": {"user_label": "beam"}},
                            "ions": {"_ions0": {"user_label": "heavy"}}},
        time_array=jnp.array([0.1, 0.2]))
    original = {key: np.asarray(value).copy() for key, value in output.items() if key != "species_parameters"}
    for _ in range(2):
        diagnostics(output)
        for key, value in original.items():
            np.testing.assert_array_equal(output[key], value)
        bulk, beam, ion = output["species"]
        assert [sp["name"] for sp in output["species"]] == ["electrons.bulk", "electrons.beam", "ions.heavy"]
        assert bulk["mass"] == beam["mass"] == 2.0
        np.testing.assert_allclose(bulk["temperature_components"] * boltzmann_constant, [[1.5, 0, 0]] * 2)
        np.testing.assert_allclose(beam["temperature"] * boltzmann_constant, [2 / 3] * 2)
        np.testing.assert_array_equal(ion["temperature"], 0.0)
        expected_ke = 0.5 * np.sum(np.asarray(physical_mass * weights)[None, :] * np.sum(np.asarray(v) ** 2, axis=-1), axis=-1)
        np.testing.assert_allclose(output["kinetic_energy"], expected_ke)
        np.testing.assert_allclose(sum(sp["kinetic_energy"] for sp in output["species"]), expected_ke)


@pytest.mark.parametrize("removed", [False, True])
def test_diagnostics_one_stored_step_is_finite(removed):
    output = _minimal_diagnostic_output(electric_field=jnp.zeros((1, 4, 3)))
    if removed:
        output.update(weights=jnp.zeros((2, 1)), masses=jnp.zeros((2, 1)),
                      charges=jnp.zeros((2, 1)), species_integer_index=jnp.arange(2))
    output["total_steps"] = 32
    output["time_array"] = jnp.array([3.2])
    diagnostics(output)
    assert output["dominant_frequency"] == 0.0
    assert np.all(np.isfinite(output["total_energy"]))
    assert all(np.all(np.isfinite(sp["temperature"])) for sp in output["species"])
    np.testing.assert_array_equal(output["time_array"], [3.2])


def test_diagnostics_frequency_uses_the_stored_uniform_times():
    t = jnp.arange(8)
    field = jnp.zeros((8, 4, 3)).at[:, 2, 0].set(jnp.sin(2 * jnp.pi * t / 8))
    output = _minimal_diagnostic_output(electric_field=field, dt=0.25)
    output.update(total_steps=24, time_array=10.0 + 0.75 * t)
    diagnostics(output)
    assert output["dominant_frequency"] == pytest.approx(2 * np.pi / (8 * 0.75))
    np.testing.assert_array_equal(output["time_array"], 10.0 + 0.75 * np.arange(8))


def test_public_diagnostics_reuses_real_simulation_output():
    from copy import deepcopy
    from jaxincell import Simulation, diagnostics as public_diagnostics
    from tests.helpers import base_simulation_parameters

    params = base_simulation_parameters()
    params["domain_parameters"]["total_steps"] = 2
    params["species_parameters"]["electrons"]["beam"] = deepcopy(params["species_parameters"]["electrons"]["electrons0"])
    params["species_parameters"]["electrons"]["beam"]["drift_speed_x"] = 10.0
    output = Simulation(params).run()
    keys = ("positions", "velocities", "charges", "masses", "weights", "species_integer_index", "time_array")
    raw = {key: np.asarray(output[key]).copy() for key in keys}
    expected_ke = 0.5 * np.sum(raw["masses"].reshape(-1) * np.sum(raw["velocities"] ** 2, axis=-1), axis=-1)
    for _ in range(2):
        public_diagnostics(output)
        assert [sp["name"] for sp in output["species"]] == ["electrons.electrons0", "electrons.beam", "ions.ions0"]
        np.testing.assert_allclose(output["kinetic_energy"], expected_ke, rtol=1e-12, atol=0.0)
        for key, value in raw.items():
            np.testing.assert_array_equal(output[key], value)


@pytest.mark.parametrize("rows, mode", [(3, 1), (8, 4)])
def test_diagnostics_keeps_the_last_positive_frequency_bin(rows, mode):
    dt = 0.25
    field = jnp.zeros((rows, 4, 3)).at[:, 2, 0].set(jnp.cos(2 * jnp.pi * mode * jnp.arange(rows) / rows))
    output = _minimal_diagnostic_output(electric_field=field, dt=dt)
    diagnostics(output)
    assert output["dominant_frequency"] == pytest.approx(2 * np.pi * mode / (rows * dt))


def test_diagnostics_relativistic_energy_and_momentum_at_point_six_c():
    # gamma=5/4: K=(1/4) m c^2 and p=(5/4) m v, including macro weights.
    mass = jnp.array([[2.0 * 2.0], [3.0 * 4.0]])
    v = jnp.zeros((2, 2, 3)).at[:, 0, 0].set(0.6 * speed_of_light).at[:, 1, 0].set(-0.6 * speed_of_light)
    output = _minimal_diagnostic_output(electric_field=jnp.zeros((2, 4, 3)), masses=mass, velocities=v)
    output.update(weights=jnp.array([[2.0], [4.0]]), solver_parameters={"relativistic": True})
    diagnostics(output)
    np.testing.assert_allclose(output["kinetic_energy"], [4 * speed_of_light ** 2] * 2, rtol=1e-12)
    np.testing.assert_allclose(output["total_momentum"], [[-6 * speed_of_light, 0.0, 0.0]] * 2, rtol=1e-12)
    np.testing.assert_allclose(sum(sp["kinetic_energy"] for sp in output["species"]), output["kinetic_energy"], rtol=1e-12)
    np.testing.assert_array_equal(output["momentum_error_rel"], 0.0)


def test_relativistic_kinetic_energy_has_a_stable_low_speed_limit():
    v = jnp.full((2, 2, 3), 1e-8 * speed_of_light)
    output = _minimal_diagnostic_output(electric_field=jnp.zeros((2, 4, 3)), velocities=v)
    output["solver_parameters"] = {"relativistic": True}
    diagnostics(output)
    np.testing.assert_allclose(output["kinetic_energy"], 4.5 * (1e-8 * speed_of_light) ** 2, rtol=1e-12)


def test_diagnostics_partial_wall_weights_use_stored_mass_histories():
    v = jnp.array([[[1., 0., 0.], [3., 0., 0.]], [[1., 0., 0.], [3., 0., 0.]]])
    output = _minimal_diagnostic_output(electric_field=jnp.zeros((2, 4, 3)), velocities=v,
                                        masses=jnp.array([[4.], [8.]]), charges=-jnp.ones((2, 1)))
    output.update(weights=jnp.array([[2.], [4.]]), species_integer_index=jnp.zeros(2, dtype=int),
                  mass_integer_lookup=jnp.array([2.]), charge_integer_lookup=jnp.array([-1.]),
                  masses_over_time=jnp.array([[[4.], [8.]], [[4.], [2.]]]))
    raw = np.asarray(output["masses_over_time"]).copy()
    for _ in range(2):
        diagnostics(output)
        np.testing.assert_array_equal(output["masses_over_time"], raw)
        np.testing.assert_allclose(output["kinetic_energy"], [38., 11.], atol=0)
        np.testing.assert_allclose(output["total_momentum"][:, 0], [28., 10.], atol=0)
        np.testing.assert_allclose(output["species"][0]["temperature_components"][:, 0] * boltzmann_constant,
                                   [16/9, 16/9], rtol=1e-14, atol=0)


@pytest.mark.parametrize("transverse_shape", [(2,), (2, 3)])
def test_diagnostics_tensor_external_energy_averages_transverse_coordinates(transverse_shape):
    shape = (4, *transverse_shape, 3)
    field = jnp.arange(np.prod(shape), dtype=float).reshape(shape)
    output = _minimal_diagnostic_output(electric_field=jnp.zeros((2, 4, 3)), external_magnetic_field=field)
    output["dimensions"] = ("x", "y", "z")[:len(transverse_shape)+1]
    diagnostics(output)
    density = np.mean(np.sum(np.asarray(field)**2, axis=-1), axis=tuple(range(1, field.ndim-1))) / (2*mu_0)
    np.testing.assert_allclose(output["external_magnetic_field_energy_density"], density, rtol=1e-14, atol=0)
    np.testing.assert_allclose(output["total_energy"], np.sum(density)*output["dx"], rtol=1e-14, atol=0)


@pytest.mark.parametrize("scale", [1., 1e-12])
def test_diagnostics_nonuniform_snapshot_times_do_not_report_an_fft_frequency(scale):
    output = _minimal_diagnostic_output(electric_field=jnp.ones((4, 4, 3)), dt=scale)
    output["time_array"] = scale*jnp.array([1., 2., 4., 5.])
    diagnostics(output)
    assert np.isnan(output["dominant_frequency"])
    assert np.isfinite(output["total_energy"]).all()
    np.testing.assert_array_equal(output["time_array"], scale*np.array([1., 2., 4., 5.]))


@pytest.mark.parametrize("transverse_shape", [(3,), (3, 2)])
def test_prescribed_energy_averages_transverse_centres(transverse_shape):
    # PIC still represents a unit-area planar column; tensor sampling adds no volume.
    electric = jnp.zeros((2, 4, 3))
    field = np.zeros((4, *transverse_shape, 3))
    field[..., 0] = np.arange(field[..., 0].size).reshape(field.shape[:-1])
    output = _minimal_diagnostic_output(electric_field=electric, external_electric_field=field)
    output["dimensions"] = ("x", "y") if len(transverse_shape) == 1 else ("x", "y", "z")
    diagnostics(output)
    transverse_count = np.prod(transverse_shape)
    expected = epsilon_0 / 2 * .5 * np.sum(field[..., 0]**2) / transverse_count
    assert float(output["external_electric_field_energy"]) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("transverse_axis", ["y", "z"])
def test_real_planar_run_preserves_axes_energy_and_optional_writer(tmp_path, transverse_axis):
    import importlib.util
    from jaxincell import Simulation
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=2, number_grid_points=4, number_pseudoparticles=2)
    p["domain_parameters"].update(number_grid_points_y=0, number_grid_points_z=0, length_y=.03, length_z=.05)
    p["domain_parameters"][f"number_grid_points_{transverse_axis}"] = 3
    field = np.zeros((4, 3, 3))
    field[..., 2] = np.arange(1, 4) * .001
    p["external_field_parameters"] = {"external_magnetic_field": {"B": field}}
    output = Simulation(p).run()
    assert output["dimensions"] == ("x", transverse_axis)
    # Exercise the merged optional writer when present; this feature alone stays independent of I/O.
    if importlib.util.find_spec("jaxincell.openpmd") and importlib.util.find_spec("openpmd_api"):
        import openpmd_api as io
        from jaxincell.openpmd import write_openpmd
        paths = write_openpmd(output, openpmd_filename=str(tmp_path / "planar.json"))
        series = io.Series(paths["data"]["combined"], io.Access.read_only)
        mesh = series.iterations[0].meshes["external_B"]
        assert mesh.axis_labels == ["x", transverse_axis]
        expected_spacing = [.01/4, (.03 if transverse_axis == "y" else .05)/3]
        np.testing.assert_allclose(mesh.grid_spacing, expected_spacing, rtol=1e-14, atol=0)
        values = mesh["z"].load_chunk()
        series.flush()
        np.testing.assert_array_equal(values, field[..., 2])
        series.close()
    diagnostics(output)
    expected = .01 * np.mean(np.sum(field**2, axis=-1)) / (2*mu_0)
    assert np.ndim(output["external_magnetic_field_energy"]) == 0
    np.testing.assert_allclose(output["external_magnetic_field_energy"], expected, rtol=2e-14, atol=0)
