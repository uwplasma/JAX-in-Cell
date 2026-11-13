"""Degenerate inputs: one particle, and a vanishing or enormous density. The expected values
are Newton's first law and the plasma frequency, not outputs of the code under test."""
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import electron_plasma

from jaxincell import Domain, Simulation, Solver, Species


def test_a_lone_electron_in_a_negligible_density_moves_in_a_straight_line():
    """With one marker per species and a density of 1e3 m^-3 the self-field is far below what
    changes the velocity in 10 steps, so the velocity is constant and x advances by v t."""
    velocity = np.array([1e6, 2e5, 0.0])
    electrons = Species.electrons(n=1, density=1e3, drift=tuple(velocity))
    ions = Species.ions(n=1, density=1e3, electrons=electrons)
    out = Simulation(Domain(length=1.0, cells=4, dt_over_dx_c=0.5), [electrons, ions], Solver()).run(
        10, seed=0, store_every=1, store_particles=True)
    elapsed = float(out.t[-1] - out.t[0])
    assert np.allclose(np.asarray(out.v[:, 0]), velocity, rtol=1e-9, atol=1.0)
    assert np.allclose(np.asarray(out.x[-1, 0] - out.x[0, 0]), velocity * elapsed, rtol=1e-9, atol=1e-15)


def test_a_vanishing_density_carries_no_field_and_does_not_underflow_to_nan():
    electrons = Species.electrons(n=64, density=1e-30, vth=(3e6, 0.0, 0.0))
    ions = Species.ions(n=64, density=1e-30, electrons=electrons)
    out = Simulation(Domain(length=1e-3, cells=8, dt_over_dx_c=0.5), [electrons, ions], Solver()).run(20, seed=0)
    assert bool(jnp.isfinite(out.E).all()) and float(jnp.abs(out.E).max()) == pytest.approx(0.0, abs=1e-20)


def test_an_enormous_density_is_stable_when_the_time_step_resolves_the_plasma_frequency():
    """A 1e31 m^-3 plasma in a nanometre box: omega_pe dt = 0.1 is below the leapfrog limit
    of 2 (Birdsall and Langdon, Plasma Physics via Computer Simulation, sec. 4-3), so the
    run stays finite."""
    simulation, omega_pe = electron_plasma(64, 1e-9, 8, 0.1, 0.01, amplitude_k=1e-3)
    assert float(simulation.species[0].density) > 1e30
    assert bool(jnp.isfinite(simulation.run(40, seed=0).E).all())
