"""The Coulomb logarithm of the binary collision operator."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import Collisions, Domain, Simulation, Solver, Species
from jaxincell._collisions import coulomb_logarithm
from jaxincell._constants import elementary_charge as e_charge, mass_electron


def test_coulomb_logarithm_is_floored_and_survives_zero_temperature():
    """The formulary goes negative in cold, dense plasma (-0.03 at 1e20 m^-3 and 10 meV)
    and divides by zero at T = 0. Both give the floor of 2, with finite gradients."""
    assert float(coulomb_logarithm(1e20, 0.01)) == 2.0
    assert float(coulomb_logarithm(1e20, 1e-3)) == 2.0
    assert float(coulomb_logarithm(1e20, 0.0)) == 2.0
    assert abs(float(coulomb_logarithm(1e20, 0.01, floor=-np.inf)) + 0.026) < 1e-3
    grads = jax.grad(lambda n, t: coulomb_logarithm(n, t), argnums=(0, 1))(1e20, 0.0)
    assert all(np.isfinite(float(g)) for g in grads)


def test_default_coulomb_logarithm_comes_from_the_electrons_wherever_they_are_listed():
    """`Collisions()` takes ln(Lambda) at the density and temperature of the lightest
    negatively charged species, not of whichever species comes first; the temperature
    is m v_th^2 / 2 of the largest thermal-speed component."""
    electrons = Species.electrons(n=400, density=1e20, vth=(2e6, 2e6, 1e6), quiet=True)
    ions = Species.ions(n=400, density=1e20, electrons=electrons, temperature_ratio=0.01, quiet=True)
    domain = Domain(length=1e-4, cells=4, dt_over_dx_c=1.0)
    ln_lambda = float(coulomb_logarithm(1e20, mass_electron * 2e6 ** 2 / 2 / e_charge))
    explicit = Simulation(domain, [ions, electrons], Solver(), Collisions(coulomb_log=ln_lambda)).run(5, seed=0)
    default = Simulation(domain, [ions, electrons], Solver(), Collisions()).run(5, seed=0)
    assert np.allclose(np.asarray(default.v), np.asarray(explicit.v), rtol=1e-9, atol=0)


def test_default_coulomb_logarithm_needs_an_electron_species():
    ions = Species.ions(n=40, density=1e20, vth=(1e4, 1e4, 1e4), quiet=True)
    simulation = Simulation(Domain(length=1e-4, cells=4, dt_over_dx_c=1.0), [ions], Solver(), Collisions())
    x, v = jnp.zeros((40, 3)), jnp.ones((40, 3))
    with pytest.raises(ValueError, match="negative charge"):
        simulation._collide(random.PRNGKey(0), x, v, jnp.ones(40), jnp.ones(40), jnp.ones(40), 1e-12)
