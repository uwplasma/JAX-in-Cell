"""Short PIC runs for the self-hosted GPU lane (``pytest -m gpu``).

They exercise only existing code, so they also pass on CPU and add no uncovered branches
to the coverage run. Compilation dominates on GPU, so the suite compiles one configuration.
"""
import jax
import numpy as np
import pytest

from jaxincell import Domain, Simulation, Solver, Species, speed_of_light as c

pytestmark = pytest.mark.gpu

STEPS = 5


@pytest.fixture(scope="module")
def simulation():
    e = Species.electrons(n=50, density=4e17, vth=(0.05 * c, 0, 0), plus_minus=True)
    i = Species.ions(n=50, density=4e17, electrons=e)
    return Simulation(Domain(length=0.01, cells=8, dt_over_dx_c=2.0), [e, i], Solver())


def test_steps_are_finite(simulation):
    out = simulation.run(STEPS)
    assert out.E.shape[0] == len(out.t)
    for field in (out.x, out.v, out.E, out.B):
        assert np.all(np.isfinite(np.asarray(field)))


def test_rerun_agrees_to_roundoff(simulation):
    # GPU scatter-adds are not bit-reproducible, so compare to round-off.
    first, second = simulation.run(STEPS).E, simulation.run(STEPS).E
    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-9 * np.abs(first).max())


def test_output_is_on_default_device(simulation):
    assert simulation.run(STEPS).x.devices() <= set(jax.devices())
