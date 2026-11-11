"""Short PIC runs for the self-hosted GPU lane (``pytest -m gpu``).

They exercise only existing code, so they also pass on CPU and add no
uncovered branches to the coverage run. Compilation dominates on GPU, so
the suite compiles one configuration.
"""
from copy import deepcopy

import jax
import numpy as np
import pytest

from jaxincell._simulation import Simulation
from tests.helpers import base_simulation_parameters

pytestmark = pytest.mark.gpu

STEPS = 5


def make_simulation(**solver):
    parameters = deepcopy(base_simulation_parameters())
    parameters["domain_parameters"].update({"total_steps": STEPS, "number_grid_points": 8})
    for species in ("electrons", "ions"):
        for block in parameters["species_parameters"][species].values():
            block["number_pseudoparticles"] = 50
    parameters["solver_parameters"].update(solver)
    return Simulation(parameters)


def assert_finite(output):
    for key in ("positions", "velocities", "electric_field", "magnetic_field"):
        assert np.all(np.isfinite(np.asarray(output[key]))), key


@pytest.fixture(scope="module")
def gauss_simulation():
    return make_simulation(field_solver=0)


def test_gauss_steps_are_finite(gauss_simulation):
    output = gauss_simulation.run()
    assert output["positions"].shape[0] == STEPS
    assert_finite(output)


def test_rerun_agrees_to_roundoff(gauss_simulation):
    # GPU scatter-adds are not bit-reproducible, so compare to round-off.
    first, second = gauss_simulation.run(), gauss_simulation.run()
    field = np.asarray(first["electric_field"])
    np.testing.assert_allclose(field, np.asarray(second["electric_field"]), rtol=1e-6, atol=1e-9 * np.abs(field).max())


def test_output_is_on_default_device_and_neutral(gauss_simulation):
    output = gauss_simulation.run()
    assert output["positions"].devices() <= set(jax.devices())
    charge = np.asarray(output["charge_density"])
    assert np.abs(charge.sum(axis=1)).max() <= 1e-6 * np.abs(charge).max()
