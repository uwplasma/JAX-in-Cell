"""Short PIC runs for the self-hosted GPU lane (``pytest -m gpu``).

They exercise only existing code, so they also pass on CPU and add no
uncovered branches to the coverage run. Compilation dominates on GPU, so
the suite compiles two configurations in total.
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


def test_rerun_is_deterministic(gauss_simulation):
    first, second = gauss_simulation.run(), gauss_simulation.run()
    np.testing.assert_array_equal(np.asarray(first["electric_field"]), np.asarray(second["electric_field"]))


def test_fdtd_steps_are_finite_on_default_device():
    output = make_simulation(field_solver=1).run()
    assert_finite(output)
    assert output["positions"].devices() <= set(jax.devices())
