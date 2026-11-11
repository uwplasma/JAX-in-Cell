"""Short PIC runs for the self-hosted GPU lane (``pytest -m gpu``).

They exercise only existing code, so they also pass on CPU and add no
uncovered branches to the coverage run.
"""
from copy import deepcopy

import jax
import numpy as np
import pytest

from jaxincell._simulation import Simulation
from tests.helpers import base_simulation_parameters

pytestmark = pytest.mark.gpu


def run_small(**solver):
    parameters = deepcopy(base_simulation_parameters())
    parameters["domain_parameters"].update({"total_steps": 20, "number_grid_points": 16})
    for species in ("electrons", "ions"):
        for block in parameters["species_parameters"][species].values():
            block["number_pseudoparticles"] = 200
    parameters["solver_parameters"].update(solver)
    return Simulation(parameters).run()


def finite(output):
    return all(np.all(np.isfinite(np.asarray(output[k]))) for k in ("positions", "velocities", "electric_field"))


def test_boris_gauss_steps_are_finite():
    output = run_small(field_solver=0)
    assert output["positions"].shape[0] == 20
    assert finite(output)


def test_boris_fdtd_steps_are_finite():
    assert finite(run_small(field_solver=1))


def test_relativistic_boris_steps_are_finite():
    assert finite(run_small(relativistic=True))


def test_run_is_deterministic():
    first, second = run_small(), run_small()
    np.testing.assert_array_equal(np.asarray(first["electric_field"]), np.asarray(second["electric_field"]))


def test_output_lives_on_default_device():
    output = run_small()
    assert output["positions"].devices() <= set(jax.devices())
