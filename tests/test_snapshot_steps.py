# tests/test_snapshot_steps.py

import subprocess
import sys
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell._simulation import Simulation
from tests.helpers import scalar
from tests.test_simulation import small_simulation_parameters


def test_snapshot_steps_matches_full_history_final_state():
    """Test solver_parameters["snapshot_steps"].

    Cases:
    - unset (default) keeps behavior unchanged: full per-step history.
    - an explicit list of step indices only keeps those snapshots, with
      output shapes reduced from (total_steps, ...) to (len(snapshot_steps), ...).
    - every recorded snapshot exactly matches the corresponding step of the
      equivalent full-history run (same seed).
    - time_array is realigned to the requested snapshot steps rather than
      staying at full length.
    """
    total_steps = 12
    number_grid_points = 6
    number_pseudoparticles = 6

    base_parameters = small_simulation_parameters(
        total_steps=total_steps,
        number_grid_points=number_grid_points,
        number_pseudoparticles=number_pseudoparticles,
    )
    base_parameters["solver_parameters"]["seed"] = 4242

    full_output = Simulation(deepcopy(base_parameters)).run()
    assert full_output["electric_field"].shape == (total_steps, number_grid_points, 3)
    assert full_output["time_array"].shape == (total_steps,)

    snapshot_steps = [0, 5, total_steps - 1]
    snap_parameters = deepcopy(base_parameters)
    snap_parameters["solver_parameters"]["snapshot_steps"] = snapshot_steps
    snap_output = Simulation(snap_parameters).run()

    n_particles = 2 * number_pseudoparticles
    num_snapshots = len(snapshot_steps)
    assert snap_output["electric_field"].shape == (num_snapshots, number_grid_points, 3)
    assert snap_output["magnetic_field"].shape == (num_snapshots, number_grid_points, 3)
    assert snap_output["current_density"].shape == (num_snapshots, number_grid_points, 3)
    assert snap_output["charge_density"].shape == (num_snapshots, number_grid_points)
    assert snap_output["positions"].shape == (num_snapshots, n_particles, 3)
    assert snap_output["velocities"].shape == (num_snapshots, n_particles, 3)

    dt = scalar(full_output["dt"])
    np.testing.assert_allclose(
        np.asarray(snap_output["time_array"]),
        np.asarray(snapshot_steps) * dt,
        rtol=1e-12,
    )

    for snap_index, step in enumerate(snapshot_steps):
        assert jnp.allclose(snap_output["electric_field"][snap_index], full_output["electric_field"][step])
        assert jnp.allclose(snap_output["magnetic_field"][snap_index], full_output["magnetic_field"][step])
        assert jnp.allclose(snap_output["positions"][snap_index], full_output["positions"][step])
        assert jnp.allclose(snap_output["velocities"][snap_index], full_output["velocities"][step])


def _peak_gpu_bytes_in_subprocess(snapshot_steps, total_steps, number_grid_points, number_pseudoparticles, seed):
    """Run a small simulation in an isolated subprocess and return the peak
    GPU memory (bytes) JAX reported for it.

    A fresh process is required because jax.devices()[0].memory_stats()
    ["peak_bytes_in_use"] is a monotonically non-decreasing high-water mark
    for the whole process, so two in-process measurements can't be compared
    directly (the second call's number would already include the first).
    """
    code = f"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
from jax import block_until_ready
from jaxincell import Simulation

parameters = {{
    "domain_parameters": {{
        "total_steps": {total_steps},
        "number_grid_points": {number_grid_points},
        "length": 0.01,
    }},
    "species_parameters": {{
        "electrons": {{"electrons0": {{"number_pseudoparticles": {number_pseudoparticles}, "grid_points_per_Debye_length": 1.0}}}},
        "ions": {{"ions0": {{"number_pseudoparticles": {number_pseudoparticles}, "grid_points_per_Debye_length": 1.0}}}},
    }},
    "solver_parameters": {{"field_solver": 0, "print_info": False, "seed": {seed}}},
}}
snapshot_steps = {snapshot_steps!r}
if snapshot_steps is not None:
    parameters["solver_parameters"]["snapshot_steps"] = snapshot_steps

output = block_until_ready(Simulation(parameters).run())
print(jax.devices()[0].memory_stats()["peak_bytes_in_use"])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip().splitlines()[-1])


def test_snapshot_steps_reduces_peak_gpu_memory():
    """Test that solver_parameters["snapshot_steps"] reduces peak device
    memory relative to full per-step history, on a problem large enough for
    the difference to be measurable.

    Requires a GPU backend (peak_bytes_in_use isn't a meaningful comparison
    on CPU); skipped automatically otherwise, e.g. on CPU-only CI runners.
    """
    if jax.default_backend() != "gpu":
        pytest.skip("peak GPU memory comparison requires a GPU backend")

    total_steps = 4000
    number_grid_points = 4000
    seed = 4242

    full_peak = _peak_gpu_bytes_in_subprocess(None, total_steps, number_grid_points, 50, seed)
    final_only_peak = _peak_gpu_bytes_in_subprocess([total_steps - 1], total_steps, number_grid_points, 50, seed)

    assert final_only_peak < full_peak
