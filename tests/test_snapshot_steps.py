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


@pytest.mark.parametrize("snapshot_steps", [[0], [0, 5], [0, 5, 11], [11], []])
@pytest.mark.parametrize("time_evolution_algorithm", [0, 1])
def test_snapshot_steps_matches_full_history_final_state(snapshot_steps, time_evolution_algorithm):
    """Test solver_parameters["snapshot_steps"].

    Cases:
    - unset (default) keeps behavior unchanged: full per-step history.
    - an explicit list of step indices only keeps those snapshots, with
      output shapes reduced from (total_steps, ...) to (len(snapshot_steps), ...).
    - every recorded snapshot exactly matches the corresponding step of the
      equivalent full-history run (same seed).
    - Boris and Crank-Nicolson both handle schedules ending before the final step.
    - schedules can select only the first or last step.
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
    base_parameters["solver_parameters"]["time_evolution_algorithm"] = time_evolution_algorithm
    base_parameters["solver_parameters"]["number_of_particle_substeps_implicit_CN"] = 1
    base_parameters["solver_parameters"]["max_number_of_Picard_iterations_implicit_CN"] = 2

    full_output = Simulation(deepcopy(base_parameters)).run()
    assert full_output["electric_field"].shape == (total_steps, number_grid_points, 3)
    assert full_output["time_array"].shape == (total_steps,)

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
        (np.asarray(snapshot_steps) + 1) * dt,
        rtol=1e-12,
    )

    for snap_index, step in enumerate(snapshot_steps):
        assert jnp.allclose(snap_output["electric_field"][snap_index], full_output["electric_field"][step])
        assert jnp.allclose(snap_output["magnetic_field"][snap_index], full_output["magnetic_field"][step])
        assert jnp.allclose(snap_output["positions"][snap_index], full_output["positions"][step])
        assert jnp.allclose(snap_output["velocities"][snap_index], full_output["velocities"][step])
        assert jnp.allclose(snap_output["current_density"][snap_index], full_output["current_density"][step])
        assert jnp.allclose(snap_output["charge_density"][snap_index], full_output["charge_density"][step])
    for name, value in snap_output["final_state"].items():
        np.testing.assert_allclose(value, full_output["final_state"][name], rtol=1e-12, atol=1e-12)
    assert scalar(snap_output["final_state"]["time"]) == pytest.approx(total_steps * dt)


@pytest.mark.parametrize("schedule", [[12], [0, 12], [-1]])
def test_snapshot_steps_reject_out_of_bounds_indices(schedule):
    parameters = small_simulation_parameters(total_steps=12, number_grid_points=4, number_pseudoparticles=2)
    parameters["solver_parameters"]["snapshot_steps"] = schedule
    with pytest.raises(AssertionError, match="Snapshot steps"):
        Simulation(parameters)


def test_snapshot_steps_are_reachable_from_toml(tmp_path):
    path = tmp_path / "snapshots.toml"
    path.write_text("""[domain_parameters]
total_steps = 3
number_grid_points = 4
[solver_parameters]
print_info = false
snapshot_steps = [2, 0, 2]
[species_parameters.electrons.electrons0]
number_pseudoparticles = 2
[species_parameters.ions.ions0]
number_pseudoparticles = 2
""")
    out = Simulation(path).run()
    np.testing.assert_allclose(out["time_array"], np.array([1, 3]) * scalar(out["dt"]), rtol=1e-12, atol=0)
    assert out["positions"].shape == (2, 4, 3)


@pytest.mark.parametrize("schedule", [[], [0]])
def test_snapshot_cli_reports_completed_time_and_skips_empty_histories(tmp_path, monkeypatch, capsys, schedule):
    from jaxincell.__main__ import main
    parameters = small_simulation_parameters(total_steps=3, number_grid_points=4, number_pseudoparticles=2)
    parameters["solver_parameters"]["snapshot_steps"] = schedule
    simulation = Simulation(parameters)
    monkeypatch.setattr("jaxincell.__main__.load_parameters", lambda path: parameters)
    monkeypatch.setattr("jaxincell.__main__.Simulation", lambda given: simulation)
    seen = []
    monkeypatch.setattr("jaxincell.__main__.diagnostics", lambda out: seen.append("diagnostics"))
    monkeypatch.setattr("jaxincell.__main__.plot", lambda out: seen.append("plot"))
    main(["input.toml"])
    assert seen == (["diagnostics", "plot"] if schedule else [])
    text = capsys.readouterr().out
    assert "steps 3" in text and f"final time {3 * scalar(simulation.dt):.3e} s" in text


def test_snapshot_steps_preserves_gradients():
    """Test differentiation through snapshot recording.

    Cases:
    - an early-ending sparse history has the same objective and gradient as selected full-history rows.
    - runtime differentiable inputs remain supported by the snapshot cursor.
    """
    parameters = small_simulation_parameters(total_steps=4, number_grid_points=4, number_pseudoparticles=2)
    full_simulation = Simulation(deepcopy(parameters))
    snapshot_steps = jnp.array([0, 2])
    parameters["solver_parameters"]["snapshot_steps"] = [0, 2]
    snapshot_simulation = Simulation(parameters)

    def full_objective(drift_speed):
        output = full_simulation.run({"electrons": {"electrons0": {"drift_speed_x": drift_speed}}})
        return jnp.mean(output["velocities"][snapshot_steps, :, 0])

    def snapshot_objective(drift_speed):
        output = snapshot_simulation.run({"electrons": {"electrons0": {"drift_speed_x": drift_speed}}})
        return jnp.mean(output["velocities"][:, :, 0])

    full_value, full_gradient = jax.value_and_grad(full_objective)(1e5)
    snapshot_value, snapshot_gradient = jax.value_and_grad(snapshot_objective)(1e5)
    np.testing.assert_allclose(snapshot_value, full_value, rtol=1e-10, atol=0)
    np.testing.assert_allclose(snapshot_gradient, full_gradient, rtol=1e-10, atol=0)


@pytest.mark.parametrize("feature", ["collisions", "tensor", "mixed"])
@pytest.mark.parametrize("schedule", [[0], []])
def test_sparse_histories_preserve_new_features_and_final_state(feature, schedule):
    parameters = small_simulation_parameters(total_steps=3, number_grid_points=6, number_pseudoparticles=4)
    if feature == "collisions":
        parameters["solver_parameters"].update(collisions=True, coulomb_logarithm=10.)
    elif feature == "tensor":
        parameters["domain_parameters"].update(number_grid_points_y=2, number_grid_points_z=3)
        parameters["external_field_parameters"] = {
            "external_magnetic_field": {"B": jnp.zeros((6, 2, 3, 3)).at[..., 2].set(.001)}}
    else:
        initial = Simulation(parameters)
        length, dt = float(initial.box_size[0]), float(initial.dt)
        speed = .02*299792458.
        for population in parameters["species_parameters"].values():
            for species in population.values():
                species.update(initial_positions=jnp.zeros((4, 3)).at[:, 0].set(length/2-dt*speed/4),
                               initial_velocities=jnp.zeros((4, 3)).at[:, 0].set(speed))
        parameters["domain_parameters"].update(particle_BC_left=3, particle_BC_right=3,
                                                 field_BC_left=1, field_BC_right=1,
                                                 mixed_BC_weight=.3, COR_left=.5, COR_right=.5)
        parameters["solver_parameters"]["field_solver"] = 2
    full = Simulation(deepcopy(parameters)).run()
    parameters["solver_parameters"]["snapshot_steps"] = schedule
    sparse = Simulation(parameters).run()
    for key in ("positions", "velocities", "electric_field", "magnetic_field", "mus",
                *(("masses_over_time", "charges_over_time") if feature == "mixed" else ())):
        np.testing.assert_array_equal(sparse[key], np.asarray(full[key])[schedule])
    for key in full["final_state"]:
        np.testing.assert_array_equal(sparse["final_state"][key], full["final_state"][key])


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
