"""Behaviour of the public interface: reproducibility, differentiation,
storage options, restarts, input files and the command line."""
import os
import shutil
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, load_toml
from jaxincell import speed_of_light as c
from jaxincell.__main__ import main


def small_simulation(n=400, **solver):
    e = Species.electrons(n=n, density=4e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0), plus_minus=True,
                          perturbation_amplitude=5e-7, perturbation_mode=1)
    i = Species.ions(n=n, density=4e17, electrons=e)
    return Simulation(Domain(length=0.01, cells=16, dt_over_dx_c=2.0), [e, i], Solver(**solver))


def test_runs_are_reproducible_and_seeds_matter():
    sim = small_simulation()
    a, b, other = sim.run(20, seed=5), sim.run(20, seed=5), sim.run(20, seed=6)
    assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
    assert not np.array_equal(np.asarray(a.x), np.asarray(other.x))
    assert a.x.shape == (20, 800, 3) and a.E.shape == (20, 16, 3) and float(a.t[0] / a.dt) == 1.0


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_gradient_matches_central_finite_difference(algorithm):
    """Reverse-mode gradients of a time-averaged field energy with respect to
    the electron drift agree with a central finite difference; the implicit
    scheme is differentiable because its Picard loop has a fixed length."""
    sim = small_simulation(algorithm=algorithm, picard_iterations=4)
    e, i = sim.species

    def objective(drift):
        return jnp.mean(sim.replace(species=(e.replace(drift=(drift, 0.0, 0.0)), i)).run(20, seed=1).E[:, :, 0] ** 2)

    gradient = float(jax.grad(objective)(6e7))
    h = 2e3
    fd = float((objective(6e7 + h) - objective(6e7 - h)) / (2 * h))
    assert abs(gradient - fd) < 1e-4 * abs(fd)


def test_gradients_with_respect_to_domain_and_species_parameters_are_finite():
    sim = small_simulation()
    grads = jax.grad(lambda s: jnp.sum(s.run(10, seed=1).E ** 2))(sim)
    assert np.isfinite(float(grads.domain.length)) and np.isfinite(float(grads.species[0].density))
    assert np.isfinite(float(grads.species[1].vth[0])) and np.isfinite(float(grads.species[0].perturbation_amplitude))


def test_vmap_over_seeds_gives_an_ensemble_from_one_program():
    sim = small_simulation()
    fields = jax.vmap(lambda seed: sim.run(10, seed=seed).E[-1, :, 0])(jnp.arange(4))
    assert fields.shape == (4, 16) and float(jnp.std(fields, axis=0).max()) > 0


def test_store_every_store_particles_and_restart():
    sim = small_simulation()
    full = sim.run(40, seed=2)
    thin = sim.run(40, seed=2, store_every=10)
    assert thin.E.shape[0] == 4 and np.allclose(np.asarray(thin.t), np.asarray(full.t[9::10]))
    assert np.allclose(np.asarray(thin.E), np.asarray(full.E[9::10]), rtol=1e-12, atol=0)
    first = sim.run(20, seed=2, store_every=10)
    second = sim.run(20, seed=2, store_every=10, state=first.state)
    assert np.allclose(np.asarray(second.E[-1]), np.asarray(full.E[-1]), rtol=1e-10, atol=0)
    light = sim.run(10, seed=2, store_particles=False)
    assert light.x is None and light.E.shape == (10, 16, 3)


def test_changing_a_physical_parameter_does_not_recompile():
    sim = small_simulation()
    e, i = sim.species
    base = jax.jit(lambda s: s.run(5, seed=0).E)
    first = base.lower(sim).compile()
    varied = sim.replace(domain=sim.domain.replace(length=0.011), species=(e.replace(density=5e17), i))
    assert jax.tree_util.tree_structure(varied) == jax.tree_util.tree_structure(sim)
    assert first(varied).shape == (5, 16, 3)


def test_toml_input_and_command_line():
    text = """
[domain]
length = 0.01
cells = 16
dt_over_dx_c = 2.0
[solver]
filter_passes = 1
[[species]]
name = "electrons"
n = 300
charge = -1
mass = "electron"
density = 4e17
vth = [1.5e7, 0.0, 0.0]
drift = [6e7, 0.0, 0.0]
plus_minus = true
[[species]]
name = "ions"
n = 300
charge = 1
mass = "proton"
density = 4e17
vth = [3.5e5, 0.0, 0.0]
[run]
steps = 10
seed = 4
plot = false
"""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "input.toml")
        with open(path, "w") as f:
            f.write(text)
        sim, run = load_toml(path)
        assert sim.species[1].mass > 1e-27 and run["steps"] == 10
        out = sim.run(run["steps"], seed=run["seed"])
        assert out.E.shape == (10, 16, 3)
        assert main([path]) == 0


def test_diagnostics_keys_and_species_views():
    sim = small_simulation()
    out = sim.run(10, seed=0)
    d = diagnostics(out)
    for key in ("electric", "magnetic", "kinetic", "kinetic_electrons", "kinetic_ions", "total", "momentum",
                "gauss_residual", "dominant_frequency", "temperatures"):
        assert key in d
    x_e, v_e = out.particles("electrons")
    assert x_e.shape == (10, 400, 3) and v_e.shape == (10, 400, 3)
    assert np.allclose(np.asarray(d["total"]), np.asarray(d["electric"] + d["magnetic"] + d["kinetic"]))


def test_plot_builds_every_panel_and_writes_a_movie(tmp_path):
    """The overview figure has one panel per non-zero field component, one per
    velocity direction and one phase space per species, and the movie writer
    produces a playable file when ffmpeg is available."""
    import matplotlib
    matplotlib.use("Agg")
    from jaxincell import plot

    out = small_simulation(n=200).run(12, seed=0)
    figure = plot(out, direction="x", show=False)
    drawn = [ax.get_title() for ax in figure.axes if ax.get_title()]
    # E_x and rho are non-zero, f(v_x), and one phase space per species
    assert any("$E_x$" == title for title in drawn) and any(r"$\rho$" == title for title in drawn)
    assert any("f(v_x)" in title.replace("$", "").replace("\\", "") for title in drawn)
    assert sum("(x, v_x)" in title.replace("$", "").replace("\\", "") for title in drawn) == 2
    assert len(plot(out, direction="xz", show=False).axes) > len(figure.axes)
    with pytest.raises(ValueError):
        plot(out, direction="q", show=False)

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    movie = tmp_path / "run.mp4"
    plot(out, direction="x", save=str(movie), show=False, fps=5)
    assert movie.stat().st_size > 1000
    assert movie.read_bytes()[4:8] == b"ftyp"        # an ISO base media file


def test_openpmd_export_round_trips():
    """The exported series carries one iteration per stored step, the meshes on
    the grids they live on, and one particle species per name."""
    io = pytest.importorskip("openpmd_api")
    from jaxincell.openpmd import write_openpmd

    sim = small_simulation(n=200)
    out = sim.run(6, seed=0)
    with tempfile.TemporaryDirectory() as folder:
        path = write_openpmd(out, os.path.join(folder, "run.json"), every=2)
        series = io.Series(path, io.Access.read_only)
        assert list(series.iterations) == [0, 2, 4]
        iteration = series.iterations[4]
        assert float(iteration.time) == pytest.approx(float(out.t[4]))
        assert set(iteration.meshes) == {"E", "B", "J", "rho"}
        assert iteration.meshes["E"]["x"].position == [0.5]     # faces
        assert iteration.meshes["B"]["x"].position == [0.0]     # centres
        assert set(iteration.particles) == set(out.names)
        electrons = iteration.particles["electrons"]
        position = electrons["position"]["x"].load_chunk()
        series.flush()
        assert position.shape == (out.counts[0],)
        assert np.allclose(position, np.asarray(out.x[4, : out.counts[0], 0]))
        series.close()
