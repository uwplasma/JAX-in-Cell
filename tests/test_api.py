"""Behaviour of the public interface: reproducibility, differentiation,
storage options, restarts, input files and the command line."""
import ast
import builtins
import gc
import os
import pathlib
import re
import shutil
import sys
import tempfile
import warnings
from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxincell import (Collisions, Domain, Simulation, Solver, Species, diagnostics, elementary_charge, epsilon_0,
                       load_toml, main, mass_electron, quiet_start, speed_of_light as c)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def small_simulation(n=400, **solver):
    e = Species.electrons(n=n, density=4e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0), plus_minus=True,
                          perturbation_amplitude=5e-7, perturbation_mode=1)
    i = Species.ions(n=n, density=4e17, electrons=e)
    return Simulation(Domain(length=0.01, cells=16, dt_over_dx_c=2.0), [e, i], Solver(**solver))


def input_file(path, n=200, steps=4, plot=False):
    """Write the small two-stream input file the command-line tests read, and return
    its path as a string. The tests differ only in the particle count, the number of
    steps and whether the run plots."""
    path.write_text(f"""
[domain]
length = 0.01
cells = 16
dt_over_dx_c = 2.0
[solver]
filter_passes = 1
[[species]]
name = "electrons"
n = {n}
charge = -1
mass = "electron"
density = 4e17
vth = [1.5e7, 0.0, 0.0]
drift = [6e7, 0.0, 0.0]
plus_minus = true
[[species]]
name = "ions"
n = {n}
charge = 1
mass = "proton"
density = 4e17
vth = [3.5e5, 0.0, 0.0]
[run]
steps = {steps}
seed = 4
plot = {str(plot).lower()}
""")
    return str(path)


def test_runs_are_reproducible_and_seeds_matter():
    """One seed gives one run: bit for bit on the CPU, and to round-off on a GPU, whose
    scatter kernels do not promise the same order of summation twice."""
    sim = small_simulation()
    a, b, other = sim.run(20, seed=5), sim.run(20, seed=5), sim.run(20, seed=6)
    if jax.default_backend() == "cpu":
        assert np.array_equal(np.asarray(a.x), np.asarray(b.x))
    else:
        assert np.allclose(np.asarray(a.x), np.asarray(b.x), rtol=1e-12, atol=1e-12 * sim.domain.length)
    assert not np.allclose(np.asarray(a.x), np.asarray(other.x), rtol=1e-6, atol=0)
    assert a.x.shape == (20, 800, 3) and a.E.shape == (20, 16, 3)
    # the first stored time is one step. `Output.dt` is worked out on the host and the clock
    # inside the loop by XLA, from the same expression on the same numbers, so the two can differ
    # in the last bit and comparing them for equality would be comparing evaluators
    assert float(a.t[0] / a.dt) == pytest.approx(1.0, rel=1e-15, abs=0)


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
    scale = float(np.abs(np.asarray(full.E)).max())   # components that vanish are compared on this scale
    thin = sim.run(40, seed=2, store_every=10)
    assert thin.E.shape[0] == 4
    assert np.allclose(np.asarray(thin.t), np.asarray(full.t[9::10]), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(thin.E), np.asarray(full.E[9::10]), rtol=1e-12, atol=1e-12 * scale)
    first = sim.run(20, seed=2, store_every=10)
    second = sim.run(20, seed=2, store_every=10, state=first.state)
    assert np.allclose(np.asarray(second.E[-1]), np.asarray(full.E[-1]), rtol=1e-10, atol=1e-10 * scale)
    light = sim.run(10, seed=2, store_particles=False)
    assert light.x is None and light.E.shape == (10, 16, 3)


def test_a_progress_meter_reports_without_changing_the_run(capsys):
    """A long run has to say how far it has got, and saying so must not change it.

    The meter lives on the host, outside the traced region: the run is split into groups, each
    the same compiled program, and the state makes a grouped run the ungrouped run. So a verbose
    run is a silent run **bit for bit**, which is what the assertions below are. Building a bar
    inside the jitted loop instead makes it trace-time state -- a second identical call reuses
    the compiled program and the bar closed at the end of the first, and reports nothing.
    """
    import io

    domain = Domain(length=1e-2, cells=16, dt_over_dx_c=1.0)
    electrons = Species.electrons(n=2000, density=1e14, vth=(1e6, 0, 0), quiet=True)
    ions = Species.ions(n=500, density=1e14, mass_ratio=1e9, vth=0.0, quiet=True)
    sim = Simulation(domain, [electrons, ions], Solver())

    silent = sim.run(60, seed=3, store_every=5)
    seen = []
    loud = sim.run(60, seed=3, store_every=5, verbose=lambda done, total: seen.append((done, total)))
    for name in ("t", "steps", "x", "v", "E", "B", "rho", "sigma"):
        assert np.array_equal(np.asarray(getattr(silent, name)), np.asarray(getattr(loud, name))), name
    assert np.array_equal(np.asarray(silent.state.u), np.asarray(loud.state.u))
    assert seen[-1] == (60, 60) and all(total == 60 for _, total in seen)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)

    # twice in a row, because a bar built at trace time would report only the first time
    again = []
    sim.run(60, seed=3, store_every=5, verbose=lambda done, total: again.append(done))
    assert again == [done for done, _ in seen] and again

    # the cadence is its own, not the snapshot schedule's: 200 chunks still give about twenty
    ticks = []
    sim.run(200, seed=3, store_every=1, verbose=lambda done, total: ticks.append(done))
    assert 10 <= len(ticks) <= 25 and ticks[-1] == 200
    dense = []
    sim.run(200, seed=3, store_every=50, verbose=lambda done, total: dense.append(done))
    assert dense == [50, 100, 150, 200]                 # four chunks cannot be split further

    # tracing turns it off without being asked: no side effect goes in the differentiated path
    quiet = []
    watch = lambda done, total: quiet.append(done)      # noqa: E731
    jax.jit(lambda vth: sim.replace(species=(electrons.replace(vth=(vth, 0, 0)), ions)).run(
        20, seed=3, store_every=20, verbose=watch).E)(1e6).block_until_ready()
    jax.grad(lambda vth: jnp.sum(sim.replace(species=(electrons.replace(vth=(vth, 0, 0)), ions)).run(
        20, seed=3, store_every=20, store_particles=False, verbose=watch).E ** 2))(1e6)
    jax.vmap(lambda s: sim.run(20, seed=s, store_every=20, store_particles=False,
                               verbose=watch).E)(jnp.arange(3))
    assert quiet == []

    # what the built-in one writes, to a terminal and to a log
    from jaxincell._progress import Progress, reporter

    for interactive, ending in ((True, "   "), (False, "\n")):
        stream = io.StringIO()
        meter = Progress(100, stream=stream, interactive=interactive)
        meter(0)                                        # nothing done yet, so no rate and no estimate
        assert "--:--" in stream.getvalue()
        meter(50)
        assert stream.getvalue().endswith(ending) and "50/100 steps" in stream.getvalue()
        assert ("\r" in stream.getvalue()) is interactive
        meter(100)
        assert stream.getvalue().count("\n") == (1 if interactive else 3)
        meter.close()                                   # closing twice adds nothing
        assert stream.getvalue().count("\n") == (1 if interactive else 3)
    # an interrupted run ends its line
    stream = io.StringIO()
    meter = Progress(100, stream=stream, interactive=True)
    meter(30)
    meter.close()
    assert stream.getvalue().endswith("\n")
    assert reporter(False, 10) is None and reporter(None, 10) is None
    assert isinstance(reporter(True, 10), Progress)
    with pytest.raises(ValueError, match="something to call with"):
        reporter("loudly", 10)
    capsys.readouterr()


def test_changing_a_physical_parameter_does_not_recompile():
    sim = small_simulation()
    e, i = sim.species
    base = jax.jit(lambda s: s.run(5, seed=0).E)
    first = base.lower(sim).compile()
    varied = sim.replace(domain=sim.domain.replace(length=0.011), species=(e.replace(density=5e17), i))
    assert jax.tree_util.tree_structure(varied) == jax.tree_util.tree_structure(sim)
    assert first(varied).shape == (5, 16, 3)


def test_toml_input_and_command_line(tmp_path):
    path = input_file(tmp_path / "input.toml", n=300, steps=10)
    sim, run = load_toml(path)
    assert sim.species[1].mass > 1e-27 and run["steps"] == 10 and sim.solver.filter_passes == 1
    out = sim.run(run["steps"], seed=run["seed"])
    assert out.E.shape == (10, 16, 3)
    assert main([path]) == 0


def test_diagnostics_keys_and_species_views():
    """The per-species energies and the species views select the same particles: the
    electron energy recomputed from the view is the one the diagnostics report, and
    the species energies add up to the total kinetic energy."""
    sim = small_simulation()
    out = sim.run(10, seed=0)
    d = diagnostics(out)
    for key in ("electric", "magnetic", "kinetic", "kinetic_electrons", "kinetic_ions", "total", "momentum",
                "gauss_residual", "dominant_frequency", "temperatures"):
        assert key in d
    x_e, v_e = out.particles("electrons")
    assert x_e.shape == (10, 400, 3) and v_e.shape == (10, 400, 3)
    from_view = 0.5 * mass_electron * np.sum(np.asarray(out.weight[:, :400]) * np.sum(np.asarray(v_e) ** 2, axis=-1),
                                             axis=1)
    assert np.allclose(np.asarray(d["kinetic_electrons"]), from_view, rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(d["kinetic_electrons"] + d["kinetic_ions"]), np.asarray(d["kinetic"]),
                       rtol=1e-12, atol=0)


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
        # where the standard says each component sits, worked out from the attributes the file
        # carries and nothing else: x_i = (gridGlobalOffset + (i + position) * gridSpacing)
        # * gridUnitSI, with position in [0, 1) from the lower corner of the cell. A centre is
        # therefore 0.5 and not 0.0; this was written the other way round, which put the centres
        # on the left faces and the faces on the centres, a whole cell apart.
        cells = out.E.shape[1]
        for name, want in (("B", np.asarray(out.grid)), ("rho", np.asarray(out.grid)),
                           ("E", np.asarray(out.faces)), ("J", np.asarray(out.faces))):
            mesh = iteration.meshes[name]
            component = mesh["x"] if name in ("E", "B", "J") else mesh[io.Record_Component.SCALAR]
            offset, spacing = mesh.grid_global_offset[0], mesh.grid_spacing[0]
            where = (offset + (np.arange(cells) + component.position[0]) * spacing) * mesh.grid_unit_SI
            # a face sits at x = 0 in this box, so the tolerance needs a length to be relative to
            assert np.allclose(where, want, rtol=1e-12, atol=1e-12 * float(out.length)), name
            assert 0.0 <= component.position[0] < 1.0, name
        assert set(iteration.particles) == set(out.names)
        electrons = iteration.particles["electrons"]
        position = electrons["position"]["x"].load_chunk()
        series.flush()
        assert position.shape == (out.counts[0],)
        assert np.allclose(position, np.asarray(out.x[4, : out.counts[0], 0]))
        series.close()


def test_courant_warning_fires_only_when_a_light_wave_can_be_seeded():
    """Stepping above c dt = dx is safe for an electrostatic run and diverges as
    soon as the particles carry transverse velocity, so the warning has to
    distinguish the two rather than firing on the time step alone."""
    longitudinal = Species.electrons(n=100, density=1e17, vth=(1e6, 0, 0))
    isotropic = Species.electrons(n=100, density=1e17, vth=(1e6, 1e6, 1e6))
    above = Domain(length=0.01, cells=16, dt_over_dx_c=4.5)
    below = Domain(length=0.01, cells=16, dt_over_dx_c=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")               # no warning in these three
        Simulation(above, [longitudinal], Solver())
        Simulation(below, [isotropic], Solver())
        Simulation(above, [isotropic], Solver(algorithm="implicit"))
    with pytest.warns(UserWarning, match="exceeds one"):
        Simulation(above, [isotropic], Solver())


def test_command_line_reports_usage_without_a_file(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_quiet_start_samples_the_maxwellian_and_fills_the_box():
    """The public helper that custom initial conditions build on: equally spaced
    positions inside the box and velocities at the quantiles of the Maxwellian,
    with vth the sqrt(2) kT/m convention, so the standard deviation is vth/sqrt2."""
    n, length, vth, drift = 20000, 0.5, (1e6, 0.0, 5e6), (2e6, 0.0, 0.0)
    x, v = quiet_start(n, length, vth=vth, drift=drift)
    assert x.shape == v.shape == (n, 3)
    assert np.abs(x[:, 0]).max() < length / 2
    spacing = np.diff(x[:, 0])
    assert np.allclose(spacing, length / n)                  # equally spaced
    assert np.allclose(v.mean(axis=0), drift, atol=1e-3 * max(vth))
    assert np.allclose(v.std(axis=0), np.asarray(vth) / np.sqrt(2), rtol=2e-3)


def test_plasma_frequency_and_debye_length_by_species_name():
    """Both accept a species name and default to the first species."""
    density = 4e17
    electrons = Species.electrons(n=100, density=density, vth=(1e6, 0, 0), name="electrons")
    ions = Species.ions(n=100, density=density, electrons=electrons, name="ions")
    sim = Simulation(Domain(length=0.01, cells=16), [electrons, ions], Solver())
    expected = np.sqrt(density * elementary_charge ** 2 / (epsilon_0 * mass_electron))
    assert abs(float(sim.plasma_frequency()) / expected - 1) < 1e-12
    assert abs(float(sim.plasma_frequency("electrons")) / expected - 1) < 1e-12
    assert float(sim.plasma_frequency("ions")) < 0.05 * expected          # heavier, slower
    assert abs(float(sim.debye_length()) - 1e6 / (np.sqrt(2) * expected)) < 1e-12 * float(sim.debye_length())
    # ions derived from these electrons are at the same temperature, and the Debye
    # length depends on the temperature and the density, not on the mass
    assert abs(float(sim.debye_length("ions")) / float(sim.debye_length("electrons")) - 1) < 1e-12
    colder = Species.ions(n=100, density=density, electrons=electrons, temperature_ratio=0.25, name="cold")
    sim = Simulation(Domain(length=0.01, cells=16), [electrons, colder], Solver())
    assert abs(float(sim.debye_length("cold")) / float(sim.debye_length("electrons")) - 0.5) < 1e-9


def test_random_positions_and_a_scalar_thermal_speed():
    """`random_positions` places particles uniformly instead of on a lattice, and
    a bare number for vth or drift is taken as the x component."""
    assert Species.electrons(n=10, density=1e17, vth=2e6).vth == (2e6, 0.0, 0.0)
    assert Species.electrons(n=10, density=1e17, drift=3e6).drift == (3e6, 0.0, 0.0)

    # cold, so that the half-step displacement does not disturb the spacing
    domain = Domain(length=0.01, cells=16)
    spread = {}
    for name, random_positions in (("random", True), ("lattice", False)):
        species = Species.electrons(n=1000, density=1e17, random_positions=random_positions)
        sim = Simulation(domain, [species], Solver())
        x = sim.initial_state(jax.random.PRNGKey(0))[0].x
        spacing = np.diff(np.sort(np.asarray(x[:, 0])))
        spread[name] = spacing.std() / spacing.mean()
    assert spread["lattice"] < 1e-9 < 0.1 < spread["random"]


def test_coulomb_logarithm_follows_the_formulary_in_both_regimes():
    """The NRL electron-ion Coulomb logarithm switches formula at 10 Z^2 eV; both
    branches are used, and it is what `Collisions()` falls back on."""
    from jaxincell._collisions import coulomb_logarithm

    density = 1e19                                            # 1e13 cm^-3
    cold, hot = float(coulomb_logarithm(density, 1.0)), float(coulomb_logarithm(density, 1000.0))
    assert abs(cold - (23.0 - np.log(np.sqrt(1e13) * 1.0 ** -1.5))) < 1e-9
    assert abs(hot - (24.0 - np.log(np.sqrt(1e13) / 1000.0))) < 1e-9
    assert 5 < cold < 25 and 5 < hot < 25 and hot > cold       # both physically sensible


def test_collisions_default_to_every_pair_and_the_formulary_logarithm():
    """`Collisions()` with no arguments collides every combination and takes the
    Coulomb logarithm from the lightest negatively charged species, rather than needing
    either spelled out."""
    electrons = Species.electrons(n=800, density=1e20, vth=(2e6, 2e6, 2e6), quiet=True)
    ions = Species.ions(n=800, density=1e20, electrons=electrons, quiet=True)
    domain = Domain(length=1e-4, cells=8, dt_over_dx_c=1.0)
    quiet = Simulation(domain, [electrons, ions], Solver()).run(20, seed=0)
    collided = Simulation(domain, [electrons, ions], Solver(),
                          Collisions()).run(20, seed=0)
    assert np.isfinite(np.asarray(collided.v)).all()
    assert not np.allclose(np.asarray(collided.v), np.asarray(quiet.v))
    momentum = np.asarray(diagnostics(collided)["momentum"])[:, 0]
    content = float(np.sum(np.asarray(collided.mass * collided.weight[0]) * np.abs(np.asarray(collided.v[0, :, 0]))))
    assert float(np.abs(momentum - momentum[0]).max()) < 1e-4 * content


def test_plot_animates_on_screen_and_warns_instead_of_failing_without_ffmpeg(monkeypatch):
    """Without `save` the figure carries a live animation; with `save` but no
    ffmpeg on the path the call warns and returns rather than raising, so a long
    run is not lost to a missing tool."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from jaxincell import plot

    out = small_simulation(n=200).run(8, seed=0)
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    figure = plot(out, direction="x", show=True)
    assert shown == [True]
    assert isinstance(figure.animation, FuncAnimation)
    with warnings.catch_warnings():                  # never rendered, which is the point
        warnings.simplefilter("ignore", UserWarning)
        figure.animation = None
        plt.close(figure)
        gc.collect()

    def no_ffmpeg(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("jaxincell._plot.subprocess.Popen", no_ffmpeg)
    with pytest.warns(RuntimeWarning, match="ffmpeg was not found"):
        plot(out, direction="x", save="never-written.mp4", show=False)
    assert not os.path.exists("never-written.mp4")


def test_command_line_plots_when_the_input_asks_for_it(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    assert main([input_file(tmp_path / "run.toml", steps=6, plot=True)]) == 0
    with warnings.catch_warnings():                  # the animation is never rendered here
        warnings.simplefilter("ignore", UserWarning)
        plt.close("all")
        gc.collect()


def test_openpmd_says_what_to_install_when_the_dependency_is_missing():
    from jaxincell.openpmd import write_openpmd

    with mock.patch.dict(sys.modules, {"openpmd_api": None}):
        with pytest.raises(ImportError, match="pip install openpmd-api"):
            write_openpmd(small_simulation(n=100).run(2, seed=0), "unused.json")


def test_configuration_objects_normalise_what_they_are_given():
    """The constructors accept the shapes a user naturally writes and store one
    canonical form, so that the pytree structure does not depend on how a value
    was spelled."""
    from jaxincell._config import BOUNDARIES, _float

    # bool is a subclass of int in Python; without a guard, a flag handed to a
    # float field would silently become 1.0 and change the leaf's dtype
    assert _float(True) is True and _float(1) == 1.0 and isinstance(_float(1), float)

    # species pairs may be any sequence and are stored as hashable tuples, since
    # they are static and become part of the compiled program's cache key
    collisions = Collisions(pairs=[["electrons", "ions"], ("electrons", "electrons")])
    assert collisions.pairs == (("electrons", "ions"), ("electrons", "electrons"))
    assert hash(collisions.pairs)

    # walls may be one name for both ends or a pair, and are stored as codes
    assert Domain(particle_bc="reflective").particle_bc == (BOUNDARIES["reflective"],) * 2
    assert Domain(particle_bc=("reflective", "absorbing")).particle_bc == (1, 2)
    with pytest.raises(ValueError, match="must have shape"):
        Species.electrons(n=10, density=1e17, vth=(1e6, 0, 0)).replace(x=np.zeros((9, 3)))

    # a wall coefficient is one number or a (left, right) pair; a reflection law is a
    # function, which cannot be an array and so is kept out of the leaves
    assert Domain(restitution=0.5).restitution == (0.5, 0.5)
    assert Domain(restitution=[1, 0.5]).restitution == (1.0, 0.5)

    def law(speed):
        return np.exp(-speed)
    species = Species.electrons(n=10, density=1e17, reflection=(law, 0.2))
    assert species.reflection == (law, 0.2)
    assert not any(callable(leaf) for leaf in jax.tree_util.tree_leaves(species))
    assert jax.tree_util.tree_map(lambda leaf: leaf, species).reflection == (law, 0.2)
    with pytest.raises(ValueError, match="reflection takes a number in"):
        Species.electrons(n=10, density=1e17, reflection=1.5)
    assert Domain(particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing")).particle_bc == (3, 2)


def test_single_precision_is_left_to_jax_s_own_switch(monkeypatch):
    """Double precision is switched on at import unless JAX_ENABLE_X64 is set, so a
    user who wants single precision chooses it with JAX's own switch rather than
    with one the package would have to invent."""
    from jaxincell._simulation import _enable_double_precision

    calls = []
    monkeypatch.setattr(jax.config, "update", lambda name, value: calls.append((name, value)))
    _enable_double_precision({"JAX_ENABLE_X64": "0"})
    assert calls == []
    _enable_double_precision({})
    assert calls == [("jax_enable_x64", True)]


def test_every_example_states_its_precision():
    """Each example sets JAX_ENABLE_X64 before anything imports JAX, so the precision of
    a run is written in the script and can be changed from the shell."""
    examples = sorted(ROOT.glob("examples/*/*.py"))
    assert len(examples) >= 13
    for path in examples:
        body = ast.parse(path.read_text()).body
        sets = [i for i, node in enumerate(body)
                if isinstance(node, ast.Expr) and "JAX_ENABLE_X64" in ast.unparse(node)]
        imports = [i for i, node in enumerate(body) if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (getattr(node, "module", None) or alias.name).startswith(("jax", "jaxincell")) for alias in node.names)]
        assert sets and imports and sets[0] < imports[0], path.name


def test_every_example_the_documentation_names_exists():
    """The README and the docs name examples by path, and on a case-sensitive file system a
    name that differs only in case does not run. The names are compared with the directory
    listing, because Path.exists ignores case on macOS and Windows. The scripts sit in
    `1_basic`, `2_intermediate` and `3_advanced`, so the listing is recursive."""
    scripts = {path.name for path in (ROOT / "examples").rglob("*") if path.is_file()}
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    named = {name.rsplit("/", 1)[-1] for page in pages
             for name in re.findall(r"examples/([\w/]+\.(?:py|toml))", page.read_text())}
    for page in ("docs/examples/index.md", "docs/getting_started/quickstart.md"):
        named |= set(re.findall(r"`(\w+\.py)`", (ROOT / page).read_text()))
    assert len(named) >= 13 and named <= scripts, sorted(named - scripts)


def test_the_citation_file_and_the_readme_name_the_same_authors():
    """CITATION.cff is what GitHub's "Cite this repository" reads, and the README has the
    same names in a BibTeX entry. Two lists of authors drift apart unless something holds
    them together."""
    citation = (ROOT / "CITATION.cff").read_text()
    surnames = re.findall(r"family-names: (\w+)", citation)
    assert surnames and surnames[:len(surnames) // 2] == surnames[len(surnames) // 2:]   # both blocks agree
    readme = re.search(r"author = \{([^}]*)\}", (ROOT / "README.md").read_text()).group(1)
    assert {name.split(",")[0].strip() for name in readme.split(" and ")} == set(surnames)
    assert "MIT" in citation and (ROOT / "LICENSE").read_text().startswith("MIT")


def test_diagnostics_without_particles_gives_the_field_quantities_only():
    """`store_particles=False` keeps the fields and drops the particle history, so
    the diagnostics that need velocities are absent rather than wrong. That is the
    trade a long run makes, and the reason the docs list which keys survive."""
    out = small_simulation(n=200).run(10, seed=0, store_particles=False)
    assert out.x is None and out.v is None
    d = diagnostics(out)
    for key in ("electric", "magnetic", "gauss_residual", "dominant_frequency"):
        assert key in d
    for key in ("kinetic", "total", "momentum", "temperatures"):
        assert key not in d


def test_openpmd_export_can_leave_out_the_meshes_or_the_particles():
    """The two switches, and a run stored without particles, which has none to write."""
    io = pytest.importorskip("openpmd_api")
    from jaxincell.openpmd import write_openpmd

    sim = small_simulation(n=200)
    out = sim.run(4, seed=0)
    with tempfile.TemporaryDirectory() as folder:
        fields_only = write_openpmd(out, os.path.join(folder, "fields"), particles=False)
        series = io.Series(fields_only, io.Access.read_only)
        assert set(series.iterations[0].meshes) and not len(series.iterations[0].particles)
        series.close()

        particles_only = write_openpmd(out, os.path.join(folder, "particles.json"), meshes=False)
        series = io.Series(particles_only, io.Access.read_only)
        assert not len(series.iterations[0].meshes) and set(series.iterations[0].particles)
        series.close()

        light = sim.run(4, seed=0, store_particles=False)
        path = write_openpmd(light, os.path.join(folder, "light.json"))
        series = io.Series(path, io.Access.read_only)
        assert set(series.iterations[0].meshes) and not len(series.iterations[0].particles)
        series.close()

    assert fields_only.endswith(".json")          # the extension is supplied when omitted


def test_toml_loading_falls_back_to_tomli_before_python_311(tmp_path):
    """`tomllib` arrived in 3.11; below that the loader uses the tomli backport, which
    is declared as a conditional dependency. Hiding tomllib takes that path on a newer
    interpreter, and on 3.10 it is the only path there is."""
    try:
        import tomllib as parser
    except ModuleNotFoundError:                  # Python 3.10, where tomli is the real one
        import tomli as parser

    path = input_file(tmp_path / "in.toml", n=100, steps=3)
    with mock.patch.dict(sys.modules, {"tomllib": None, "tomli": parser}):
        sim, run = load_toml(path)
    assert run["steps"] == 3 and sim.species[0].n == 100


def test_the_package_reports_an_unknown_version_from_a_bare_source_tree():
    """`jaxincell/version.py` is generated at build time and is not in the
    repository, so a fresh clone that has not been installed does not have one.
    The package, and the openPMD exporter that records it, still import, with the
    version left unknown."""
    import jaxincell

    source = pathlib.Path(jaxincell.__file__).read_text()
    namespace = {"__name__": "jaxincell_bare", "__package__": "jaxincell"}
    real_import = builtins.__import__

    def refuse_version(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "version" or name.endswith(".version"):
            raise ImportError("no generated version file")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch.object(builtins, "__import__", refuse_version):
        exec(compile(source, jaxincell.__file__, "exec"), namespace)
    assert namespace["__version__"] == "unknown"
