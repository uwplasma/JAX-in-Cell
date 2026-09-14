"""The field walls, the gather, and the time loop: mirror symmetry of the wall closures,
the force on a charge sheet next to each wall, the random keys each step hands out, the
switches the implicit scheme refuses, and the bookkeeping that keeps the loop cheap."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

import jaxincell._simulation as simulation_module
from jaxincell import (Collisions, Domain, Simulation, Solver, Species, elementary_charge, epsilon_0,
                       speed_of_light as c)
from jaxincell._core import E_x_from_rho, apply_particle_bc, current_from_continuity, deposit, wrap_positions

L, CELLS = 1.0, 32
DX = L / CELLS
CENTRE0 = -L / 2 + DX / 2
WALLS = [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2)]           # every pair Domain accepts


def left_wall_value(F, s, bc):
    """F_{-1/2}, which the grid does not store, from the first cell of the difference equation."""
    return F[-1] if bc == (0, 0) else F[0] - DX * s[0]


def reflect_faces(F, F_left):
    """Faces mirrored about the box centre: face i+1/2 goes to face (N-2-i)+1/2, and the
    left wall face, which is not stored, becomes the stored right wall face."""
    return jnp.concatenate([F[:-1][::-1], F_left[None]])


@pytest.mark.parametrize("bc", WALLS)
def test_wall_closures_are_mirror_images_and_satisfy_gauss(bc):
    """For every pair of field walls, the Gauss solve and the continuity current of a
    charge distribution and of its mirror image, with the walls swapped, are mirror
    images: E_x and J_x change sign under the reflection. Both satisfy their difference
    equation in every cell, E_x vanishes at a reflective wall, two absorbing walls hold
    no potential difference across the box, and the periodic current carries the mean
    current it is given."""
    rng = np.random.default_rng(7)
    rho = jnp.asarray(rng.normal(size=CELLS) + 0.3)        # not neutral, so the closures matter
    drho = jnp.asarray(rng.normal(size=CELLS))
    solves = ((lambda r, b, sign: E_x_from_rho(r, DX, b), rho, rho / epsilon_0, 0.0),
              (lambda r, b, sign: current_from_continuity(0 * r, r, 1.0, DX, sign * 0.25, b), drho, -drho, 0.25))
    for field, data, source, mean in solves:
        F = field(data, bc, 1.0)
        s = source - jnp.mean(source) if bc in ((0, 0), (1, 1)) else source
        F_left = left_wall_value(F, s, bc)
        F_before = jnp.concatenate([F_left[None], F[:-1]])
        assert np.allclose(np.asarray((F - F_before) / DX), np.asarray(s), rtol=1e-12,
                           atol=1e-12 * float(jnp.abs(s).max()))
        G = field(data[::-1], bc[::-1], -1.0)             # the mean current reverses with the charges
        assert np.allclose(np.asarray(G), -np.asarray(reflect_faces(F, F_left)), rtol=1e-12,
                           atol=1e-12 * float(jnp.abs(F).max()))
        scale = float(jnp.abs(F).max())
        if bc[0] == 1:
            assert abs(float(F_left)) < 1e-12 * scale
        if bc[1] == 1:
            assert abs(float(F[-1])) < 1e-12 * scale
        if bc == (2, 2):
            assert abs(float(0.5 * F_left + jnp.sum(F[:-1]) + 0.5 * F[-1])) < 1e-12 * CELLS * scale
        if bc == (0, 0):
            assert abs(float(jnp.mean(F)) - mean) < 1e-12 * scale


class KeyLedger:
    """Stands in for ``jax.random`` in the simulation module and records every concrete
    key that is split, folded or drawn from. Inside a scan the keys are tracers and are
    not recorded; the keys handed into a scan are."""

    def __init__(self):
        self.consumed = []

    def note(self, key):
        if not isinstance(key, jax.core.Tracer):
            self.consumed.append(tuple(np.asarray(key).ravel().tolist()))

    def split(self, key, num=2):
        self.note(key)
        return random.split(key, num)

    def fold_in(self, key, data):
        self.note(key)
        return random.fold_in(key, data)

    def uniform(self, key, *args, **kwargs):
        self.note(key)
        return random.uniform(key, *args, **kwargs)

    def normal(self, key, *args, **kwargs):
        self.note(key)
        return random.normal(key, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(random, name)


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_every_random_key_is_used_once(monkeypatch, algorithm):
    """Each key a step derives goes to one consumer -- the next step, the collisions,
    the thermal wall, the sub-steps of the implicit scheme -- and is either split or
    drawn from, once. A key used twice hands two consumers the same random numbers:
    in JAX's threefry keys ``fold_in(k, 1)`` equals ``split(k)[1]``, which is how the
    thermal wall once drew the numbers the next step's collisions drew."""
    ledger = KeyLedger()
    monkeypatch.setattr(simulation_module, "random", ledger)
    monkeypatch.setattr(Simulation, "_collide", lambda self, key, x, v, *rest: (ledger.note(key), v)[1])
    electrons = Species.electrons(n=40, density=1e6, vth=(1e6, 1e6, 1e6), drift=(-2e6, 0, 0))
    domain = Domain(length=1e-3, cells=8, particle_bc=("thermal", "reflective"), field_bc="reflective")
    sim = Simulation(domain, [electrons], Solver(algorithm=algorithm, picard_iterations=1),
                     Collisions(coulomb_log=10.0))
    carry, extra = sim.initial_state(random.PRNGKey(0))
    step = sim._explicit_step if algorithm == "explicit" else sim._implicit_step
    for _ in range(3):
        carry, _ = step(carry, extra)
    assert len(ledger.consumed) > 6
    assert len(set(ledger.consumed)) == len(ledger.consumed)


@pytest.mark.parametrize("switch", [{"filter_passes": 2}, {"field_solver": "gauss"}])
def test_the_implicit_scheme_refuses_the_switches_it_would_ignore(switch):
    """A filter and the Gauss solve both belong to the explicit scheme. The implicit one
    used to accept and silently skip them; it now says so when it is built."""
    electrons = Species.electrons(n=10, density=1e10)
    with pytest.raises(ValueError, match="implicit"):
        Simulation(Domain(), [electrons], Solver(algorithm="implicit", **switch))
    Simulation(Domain(), [electrons], Solver(algorithm="explicit", **switch))


def test_absorbed_particles_are_parked_symmetrically_off_both_grids():
    """A particle with no weight left is parked the same distance beyond either wall,
    one and a half cells, the half-width of the spline, so that neither the centred nor
    the staggered stencil reaches back into the box from there."""
    x = jnp.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    v = jnp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    nothing = (jnp.zeros(2), jnp.zeros(2))
    parked, _, w, _ = apply_particle_bc(x, v, jnp.ones(2), jnp.ones(2), (L, L, L), (2, 2), (1.0, 1.0), nothing, DX)
    assert float(parked[0, 0]) == pytest.approx(-L / 2 - 1.5 * DX)
    assert float(parked[1, 0]) == pytest.approx(L / 2 + 1.5 * DX)
    for origin in (CENTRE0, CENTRE0 + DX / 2):
        assert float(jnp.abs(deposit(parked[:, 0], jnp.ones(2), origin, DX, CELLS, (2, 2))).max()) == 0.0


def slow_ones(speed):
    """A reflection law that returns the electrons slower than about 1e6 m/s."""
    return jnp.exp(-speed ** 2 / (2 * 1e12))


@pytest.mark.parametrize("bc", [(0, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3)])
def test_wrap_positions_is_the_position_map_of_the_particle_walls(bc):
    """wrap_positions repeats the position rules of apply_particle_bc to be cheaper. On
    particles beyond either wall and in all three coordinates, with and without weight,
    the two give identical positions."""
    rng = np.random.default_rng(3)
    n = 400
    x = jnp.asarray(rng.uniform(-0.6 * L, 0.6 * L, (n, 3)))
    w = jnp.asarray(np.where(rng.uniform(size=n) < 0.3, 0.0, 1.0))
    unchanged = (jnp.ones(n), jnp.ones(n))
    box = (L, 0.7, 0.9)
    full, _, _, _ = apply_particle_bc(x, jnp.zeros_like(x), w, jnp.ones(n), box, bc, (1.0, 1.0), unchanged, DX)
    assert np.array_equal(np.asarray(wrap_positions(x, w, box, bc, DX)), np.asarray(full))


def walled_simulation(**solver):
    """Electrostatic (no transverse velocity), so that a step far above the light-wave limit is stable."""
    e = Species.electrons(n=600, density=1e15, vth=(2e6, 0, 0), drift=(1e6, 0, 0), reflection=(0.0, slow_ones))
    i = Species.ions(n=600, density=1e15, electrons=e, mass_ratio=50.0, reflection=(0.0, 0.5))
    domain = Domain(length=1e-3, cells=24, dt_over_dx_c=20.0, particle_bc=("thermal", "absorbing"),
                    field_bc=("reflective", "absorbing"))
    return Simulation(domain, [e, i], Solver(filter_passes=2, **solver))


def test_the_carried_density_is_the_density_at_the_integer_time_positions():
    """The explicit step no longer deposits the density at x^n: it carries the one the
    previous step ended on. That is only right if the carried density is the deposit at
    the reconstructed positions wrap(x^{n+1/2} - dt v/2), with the weights the walls have
    left, which this checks after a run with a thermal wall, partial reflection and a
    filter."""
    sim = walled_simulation()
    d = sim.domain
    out = sim.run(40, seed=1, store_every=4)
    E, B, x_half, v, w, qm, rho, key = out.state
    x_n = wrap_positions(x_half - 0.5 * d.dt * v, w, (d.length, d.length_y, d.length_z), d.particle_bc, d.dx)
    expected = sim._smooth(deposit(x_n[:, 0], out.charge * w, d.grid[0], d.dx, d.cells, d.particle_bc))
    assert np.allclose(np.asarray(rho), np.asarray(expected), rtol=1e-12, atol=1e-12 * float(jnp.abs(expected).max()))
    assert np.allclose(np.asarray(out.rho[-1]), np.asarray(rho), rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["explicit", "implicit"])
def test_store_every_traces_the_step_once(monkeypatch, algorithm):
    """With ``store_every > 1`` the loop carries the last output along with the state
    through one scan, so the step is traced once rather than twice, which is most of the
    compile time."""
    traces = []
    method = "_explicit_step" if algorithm == "explicit" else "_implicit_step"
    original = getattr(Simulation, method)

    def counted(self, carry, extra):
        traces.append(1)
        return original(self, carry, extra)

    monkeypatch.setattr(Simulation, method, counted)
    electrons = Species.electrons(n=17, density=1e10, vth=(1e5, 0, 0))
    sim = Simulation(Domain(cells=11), [electrons], Solver(algorithm=algorithm, picard_iterations=1))
    out = sim.run(6, seed=0, store_every=3)
    assert len(traces) == 1 and out.E.shape == (2, 11, 3)


CODES = {0: "periodic", 1: "reflective", 2: "absorbing"}


def field_on_sheets(positions, bc, length=L, cells=CELLS):
    """E_x that sheets of electrons at rest, one pseudo-particle each, feel from their own
    fields and those of their images, read off the first explicit step: from rest nothing
    moves before the push, so the velocity after one step is (q/m) E dt exactly. Returns
    the fields at the sheets and the surface charge sigma of one sheet."""
    n = len(positions)
    x = np.zeros((n, 3))
    x[:, 0] = positions
    density = 1e12
    electrons = Species.electrons(n=n, density=density).replace(x=x, v=np.zeros((n, 3)))
    walls = tuple(CODES[code] for code in bc)
    sim = Simulation(Domain(length=length, cells=cells, particle_bc=walls, field_bc=walls), [electrons], Solver())
    out = sim.run(1, seed=0)
    qm = electrons.charge_si / electrons.mass
    sigma = electrons.charge_si * density * length / n
    return np.asarray(out.v[0, :, 0]) / (qm * sim.domain.dt), sigma


@pytest.mark.parametrize("fraction", [0.0, 0.2, 0.5, 0.77])
def test_a_charge_sheet_feels_no_force_of_its_own_in_a_periodic_box(fraction):
    """Gathering with the transpose of the deposit leaves a lone particle in a periodic
    box without a self-force, wherever it sits in its cell. Gathering E_x straight from
    the faces pushed it with up to 8 % of its own field."""
    field, sigma = field_on_sheets([CENTRE0 + (5 + fraction) * DX], (0, 0))
    assert abs(field[0]) < 1e-12 * abs(sigma) / epsilon_0


@pytest.mark.parametrize("bc", WALLS[1:])
@pytest.mark.parametrize("distance", [1.3 * DX, 5.4 * DX])
def test_a_charge_sheet_feels_its_images_as_the_one_dimensional_theory_says(bc, distance):
    """A sheet of charge sigma a distance a from the left wall, with its cloud inside the
    box, feels the field of its images (the 1D electrostatics of a sheet between two
    walls): between two symmetry planes, sigma/eps0 (1/2 - a/L), pushed from the nearer;
    from a symmetry plane towards a floating conductor, +-sigma/2 eps0 whatever a; between
    two short-circuited conductors, sigma/eps0 (a/L - 1/2), pulled to the nearer."""
    field, sigma = field_on_sheets([-L / 2 + distance], bc)
    expected = {(1, 1): 0.5 - distance / L, (1, 2): 0.5, (2, 1): -0.5, (2, 2): distance / L - 0.5}[bc]
    assert field[0] == pytest.approx(expected * sigma / epsilon_0, rel=1e-10)


@pytest.mark.parametrize("bc", WALLS)
@pytest.mark.parametrize("distance", [0.0, 0.4 * DX, 0.9 * DX, 1.6 * DX])
def test_the_force_near_a_wall_is_the_mirror_image_of_the_force_near_the_opposite_wall(bc, distance):
    """A sheet a distance a from the left wall and one a from the right wall, with the
    walls swapped, feel opposite fields -- also inside the last cell, where the cloud
    reaches beyond the wall and the value the gather assumes there matters."""
    left, _ = field_on_sheets([-L / 2 + distance], bc)
    right, sigma = field_on_sheets([L / 2 - distance], bc[::-1])
    assert abs(left[0] + right[0]) < 1e-11 * abs(sigma) / epsilon_0


@pytest.mark.parametrize("distance", [0.1 * DX, 0.7 * DX, 2.2 * DX])
def test_a_reflective_box_gathers_like_the_periodic_box_twice_as_long_with_the_images(distance):
    """Two reflective walls are two symmetry planes: the box is half of a periodic box
    twice as long holding each particle and its mirror image. The field on a sheet is the
    same in both, including within a cell of the wall, where the gather reads the mirror
    image of the first centre with the parity of E_x."""
    field, _ = field_on_sheets([-L / 2 + distance], (1, 1))
    doubled, sigma = field_on_sheets([distance, -distance], (0, 0), length=2 * L, cells=2 * CELLS)
    assert abs(field[0] - doubled[0]) < 1e-11 * abs(sigma) / epsilon_0


def lone_electron(drift, relativistic, length=1e-2):
    """One electron in a periodic box, so tenuous that its own field changes nothing."""
    electrons = Species.electrons(n=1, density=1.0, drift=drift, quiet=True)
    return Simulation(Domain(length=length, cells=8), [electrons], Solver(relativistic=relativistic))


def test_speeds_past_light_are_brought_back_only_in_a_relativistic_run():
    """A relativistic run cannot represent a speed at or above c, so a drawn velocity
    past sqrt(1 - 1e-5) c is scaled back to that speed along its own direction, and not
    per component, which once let (0.9c, 0.9c) through as 1.27c. A Newtonian run has no
    such limit: its velocities, and the derivatives with respect to them, are left alone."""
    out = lone_electron((0.9 * c, 0.9 * c, 0.0), relativistic=True).run(1, seed=0)
    v = np.asarray(out.v[0, 0])
    assert np.sum(v ** 2) / c ** 2 == pytest.approx(1 - 1e-5, rel=1e-12)
    assert v[0] == pytest.approx(v[1], rel=1e-12) and v[2] == 0.0

    sim = lone_electron((1.5 * c, 0.0, 0.0), relativistic=False)
    assert float(sim.run(1, seed=0).v[0, 0, 0]) == pytest.approx(1.5 * c, rel=1e-12)
    electrons = sim.species[0]
    slope = jax.grad(lambda d: sim.replace(species=(electrons.replace(drift=(d, 0.0, 0.0)),)).run(1, seed=0).v[0, 0, 0])
    assert float(slope(1.5 * c)) == pytest.approx(1.0, rel=1e-9)


F32_RUN = """
import json, sys
import numpy as np
from jaxincell import Domain, Simulation, Solver, Species, speed_of_light as c
gammas = {}
for gamma in (100.0, 300.0):
    speed = c * np.sqrt(1 - 1 / gamma ** 2)
    electrons = Species.electrons(n=1, density=1.0, drift=(speed, 0.0, 0.0), quiet=True)
    sim = Simulation(Domain(length=1e-2, cells=8, dt_over_dx_c=0.5), [electrons], Solver(relativistic=True))
    v = np.asarray(sim.run(1000, seed=0, store_every=100).v[:, 0, 0], dtype=np.float64)
    gammas[gamma] = list(1 / np.sqrt(1 - (v / c) ** 2))
print(json.dumps({"dtype": str(sim.run(1, seed=0).v.dtype), "gammas": gammas}))
"""


def test_a_relativistic_particle_keeps_its_lorentz_factor_in_single_precision():
    """A relativistic run carries u = gamma v, not v. Converting v to u and back every
    step loses a fraction gamma^2 epsilon of gamma each time, which in single precision
    walked a particle at gamma = 1000 to 1423 in a thousand steps with no field. Carried,
    u does not change at all when there is no force, so the stored velocities are
    identical from step to step and gamma is as exact as the start allows: to
    gamma^2 epsilon/2, 0.5 % at gamma = 300 in single precision."""
    import json
    import os
    import subprocess
    import sys

    env = dict(os.environ, JAX_ENABLE_X64="0")
    result = subprocess.run([sys.executable, "-c", F32_RUN], env=env, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["dtype"] == "float32"
    for gamma, history in report["gammas"].items():
        history = np.array(history)
        assert np.ptp(history) == 0.0, (gamma, history)
        assert history[0] == pytest.approx(float(gamma), rel=6e-3)


def test_collisions_without_a_coulomb_logarithm_need_a_negative_species():
    """The default Coulomb logarithm is the one of the lightest negatively charged
    species; without one, Collisions() has to be given coulomb_log, and says so when the
    simulation is built rather than failing inside the loop."""
    ions = Species.ions(n=10, density=1e20, vth=(1e4, 0, 0))
    with pytest.raises(ValueError, match="coulomb_log"):
        Simulation(Domain(), [ions], Solver(), Collisions())
    Simulation(Domain(), [ions], Solver(), Collisions(coulomb_log=10.0))
    Simulation(Domain(), [ions, Species.electrons(n=10, density=1e20)], Solver(), Collisions())
    # a traced charge has no sign until the program runs, so construction leaves it alone
    built = jax.jit(lambda q: Simulation(Domain(), [ions.replace(charge=q)], Solver(), Collisions()).species[0].charge)
    assert float(built(elementary_charge)) == elementary_charge


def test_simulation_and_run_refuse_bad_arguments_with_value_errors():
    """Argument checks raise ValueError, which ``python -O`` keeps, where assert would vanish."""
    electrons = Species.electrons(n=10, density=1e10)
    with pytest.raises(ValueError, match="species"):
        Simulation(Domain(), [], Solver())
    with pytest.raises(ValueError, match="distinct"):
        Simulation(Domain(), [electrons, electrons], Solver())
    sim = Simulation(Domain(), [electrons], Solver())
    for steps, store_every in ((10, 3), (10, 0)):
        with pytest.raises(ValueError, match="store_every"):
            sim.run(steps, store_every=store_every)
