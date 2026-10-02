"""Exact properties of the numerical kernels: conservation to round-off,
agreement with closed-form results, and the analytic response of the filter.

Every comparison that claims round-off passes ``rtol=0`` and an absolute tolerance
on the scale of the quantity compared. ``numpy.allclose`` otherwise adds a relative
tolerance of 1e-5, which turns a round-off check into a five-digit one, and a
default ``atol=1e-8``, which cannot fail for quantities smaller than that. Each test
draws its random data from its own generator, so what it checks does not depend on
which tests ran before it."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline
from jax import lax

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mu_0,
                       elementary_charge, load_state, mass_electron, quiet_start, save_state)
from jaxincell import speed_of_light as c
from jaxincell._core import (E_x_from_rho, apply_particle_bc, boris, boris_relativistic,
                             current_from_continuity, deposit, gather, gather_xyz, half_step_fields,
                             orbit_field_average, s2_weights, shape_weights, smooth, with_ghosts, wrap_positions)
from jaxincell._simulation import _implicit_fields

L, G = 1.0, 32
dx = L / G
x0 = -L / 2 + dx / 2


def lorentz_factor(v):
    return 1 / jnp.sqrt(1 - jnp.sum(v ** 2, 1) / c ** 2)


def quadratic_spline(u):
    """The quadratic B-spline M(u) in cell units, written out piece by piece rather
    than taken from the code under test."""
    u = np.abs(u)
    return np.where(u <= 0.5, 0.75 - u ** 2, np.where(u <= 1.5, 0.5 * (1.5 - u) ** 2, 0.0))


def periodic_spline_matrix(position, cells, order):
    """Independent Cox-de Boor basis, period one, including overlapping periodic images."""
    centre = (np.arange(cells) + .5) / cells - .5
    offset = (np.asarray(position)[:, None, None] - centre[None, :, None]
              + np.arange(-3, 4)[None, None, :]) * cells
    basis = BSpline.basis_element(np.arange(order + 2) - (order + 1) / 2, extrapolate=False)
    return np.nan_to_num(basis(offset)).sum(axis=-1)


def spline_orbit_field(field, x, shift, order=5):
    """Independent integral of the linear/quartic face interpolant, period one.

    Three Gauss points integrate each quartic piece exactly. Splitting at centre
    knots avoids large-potential subtraction at zero/tiny shifts and periodic wraps.
    """
    cells = len(field)
    faces = (np.arange(cells) + 1) / cells - .5
    knots = (np.arange(-3 * cells, 3 * cells) + (0. if order == 2 else .5)) / cells - .5
    basis = BSpline.basis_element(np.arange(order + 1) - order / 2, extrapolate=False)
    nodes, weights = np.polynomial.legendre.leggauss(3)

    def value(position):
        offset = (np.asarray(position)[..., None, None] - faces[:, None] + np.arange(-3, 4)) * cells
        return np.nan_to_num(basis(offset)).sum(axis=-1) @ field

    result = []
    for start, displacement in zip(x, shift):
        if displacement == 0:
            result.append(float(value(start)))
            continue
        fractions = (knots - start) / displacement
        cuts = np.r_[0., fractions[(fractions > 0) & (fractions < 1)], 1.]
        cuts.sort()
        width = np.diff(cuts)
        points = start + displacement * ((cuts[:-1, None] + cuts[1:, None]) / 2 + width[:, None] * nodes / 2)
        result.append(float(np.sum(width * (value(points) @ weights) / 2)))
    return np.asarray(result)


def quintic_midpoint_orbits(field, x, u, charge, mass, dt, substeps=2):
    """Independent frozen-field 1V relativistic secant-velocity fixed point."""
    x, u = np.array(x), np.array(u)
    displacement = np.zeros_like(x)
    h = dt / substeps
    for _ in range(substeps):
        old_x, old_u = x.copy(), u.copy()
        shift = h * old_u / np.hypot(1., old_u)
        for _ in range(64):
            new_u = old_u + h * charge / mass * spline_orbit_field(field, old_x, shift)
            new_shift = h * (old_u + new_u) / (np.hypot(1., old_u) + np.hypot(1., new_u))
            residual = max(np.max(abs(new_u - u)), np.max(abs(new_shift - shift)))
            u, shift = new_u, new_shift
            if residual < 2e-14:
                break
        assert residual < 2e-14, 'independent frozen-field orbit did not close'
        displacement += shift
        x = (old_x + shift + .5) % 1 - .5
    return x, u, displacement


def implicit_quintic_box(dt=.04, iterations=8, fraction=0., transverse=False, orbit_force='secant', order=5):
    """A neutral relativistic box with fixed physical markers and a density wave."""
    rng = np.random.default_rng(27)
    x = rng.uniform(-.5, .5, 32)
    velocity = rng.normal(size=(32, 3)) * .025 * c
    if not transverse:
        velocity[:, 1:] = 0
    density = epsilon_0 * mass_electron * c**2 / elementary_charge**2
    species = []
    for name, charge, mass in (('electrons', -1, mass_electron), ('ions', 1, 1836 * mass_electron)):
        position = x + (.035 * np.sin(2 * np.pi * x + .3) if charge < 0 else 0.) + fraction / 8
        position = jnp.zeros((32, 3)).at[:, 0].set((position + .5) % 1 - .5)
        species.append(Species(name, 32, charge, mass, density, x=position,
                               v=jnp.asarray(velocity if charge < 0 else velocity / 1836)))
    return Simulation(Domain(1., 8, time_step=dt / c), tuple(species),
                      Solver(algorithm='implicit', shape_order=order, relativistic=True, orbit_force=orbit_force,
                             picard_iterations=iterations))


def test_implicit_quintic_orbit_transpose_matches_independent_integral_and_work():
    rng = np.random.default_rng(28)
    field = jnp.asarray(rng.normal(size=G) + .7)
    position = jnp.asarray(np.r_[-.5, -.499, -.1, .5 - 1e-13, .3, .03, .42])
    shift = jnp.asarray(dx * np.array([0., 1e-14, -1e-12, 1e-9, .7, -2.3, G + .2]))
    end = (position + shift + .5) % 1 - .5
    amount = jnp.asarray(rng.normal(size=len(position)))

    def to_current(rho):
        return current_from_continuity(jnp.zeros(G), rho, 1., dx, 0., (0, 0))
    phi = jax.linear_transpose(to_current, jnp.zeros(G))(field)[0]

    def potential(x):
        return dx * jax.linear_transpose(lambda q: deposit(x, q, x0, dx, G, (0, 0), 5), amount)(phi)[0]

    midpoint = (position + shift / 2 + .5) % 1 - .5
    slope = jax.jvp(potential, (midpoint,), (jnp.ones_like(position),))[1]
    small = abs(shift) < np.sqrt(np.finfo(float).eps) * dx
    actual = jnp.where(small, slope, (potential(end) - potential(position)) / jnp.where(small, 1., shift))
    actual += jnp.mean(field)
    expected = spline_orbit_field(np.asarray(field), np.asarray(position), np.asarray(shift))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
    rho0, rho1 = (deposit(x, amount, x0, dx, G, (0, 0), 5) for x in (position, end))
    current = current_from_continuity(rho0, rho1, 1., dx, jnp.sum(amount * shift), (0, 0))
    np.testing.assert_allclose(jnp.mean(current), jnp.sum(amount * shift), rtol=0, atol=2e-14)
    np.testing.assert_allclose(dx * current @ field, jnp.sum(amount * shift * expected), rtol=0, atol=2e-13)


@pytest.mark.parametrize('fraction', [0., .25, .5])
@pytest.mark.parametrize('orbit_force', ['secant', 'integral'])
def test_implicit_quintic_accepted_force_current_and_fractional_grid_bias(fraction, orbit_force):
    sim = implicit_quintic_box(fraction=fraction, orbit_force=orbit_force)
    initial, (mass, charge) = sim.initial_state(jax.random.PRNGKey(0))
    final, output = jax.jit(sim._implicit_step)(initial, (mass, charge))
    scale = mass_electron * c**2 / elementary_charge
    field = np.asarray((initial.E[:, 0] + final.E[:, 0]) / 2) / scale
    x, u, shift = quintic_midpoint_orbits(
        field, np.asarray(initial.x[:, 0]), np.asarray(initial.u[:, 0]) / c,
        np.asarray(charge) / elementary_charge, np.asarray(mass) / mass_electron, .04)
    weights = np.asarray(initial.w) / (sim.species[0].density * L)
    amounts = np.asarray(charge) / elementary_charge * weights
    np.testing.assert_allclose(final.u[:, 0] / c, u, rtol=0, atol=2e-13 if orbit_force == 'integral' else 3e-12)
    np.testing.assert_allclose(final.x[:, 0], x, rtol=0, atol=3e-13)
    expected_rho = amounts @ periodic_spline_matrix(x, 8, 5) * 8
    rho_scale = elementary_charge * sim.species[0].density
    np.testing.assert_allclose(final.rho / rho_scale, expected_rho, rtol=0, atol=2e-14)
    current = np.asarray(output[5][:, 0]) / (rho_scale * c)
    np.testing.assert_allclose(current.mean(), np.sum(amounts * shift) / .04, rtol=0, atol=2e-13)
    continuity = ((final.rho - initial.rho) / sim.domain.dt
                  + (output[5][:, 0] - jnp.roll(output[5][:, 0], 1)) / sim.domain.dx)
    assert float(jnp.max(abs(continuity)) / (rho_scale * c)) < 2e-12
    impulse = np.sum(np.asarray(mass) / mass_electron * weights * np.asarray(final.u[:, 0] - initial.u[:, 0]) / c)
    reference_impulse = np.sum(np.asarray(mass) / mass_electron * weights * (u - np.asarray(initial.u[:, 0]) / c))
    np.testing.assert_allclose(impulse, reference_impulse, rtol=0, atol=2e-12)
    assert abs(impulse) > 1e-12  # This mesh-conjugate force is not an exact continuum momentum map.


def test_implicit_quintic_picard_closure_and_transverse_work_converge():
    errors = []
    for iterations in (2, 4, 8):
        sim = implicit_quintic_box(dt=.2, iterations=iterations)
        initial, extra = sim.initial_state(jax.random.PRNGKey(0))
        final, _ = jax.jit(sim._implicit_step)(initial, extra)
        field = np.asarray((initial.E[:, 0] + final.E[:, 0]) / 2) * elementary_charge / (mass_electron * c**2)
        _, reference, _ = quintic_midpoint_orbits(field, np.asarray(initial.x[:, 0]), np.asarray(initial.u[:, 0]) / c,
                                                  np.asarray(extra[1]) / elementary_charge,
                                                  np.asarray(extra[0]) / mass_electron, .2)
        errors.append(np.max(abs(reference - np.asarray(final.u[:, 0]) / c)))
    assert errors[1] < errors[0] / 5 and errors[2] < errors[1] / 5
    assert errors[2] < 2e-12
    sim = implicit_quintic_box(transverse=True)
    initial, (mass, charge) = sim.initial_state(jax.random.PRNGKey(0))
    scale = mass_electron * c**2 / elementary_charge
    initial = initial.replace(E=initial.E.at[:, 1].set(.02 * scale * jnp.cos(2 * jnp.pi * sim.domain.faces)),
                              B=initial.B.at[:, 2].set(.1 * scale / c))
    final, output = jax.jit(sim._implicit_step)(initial, (mass, charge))

    def energy(state):
        u = np.asarray(state.u) / c
        gamma = np.sqrt(1 + np.sum(u**2, axis=1))
        kinetic = np.sum(np.asarray(mass * state.w) * c**2 * np.sum(u**2, axis=1) / (gamma + 1))
        return kinetic + epsilon_0 * sim.domain.dx / 2 * np.sum(np.asarray(state.E)**2 + c**2 * np.asarray(state.B)**2)

    unit = sim.species[0].density * mass_electron * c**2
    assert abs(energy(final) - energy(initial)) / unit < 2e-12
    work = sim.domain.dt * sim.domain.dx * jnp.sum(output[5] * (initial.E + final.E) / 2)
    field_loss = epsilon_0 * sim.domain.dx / 2 * jnp.sum(
        initial.E**2 - final.E**2 + c**2 * (initial.B**2 - final.B**2))
    np.testing.assert_allclose(work / unit, field_loss / unit, rtol=0, atol=2e-13)


def uniform_quintic_box(dt, parameters=(1., 1., 1.), orbit_force='secant', order=5):
    """Cold mobile species isolate the mean current and its exact midpoint phase."""
    weight, electric, seed = parameters
    density = epsilon_0 * mass_electron * c**2 / elementary_charge**2
    x, _ = quiet_start(16, L)
    electrons = Species('electrons', 16, -1, mass_electron, density * weight, x=x,
                        v=jnp.zeros((16, 3)).at[:, 0].set(.005 * c * seed))
    ions = Species('ions', 16, 1, 1836 * mass_electron, density * weight, x=x, v=jnp.zeros((16, 3)))
    sim = Simulation(Domain(L, 8, time_step=dt / c), (electrons, ions),
                     Solver(algorithm='implicit', shape_order=order, orbit_force=orbit_force, picard_iterations=8))
    initial, _ = sim.initial_state(jax.random.PRNGKey(0))
    scale = mass_electron * c**2 / elementary_charge
    return sim, initial.replace(E=initial.E.at[:, 0].add(.01 * scale * electric))


def uniform_midpoint_answer(parameters, dt, steps):
    """Independent 2x2 Cayley map and Newtonian kinetic energy; analytic in inputs."""
    weight, electric, seed = parameters
    omega2 = weight * (1 + 1 / 1836)
    generator = np.array([[0., -1.], [omega2, 0.]])
    midpoint = np.linalg.solve(np.eye(2) - dt * generator / 2, np.eye(2) + dt * generator / 2)
    initial = np.array([.01 * electric, -.005 * seed * weight])
    field, current = np.linalg.matrix_power(midpoint, steps) @ initial
    impulse = (current - initial[1]) / omega2
    electron, ion = .005 * seed - impulse, impulse / 1836
    return field, current, .5 * weight * (electron**2 + 1836 * ion**2)


@pytest.mark.parametrize('orbit_force', ['secant', 'integral'])
def test_implicit_quintic_mean_phase_refinement_and_complete_restart(tmp_path, orbit_force):
    phase_errors = []
    for dt, steps in ((.2, 8), (.1, 16)):
        sim, initial = uniform_quintic_box(dt, orbit_force=orbit_force)
        out = sim.run(steps, state=initial, store_particles=False, store_every=4)
        scale = mass_electron * c**2 / elementary_charge
        field = float(jnp.mean(out.state.E[:, 0]) / scale)
        mass, charge = sim.per_particle
        density = sim.species[0].density
        current = float(jnp.sum(charge * out.state.w * out.state.u[:, 0]) / (elementary_charge * density * c * L))
        expected_field, expected_current, _ = uniform_midpoint_answer((1., 1., 1.), dt, steps)
        np.testing.assert_allclose([field, current], [expected_field, expected_current], rtol=0, atol=2e-13)
        omega = np.sqrt(1 + 1 / 1836)
        initial_phase = np.angle(.01 - .005j / omega)
        phase_errors.append(abs(np.angle(field + 1j * current / omega) - initial_phase - omega * dt * steps))
        if dt == .2:
            first = sim.run(4, state=initial, store_particles=False, store_every=4)
            path = save_state(tmp_path / 'quintic.npz', first.state, sim)
            continued = sim.run(4, state=load_state(path, sim), store_particles=False, store_every=4)
            for whole, resumed in zip(jax.tree.leaves(out.state), jax.tree.leaves(continued.state)):
                np.testing.assert_array_equal(whole, resumed)
            with pytest.raises(ValueError, match='shape_order'):
                load_state(path, sim.replace(solver=sim.solver.replace(shape_order=2)))
            assert abs(float(jnp.sum(charge * initial.w))) / (elementary_charge * density * L) < 2e-15
    assert phase_errors[0] > 1e-4 and 3.9 < phase_errors[0] / phase_errors[1] < 4.1


@pytest.mark.parametrize('orbit_force', ['secant', 'integral'])
@pytest.mark.parametrize('order', [2, 5])
def test_implicit_quintic_physical_objective_ad_matches_independent_frechet_and_finite_differences(orbit_force, order):
    def objective(parameters):
        sim, initial = uniform_quintic_box(.1, parameters, orbit_force, order)
        final = sim.run(4, state=initial, store_particles=False, store_every=4).state
        field = jnp.mean(final.E[:, 0]) * elementary_charge / (mass_electron * c**2)
        density = epsilon_0 * mass_electron * c**2 / elementary_charge**2
        kinetic = jnp.sum(sim.per_particle[0] / mass_electron * final.w / (density * L)
                          * jnp.sum((final.u / c)**2, axis=1) / 2)
        return field + .3 * kinetic

    point = jnp.array([1., 1., 1.])
    reverse = np.asarray(jax.jit(jax.grad(objective))(point))
    forward = np.array([float(jax.jvp(objective, (point,), (direction,))[1]) for direction in jnp.eye(3)])
    independent, finite = [], []
    measured = jax.jit(objective)
    for direction in np.eye(3):
        field, _, kinetic = uniform_midpoint_answer(np.asarray(point, dtype=complex) + 1e-30j * direction, .1, 4)
        independent.append(np.imag(field + .3 * kinetic) / 1e-30)
        finite.append(float(measured(point + 1e-5 * direction) - measured(point - 1e-5 * direction)) / 2e-5)
    assert min(abs(reverse)) > 1e-6
    np.testing.assert_allclose(reverse, independent, rtol=2e-9, atol=0)
    np.testing.assert_allclose(forward, reverse, rtol=2e-10, atol=0)
    np.testing.assert_allclose(finite, reverse, rtol=2e-8, atol=0)


@pytest.mark.parametrize('order', [2, 5])
def test_short_orbit_integral_matches_independent_quadrature_and_zero_shift_derivative(order):
    rng = np.random.default_rng(61)
    field = rng.normal(size=G) + .7
    position = np.r_[x0 + dx / 2, x0, x0 + .37 * dx, .5 - 1e-13, -.5 + 1e-13, .13, -.23, .31]
    shift = dx * np.array([0., 0., 1e-20, .7, -.7, -1e-8, 1., -1.])
    average = orbit_field_average(jnp.asarray(field), jnp.asarray(position), jnp.asarray(shift), x0 + dx / 2, dx, order)
    expected = spline_orbit_field(field, position, shift, order)
    np.testing.assert_allclose(average, expected, rtol=0, atol=5e-15)
    np.testing.assert_allclose(orbit_field_average(jnp.ones(G), position, shift, x0 + dx / 2, dx, order),
                               1, rtol=0, atol=5e-16)  # Preserve the mean field exactly once.
    point = x0 + .37 * dx  # Off knots: the quadratic shape's face-field slope has jumps.
    basis = BSpline.basis_element(np.arange(order + 1) - order / 2, extrapolate=False).derivative()
    offset = (point - (np.arange(G) + 1) / G + .5 + np.arange(-3, 4)[:, None]) * G
    derivative = np.nan_to_num(basis(offset)).sum(axis=0) @ field * G / 2

    def force(s):
        return orbit_field_average(jnp.asarray(field), jnp.asarray(point), s, x0 + dx / 2, dx, order)
    np.testing.assert_allclose(jax.grad(force)(0.), derivative, rtol=0, atol=2e-13)
    h = dx * 1e-5
    independent_fd = (spline_orbit_field(field, [point], [h], order)
                      - spline_orbit_field(field, [point], [-h], order))[0] / (2 * h)
    np.testing.assert_allclose(jax.grad(force)(0.), independent_fd, rtol=2e-6, atol=2e-10)
    # Work uses an independent charge spline and the accepted endpoint continuity current.
    amount = rng.normal(size=len(position))
    end = (position + shift + .5) % 1 - .5
    rho0, rho1 = (amount @ periodic_spline_matrix(p, G, order) / dx for p in (position, end))
    current = current_from_continuity(rho0, rho1, 1., dx, np.sum(amount * shift), (0, 0))
    np.testing.assert_allclose(dx * current @ field, np.sum(amount * shift * average), rtol=0, atol=2e-13)


@pytest.mark.parametrize('order', [2, 5])
def test_short_orbit_integral_avoids_potential_cancellation(order):
    rng = np.random.default_rng(62)
    field = jnp.asarray(rng.normal(size=G) + .7)
    position = jnp.asarray(rng.uniform(-.5, .5, 16))
    shift = dx * np.sqrt(np.finfo(float).eps) * jnp.linspace(2., 4., 16)
    phi = jax.linear_transpose(lambda rho: current_from_continuity(jnp.zeros(G), rho, 1., dx, 0., (0, 0)),
                               jnp.zeros(G))(field)[0]

    def potential(x):
        return dx * jax.linear_transpose(lambda q: deposit(x, q, x0, dx, G, (0, 0), order), jnp.ones(16))(phi)[0]
    end = (position + shift + .5) % 1 - .5
    secant = (potential(end) - potential(position)) / shift + jnp.mean(field)
    integral = orbit_field_average(field, position, shift, x0 + dx / 2, dx, order)
    expected = spline_orbit_field(np.asarray(field), np.asarray(position), np.asarray(shift), order)
    assert np.max(abs(np.asarray(secant) - expected)) > 1e-9
    np.testing.assert_allclose(integral, expected, rtol=0, atol=5e-15)


@pytest.mark.parametrize('order', [2, 5])
def test_integral_force_all_components_work_charge_wrap_and_complete_restart(order, tmp_path):
    sim = implicit_quintic_box(dt=.02, transverse=True, orbit_force='integral', order=order)
    initial, (mass, charge) = sim.initial_state(jax.random.PRNGKey(0))
    initial = initial.replace(x=initial.x.at[0, 0].set(.5 - 1e-5), u=initial.u.at[0, 0].set(.03 * c))
    rho = deposit(initial.x[:, 0], charge * initial.w, sim.domain.grid[0], sim.domain.dx, 8, (0, 0), order)
    scale = mass_electron * c**2 / elementary_charge
    electric = jnp.stack((E_x_from_rho(rho, sim.domain.dx, (0, 0)) + .006 * scale,
                          .02 * scale * jnp.cos(2 * jnp.pi * sim.domain.faces),
                          .011 * scale * jnp.sin(2 * jnp.pi * sim.domain.faces)), axis=1)
    magnetic = jnp.broadcast_to(jnp.array([.04, -.03, .1]) * scale / c, (8, 3))
    initial = initial.replace(rho=rho, E=electric, B=magnetic)
    final, output = jax.jit(sim._implicit_step)(initial, (mass, charge))
    assert float(final.x[0, 0]) < -.49  # An actual accepted periodic crossing.
    unit = sim.species[0].density * mass_electron * c**2 * L

    def kinetic(st):
        return jnp.sum(mass * st.w * jnp.sum(st.u**2, axis=1) / (sim._gamma(st.u)[:, 0] + 1))
    work = sim.domain.dt * sim.domain.dx * jnp.sum(output[5] * (initial.E + final.E) / 2)
    loss = epsilon_0 * sim.domain.dx / 2 * jnp.sum(initial.E**2 - final.E**2 + c**2 * (initial.B**2 - final.B**2))
    np.testing.assert_allclose((kinetic(final) - kinetic(initial)) / unit, work / unit, rtol=0, atol=2e-13)
    np.testing.assert_allclose(work / unit, loss / unit, rtol=0, atol=2e-13)
    density_scale = elementary_charge * sim.species[0].density
    continuity = ((final.rho - initial.rho) / sim.domain.dt
                  + (output[5][:, 0] - jnp.roll(output[5][:, 0], 1)) / sim.domain.dx)
    np.testing.assert_allclose(continuity / (density_scale * c), 0, rtol=0, atol=2e-13)
    gauss = ((final.E[:, 0] - jnp.roll(final.E[:, 0], 1)) / sim.domain.dx
             - (final.rho - jnp.mean(initial.rho)) / epsilon_0)
    np.testing.assert_allclose(gauss * epsilon_0 / density_scale, 0, rtol=0, atol=2e-13)
    shift = (final.x[:, 0] - initial.x[:, 0] + .5) % 1 - .5
    mean = jnp.sum(charge * initial.w * shift) / (L * sim.domain.dt)
    np.testing.assert_allclose((jnp.mean(output[5][:, 0]) - mean) / (density_scale * c), 0, rtol=0, atol=2e-13)
    np.testing.assert_allclose(sim.domain.dx * jnp.sum(final.rho) / (density_scale * L), 0, rtol=0, atol=2e-13)
    whole = sim.run(4, state=initial, moments='flux', store_particles=False).state
    first = sim.run(2, state=initial, moments='flux', store_particles=False).state
    path = save_state(tmp_path / 'integral', first, sim)
    restored = load_state(path, sim)
    for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(first)):
        np.testing.assert_array_equal(actual, expected)
    resumed = sim.run(2, state=restored, moments='flux', store_particles=False).state
    for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(whole)):
        np.testing.assert_array_equal(actual, expected)
    with np.load(path) as data:
        stored = dict(data)
    assert int(stored['format']) == 3 and str(stored['orbit_force']) == 'integral'
    with pytest.raises(ValueError, match='orbit_force'):
        load_state(path, sim.replace(solver=sim.solver.replace(orbit_force='secant')))
    malformed = [{k: v for k, v in stored.items() if k != key} for key in ('orbit_force', 'shape_order', 'algorithm')]
    malformed += [{**stored, key: np.asarray(value)} for key, value in
                  (('orbit_force', 'secant'), ('algorithm', 'explicit'), ('shape_order', [order]))]
    for archive in malformed:
        np.savez(tmp_path / 'malformed.npz', **archive)
        with pytest.raises(ValueError, match='format 3 requires'):
            load_state(tmp_path / 'malformed.npz')
    secant = sim.replace(solver=sim.solver.replace(orbit_force='secant'))
    with pytest.raises(ValueError, match='orbit_force'):
        load_state(save_state(tmp_path / 'secant', first, secant), sim)


@pytest.mark.parametrize('order', [2, 5])
def test_integral_force_retains_the_long_orbit_secant_and_default_path(order):
    sim, initial = uniform_quintic_box(.3, order=order)
    sim = sim.replace(solver=sim.solver.replace(substeps=1))
    initial = initial.replace(u=initial.u.at[:, 0].set(.6 * c))
    extra = sim.per_particle
    # All accepted shifts exceed a cell. Disable jit to isolate unchanged arithmetic
    # from independently compiled programs' permissible accumulation roundoff.
    with jax.disable_jit():
        default, _ = sim._implicit_step(initial, extra)
        named, _ = sim.replace(solver=sim.solver.replace(orbit_force='secant'))._implicit_step(initial, extra)
        integral, _ = sim.replace(solver=sim.solver.replace(orbit_force='integral'))._implicit_step(initial, extra)
    assert np.min(abs(np.asarray(default.x[:, 0] - initial.x[:, 0]))) > sim.domain.dx
    for want, explicit_default, fallback in zip(jax.tree.leaves(default), jax.tree.leaves(named),
                                                jax.tree.leaves(integral)):
        np.testing.assert_array_equal(want, explicit_default)
        np.testing.assert_array_equal(want, fallback)


@pytest.mark.parametrize('order', [2, 5])
def test_integral_force_differentiates_initialized_positions_weights_fields_and_dynamics(order):
    base = implicit_quintic_box(dt=.02, orbit_force='integral', order=order)
    density = base.species[0].density
    scale = mass_electron * c**2 / elementary_charge

    def objective(parameters):
        displacement, ripple, amount, electric = parameters
        electrons, ions = base.species
        x = electrons.x.at[:, 0].add(displacement * .01 * jnp.sin(2 * jnp.pi * electrons.x[:, 0]))
        sim = base.replace(species=(electrons.replace(x=x, density=density * amount),
                                    ions.replace(density=density * amount)))
        state, extra = sim.initial_state(jax.random.PRNGKey(0))
        modulation = 1 + ripple * jnp.tile(jnp.cos(2 * jnp.pi * ions.x[:, 0]), 2)
        weights = state.w * modulation  # The two species retain exactly matched total weight.
        rho = deposit(state.x[:, 0], extra[1] * weights, sim.domain.grid[0], sim.domain.dx, 8, (0, 0), order)
        field = state.E.at[:, 0].set(E_x_from_rho(rho, sim.domain.dx, (0, 0)) + .006 * scale * electric)
        final, _ = sim._implicit_step(state.replace(w=weights, rho=rho, E=field), extra)
        kinetic = jnp.sum(extra[0] / mass_electron * final.w / (density * L)
                          * jnp.sum((final.u / c)**2, axis=1) / (sim._gamma(final.u)[:, 0] + 1))
        mode = jnp.mean(final.E[:, 0] / scale * jnp.cos(2 * jnp.pi * sim.domain.faces))
        return jnp.mean(final.E[:, 0]) / scale + .3 * kinetic + mode

    point = jnp.array([.4, .15, 1., 1.])
    gradient = np.asarray(jax.jit(jax.grad(objective))(point))
    measured = jax.jit(objective)
    h = 1e-5
    finite = [float(measured(point + h * axis) - measured(point - h * axis)) / (2 * h) for axis in jnp.eye(4)]
    assert min(abs(gradient)) > 1e-6
    np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=2e-10)


@pytest.mark.parametrize("order", [2, 5])
def test_cardinal_shape_matches_independent_basis_and_moments(order):
    """A cardinal degree-p spline convolves p+1 boxes; its variance is (p+1)/12."""
    x = jnp.asarray(dx * np.linspace(-7, 7, 281) + x0)
    index, weights = shape_weights(x, x0, dx, order)
    distance = (np.asarray(x)[:, None] - x0) / dx - np.asarray(index)
    basis = BSpline.basis_element(np.arange(order + 2) - (order + 1) / 2, extrapolate=False)
    expected = np.nan_to_num(basis(distance))  # outside the compact support
    np.testing.assert_allclose(weights, expected, rtol=0, atol=2e-15)
    # SHARP's W^p integrates its degree-(p-1) raw particle shape across one cell.
    raw = BSpline.basis_element(np.arange(order + 1) - order / 2, extrapolate=False).antiderivative()
    integrated = (raw(np.clip(distance + .5, -order / 2, order / 2))
                  - raw(np.clip(distance - .5, -order / 2, order / 2)))
    np.testing.assert_allclose(weights, integrated, rtol=0, atol=2e-15)
    assert np.min(weights) >= 0
    np.testing.assert_allclose(np.sum(weights, axis=1), 1, rtol=0, atol=2e-15)
    np.testing.assert_allclose(np.sum(weights * distance, axis=1), 0, rtol=0, atol=3e-15)
    np.testing.assert_allclose(np.sum(weights * distance**2, axis=1), (order + 1) / 12,
                               rtol=0, atol=3e-15)
    _, slope = jax.jvp(lambda position: shape_weights(position, x0, dx, order)[1],
                       (x,), (jnp.full_like(x, dx),))
    np.testing.assert_allclose(slope, np.nan_to_num(basis.derivative()(distance)), rtol=0, atol=4e-15)


@pytest.mark.parametrize("order", [2, 5])
def test_periodic_shape_gather_is_the_deposit_transpose_for_every_component(order):
    rng = np.random.default_rng(15)
    x = jnp.asarray(np.r_[-L / 2, L / 2, rng.uniform(-L / 2, L / 2, 46)])
    q, field = jnp.asarray(rng.normal(size=len(x))), jnp.asarray(rng.normal(size=(G, 6)))
    rho = deposit(x, q, x0, dx, G, (0, 0), order)
    gathered = gather(with_ghosts(field, (0, 0), shape_order=order), x, x0, dx, order)
    np.testing.assert_allclose(dx * jnp.sum(rho), jnp.sum(q), rtol=0, atol=2e-14 * float(jnp.sum(abs(q))))
    np.testing.assert_allclose(jnp.sum(q[:, None] * gathered, axis=0), dx * rho @ field,
                               rtol=0, atol=2e-14 * float(jnp.sum(abs(q))))


@pytest.mark.parametrize("order", [2, 5])
def test_periodic_xyz_gather_matches_independent_tensor_product(order):
    rng = np.random.default_rng(16)
    n, ny, nz = 8, 3, 4
    position = rng.uniform(-.5, .5, size=(7, 3))
    position[:2, 0] = [-.5, .5]
    field = jnp.asarray(rng.normal(size=(n, ny, nz, 6)))
    matrices = [periodic_spline_matrix(position[:, axis], count, order)
                for axis, count in enumerate((n, ny, nz))]
    expected = np.einsum('pa,pb,pc,abcf->pf', *matrices, field)
    ghosts = with_ghosts(field, (0, 0), shape_order=order)
    actual = gather_xyz(ghosts, jnp.asarray(position), -.5 + .5 / n, 1 / n, (1., 1.), order)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-15)
    flat = field[:, 0, 0]
    scalar = with_ghosts(flat, (0, 0), shape_order=order)
    np.testing.assert_allclose(gather_xyz(scalar[:, None, None], jnp.asarray(position),
                                          -.5 + .5 / n, 1 / n, (1., 1.), order),
                               gather(scalar, jnp.asarray(position[:, 0]), -.5 + .5 / n, 1 / n, order),
                               rtol=0, atol=5e-15)


@pytest.mark.parametrize("cells", [1, 2])
def test_quintic_gather_wraps_all_six_centres_on_tiny_periodic_grids(cells):
    position = jnp.asarray([-.5, -.499, .07, .499, .5])
    field = jnp.arange(cells * 6, dtype=float).reshape(cells, 6)
    ghosts = with_ghosts(field, (0, 0), shape_order=5)
    np.testing.assert_array_equal(ghosts, field[jnp.arange(-3, cells + 3) % cells])
    actual = gather(ghosts, position, -.5 + .5 / cells, 1 / cells, 5)
    np.testing.assert_allclose(actual, periodic_spline_matrix(position, cells, 5) @ field,
                               rtol=0, atol=5e-15)


def test_quintic_simulation_sources_moments_and_external_gathers_use_the_selected_shape():
    rng = np.random.default_rng(18)
    position, velocity = jnp.asarray(rng.uniform(-.5, .5, (12, 3))), jnp.asarray(rng.normal(size=(12, 3)))
    amount = jnp.asarray(rng.uniform(.5, 1.5, 12))
    field = jnp.asarray(rng.normal(size=(G, 3, 4, 3)))
    magnetic = jnp.asarray(rng.normal(size=(G, 3)))
    sim = Simulation(Domain(L, G, dt_over_dx_c=.2, length_y=L, length_z=L), (Species.electrons(12, 1),),
                     Solver(shape_order=5), external_E=field, external_B=magnetic)
    matrices = [periodic_spline_matrix(position[:, axis], cells, 5)
                for axis, cells in enumerate((G, 3, 4))]
    old = (position[:, 0] - .01 * velocity[:, 0] + .5) % L - .5
    rho0 = amount @ periodic_spline_matrix(old, G, 5) / dx
    mean_current = jnp.sum(amount * velocity[:, 0]) / L
    rho, current = sim._sources(position, velocity, amount, .01, mean_current, rho0)
    centred = np.stack([amount * velocity[:, axis] @ matrices[0] / dx for axis in (1, 2)], axis=1)
    np.testing.assert_allclose(rho, amount @ matrices[0] / dx, rtol=0, atol=2e-13)
    np.testing.assert_allclose(current[:, 1:], (centred + np.roll(centred, -1, axis=0)) / 2,
                               rtol=0, atol=2e-13)
    np.testing.assert_allclose(jnp.mean(current[:, 0]), mean_current, rtol=0, atol=2e-13)
    expected_moments = np.stack([amount @ matrices[0]]
                                + [amount * velocity[:, axis] @ matrices[0] for axis in range(3)]) / dx
    np.testing.assert_allclose(sim.moments(position, velocity, amount, 4)[0], expected_moments,
                               rtol=0, atol=2e-13)
    expected = np.concatenate([np.einsum('pa,pb,pc,abcf->pf', *matrices, field), matrices[0] @ magnetic], axis=1)
    np.testing.assert_allclose(sim.external_fields_at(position), expected, rtol=0, atol=5e-15)
    np.testing.assert_allclose(sim._fields_at(position, jnp.zeros((G, 3)), jnp.zeros((G, 3)), jnp.zeros(G)),
                               expected, rtol=0, atol=5e-15)
    electric = field[:, 0, 0]
    sim = sim.replace(external_E=electric)
    expected = np.concatenate([matrices[0] @ ((electric + np.roll(electric, 1, axis=0)) / 2),
                               matrices[0] @ magnetic], axis=1)
    np.testing.assert_allclose(sim.external_fields_at(position), expected, rtol=0, atol=5e-15)
    np.testing.assert_allclose(sim._fields_at(position, jnp.zeros((G, 3)), jnp.zeros((G, 3)), jnp.zeros(G)),
                               expected, rtol=0, atol=5e-15)


def test_default_shape_keeps_the_quadratic_kernel_arithmetic():
    x, q = jnp.asarray([-.5, -.499, .003, .498, .5]), jnp.asarray([1., -2., 3., -4., 5.])
    index, weights = s2_weights(x, x0, dx)
    legacy_deposit = jnp.zeros(G).at[index % G].add(weights * (q / dx)[:, None])
    np.testing.assert_array_equal(deposit(x, q, x0, dx, G, (0, 0)), legacy_deposit)
    field = jnp.arange(G * 6, dtype=float).reshape(G, 6)
    ghosts = jnp.concatenate([field[-1:], field, field[:1]])
    index, weights = s2_weights(x, x0 - dx, dx)
    legacy_gather = jnp.einsum('nk,nkc->nc', weights, ghosts[jnp.clip(index, 0, G + 1)])
    np.testing.assert_array_equal(with_ghosts(field, (0, 0)), ghosts)
    np.testing.assert_array_equal(gather(ghosts, x, x0, dx), legacy_gather)


@pytest.mark.parametrize("order", [2, 5])
def test_periodic_shape_current_gauss_and_translated_force_identity(order):
    rng = np.random.default_rng(17)
    x = rng.uniform(-L / 2, L / 2, 64)
    q = jnp.asarray(np.tile([1e-3, -1e-3], 32))
    v = jnp.asarray(rng.normal(size=64)) * dx / 1e-9
    dt = 1e-9
    for fraction in (0., .25, .5):
        old = jnp.asarray((x + fraction * dx + L / 2) % L - L / 2)
        new = (old + dt * v + L / 2) % L - L / 2
        rho0 = deposit(old, q, x0, dx, G, (0, 0), order)
        rho1 = deposit(new, q, x0, dx, G, (0, 0), order)
        mean = jnp.sum(q * v) / L
        current = current_from_continuity(rho0, rho1, dt, dx, mean, (0, 0))
        source_scale = float(jnp.max(abs((rho1 - rho0) / dt)))
        continuity = (rho1 - rho0) / dt + (current - jnp.roll(current, 1)) / dx
        np.testing.assert_allclose(continuity, 0, rtol=0, atol=2e-13 * source_scale)
        np.testing.assert_allclose(jnp.mean(current), mean, rtol=0, atol=2e-13 * source_scale * L)
        field = E_x_from_rho(rho0, dx, (0, 0))
        advanced = field - dt * current / epsilon_0
        gauss = (advanced - jnp.roll(advanced, 1)) / dx - (rho1 - jnp.mean(rho1)) / epsilon_0
        np.testing.assert_allclose(gauss, 0, rtol=0, atol=3e-13 * float(jnp.max(abs(rho1))) / epsilon_0)
        centred = .5 * (field + jnp.roll(field, 1))
        force = q * gather(with_ghosts(centred[:, None], (0, 0), shape_order=order),
                           old, x0, dx, order)[:, 0]
        np.testing.assert_allclose(jnp.sum(force), 0, rtol=0, atol=2e-13 * float(jnp.sum(abs(force))))


@pytest.mark.parametrize("order", [2, 5])
def test_selected_shape_periodic_step_preserves_charge_gauss_and_total_impulse(order):
    omega, count = 1e9, 24
    density = epsilon_0 * mass_electron * omega**2 / elementary_charge**2
    length = 2 * np.pi * c / omega
    x, v = quiet_start(count, length, drift=(.03 * c, 0., 0.))
    x = x.at[:, 0].add(.03 * length * jnp.sin(2 * jnp.pi * x[:, 0] / length))
    electrons = Species.electrons(count, density).replace(x=x, v=v)
    ions = Species("positive", count, 1, mass_electron, density, x=x + .02 * length, v=-v)
    sim = Simulation(Domain(length, 16, time_step=.01 / omega), (electrons, ions),
                     Solver(shape_order=order, relativistic=True))
    step = jax.jit(sim._explicit_step)
    for fraction in (0., .25, .5):
        populations = tuple(sp.replace(x=sp.x.at[:, 0].add(fraction * sim.domain.dx)) for sp in sim.species)
        initial, extra = sim.replace(species=populations).initial_state(jax.random.PRNGKey(0))
        mass, charge = extra
        physical = np.concatenate([np.asarray(sp.x[:, 0]) for sp in populations]) / length
        expected = (charge * initial.w) @ periodic_spline_matrix(physical, sim.domain.cells, order) / sim.domain.dx
        np.testing.assert_allclose(initial.rho, expected, rtol=0, atol=2e-13 * density * elementary_charge)
        final, _ = step(initial, extra)
        momentum = jnp.sum(mass[:, None] * initial.w[:, None] * (final.u - initial.u), axis=0)
        np.testing.assert_allclose(momentum, 0, rtol=0, atol=2e-13 * density * mass_electron * c * length)
        np.testing.assert_allclose(sim.domain.dx * jnp.sum(final.rho), jnp.sum(charge * final.w),
                                   rtol=0, atol=2e-13 * density * elementary_charge * length)
        gauss = ((final.E[:, 0] - jnp.roll(final.E[:, 0], 1)) / sim.domain.dx
                 - (final.rho - jnp.mean(final.rho)) / epsilon_0)
        np.testing.assert_allclose(gauss, 0, rtol=0, atol=2e-13 * density * elementary_charge / epsilon_0)
        np.testing.assert_array_equal(final.u[:, 1:], 0)


def test_quintic_rejects_unsupported_solver_and_walls_and_checks_restart_shape(tmp_path):
    particle = Species.electrons(4, 1)
    Simulation(Domain(), (Species.electrons(8, 1),), Solver(shape_order=5, algorithm='implicit'))
    for solver, domain in ((Solver(shape_order=5), Domain(particle_bc='reflective')),
                           (Solver(shape_order=5), Domain(field_bc='reflective')),
                           (Solver(shape_order=5, algorithm='implicit'), Domain(field_bc='reflective'))):
        with pytest.raises(ValueError, match='shape_order=5'):
            Simulation(domain, (particle,), solver)
    with pytest.raises(ValueError, match='shape_order'):
        Solver(shape_order=3)
    with pytest.raises(ValueError, match='shape_order'):
        shape_weights(jnp.zeros(1), x0, dx, 3)
    with pytest.raises(ValueError, match='periodic'):
        with_ghosts(jnp.zeros((G, 3)), (1, 1), shape_order=5)
    sim = Simulation(Domain(cells=4), (particle,))
    state, _ = sim.initial_state(jax.random.PRNGKey(0))
    path = save_state(tmp_path / 'quadratic', state, sim)
    quintic = sim.replace(solver=Solver(shape_order=5))
    archive = dict(np.load(path))
    assert int(archive['format']) == 1 and 'shape_order' not in archive
    np.testing.assert_array_equal(load_state(path, sim).rho, state.rho)
    with pytest.raises(ValueError, match='shape_order'):
        load_state(path, quintic)
    state5, _ = quintic.initial_state(jax.random.PRNGKey(0))
    save_state(path, state5, quintic)
    archive = dict(np.load(path))
    assert int(archive['format']) == 2 and int(archive['shape_order']) == 5
    with pytest.raises(ValueError, match='shape_order'):
        load_state(path, sim)
    np.testing.assert_array_equal(load_state(path, quintic).rho, state5.rho)
    for malformed in ({key: value for key, value in archive.items() if key != 'shape_order'},
                      {**archive, 'shape_order': np.asarray(2)}):
        np.savez(path, **malformed)
        with pytest.raises(ValueError, match='format 2 requires stored shape_order=5'):
            load_state(path)  # validation also applies without a supplied simulation


def test_shape_function_partition_of_unity_and_charge_conservation():
    """The three quadratic-spline weights sum to one at every position, so the
    deposited charge equals the particle charge for periodic and reflective walls.
    The weights of a sum of three terms of order one differ from one by a few
    units of the double-precision epsilon, hence 1e-15."""
    rng = np.random.default_rng(11)
    x = jnp.array(rng.uniform(-L / 2, L / 2, 2000))
    _, w = s2_weights(x, x0, dx)
    assert np.allclose(np.asarray(w).sum(1), 1.0, rtol=0, atol=1e-15)
    q = jnp.array(rng.choice([-1e-3, 1e-3], x.shape[0]))
    for bc in ((0, 0), (1, 1)):
        # 6000 weights of 1e-3 summed: round-off of a few parts in 1e16 of sum |q| = 2
        assert abs(float(deposit(x, q, x0, dx, G, bc).sum() * dx - q.sum())) < 1e-15


def test_an_absorbing_wall_keeps_exactly_the_part_of_the_cloud_inside_the_box():
    """A particle closer than one and a half cells to a wall has a cloud that reaches
    past it. An absorbing wall drops the weights that fall on cells beyond the wall
    and keeps the rest, so the charge the particle deposits is its charge times the
    sum of the quadratic-spline weights on the cells of the box -- one half for a
    particle on the wall, all of it from one cell in. A reflective wall clamps the
    stencil back into the box and keeps all of it."""
    offsets = dx * np.array([1e-3, 0.25, 0.5, 0.75, 0.999, 1.25, 1.5])
    positions = np.concatenate([-L / 2 + offsets, L / 2 - offsets])
    centres = x0 + dx * np.arange(G)
    kept = quadratic_spline((positions[:, None] - centres[None, :]) / dx).sum(axis=1)
    assert kept.min() < 0.51 and kept.max() == 1.0          # the offsets span both cases

    def deposited(bc):
        one = jax.vmap(lambda xp: deposit(xp[None], jnp.ones(1), x0, dx, G, bc).sum() * dx)
        return np.asarray(one(jnp.asarray(positions)))

    # three weights divided by dx, scattered and summed over the grid, times dx: a few
    # units of epsilon on a fraction of order one
    assert np.allclose(deposited((2, 2)), kept, rtol=0, atol=1e-14)
    assert np.allclose(deposited((1, 1)), 1.0, rtol=0, atol=1e-14)


def test_gather_reproduces_a_constant_and_a_linear_field():
    """Interpolation with the same spline reproduces constants exactly and linear
    fields to round-off away from the walls: three weighted terms of order one."""
    rng = np.random.default_rng(12)
    x = jnp.array(rng.uniform(-L / 2 + 2 * dx, L / 2 - 2 * dx, 500))
    centres = x0 + jnp.arange(G) * dx
    F = jnp.stack([jnp.full(G, 3.0), 2.0 * centres, jnp.zeros(G)], axis=1)
    got = gather(with_ghosts(F, (1, 1)), x, x0, dx)
    assert np.allclose(np.asarray(got[:, 0]), 3.0, rtol=0, atol=4e-15)
    assert np.allclose(np.asarray(got[:, 1]), 2.0 * np.asarray(x), rtol=0, atol=4e-15)


@pytest.mark.parametrize("bc", [(0, 0), (1, 1)])
def test_current_satisfies_the_discrete_continuity_equation(bc):
    """(rho_new - rho_old)/dt + (J_{i+1/2} - J_{i-1/2})/dx = 0 in every cell, to
    round-off, for displacements of a fraction of a cell (Villasenor and Buneman 1992)."""
    rng = np.random.default_rng(13)
    n = 1000
    x = jnp.array(rng.uniform(-L / 2, L / 2, n))
    q = jnp.array(rng.choice([-1e-3, 1e-3], n))
    v = jnp.array(rng.normal(0, 0.3, n)) * dx / 1e-9
    dt = 1e-9
    x_new = wrap_positions(jnp.stack([x + v * dt, 0 * x, 0 * x], 1), jnp.ones_like(x), (L, L, L), bc, dx)[:, 0]
    rho_old, rho_new = deposit(x, q, x0, dx, G, bc), deposit(x_new, q, x0, dx, G, bc)
    J = current_from_continuity(rho_old, rho_new, dt, dx, jnp.sum(q * v) / L, bc)
    J_left = jnp.roll(J, 1) if bc == (0, 0) else jnp.concatenate([jnp.zeros(1), J[:-1]])
    residual = (rho_new - rho_old) / dt + (J - J_left) / dx
    assert float(jnp.abs(residual).max()) < 1e-12 * float(jnp.abs((rho_new - rho_old) / dt).max())
    if bc == (0, 0):
        assert abs(float(J.mean() - jnp.sum(q * v) / L)) < 1e-12 * abs(float(J.mean()))


def test_gauss_solver_matches_the_finite_difference_divergence():
    """The electrostatic solver returns the field whose backward difference is
    rho/epsilon_0 exactly, for a neutral periodic box and for a box with walls."""
    rng = np.random.default_rng(14)
    rho = jnp.array(rng.normal(size=G)) * 1e-3
    rho = rho - rho.mean()
    E = E_x_from_rho(rho, dx, (0, 0))
    assert np.allclose(np.asarray((E - jnp.roll(E, 1)) / dx), np.asarray(rho / epsilon_0), rtol=1e-12, atol=0)
    assert abs(float(E.mean())) < 1e-12 * float(jnp.abs(E).max())
    E = E_x_from_rho(rho, dx, (1, 1))
    E_left = jnp.concatenate([jnp.zeros(1), E[:-1]])
    assert np.allclose(np.asarray((E - E_left) / dx), np.asarray(rho / epsilon_0), rtol=1e-12, atol=0)


def test_boris_pusher_rotation_and_kicks():
    """In a pure magnetic field the Boris step conserves speed to round-off and
    rotates the velocity by exactly 2 arctan(q B dt / 2 m) (Boris 1970; Qin et
    al. 2013); in a pure electric field it is the exact kick q E dt / m."""
    qm = -elementary_charge / mass_electron
    B = jnp.array([[0.0, 0.0, 0.1]])
    v = jnp.array([[1e6, 0.0, 0.0]])
    dt = 5e-12
    v_new = boris(v, jnp.zeros((1, 3)), B, jnp.array([[qm]]), dt)
    assert abs(float(jnp.linalg.norm(v_new) / jnp.linalg.norm(v)) - 1) < 1e-14
    angle = float(jnp.arctan2(v_new[0, 1], v_new[0, 0]))
    assert abs(abs(angle) - 2 * np.arctan(abs(qm) * 0.1 * dt / 2)) < 1e-14
    E = jnp.array([[2e5, -1e5, 3e5]])
    v_new = boris(v, E, jnp.zeros((1, 3)), jnp.array([[qm]]), dt)
    assert np.allclose(np.asarray(v_new), np.asarray(v + qm * E * dt), rtol=1e-14, atol=0)


def test_relativistic_pusher_conserves_energy_in_a_magnetic_field():
    """The relativistic Boris step, which advances u = gamma v, conserves the Lorentz
    factor in a pure magnetic field, and in a pure electric field reproduces
    p = p0 + q E t. The momenta are of order 1e-22 kg m/s, so they are compared relative
    to their own magnitude; the kick is 8 % of p0, and 1e-12 of p is far above the
    round-off of one step, which no longer converts between v and u."""
    q, m = -elementary_charge, mass_electron
    v = jnp.array([[0.6 * c, 0.3 * c, 0.0]])
    u = lorentz_factor(v)[:, None] * v
    u_new = boris_relativistic(u, jnp.zeros((1, 3)), jnp.array([[0.0, 0.0, 1.0]]), q / m, 1e-12)
    assert abs(float(jnp.linalg.norm(u_new) / jnp.linalg.norm(u)) - 1) < 1e-13
    E = jnp.array([[1e8, 0.0, 0.0]])
    u_new = boris_relativistic(u, E, jnp.zeros((1, 3)), q / m, 1e-12)
    p_expected = np.asarray(m * u[0] + q * E[0] * 1e-12)
    p_new = np.asarray(m * u_new[0])
    assert np.all(np.abs(p_new - p_expected) <= 1e-12 * np.abs(p_expected).max())


def test_filter_transfer_function():
    """The compensated binomial filter has the analytic response
    H(k) = [a + (1-a) cos(k dx)]^p [a_c + (1-a_c) cos(k dx)] with a_c = 1 + p(1-a),
    which is 1 - O(k^4) at long wavelength and vanishes at the Nyquist wavenumber.
    The signal is of order one and four passes of a three-point stencil lose a
    few parts in 1e16 of it."""
    a, p = 0.5, 3
    for mode in (1, 5, 16):
        theta = 2 * np.pi * mode / G
        f = jnp.cos(theta * jnp.arange(G))
        H = (a + (1 - a) * np.cos(theta)) ** p * (1 + p * (1 - a) - p * (1 - a) * np.cos(theta))
        assert np.allclose(np.asarray(smooth(f, p, a, (1,), (0, 0))), H * np.asarray(f), rtol=0, atol=1e-14)


def field_energy(E, B, h):
    return float(0.5 * epsilon_0 * jnp.sum(E ** 2) * h + 0.5 / mu_0 * jnp.sum(B ** 2) * h)


def maxwell_matrix(n, bc):
    """Independent dimensionless generator for (E_y, c B_z), including wall ghosts."""
    def derivative(y):
        e, b = y[:n], y[n:]
        left = e[-1] if bc[0] == 0 else e[0] if bc[0] == 1 else -2 * b[0] - e[0]
        right = b[0] if bc[1] == 0 else b[-1] if bc[1] == 1 else 2 * e[-1] - b[-1]
        return np.r_[-np.diff(np.r_[b, right]), -np.diff(np.r_[left, e])]
    return np.column_stack([derivative(y) for y in np.eye(2 * n)])


@pytest.mark.parametrize("n, mode, courant", [
    (4, 0, 4.5), (4, 2, 1.), (9, 4, 1.1), (9, 1, 20.), (32, 16, 4.5), (32, 3, .5)])
def test_implicit_maxwell_matches_the_discrete_fourier_CN_solution(n, mode, courant):
    """The two-by-two Cayley transform uses the staggered difference symbol, not ik.
    It tests both polarizations, zero/odd/Nyquist modes and prescribed current."""
    theta = 2 * np.pi * mode / n
    phase = np.exp(1j * theta * np.arange(n))
    E = np.zeros((n, 3))
    B = np.zeros_like(E)
    forcing = np.zeros_like(E)
    dt, h = courant / (n * c), 1 / n
    operator = np.array([[0, -(np.exp(1j * theta) - 1)], [-(1 - np.exp(-1j * theta)), 0]])
    expected_E, expected_B = E.copy(), B.copy()
    for ei, bi, sign in [(1, 2, 1), (2, 1, -1)]:
        amplitudes = np.array([.7 + .2j * ei, -.3 + .1j * bi])
        current = .2 - .15j * ei
        E[:, ei] = np.real(amplitudes[0] * phase)
        B[:, bi] = sign * np.real(amplitudes[1] * phase) / c
        forcing[:, ei] = np.real(current * phase)
        evolved = np.linalg.solve(np.eye(2) - courant * operator / 2,
                                  (np.eye(2) + courant * operator / 2) @ amplitudes - [current, 0])
        expected_E[:, ei] = np.real(evolved[0] * phase)
        expected_B[:, bi] = sign * np.real(evolved[1] * phase) / c
    E[:, 0], B[:, 0], forcing[:, 0] = .4, .3 / c, .1
    expected_E[:, 0], expected_B[:, 0] = .3, .3 / c
    actual_E, actual_B = _implicit_fields(jnp.asarray(E), jnp.asarray(B),
                                          jnp.asarray(forcing) * epsilon_0 / dt, dt, h, (0, 0))
    assert np.allclose(actual_E, expected_E, rtol=0, atol=2e-14)
    assert np.allclose(c * actual_B, c * expected_B, rtol=0, atol=2e-14)


@pytest.mark.parametrize("bc", [(1, 1), (2, 2), (1, 2), (2, 1)])
@pytest.mark.parametrize("n", [4, 9])
@pytest.mark.parametrize("courant", [.5, 4.5, 20.])
def test_implicit_maxwell_wall_equations_and_boundary_work(bc, n, courant):
    """Dense CN is independent of the tridiagonal elimination. Wall work is retained,
    rather than assuming that every nonperiodic ghost conserves the stored energy."""
    rng = np.random.default_rng(23)
    E, scaled_B, forcing = rng.normal(size=(3, n, 3))
    A = maxwell_matrix(n, bc)
    dt, h = courant / (n * c), 1 / n
    actual_E, actual_B = _implicit_fields(jnp.asarray(E), jnp.asarray(scaled_B / c),
                                          jnp.asarray(forcing) * epsilon_0 / dt, dt, h, bc)
    actual_E, actual_B = np.asarray(actual_E), c * np.asarray(actual_B)
    assert np.allclose(actual_E[:, 0], E[:, 0] - forcing[:, 0], rtol=0, atol=1e-14)
    assert np.array_equal(actual_B[:, 0], scaled_B[:, 0])
    for ei, bi, sign in [(1, 2, 1), (2, 1, -1)]:
        y = np.r_[E[:, ei], sign * scaled_B[:, bi]]
        new = np.r_[actual_E[:, ei], sign * actual_B[:, bi]]
        source = np.r_[forcing[:, ei], np.zeros(n)]
        expected = np.linalg.solve(np.eye(2 * n) - courant * A / 2,
                                   (np.eye(2 * n) + courant * A / 2) @ y - source)
        assert np.allclose(new, expected, rtol=0, atol=3e-14)
        midpoint = .5 * (y + new)
        assert np.max(np.abs(new - y - courant * A @ midpoint + source)) < 8e-14
        energy_change = .5 * (np.dot(new, new) - np.dot(y, y))
        work = np.dot(midpoint, courant * A @ midpoint - source)
        assert energy_change == pytest.approx(work, abs=1e-12, rel=0)


@pytest.mark.parametrize("bc", [(0, 0), (1, 1), (2, 2), (1, 2), (2, 1)])
def test_implicit_maxwell_time_step_gradient(bc):
    """Differentiate the independently assembled CN equation analytically, with a
    fixed current rate; both JVP and VJP must agree, including the radiating rows."""
    n, h = 9, 1 / 9
    rng = np.random.default_rng(81)
    y, rate, weights = rng.normal(size=(3, 2 * n))
    rate[n:] = 0
    A, courant = maxwell_matrix(n, bc), 1.1
    new = np.linalg.solve(np.eye(2 * n) - courant * A / 2,
                          (np.eye(2 * n) + courant * A / 2) @ y - courant * rate)
    derivative = np.linalg.solve(np.eye(2 * n) - courant * A / 2, .5 * A @ (new + y) - rate)
    E = jnp.zeros((n, 3)).at[:, 1].set(y[:n])
    B = jnp.zeros_like(E).at[:, 2].set(y[n:] / c)
    J = jnp.zeros_like(E).at[:, 1].set(epsilon_0 * c / h * rate[:n])

    def objective(C):
        e, b = _implicit_fields(E, B, J, C * h / c, h, bc)
        return jnp.dot(jnp.asarray(weights), jnp.concatenate([e[:, 1], c * b[:, 2]]))

    expected = np.dot(weights, derivative)
    assert float(jax.grad(objective)(courant)) == pytest.approx(expected, abs=3e-13, rel=0)
    assert float(jax.jacfwd(objective)(courant)) == pytest.approx(expected, abs=3e-13, rel=0)
    finite_difference = (float(objective(courant + 1e-5)) - float(objective(courant - 1e-5))) / 2e-5
    assert finite_difference == pytest.approx(expected, abs=5e-9, rel=0)


def test_implicit_vacuum_time_and_gradient_refinement():
    """At a fixed grid compare with the exact semi-discrete wave, separating time
    error from spatial dispersion. Both the field and its time derivative converge at order two."""
    n, mode, total = 32, 3, 16.
    theta, h = 2 * np.pi * mode / n, 1 / n
    profile = jnp.cos(theta * jnp.arange(n))
    magnetic = jnp.sin(theta * (jnp.arange(n) - .5))
    E = jnp.zeros((n, 3)).at[:, 1].set(profile)
    B = jnp.zeros_like(E)
    frequency = 2 * np.sin(theta / 2)
    expected = np.array([np.cos(frequency * total), np.sin(frequency * total)])
    expected_derivative = frequency * np.array([-np.sin(frequency * total), np.cos(frequency * total)])
    errors, derivative_errors = [], []
    for steps in [20, 40, 80, 160]:
        def amplitudes(t):
            def step(_, fields):
                return _implicit_fields(*fields, jnp.zeros_like(E), (t / steps) * h / c, h, (0, 0))
            e, b = lax.fori_loop(0, steps, step, (E, B))
            return jnp.array([jnp.dot(e[:, 1], profile) / jnp.dot(profile, profile),
                              c * jnp.dot(b[:, 2], magnetic) / jnp.dot(magnetic, magnetic)])
        value = np.asarray(jax.jit(amplitudes)(total))
        derivative = np.asarray(jax.jit(jax.jacfwd(amplitudes))(total))
        reverse = np.asarray(jax.jit(jax.jacrev(amplitudes))(total))
        assert np.allclose(derivative, reverse, rtol=0, atol=1e-13)
        assert np.dot(value, value) == pytest.approx(1, abs=2e-13, rel=0)
        errors.append(np.linalg.norm(value - expected))
        derivative_errors.append(np.linalg.norm(derivative - expected_derivative))
    assert np.all((np.asarray(errors[:-1]) / errors[1:] > 3.8)
                  & (np.asarray(errors[:-1]) / errors[1:] < 4.2))
    assert np.all((np.asarray(derivative_errors[:-1]) / derivative_errors[1:] > 3.8)
                  & (np.asarray(derivative_errors[:-1]) / derivative_errors[1:] < 4.2))


@pytest.mark.parametrize("bc", [(0, 0), (2, 2)])
def test_implicit_maxwell_keeps_single_precision(bc):
    n, h, courant = 9, 1 / 9, .75
    E = jnp.zeros((n, 3), dtype=jnp.float32).at[:, 1].set(jnp.cos(2 * jnp.pi * jnp.arange(n) / n))
    B = jnp.zeros_like(E)
    e, b = _implicit_fields(E, B, jnp.zeros_like(E), courant * h / c, h, bc)
    assert e.dtype == b.dtype == jnp.float32
    y = np.r_[np.asarray(E[:, 1]), np.zeros(n)]
    A = maxwell_matrix(n, bc)
    expected = np.linalg.solve(np.eye(2 * n) - courant * A / 2, (np.eye(2 * n) + courant * A / 2) @ y)
    assert np.allclose(np.r_[np.asarray(e[:, 1]), c * np.asarray(b[:, 2])], expected, rtol=0, atol=5e-7)


@pytest.mark.parametrize("courant", [0.5, 1.0])
def test_vacuum_light_wave(courant):
    """At C=1 a right-moving discrete eigenstate translates exactly: its E face
    is the average of neighboring cB centres, rather than a co-located copy.
    Seventeen steps avoid the half-box recurrence that hides an impedance error.
    Below C=1 the physical Gaussian is dispersive and the test tracks its peak."""
    n = 128
    h = 1.0 / n
    dt = courant * h / c
    xs = jnp.arange(n) * h
    profile = jnp.exp(-((xs + h / 2 - 0.5) / 0.05) ** 2)
    electric = (.5 * (profile + jnp.roll(profile, -1)) if courant == 1.0 else
                jnp.exp(-((xs + h - 0.5) / 0.05) ** 2))
    E = jnp.zeros((n, 3)).at[:, 1].set(electric)
    B = jnp.zeros((n, 3)).at[:, 2].set(profile / c)
    e0 = field_energy(E, B, h)
    steps = 17
    for _ in range(steps):
        E, B = half_step_fields(E, B, jnp.zeros((n, 3)), dt / 2, h, (0, 0), True)
        E, B = half_step_fields(E, B, jnp.zeros((n, 3)), dt / 2, h, (0, 0), False)
    assert abs(field_energy(E, B, h) / e0 - 1) < (1e-12 if courant == 1.0 else 1e-3)
    shift = int(round(courant * steps))
    if courant == 1.0:
        assert np.allclose(np.asarray(E[:, 1]), np.roll(np.asarray(electric), shift), rtol=0, atol=1e-13)
        assert np.allclose(c * np.asarray(B[:, 2]), np.roll(np.asarray(profile), shift), rtol=0, atol=1e-13)
    else:
        peak = float(xs[jnp.argmax(E[:, 1])])
        assert abs(peak + h - (0.5 + c * steps * dt)) < 2 * h


def test_particle_boundaries():
    """Periodic walls wrap positions exactly. Reflective walls mirror the position
    and multiply the normal velocity by -restitution, each wall by its own
    coefficient. An absorbing wall keeps the reflected fraction of each particle's
    weight, bouncing it the same way, and stops and parks a particle with nothing
    reflected -- for good: a parked particle is never brought back. Positions of
    order one are mirrored with one subtraction, so 1e-15 is round-off."""
    def same(actual, expected):
        return np.allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-15)

    x = jnp.array([[-0.6, 0.0, 0.0], [0.7, 0.0, 0.0], [0.1, 0.0, 0.0]])
    v = jnp.array([[-1.0, 2.0, 0.0], [3.0, 0.0, 1.0], [0.5, 0.0, 0.0]])
    w, qm, nothing = jnp.ones(3), jnp.full(3, 2.0), (jnp.zeros(3), jnp.zeros(3))
    xp, vp, _, _, _ = apply_particle_bc(x, v, w, qm, (L, L, L), (0, 0), (1.0, 1.0), nothing, dx)
    assert same(xp[:, 0], [0.4, -0.3, 0.1]) and same(vp, v)
    xr, vr, wr, _, _ = apply_particle_bc(x, v, w, qm, (L, L, L), (1, 1), (0.5, 0.25), nothing, dx)
    assert same(xr[:, 0], [-0.4, 0.3, 0.1]) and same(wr, 1.0)
    assert same(vr[:, 0], [0.5, -0.75, 0.5])
    assert same(vr[:, 1:], v[:, 1:])
    xa, va, wa, qma, _ = apply_particle_bc(x, v, w, qm, (L, L, L), (2, 2), (1.0, 1.0), nothing, dx)
    assert same(wa, [0.0, 0.0, 1.0]) and same(va[:2], 0.0)
    assert float(xa[0, 0]) < -L / 2 and float(xa[1, 0]) > L / 2 and same(qma, [0.0, 0.0, 2.0])

    # 30 % of the left particle and 60 % of the right one come back, bounced
    reflect = (jnp.full(3, 0.3), jnp.full(3, 0.6))
    xm, vm, wm, qmm, _ = apply_particle_bc(x, v, w, qm, (L, L, L), (2, 2), (0.5, 1.0), reflect, dx)
    assert same(wm, [0.3, 0.6, 1.0]) and same(xm[:, 0], [-0.4, 0.3, 0.1])
    assert same(vm[:, 0], [0.5, -3.0, 0.5]) and same(qmm, 2.0)
    xs, _, ws, _, _ = apply_particle_bc(xa, va, wa, qma, (L, L, L), (2, 2), (1.0, 1.0), reflect, dx)
    assert same(xs, xa) and same(ws, wa)

    # reconstructed positions: what still has weight was reflected and is mirrored
    xw = wrap_positions(x, jnp.array([1.0, 0.0, 1.0]), (L, L, L), (2, 2), dx)
    assert abs(float(xw[0, 0]) + 0.4) < 1e-15 and float(xw[1, 0]) > L / 2
