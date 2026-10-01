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

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mu_0,
                       elementary_charge, load_state, mass_electron, quiet_start, save_state)
from jaxincell import speed_of_light as c
from jaxincell._core import (E_x_from_rho, apply_particle_bc, boris, boris_relativistic,
                             current_from_continuity, deposit, gather, gather_xyz, half_step_fields,
                             s2_weights, shape_weights, smooth, with_ghosts, wrap_positions)

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
    for solver, domain in ((Solver(shape_order=5, algorithm='implicit'), Domain()),
                           (Solver(shape_order=5), Domain(particle_bc='reflective')),
                           (Solver(shape_order=5), Domain(field_bc='reflective'))):
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


@pytest.mark.parametrize("courant", [0.5, 1.0])
def test_vacuum_light_wave(courant):
    """A Gaussian pulse propagates at c on the Yee grid. At Courant number one
    the scheme is exact (the magic time step) and reproduces the initial profile
    shifted by a whole number of cells, to the round-off of 64 steps on a profile
    of unit height; below it the pulse is no longer an eigenmode of the discrete
    operator, so the energy wanders by a part in 1e4 and the test tracks the peak
    instead."""
    n = 128
    h = 1.0 / n
    dt = courant * h / c
    xs = jnp.arange(n) * h
    profile = jnp.exp(-((xs + h / 2 - 0.5) / 0.05) ** 2)
    E = jnp.zeros((n, 3)).at[:, 1].set(profile)
    B = jnp.zeros((n, 3)).at[:, 2].set(profile / c)
    e0 = field_energy(E, B, h)
    steps = 64
    for _ in range(steps):
        E, B = half_step_fields(E, B, jnp.zeros((n, 3)), dt / 2, h, (0, 0), True)
        E, B = half_step_fields(E, B, jnp.zeros((n, 3)), dt / 2, h, (0, 0), False)
    assert abs(field_energy(E, B, h) / e0 - 1) < (1e-12 if courant == 1.0 else 1e-3)
    shift = int(round(courant * steps))
    if courant == 1.0:
        assert np.allclose(np.asarray(E[:, 1]), np.roll(np.asarray(profile), shift), rtol=0, atol=1e-13)
    else:
        peak = float(xs[jnp.argmax(E[:, 1])])
        assert abs(peak + h / 2 - (0.5 + c * steps * dt)) < 2 * h


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
