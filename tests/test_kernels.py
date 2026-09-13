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

from jaxincell import epsilon_0, mu_0, elementary_charge, mass_electron
from jaxincell import speed_of_light as c
from jaxincell._core import (E_x_from_rho, apply_particle_bc, boris, boris_relativistic,
                             current_from_continuity, deposit, gather, half_step_fields,
                             s2_weights, smooth, with_ghosts, wrap_positions)

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
    xp, vp, _, _ = apply_particle_bc(x, v, w, qm, (L, L, L), (0, 0), (1.0, 1.0), nothing, dx)
    assert same(xp[:, 0], [0.4, -0.3, 0.1]) and same(vp, v)
    xr, vr, wr, _ = apply_particle_bc(x, v, w, qm, (L, L, L), (1, 1), (0.5, 0.25), nothing, dx)
    assert same(xr[:, 0], [-0.4, 0.3, 0.1]) and same(wr, 1.0)
    assert same(vr[:, 0], [0.5, -0.75, 0.5])
    assert same(vr[:, 1:], v[:, 1:])
    xa, va, wa, qma = apply_particle_bc(x, v, w, qm, (L, L, L), (2, 2), (1.0, 1.0), nothing, dx)
    assert same(wa, [0.0, 0.0, 1.0]) and same(va[:2], 0.0)
    assert float(xa[0, 0]) < -L / 2 and float(xa[1, 0]) > L / 2 and same(qma, [0.0, 0.0, 2.0])

    # 30 % of the left particle and 60 % of the right one come back, bounced
    reflect = (jnp.full(3, 0.3), jnp.full(3, 0.6))
    xm, vm, wm, qmm = apply_particle_bc(x, v, w, qm, (L, L, L), (2, 2), (0.5, 1.0), reflect, dx)
    assert same(wm, [0.3, 0.6, 1.0]) and same(xm[:, 0], [-0.4, 0.3, 0.1])
    assert same(vm[:, 0], [0.5, -3.0, 0.5]) and same(qmm, 2.0)
    xs, _, ws, _ = apply_particle_bc(xa, va, wa, qma, (L, L, L), (2, 2), (1.0, 1.0), reflect, dx)
    assert same(xs, xa) and same(ws, wa)

    # reconstructed positions: what still has weight was reflected and is mirrored
    xw = wrap_positions(x, jnp.array([1.0, 0.0, 1.0]), (L, L, L), (2, 2), dx)
    assert abs(float(xw[0, 0]) + 0.4) < 1e-15 and float(xw[1, 0]) > L / 2
