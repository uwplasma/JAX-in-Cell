"""Exact properties of the numerical kernels: conservation to round-off,
agreement with closed-form results, and the analytic response of the filter."""
import numpy as np
import jax.numpy as jnp
import pytest

from jaxincell import epsilon_0, mu_0, elementary_charge, mass_electron
from jaxincell import speed_of_light as c
from jaxincell._core import (E_x_from_rho, apply_particle_bc, boris, boris_relativistic,
                             current_from_continuity, deposit, gather, half_step_fields,
                             s2_weights, smooth, wrap_positions)

L, G = 1.0, 32
dx = L / G
x0 = -L / 2 + dx / 2
rng = np.random.default_rng(1)


def test_shape_function_partition_of_unity_and_charge_conservation():
    """The three quadratic-spline weights sum to one at every position, so the
    deposited charge equals the particle charge for periodic and reflective walls;
    absorbing walls drop only the part of a cloud that lies beyond the wall."""
    x = jnp.array(rng.uniform(-L / 2, L / 2, 2000))
    _, w = s2_weights(x, x0, dx)
    assert np.allclose(np.asarray(w).sum(1), 1.0, atol=1e-14)
    q = jnp.array(rng.choice([-1e-3, 1e-3], x.shape[0]))
    for bc in ((0, 0), (1, 1)):
        assert abs(float(deposit(x, q, x0, dx, G, bc).sum() * dx - q.sum())) < 1e-15
    inside = np.abs(np.asarray(x)) < L / 2 - 1.5 * dx
    rho = deposit(x, q, x0, dx, G, (2, 2))
    assert abs(float(rho.sum() * dx - q[inside].sum())) <= float(np.abs(q[~inside]).sum())


def test_gather_reproduces_a_constant_and_a_linear_field():
    """Interpolation with the same spline reproduces constants exactly and
    linear fields to round-off away from the walls."""
    x = jnp.array(rng.uniform(-L / 2 + 2 * dx, L / 2 - 2 * dx, 500))
    faces = x0 + dx / 2 + jnp.arange(G) * dx
    F = jnp.stack([jnp.full(G, 3.0), 2.0 * faces, jnp.zeros(G)], axis=1)
    got = gather(F, x, x0 + dx / 2, dx, (1, 1))
    assert np.allclose(np.asarray(got[:, 0]), 3.0, atol=1e-13)
    assert np.allclose(np.asarray(got[:, 1]), 2.0 * np.asarray(x), atol=1e-12)


@pytest.mark.parametrize("bc", [(0, 0), (1, 1)])
def test_current_satisfies_the_discrete_continuity_equation(bc):
    """(rho_new - rho_old)/dt + (J_{i+1/2} - J_{i-1/2})/dx = 0 in every cell, to
    round-off, for displacements of a fraction of a cell (Villasenor and Buneman 1992)."""
    n = 1000
    x = jnp.array(rng.uniform(-L / 2, L / 2, n))
    q = jnp.array(rng.choice([-1e-3, 1e-3], n))
    v = jnp.array(rng.normal(0, 0.3, n)) * dx / 1e-9
    dt = 1e-9
    x_new = wrap_positions(jnp.stack([x + v * dt, 0 * x, 0 * x], 1), (L, L, L), bc, dx)[:, 0]
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
    assert np.allclose(np.asarray(v_new), np.asarray(v + qm * E * dt), rtol=1e-14)


def test_relativistic_pusher_conserves_energy_in_a_magnetic_field():
    """The relativistic Boris step conserves the Lorentz factor in a pure
    magnetic field, and in a pure electric field reproduces p = p0 + q E t."""
    q, m = -elementary_charge, mass_electron
    v = jnp.array([[0.6 * c, 0.3 * c, 0.0]])
    gamma = lambda u: 1 / jnp.sqrt(1 - jnp.sum(u ** 2, 1) / c ** 2)
    v_new = boris_relativistic(v, jnp.zeros((1, 3)), jnp.array([[0.0, 0.0, 1.0]]), q, m, 1e-12)
    assert abs(float(gamma(v_new)[0] / gamma(v)[0]) - 1) < 1e-13
    E = jnp.array([[1e8, 0.0, 0.0]])
    v_new = boris_relativistic(v, E, jnp.zeros((1, 3)), q, m, 1e-12)
    p_expected = gamma(v)[0] * m * v[0] + q * E[0] * 1e-12
    assert np.allclose(np.asarray(gamma(v_new)[0] * m * v_new[0]), np.asarray(p_expected), rtol=1e-12)


def test_filter_transfer_function():
    """The compensated binomial filter has the analytic response
    H(k) = [a + (1-a) cos(k dx)]^p [a_c + (1-a_c) cos(k dx)] with a_c = 1 + p(1-a),
    which is 1 - O(k^4) at long wavelength and vanishes at the Nyquist wavenumber."""
    a, p = 0.5, 3
    for mode in (1, 5, 16):
        theta = 2 * np.pi * mode / G
        f = jnp.cos(theta * jnp.arange(G))
        H = (a + (1 - a) * np.cos(theta)) ** p * (1 + p * (1 - a) - p * (1 - a) * np.cos(theta))
        assert np.allclose(np.asarray(smooth(f, p, a, (1,), (0, 0))), H * np.asarray(f), atol=1e-13)


@pytest.mark.parametrize("courant", [0.5, 1.0])
def test_vacuum_light_wave(courant):
    """A Gaussian pulse propagates at c on the Yee grid. At Courant number one
    the scheme is exact (the magic time step) and reproduces the initial profile
    shifted by a whole number of cells; below it the pulse is no longer an
    eigenmode of the discrete operator, so the energy wanders by a part in 1e4
    and the test tracks the peak instead."""
    n = 128
    h = 1.0 / n
    dt = courant * h / c
    xs = jnp.arange(n) * h
    profile = jnp.exp(-((xs + h / 2 - 0.5) / 0.05) ** 2)
    E = jnp.zeros((n, 3)).at[:, 1].set(profile)
    B = jnp.zeros((n, 3)).at[:, 2].set(profile / c)
    energy = lambda E, B: float(0.5 * epsilon_0 * jnp.sum(E ** 2) * h + 0.5 / mu_0 * jnp.sum(B ** 2) * h)
    e0 = energy(E, B)
    steps = 64
    for _ in range(steps):
        E, B = half_step_fields(E, B, jnp.zeros((n, 3)), dt / 2, h, (0, 0), True)
        E, B = half_step_fields(E, B, jnp.zeros((n, 3)), dt / 2, h, (0, 0), False)
    assert abs(energy(E, B) / e0 - 1) < (1e-12 if courant == 1.0 else 1e-3)
    shift = int(round(courant * steps))
    if courant == 1.0:
        assert np.allclose(np.asarray(E[:, 1]), np.roll(np.asarray(profile), shift), atol=1e-12)
    else:
        peak = float(xs[jnp.argmax(E[:, 1])])
        assert abs(peak + h / 2 - (0.5 + c * steps * dt)) < 2 * h


def test_particle_boundaries():
    """Periodic walls wrap positions exactly; reflective walls mirror the position
    and multiply the normal velocity by -restitution; absorbing walls remove the
    charge and stop the particle."""
    x = jnp.array([[-0.6, 0.0, 0.0], [0.7, 0.0, 0.0], [0.1, 0.0, 0.0]])
    v = jnp.array([[-1.0, 2.0, 0.0], [3.0, 0.0, 1.0], [0.5, 0.0, 0.0]])
    q, qm = jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])
    xp, vp, qp, _ = apply_particle_bc(x, v, q, qm, (L, L, L), (0, 0), 1.0, dx)
    assert np.allclose(np.asarray(xp[:, 0]), [0.4, -0.3, 0.1]) and np.allclose(np.asarray(vp), np.asarray(v))
    xr, vr, qr, _ = apply_particle_bc(x, v, q, qm, (L, L, L), (1, 1), 0.5, dx)
    assert np.allclose(np.asarray(xr[:, 0]), [-0.4, 0.3, 0.1])
    assert np.allclose(np.asarray(vr[:, 0]), [0.5, -1.5, 0.5]) and np.allclose(np.asarray(vr[:, 1:]), np.asarray(v[:, 1:]))
    xa, va, qa, qma = apply_particle_bc(x, v, q, qm, (L, L, L), (2, 2), 1.0, dx)
    assert np.allclose(np.asarray(qa), [0.0, 0.0, 1.0]) and np.allclose(np.asarray(va[:2]), 0.0)
    assert float(xa[0, 0]) < -L / 2 and float(xa[1, 0]) > L / 2 and float(qma[2]) == 2.0
