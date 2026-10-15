"""External fields given on an (x, y, z) grid: gathered at a particle's y and z as well as x.

Each test is a single tenuous electron in a prescribed field, checked against the guiding-centre
theory of that field: the grad-B drift at its analytic speed, the uniform field recovered from a
z-varying one as the variation goes to zero, and the magnetic moment kept through a mirror bounce.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge as e_charge, magnetic_moment,
                       mass_electron)
from jaxincell._core import gather, gather_xyz, with_ghosts

B0 = 0.01                                    # T
OMEGA = e_charge * B0 / mass_electron        # electron gyro-frequency, rad/s
SPEED = 1e5                                  # m/s
RHO = SPEED / OMEGA                          # gyro-radius at SPEED, m


def centres(n, length):
    return (np.arange(n) + 0.5) * length / n - length / 2


def electron(x, v):
    """One electron, tenuous enough that its own field is nothing beside the external one."""
    species = Species("electron", 1, -1.0, mass_electron, 1e-3, (0.0, 0.0, 0.0))
    return species.replace(x=jnp.asarray([x], float), v=jnp.asarray([v], float))


def orbit(domain, particle, steps, store_every=1, **external):
    sim = Simulation(domain, [particle], Solver(model="electrostatic"), **external)
    return sim, sim.run(steps, store_every=store_every)


def guiding_centre(sim, out):
    """R = x + m (v x B)/(q B^2), with B the external field at the particle."""
    B = np.asarray(jax.vmap(sim.external_fields_at)(out.x))[:, 0, 3:]
    v = np.asarray(out.v)[:, 0]
    return np.asarray(out.x)[:, 0] - mass_electron / e_charge * np.cross(v, B) / np.sum(B ** 2, axis=1)[:, None]


def test_gather_xyz_is_the_x_gather_when_y_and_z_have_one_cell():
    rng = np.random.default_rng(0)
    field = with_ghosts(jnp.asarray(rng.normal(size=(12, 3))), (0, 0))
    x = jnp.asarray(rng.uniform(-0.5, 0.5, size=(50, 3)))
    flat = gather(field, x[:, 0], -0.5 + 1 / 24, 1 / 12)
    grid = gather_xyz(field[:, None, None, :], x, -0.5 + 1 / 24, 1 / 12, (1.0, 1.0))
    assert np.allclose(grid, flat, rtol=1e-14, atol=1e-15)


def test_a_field_on_a_one_cell_y_z_grid_is_the_flat_field():
    """The same uniform B given flat and on an (x, 1, 1) grid moves the electron identically,
    to round-off, and a z-varying field goes over to the uniform one linearly in its amplitude."""
    cells, Lz, nz = 8, 20 * RHO, 32
    domain = Domain(length=1e-2, cells=cells, time_step=0.1 / OMEGA, length_z=Lz)
    particle = electron((0.0, 0.0, 0.0), (0.0, SPEED, 0.0))
    flat = np.zeros((cells, 3))
    flat[:, 0] = B0
    _, uniform = orbit(domain, particle, 400, external_B=flat)
    _, grid = orbit(domain, particle, 400, external_B=flat[:, None, None, :])
    assert np.max(np.abs(np.asarray(grid.x) - np.asarray(uniform.x))) < 1e-12 * RHO

    def deviation(amplitude):
        B = np.zeros((cells, 1, nz, 3))
        B[..., 0] = B0 * (1 + amplitude * np.sin(2 * np.pi * centres(nz, Lz) / Lz))
        _, varying = orbit(domain, particle, 400, external_B=B)
        return np.max(np.abs(np.asarray(varying.x) - np.asarray(uniform.x)))

    coarse, fine = deviation(1e-2), deviation(1e-3)
    assert 0 < fine < coarse
    assert coarse / fine == pytest.approx(10, rel=0.02)          # first order in the amplitude
    assert coarse < 0.5 * RHO                                    # a phase slip of 1 % of 6 turns


def test_grad_B_drift_at_the_analytic_speed():
    """B = B0 (1 + y/L) along x: the guiding centre drifts along z at
    v_d = m v_perp^2 (B x grad B)/(2 q B^3), which is v_perp (rho/L)/2 in size and, for an
    electron, along -z. The finite-Larmor-radius correction is second order in rho/L, and is
    what is left: a quarter of it when L doubles."""
    def drift_error(L):
        cells, ny, Ly = 8, 64, 1e-2
        domain = Domain(length=1e-2, cells=cells, time_step=0.1 / OMEGA, length_y=Ly)
        B = np.zeros((cells, ny, 1, 3))
        B[..., 0] = B0 * (1 + centres(ny, Ly) / L)[None, :, None]
        # started so that its guiding centre is at y = 0, where B = B0
        sim, out = orbit(domain, electron((0.0, 0.0, RHO), (0.0, SPEED, 0.0)), 3000, external_B=B)
        R, t = guiding_centre(sim, out), np.asarray(out.t)
        analytic = -SPEED ** 2 / (2 * OMEGA * L)
        assert abs(np.polyfit(t, R[:, 1], 1)[0]) < 1e-3 * abs(analytic)   # none along the gradient
        assert abs(np.polyfit(t, R[:, 0], 1)[0]) < 1e-6 * abs(analytic)
        return np.polyfit(t, R[:, 2], 1)[0] / analytic - 1

    near, far = drift_error(20 * RHO), drift_error(40 * RHO)
    assert 0 < near < 5e-3
    assert near / far == pytest.approx(4, rel=0.05)


def test_mirror_bounce_keeps_the_magnetic_moment():
    """A mirror along y, B_y = B0 (1 + y^2/L^2), closed by the radial field -r/2 dB_y/dy that
    div B = 0 asks for. An electron with a pitch angle of 45 degrees at the midplane turns where
    B = 2 B0, y = L, and its magnetic moment holds while the field it sees doubles."""
    cells, ny, nz, L = 16, 128, 16, 50 * RHO
    Lx = Lz = 16 * RHO
    Ly = 4 * L
    domain = Domain(length=Lx, cells=cells, time_step=0.1 / OMEGA, length_y=Ly, length_z=Lz)
    x, y, z = np.meshgrid(centres(cells, Lx), centres(ny, Ly), centres(nz, Lz), indexing="ij")
    B = np.stack([-x * y * B0 / L ** 2, B0 * (1 + y ** 2 / L ** 2), -z * y * B0 / L ** 2], axis=-1)
    v_perp = v_par = SPEED / np.sqrt(2)
    rho = v_perp / OMEGA
    steps = 10 * int(0.22 * np.pi * L / v_par / (0.1 / OMEGA))      # past the turning point and back
    sim, out = orbit(domain, electron((rho, 0.0, 0.0), (0.0, v_par, -v_perp)), steps, store_every=10,
                     external_B=B)
    y_path = np.asarray(out.x)[:, 0, 1]
    strength = np.linalg.norm(np.asarray(jax.vmap(sim.external_fields_at)(out.x))[:, 0, 3:], axis=1)
    assert y_path.max() == pytest.approx(L, rel=1e-3)                # turns where B = 2 B0
    assert strength.max() / strength[0] == pytest.approx(2, rel=0.05)
    assert np.asarray(out.v)[:, 0, 1].min() < -0.9 * v_par           # and comes back
    mu = np.asarray(magnetic_moment(out, sim))[:, 0]
    assert mu[0] == pytest.approx(mass_electron * v_perp ** 2 / (2 * B0), rel=1e-2, abs=0)
    assert np.ptp(mu) / mu.mean() < 1e-4


def test_magnetic_moment_without_a_field_and_relativistic():
    domain = Domain(length=1e-2, cells=8, time_step=0.1 / OMEGA)
    particle = electron((0.0, 0.0, 0.0), (0.0, 1e8, 0.0))
    sim, out = orbit(domain, particle, 3)
    assert np.all(np.asarray(magnetic_moment(out, sim)) == 0)
    B = np.zeros((8, 2, 2, 3))
    B[..., 0] = B0
    sim = Simulation(domain, [particle], Solver(model="electrostatic", relativistic=True), external_B=B)
    out = sim.run(3)
    gamma = 1 / np.sqrt(1 - np.sum(np.asarray(out.v) ** 2, axis=-1) / 299792458.0 ** 2)
    assert np.allclose(magnetic_moment(out, sim), mass_electron * gamma ** 2 * 1e16 / (2 * B0), rtol=1e-6, atol=0)


@pytest.mark.parametrize("shape", [(8, 3, 3), (7, 3), (8, 2, 2, 2), (8, 4)])
def test_external_field_shape_is_checked(shape):
    domain = Domain(length=1e-2, cells=8, time_step=1e-11)
    with pytest.raises(ValueError, match="cells, ny, nz, 3"):
        Simulation(domain, [electron((0, 0, 0), (0, 0, 0))], external_E=np.zeros(shape))


def test_an_electric_field_on_a_grid_accelerates_as_a_flat_one_does():
    """E given on an (x, y, z) grid of centres, in the implicit scheme too."""
    cells = 8
    domain = Domain(length=1e-2, cells=cells, time_step=1e-11)
    particle = electron((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    E = np.zeros((cells, 2, 3, 3))
    E[..., 1] = 100.0
    for solver in (Solver(model="electrostatic"), Solver(model="electrostatic", algorithm="implicit")):
        out = Simulation(domain, [particle], solver, external_E=E).run(10)
        v_y = np.asarray(out.v)[:, 0, 1]
        assert np.allclose(np.diff(v_y), -e_charge * 100.0 / mass_electron * 1e-11, rtol=1e-9)
