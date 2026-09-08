"""Physics tests. Each runs a small simulation and compares a measured rate,
frequency, threshold or conserved quantity with a closed-form or tabulated
result from the literature."""
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import (Collisions, Domain, Simulation, Solver, Species, diagnostics, epsilon_0,
                       elementary_charge, mass_electron, temperatures,
                       elementary_charge as e_charge, speed_of_light as c)
from jaxincell._collisions import collide
from conftest import electron_plasma, growth_rate, mode_amplitude, rate_and_frequency


@pytest.mark.parametrize("k_lambda_d", [0.05, 0.25])
def test_langmuir_wave_follows_the_bohm_gross_dispersion_relation(k_lambda_d):
    """A small density perturbation of a Maxwellian plasma oscillates at
    omega^2 = omega_pe^2 (1 + 3 k^2 lambda_D^2) (Bohm and Gross, Phys. Rev. 75,
    1851, 1949). With v_th = sqrt(2 k_B T / m), lambda_D = v_th / (sqrt2 omega_pe)."""
    L, cells = 1.0, 32
    k = 2 * np.pi / L
    vth_over_c = k_lambda_d / k * np.sqrt(2) * (0.05 * c * cells / L) / c
    sim, omega_pe = electron_plasma(20000, L, cells, 0.05, vth_over_c, amplitude_k=1e-3)
    out = sim.run(800, seed=0, store_particles=False)
    _, omega = rate_and_frequency(np.asarray(out.t) * omega_pe, np.abs(mode_amplitude(out, 1)))
    assert abs(omega / np.sqrt(1 + 3 * k_lambda_d ** 2) - 1) < 0.01


def test_landau_damping_matches_the_kinetic_root():
    """At k lambda_D = 0.5 the least damped root of 1 + (1/(k lambda_D)^2)
    [1 + zeta Z(zeta)] = 0, zeta = omega/(k v_th), is
    omega = (1.4157 - 0.1533 i) omega_pe (Landau, J. Phys. USSR 10, 25, 1946;
    roots tabulated by Canosa, J. Plasma Phys. 8, 187, 1972). A quiet start with
    a one percent seed follows it until the discrete-particle noise takes over."""
    L, cells = 1.0, 64
    k = 2 * np.pi / L
    vth_over_c = 0.5 / k * np.sqrt(2) * (0.05 * c * cells / L) / c
    sim, omega_pe = electron_plasma(150000, L, cells, 0.05, vth_over_c, amplitude_k=0.01)
    out = sim.run(500, seed=0, store_particles=False)
    amplitude = np.abs(mode_amplitude(out, 1))
    floor = amplitude[int(0.8 * amplitude.size):].mean()
    gamma, omega = rate_and_frequency(np.asarray(out.t) * omega_pe, amplitude, above=5 * floor)
    assert abs(gamma / -0.1533 - 1) < 0.05
    assert abs(omega / 1.4157 - 1) < 0.02


def test_cold_two_stream_grows_at_the_fluid_rate():
    """Two cold beams of density n/2 drifting at +-v_0 grow fastest at
    k v_0 / omega_pe = sqrt(3/8), where gamma = omega_pe / (2 sqrt2), from
    1 = (omega_pe^2/2) [(omega - k v_0)^-2 + (omega + k v_0)^-2]
    (Buneman, Phys. Rev. 115, 503, 1959; Birdsall and Langdon, section 5-2)."""
    L, cells = 1.0, 64
    k = 2 * np.pi / L
    v0 = np.sqrt(3 / 8) * (0.05 * c * cells / L) / k
    sim, omega_pe = electron_plasma(40000, L, cells, 0.05, vth_over_c=1e-3 * v0 / c, drift=v0,
                                    plus_minus=True, amplitude_k=1e-4)
    out = sim.run(600, seed=0, store_particles=False)
    t = np.asarray(out.t) * omega_pe
    amplitude = np.abs(mode_amplitude(out, 1))
    peak = int(np.argmax(amplitude))
    # fit between ten times the seed and a third of saturation, before the peak
    window = (amplitude > 10 * amplitude[0]) & (amplitude < 0.3 * amplitude[peak]) & (np.arange(t.size) < peak)
    assert abs(growth_rate(t, amplitude, window) / (1 / (2 * np.sqrt(2))) - 1) < 0.05


def test_weibel_growth_is_confined_to_the_unstable_wavenumbers():
    """A bi-Maxwellian with T_z > T_x drives the Weibel instability (Weibel,
    Phys. Rev. Lett. 2, 83, 1959). Setting omega = 0 in the transverse
    dispersion relation gives the marginal wavenumber k_c c = omega_pe
    sqrt(T_z/T_x - 1): modes well below it grow, modes above it do not. The
    growth rate vanishes as k approaches k_c, so the two sides are read off at
    0.7 k_c and 1.2 k_c."""
    ratio, n_e, n = 25.0, 1e15, 12000
    omega_pe = np.sqrt(n_e * e_charge ** 2 / (epsilon_0 * mass_electron))
    k_c = np.sqrt(ratio - 1) * omega_pe / c
    L = 4.0 * 2 * np.pi / k_c                       # modes 1-3 unstable, 5-8 stable
    vth_x = 0.02 * c
    rng = np.random.default_rng(0)
    x = np.linspace(-L / 2, L / 2, n, endpoint=False) + L / (2 * n)
    v = np.stack([vth_x / np.sqrt(2) * rng.standard_normal(n), np.zeros(n),
                  vth_x * np.sqrt(ratio / 2) * rng.standard_normal(n)], axis=1)
    electrons = Species.electrons(n=n, density=n_e, vth=(vth_x, 0.0, vth_x * np.sqrt(ratio)))
    electrons = electrons.replace(x=np.stack([x, np.zeros(n), np.zeros(n)], axis=1), v=v)
    ions = Species.ions(n=n // 4, density=n_e, mass_ratio=1e6, vth=(0.0, 0.0, 0.0), quiet=True)
    sim = Simulation(Domain(length=L, cells=128, dt_over_dx_c=0.5), [electrons, ions], Solver(filter_passes=0))
    out = sim.run(3000, seed=0, store_every=20)
    B_k = np.abs(np.fft.rfft(np.asarray(out.B[:, :, 1]), axis=1))
    modes = np.arange(1, 9)
    gain = B_k[-1, modes] / B_k[0, modes]
    unstable = [i for i, m in enumerate(modes) if 2 * np.pi * m / L < 0.7 * k_c]
    stable = [i for i, m in enumerate(modes) if 2 * np.pi * m / L > 1.2 * k_c]
    assert min(gain[unstable]) > 10.0, f"unstable modes only reached {gain[unstable]}"
    assert max(gain[stable]) < 3.0, f"stable modes reached {gain[stable]}"
    total = np.asarray(diagnostics(out)["total"])
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-3


def two_stream(algorithm, steps, n=3000, **solver):
    """The warm two-stream instability used by the conservation tests."""
    e = Species.electrons(n=n, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0), plus_minus=True,
                          perturbation_amplitude=5e-7, perturbation_mode=1)
    i = Species.ions(n=n, density=4.37e17, electrons=e)
    sim = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [e, i], Solver(algorithm=algorithm, **solver))
    return sim.run(steps, seed=3)


def test_explicit_scheme_has_bounded_energy_error_and_an_exact_gauss_law():
    """Through the growth and saturation of the two-stream instability the
    explicit scheme changes the total energy by less than one percent, and the
    discrete Gauss law holds to round-off at every step because the deposited
    current satisfies the discrete continuity equation exactly."""
    out = two_stream("explicit", 600, filter_passes=2)
    d = diagnostics(out)
    total = np.asarray(d["total"])
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-2
    assert float(np.max(np.asarray(d["gauss_residual"]))) < 1e-9
    assert float(np.asarray(d["electric"]).max()) > 1e3 * float(np.asarray(d["electric"])[0])


def test_implicit_scheme_conserves_energy_to_round_off():
    """The Crank-Nicolson scheme with the orbit-averaged current conserves the
    discrete total energy once the Picard iteration has converged (Chen, Chacon
    and Barnes, J. Comput. Phys. 230, 7018, 2011; Markidis and Lapenta 2011)."""
    out = two_stream("implicit", 150, n=2000, picard_iterations=8)
    total = np.asarray(diagnostics(out)["total"])
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-11


def test_periodic_box_conserves_charge_exactly_and_momentum_to_the_solver_error():
    """In a periodic box the deposited charge is exactly the charge carried by
    the particles. Depositing and gathering with the same shape makes the
    interparticle force antisymmetric, so the momentum drift is not the
    interpolation but the residual of the staggered field solve."""
    out = two_stream("explicit", 300, n=4000)
    charge = np.asarray(out.charge)
    on_grid = np.asarray(out.rho).sum(axis=1) * out.dx
    assert float(np.abs(on_grid - charge.sum()).max()) < 1e-12 * float(np.abs(charge).sum())
    p = np.asarray(diagnostics(out)["momentum"])[:, 0]
    momentum_content = float(np.sum(np.asarray(out.mass) * np.abs(np.asarray(out.v[0, :, 0]))))
    assert float(np.max(np.abs(p - p[0]))) < 1e-4 * momentum_content


def test_reflective_walls_hold_the_particles_and_absorbing_walls_remove_them():
    """Reflective walls keep every particle inside [-L/2, L/2] and, at unit
    restitution, conserve the total energy. Absorbing walls zero the charge of
    the particles that reach them, and the charge left on the grid then follows
    the charge left on the particles to within the part of the shape function
    that sticks out past the wall."""
    def walls(kind):
        e = Species.electrons(n=2000, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0))
        i = Species.ions(n=2000, density=1e17, electrons=e, vth=(0.02 * c, 0, 0)).replace(drift=(0.05 * c, 0, 0))
        domain = Domain(length=1e-2, cells=32, dt_over_dx_c=1.0, particle_bc=kind, field_bc=kind)
        return Simulation(domain, [e, i], Solver(filter_passes=0)).run(300, seed=0)

    out = walls("reflective")
    assert float(np.abs(np.asarray(out.x[:, :, 0])).max()) <= 5e-3 * (1 + 1e-12)
    assert int((np.asarray(out.charge) == 0).sum()) == 0
    total = np.asarray(diagnostics(out)["total"])
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-4

    out = walls("absorbing")
    charge = np.asarray(out.charge)
    assert 0 < int((charge == 0).sum()) < charge.size
    on_grid = float(np.asarray(out.rho)[-1].sum()) * out.dx
    assert abs(on_grid - charge.sum()) < 1e-3 * np.abs(charge).sum()


def test_collisions_reproduce_the_spitzer_relaxation_rates():
    """A beam much faster than the background it scatters off slows down at
    nu_s = (1 + m_a/m_b) nu_0 and spreads in angle at nu_perp = 2 nu_0, with
    nu_0 = q_a^2 q_b^2 n_b ln(Lambda) / (4 pi eps0^2 m_a^2 v^3) (Trubnikov, Rev.
    Plasma Phys. 1, 105, 1965; NRL Plasma Formulary). This limit fixes the
    variance of the Takizuka-Abe scattering angle with no free parameter."""
    n_b, coulomb_log, v_beam, n = 1e20, 12.5, 3e7, 60000
    nu_0 = e_charge ** 4 * coulomb_log * n_b / (4 * np.pi * epsilon_0 ** 2 * mass_electron ** 2 * v_beam ** 3)
    rng = np.random.default_rng(1)
    beam = np.zeros((n, 3))
    beam[:, 0] = v_beam
    v = np.concatenate([beam, 3e5 / np.sqrt(2) * rng.standard_normal((n, 3))])
    arrays = (np.zeros((2 * n, 3)), np.full(2 * n, n_b / n), np.full(2 * n, mass_electron),
              np.full(2 * n, -e_charge))
    dt, steps = 0.0005 / (2 * nu_0), 40           # the beam slows by about two percent
    key, v = random.PRNGKey(0), jnp.asarray(v)
    arrays = tuple(jnp.asarray(a) for a in arrays)
    history = []
    for step in range(steps + 1):
        history.append((step * dt, float(v[:n, 0].mean()), float((v[:n, 1] ** 2 + v[:n, 2] ** 2).mean())))
        key, sub = random.split(key)
        v = collide(sub, arrays[0], v, *arrays[1:], ((0, n), (n, n)), ((0, 1),), coulomb_log, dt, 1.0, 1.0, 1)
    t, v_parallel, v_perp2 = (np.array(column) for column in zip(*history))
    nu_slow = -np.polyfit(t, np.log(v_parallel), 1)[0]
    nu_perp = np.polyfit(t, v_perp2, 1)[0] / v_beam ** 2
    assert abs(nu_slow / (2 * nu_0) - 1) < 0.05
    assert abs(nu_perp / (2 * nu_0) - 1) < 0.05


def test_relativistic_pusher_gyrates_at_the_relativistic_frequency():
    """A charge in a uniform magnetic field gyrates at Omega = qB/(gamma m), not
    qB/m. The relativistic Boris rotation reproduces that frequency and keeps
    gamma fixed to round-off, because a magnetic field does no work."""
    from jaxincell._core import boris, boris_relativistic

    B0, speed = 0.05, 0.9 * c
    gamma = 1 / np.sqrt(1 - (speed / c) ** 2)
    omega_c = elementary_charge * B0 / (gamma * mass_electron)
    steps = 400
    dt = 2 * np.pi / omega_c / steps                       # one relativistic orbit
    field_E = jnp.zeros((1, 3))
    field_B = jnp.array([[0.0, 0.0, B0]])
    charge, mass = jnp.array([[elementary_charge]]), jnp.array([[mass_electron]])

    v = jnp.array([[speed, 0.0, 0.0]])
    speeds, angles = [], []
    for _ in range(steps):
        v = boris_relativistic(v, field_E, field_B, charge, mass, dt)
        speeds.append(float(jnp.linalg.norm(v)))
        angles.append(float(jnp.arctan2(v[0, 1], v[0, 0])))
    assert max(abs(s / speed - 1) for s in speeds) < 1e-12   # gamma is conserved
    # after one relativistic period the velocity is back where it started
    assert abs(angles[-1]) < 2 * np.pi / steps

    # the non-relativistic pusher turns gamma times too fast and so overshoots
    v = jnp.array([[speed, 0.0, 0.0]])
    for _ in range(steps):
        v = boris(v, field_E, field_B, jnp.full((1, 1), elementary_charge / mass_electron), dt)
    assert abs(float(jnp.arctan2(v[0, 1], v[0, 0]))) > 1.0


def test_collisions_through_the_simulation_conserve_momentum_and_isotropise():
    """Wired into a run, the collision operator leaves the total momentum alone
    and relaxes an anisotropic temperature towards isotropy."""
    n, density = 4000, 1e21
    electrons = Species.electrons(n=n, density=density, vth=(3e6, 3e6, 1e6), quiet=True)
    ions = Species.ions(n=n, density=density, electrons=electrons, quiet=True)
    simulation = Simulation(Domain(length=2e-5, cells=16, dt_over_dx_c=1.0), [electrons, ions],
                            Solver(filter_passes=0), Collisions(coulomb_log=1e4))
    out = simulation.run(300, seed=0)
    momentum = np.asarray(diagnostics(out)["momentum"])[:, 0]
    content = float(np.sum(np.asarray(out.mass) * np.abs(np.asarray(out.v[0, :, 0]))))
    assert float(np.abs(momentum - momentum[0]).max()) < 1e-4 * content
    T = np.asarray(temperatures(out)["electrons"])
    start, end = T[0], T[-1]
    assert end[2] > start[2] and end[0] < start[0]          # the cold axis heats
    assert abs(end[2] / end[0] - 1) < abs(start[2] / start[0] - 1)
