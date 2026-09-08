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


@pytest.mark.parametrize("wall", ["periodic", "reflective", "absorbing"])
@pytest.mark.parametrize("filter_passes", [0, 2])
def test_gauss_law_holds_at_every_wall_with_and_without_filtering(wall, filter_passes):
    """The discrete Gauss law is exact for every wall type, because the current
    is derived from the same charge density the field is checked against.

    Three things have to line up for that, and each was wrong once. The density
    at the half step has to be the one the first half of the step ended on, or
    the charge a wall absorbs disappears between the halves with no current to
    carry it. The initial field has to be built from the density the loop starts
    from, which at a wall is not where the particles were placed. And the filter
    has to be applied to the density before the current is taken from it.
    """
    e = Species.electrons(n=2000, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0), quiet=True)
    i = Species.ions(n=2000, density=1e17, electrons=e, quiet=True)
    domain = Domain(length=1e-2, cells=32, dt_over_dx_c=1.0, particle_bc=wall, field_bc=wall)
    out = Simulation(domain, [e, i], Solver(filter_passes=filter_passes,
                                            filter_strides=(1, 2))).run(120, seed=0)
    assert float(np.asarray(diagnostics(out)["gauss_residual"]).max()) < 1e-10


@pytest.mark.parametrize("wall, conserving", [("periodic", True), ("reflective", True),
                                              ("absorbing", False)])
def test_the_filter_moves_the_sources_around_without_inventing_any(wall, conserving):
    """Smoothing redistributes charge; it must not create it. A reflective wall
    mirrors the stencil back into the box, so the total is untouched; only an
    absorbing wall, which is supposed to let charge leave, may lose any."""
    from jaxincell._core import smooth

    rho = np.zeros(24)
    rho[[0, 3, 20, 23]] = 1.0                    # deliberately loaded against both walls
    code = {"periodic": (0, 0), "reflective": (1, 1), "absorbing": (2, 2)}[wall]
    total = float(jnp.sum(smooth(jnp.asarray(rho), 2, 0.5, (1, 2, 4), code)))
    if conserving:
        assert abs(total / rho.sum() - 1) < 1e-12
    else:
        assert total < 0.95 * rho.sum()


def test_electrostatic_solvers_agree_and_both_satisfy_gauss():
    """`field_solver="gauss"` recomputes E_x from the charge density instead of
    advancing it with the current. On an electrostatic problem the two must
    agree to the discretisation error, and both must satisfy the discrete Gauss
    law: one by construction, the other because the current conserves charge."""
    def run(field_solver):
        e = Species.electrons(n=4000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(5e7, 0, 0),
                              plus_minus=True, quiet=True, perturbation_amplitude=5e-7,
                              perturbation_mode=1)
        i = Species.ions(n=4000, density=4.37e17, electrons=e, quiet=True)
        return Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [e, i],
                          Solver(field_solver=field_solver)).run(300, seed=3)

    ampere, gauss = run("ampere"), run("gauss")
    for out in (ampere, gauss):
        assert float(np.asarray(diagnostics(out)["gauss_residual"]).max()) < 1e-10
    field = np.asarray(ampere.E[:, :, 0])
    assert np.abs(np.asarray(gauss.E[:, :, 0]) - field).max() < 1e-3 * np.abs(field).max()


def test_relativistic_run_conserves_the_energy_the_pusher_conserves():
    """With `relativistic=True` the kinetic energy is sum (gamma - 1) m c^2, and
    the diagnostic has to follow the solver: reporting the Newtonian energy for a
    relativistic run would show a spurious drift where there is none."""
    def run(relativistic):
        e = Species.electrons(n=2000, density=1e17, vth=(0.3 * c, 0, 0), quiet=True,
                              perturbation_amplitude=1e-5, perturbation_mode=1)
        i = Species.ions(n=2000, density=1e17, electrons=e, quiet=True)
        out = Simulation(Domain(length=0.05, cells=32, dt_over_dx_c=1.0), [e, i],
                         Solver(relativistic=relativistic)).run(200, seed=0)
        return out, np.asarray(diagnostics(out)["total"])

    out, total = run(True)
    assert float(np.abs(np.asarray(out.v)).max()) > 0.5 * c      # relativity matters here
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-3
    gamma = 1 / np.sqrt(1 - np.sum(np.asarray(out.v[0]) ** 2, axis=-1) / c ** 2)
    expected = float(np.sum((gamma - 1) * np.asarray(out.mass) * c ** 2))
    assert abs(float(diagnostics(out)["kinetic"][0]) / expected - 1) < 1e-12
    assert float(diagnostics(run(False)[0])["kinetic"][0]) < expected   # Newtonian is lower


def test_an_external_magnetic_field_magnetises_the_plasma():
    """A uniform external B_x is the one field the code cannot generate itself,
    because the curl has no x component in one dimension. Particles in it gyrate
    at Omega_c = qB/m in the y-z plane at constant speed, since a magnetic field
    does no work."""
    B0, cells, length, n = 5e-4, 32, 1.0, 2000
    omega_c = elementary_charge * B0 / mass_electron
    # tenuous and cold, so that the self-consistent fields do not compete
    e = Species.electrons(n=n, density=1e6, vth=(0, 0, 0), drift=(0, 1e5, 0), quiet=True)
    i = Species.ions(n=n, density=1e6, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
    external = np.zeros((cells, 3))
    external[:, 0] = B0
    domain = Domain(length=length, cells=cells, dt_over_dx_c=1.0)
    out = Simulation(domain, [e, i], Solver(filter_passes=0), external_B=external).run(400, seed=0)

    v_y = np.asarray(out.v[:, :n, 1]).mean(axis=1)
    v_z = np.asarray(out.v[:, :n, 2]).mean(axis=1)
    speed = np.hypot(v_y, v_z)
    assert abs(speed[-1] / speed[0] - 1) < 1e-5                      # no work done
    phase = np.unwrap(np.arctan2(v_z, v_y))
    measured = abs(phase[-1] - phase[0]) / float(out.t[-1] - out.t[0])
    assert abs(measured / omega_c - 1) < 1e-3

    # without the external field there is nothing to rotate into z
    plain = Simulation(domain, [e, i], Solver(filter_passes=0)).run(400, seed=0)
    assert float(np.abs(np.asarray(plain.v[:, :n, 2])).max()) == 0.0


def test_absorbing_walls_build_a_sheath_and_float_the_plasma():
    """A plasma between two absorbing walls charges them negative until the
    electron flux is throttled to the ion flux. What is left is a quasi-neutral
    bulk joined to each wall by a positively charged layer a few Debye lengths
    thick, with the bulk floating above the walls by about
    (T_e/2e) ln(m_i/2 pi m_e) and the ions entering the sheath at the Bohm speed
    c_s = sqrt(T_e/m_i) (Bohm 1949; Lieberman and Lichtenberg, section 6.2).

    Absorbing walls are conductors short-circuited to one another, so both stay
    at the same potential -- the standard bounded-plasma closure (Verboncoeur,
    J. Comput. Phys. 104, 321, 1993).

    The band on the drop is deliberately wide. The formula assumes a Maxwellian
    tail and m_i >> m_e, and this run has neither: nothing sustains the plasma, so
    the walls take the tail the flux balance is derived from, and the mass ratio is
    reduced to 100 to bring the ion transit within a test. Each shifts the answer by
    tens of per cent, in opposite directions -- at 100 it comes out high, at the 400
    of examples/sheath.py it comes out low. What is checked sharply is the structure
    that has no free parameters: the two walls sitting at one potential, a charged
    layer at each of them with a neutral bulk between, and ions leaving at the Bohm
    speed.
    """
    from jaxincell import potential, quiet_start

    T_e, density, mass_ratio, n, cells = 1.0, 1e16, 100.0, 20000, 120
    v_th = np.sqrt(2 * T_e * elementary_charge / mass_electron)
    omega_pe = np.sqrt(density * elementary_charge ** 2 / (epsilon_0 * mass_electron))
    debye = v_th / (np.sqrt(2) * omega_pe)
    length = 60 * debye
    v_th_ion = v_th * np.sqrt(1 / (40 * mass_ratio))
    x, v = quiet_start(n, length, vth=(v_th, 0, 0))
    electrons = Species.electrons(n=n, density=density, vth=(v_th, 0, 0)).replace(x=x, v=v)
    x, v = quiet_start(n, length, vth=(v_th_ion, 0, 0))
    ions = Species("ions", n, 1.0, mass_ratio * mass_electron, density,
                   (v_th_ion, 0, 0)).replace(x=x, v=v)
    domain = Domain(length=length, cells=cells, particle_bc="absorbing", field_bc="absorbing",
                    dt_over_dx_c=(0.2 / omega_pe) * c / (length / cells))
    out = Simulation(domain, [electrons, ions], Solver(filter_passes=4)).run(1500, seed=0,
                                                                            store_every=50)
    phi = np.asarray(potential(out))
    v_x = np.asarray(out.v[..., 0])
    bulk = np.abs(np.asarray(out.x[:, :n, 0])) < length / 5
    T_bulk = np.array([mass_electron * np.var(v_x[k, :n][bulk[k]]) / elementary_charge
                       for k in range(phi.shape[0])])

    # the two electrodes are short-circuited, so the far wall stays at zero
    assert float(np.abs(phi[:, -1]).max()) < 1e-9 * T_bulk.max()

    late = slice(phi.shape[0] // 2, None)
    drop = (phi[late, 2 * cells // 5:3 * cells // 5].mean(axis=1) / T_bulk[late]).mean()
    theory = 0.5 * np.log(mass_ratio / (2 * np.pi))
    assert 0.6 * theory < drop < 1.5 * theory

    # the sheath is where quasi-neutrality fails: net positive charge at the walls,
    # none in the middle
    rho = np.asarray(out.rho)[late].mean(axis=0) / (density * elementary_charge)
    assert rho[:3].mean() > 0.02 and rho[-3:].mean() > 0.02
    assert abs(rho[2 * cells // 5:3 * cells // 5].mean()) < 0.005

    # and the pre-sheath has accelerated the ions towards the Bohm speed
    c_s = np.sqrt(T_bulk[-1] * elementary_charge / (mass_ratio * mass_electron))
    x_i = np.asarray(out.x[-1, n:, 0])
    edge = (x_i > 0.30 * length) & (x_i < 0.40 * length)
    assert v_x[-1, n:][edge].mean() > 0.5 * c_s
