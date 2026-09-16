"""Physics tests. Each runs a small simulation and compares a measured rate,
frequency, threshold or conserved quantity with a closed-form or tabulated
result from the literature."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import (Collisions, Domain, Simulation, Solver, Species, diagnostics, epsilon_0, gauss_residual,
                       mass_electron, potential, quiet_start, temperatures, elementary_charge as e_charge,
                       speed_of_light as c)
from jaxincell._collisions import collide
from jaxincell._core import boris, boris_relativistic, smooth
from conftest import electron_plasma, growth_rate, mode_amplitude, rate_and_frequency

# Real part of the least-damped root omega/omega_pe of 1 + [1 + zeta Z(zeta)]/(k lambda_D)^2 = 0,
# zeta = omega/(k v_th), solved with the Faddeeva function (docs/scripts/dispersion.py). Hard-coded
# because scipy is not a test dependency; test_the_reference_roots_solve_the_dispersion_relation
# re-derives them wherever scipy is installed.
LANGMUIR_FREQUENCY = {0.05: 1.0038, 0.10: 1.0152, 0.15: 1.0348, 0.20: 1.0640, 0.25: 1.1057, 0.30: 1.1598}
LANDAU_ROOT = 1.4157 - 0.1533j          # k lambda_D = 0.5 (Canosa, J. Plasma Phys. 8, 187, 1972)


def kinetic_langmuir_root(k_lambda_d, guess):
    """Newton's method on the electrostatic dispersion relation of a Maxwellian, in
    units where omega_pe = lambda_D = 1, so that v_th = sqrt(2)."""
    special = pytest.importorskip("scipy.special")
    kv, omega = np.sqrt(2) * k_lambda_d, complex(guess)
    for _ in range(50):
        zeta = omega / kv
        Z = 1j * np.sqrt(np.pi) * special.wofz(zeta)
        epsilon = 1 + (1 + zeta * Z) / k_lambda_d ** 2
        d_epsilon = (Z - 2 * zeta * (1 + zeta * Z)) / (kv * k_lambda_d ** 2)     # Z' = -2 (1 + zeta Z)
        omega -= epsilon / d_epsilon
    assert abs(epsilon) < 1e-10
    return omega


def test_the_reference_roots_solve_the_dispersion_relation():
    """The frequencies the Langmuir and Landau tests compare with, to one unit in the
    last digit they are quoted to."""
    for k_lambda_d, frequency in LANGMUIR_FREQUENCY.items():
        assert abs(kinetic_langmuir_root(k_lambda_d, np.sqrt(1 + 3 * k_lambda_d ** 2)).real - frequency) <= 1e-4
    root = kinetic_langmuir_root(0.5, LANDAU_ROOT)
    assert abs(root.real - LANDAU_ROOT.real) <= 1e-4 and abs(root.imag - LANDAU_ROOT.imag) <= 1e-4


def discrete_frequency_factor(cells, mode=1):
    """omega / omega_pe of a cold plasma on this grid: S(k) sqrt(k/K), with
    S = sinc^3(k dx/2) the quadratic spline and K = (2/dx) sin(k dx/2) the staggered
    Gauss law (Birdsall and Langdon, Plasma Physics via Computer Simulation, ch. 8)."""
    theta = np.pi * mode / cells
    return np.sqrt((np.sin(theta) / theta) ** 6 * theta / np.sin(theta))


@pytest.mark.parametrize("k_lambda_d", [0.05, 0.3])
def test_langmuir_wave_follows_the_kinetic_dispersion_relation(k_lambda_d):
    """A small density perturbation of a Maxwellian oscillates at the real part of the
    least-damped kinetic root. At k lambda_D = 0.3 that is 2.9 % above the Bohm-Gross
    frequency omega_pe sqrt(1 + 3 k^2 lambda_D^2) (Bohm and Gross, Phys. Rev. 75, 1851,
    1949), so the test tells kinetic from fluid; at 0.05 the two coincide and it checks
    the plasma frequency. With v_th = sqrt(2 k_B T / m), lambda_D = v_th / (sqrt2 omega_pe).

    The grid lowers the frequency by a known amount: the spline deposit and gather and
    the staggered Gauss law give omega_pe S(k) sqrt(k/K), 0.40 % low at k dx = 2 pi/32,
    and the leapfrog raises it by (omega_pe dt)^2/24 = 1e-4. What remains is the grid's
    correction to the thermal term, a few tenths of a per cent at 0.3, and the timing of
    the maxima, 0.1 % over 1600 steps; 0.5 % covers both and is a sixth of the gap to
    Bohm-Gross. The seed a k = 1e-2 keeps the wave above the particle noise: at 1e-3
    the noise moves the maxima by a per cent or more at k lambda_D >= 0.25."""
    L, cells = 1.0, 32
    k = 2 * np.pi / L
    vth_over_c = k_lambda_d / k * np.sqrt(2) * (0.05 * c * cells / L) / c
    sim, omega_pe = electron_plasma(20000, L, cells, 0.05, vth_over_c, amplitude_k=1e-2)
    out = sim.run(1600, seed=0, store_particles=False)
    _, omega = rate_and_frequency(np.asarray(out.t) * omega_pe, np.abs(mode_amplitude(out, 1)))
    assert abs(omega / (LANGMUIR_FREQUENCY[k_lambda_d] * discrete_frequency_factor(cells)) - 1) < 5e-3


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
    assert abs(gamma / LANDAU_ROOT.imag - 1) < 0.05
    assert abs(omega / LANDAU_ROOT.real - 1) < 0.02


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


# Growth rates of modes 1 and 2 of the Weibel box below, from the transverse kinetic
# dispersion relation (docs/scripts/dispersion.py, weibel_dispersion), in units of omega_pe
WEIBEL_RATES = {1: 0.0473, 2: 0.0428}


def test_weibel_growth_is_confined_to_the_unstable_wavenumbers():
    """A bi-Maxwellian with T_z > T_x drives the Weibel instability (Weibel,
    Phys. Rev. Lett. 2, 83, 1959). Setting omega = 0 in the transverse
    dispersion relation gives the marginal wavenumber k_c c = omega_pe
    sqrt(T_z/T_x - 1): modes below it grow, modes above it do not. The box holds four
    marginal wavelengths, so modes 1-3 lie below k_c, mode 4 on it and modes 5-8 above.
    The growth rate vanishes as k approaches k_c, so the two sides are read off at
    k < 0.7 k_c (modes 1 and 2) and k > 1.2 k_c (modes 5-8).

    Every mode starts from the same coherent transverse current on a quiet start. From
    random velocities at this particle count the noise in a stable mode changes by as
    much between two windows as the slower unstable mode grows, whatever baseline is
    used. The gain is the median of |B_k| over the last ten stored samples against the
    median over 2 < t omega_pe < 6, once the field has built up from zero. A seeded
    current puts part of each unstable mode into its damped root, so an unstable mode
    has to gain a third of exp(gamma dt), 4.4 and 3.4 here, where it gains 8 and 12; a
    stable mode has no growing root and has to stay within a factor of two, where it
    stays within 1.2."""
    ratio, n_e, n = 25.0, 1e15, 12000
    omega_pe = np.sqrt(n_e * e_charge ** 2 / (epsilon_0 * mass_electron))
    k_c = np.sqrt(ratio - 1) * omega_pe / c
    L = 4.0 * 2 * np.pi / k_c
    vth = (0.02 * c, 0.0, 0.02 * c * np.sqrt(ratio))
    x, v = quiet_start(n, L, vth=vth)
    v = v.at[:, 2].add(1e-2 * vth[2] * sum(jnp.sin(2 * jnp.pi * m * x[:, 0] / L) for m in range(1, 9)))
    electrons = Species.electrons(n=n, density=n_e, vth=vth).replace(x=x, v=v)
    ions = Species.ions(n=n // 4, density=n_e, mass_ratio=1e6, vth=(0.0, 0.0, 0.0), quiet=True)
    sim = Simulation(Domain(length=L, cells=128, dt_over_dx_c=0.5), [electrons, ions], Solver(filter_passes=0))
    out = sim.run(3000, seed=0, store_every=20)
    t = np.asarray(out.t) * omega_pe
    B_k = np.abs(np.fft.rfft(np.asarray(out.B[:, :, 1]), axis=1))
    early, late = (t > 2) & (t < 6), np.arange(t.size) >= t.size - 10
    gain = np.median(B_k[late], axis=0) / np.median(B_k[early], axis=0)
    elapsed = np.median(t[late]) - np.median(t[early])
    k = 2 * np.pi * np.arange(B_k.shape[1]) / L
    unstable = [m for m in range(1, 9) if k[m] < 0.7 * k_c]
    stable = [m for m in range(1, 9) if k[m] > 1.2 * k_c]
    assert unstable == [1, 2] and stable == [5, 6, 7, 8]
    for m in unstable:
        assert gain[m] > np.exp(WEIBEL_RATES[m] * elapsed) / 3, f"mode {m} only grew by {gain[m]:.2f}"
    assert max(gain[stable]) < 2.0, f"stable modes reached {gain[stable]}"
    total = np.asarray(diagnostics(out)["total"])
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-3


def two_stream(algorithm, steps, n=3000, drift=6e7, quiet=False, **solver):
    """The warm two-stream instability used by the conservation tests."""
    e = Species.electrons(n=n, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(drift, 0, 0), plus_minus=True,
                          quiet=quiet, perturbation_amplitude=5e-7, perturbation_mode=1)
    i = Species.ions(n=n, density=4.37e17, electrons=e, quiet=quiet)
    sim = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [e, i], Solver(algorithm=algorithm, **solver))
    return sim.run(steps, seed=3)


@pytest.fixture(scope="module")
def quiet_two_stream():
    """One quiet two-stream run read by two tests: the number of steps is static, so
    every distinct run is a compilation of its own."""
    return two_stream("explicit", 300, n=4000, drift=5e7, quiet=True)


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


@pytest.mark.parametrize("drift, relativistic", [(6e7, False), (0.6 * c, True)])
def test_implicit_scheme_conserves_energy_and_charge_to_round_off(drift, relativistic):
    """Once the Picard iteration has converged the Crank-Nicolson scheme conserves the
    discrete total energy (Chen, Chacon and Barnes, J. Comput. Phys. 230, 7018, 2011)
    and, in the same run, the discrete Gauss law: E_x is the discrete gradient whose work
    is what the charge-conserving current takes from the field (Kormann and
    Sonnendruecker, J. Comput. Phys. 425, 109890, 2021). With relativity the particles
    move at (u + u')/(gamma + gamma'), along which the Boris step does exactly the work of
    E; the mean of the two velocities, which the scheme once took, is not that velocity,
    and left 7e-13 at eight iterations and at twelve. Both errors come out a few 1e-16."""
    out = two_stream("implicit", 150, n=2000, drift=drift, relativistic=relativistic, picard_iterations=8)
    d = diagnostics(out)
    assert float(np.asarray(d["energy_error"]).max()) < 1e-14
    assert float(np.asarray(d["gauss_residual"]).max()) < 1e-10


def test_the_implicit_gauss_law_holds_along_its_derivative():
    """Reverse-mode derivatives run through the implicit scheme at a wall that returns half
    of each electron, with ions that start at rest, whose zero displacement takes the slope
    of the potential in place of the quotient 0/0. The Gauss law holds for every value of a
    parameter, so it holds for the derivative too: the derivative of div E with respect to
    the electron drift is the derivative of rho/eps0, on a random combination of cells."""
    e = Species.electrons(n=300, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0), reflection=0.5)
    i = Species.ions(n=300, density=1e17, vth=(0.0, 0.0, 0.0), quiet=True)
    domain = Domain(length=1e-2, cells=16, dt_over_dx_c=2.0, particle_bc="absorbing", field_bc="absorbing")
    sim = Simulation(domain, [e, i], Solver(algorithm="implicit", picard_iterations=4))
    cells = jnp.asarray(np.random.default_rng(0).normal(size=15))

    def divergence_and_density(drift):
        out = sim.replace(species=(e.replace(drift=(drift, 0.0, 0.0)), i)).run(30, seed=0)
        E = out.E[-1, :, 0]
        return jnp.stack([jnp.sum(cells * (E[1:] - E[:-1]) / out.dx), jnp.sum(cells * out.rho[-1, 1:] / epsilon_0)])

    slopes = np.asarray(jax.jacrev(divergence_and_density)(0.05 * c))
    assert np.all(np.isfinite(slopes)) and slopes[1] != 0
    assert abs(slopes[0] - slopes[1]) < 1e-10 * abs(slopes[1])


def test_periodic_box_conserves_charge_exactly_and_momentum_to_the_solver_error(quiet_two_stream):
    """In a periodic box the deposited charge is exactly the charge carried by
    the particles. Depositing and gathering with the same shape makes the
    interparticle force antisymmetric, so the momentum drift is not the
    interpolation but the residual of the staggered field solve."""
    out = quiet_two_stream
    charge = np.asarray(out.charge * out.weight[-1])
    on_grid = np.asarray(out.rho).sum(axis=1) * out.dx
    assert float(np.abs(on_grid - charge.sum()).max()) < 1e-12 * float(np.abs(charge).sum())
    assert float(np.asarray(diagnostics(out)["momentum_error"]).max()) < 1e-4


def test_reflective_walls_hold_the_particles_and_absorbing_walls_remove_them():
    """Reflective walls keep every particle inside [-L/2, L/2] and, at unit
    restitution, conserve the total energy. Absorbing walls zero the weight of
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
    assert int((np.asarray(out.weight[-1]) == 0).sum()) == 0
    total = np.asarray(diagnostics(out)["total"])
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-4

    out = walls("absorbing")
    weight = np.asarray(out.weight[-1])
    charge = np.asarray(out.charge) * weight
    assert 0 < int((weight == 0).sum()) < weight.size
    on_grid = float(np.asarray(out.rho)[-1].sum()) * out.dx
    assert abs(on_grid - charge.sum()) < 1e-3 * np.abs(charge).sum()


def test_a_wall_returns_the_flux_average_of_its_reflection_law():
    """A wall samples the flux, not the distribution: from a Maxwellian of spread
    sigma the particles reaching it with normal speed v are in proportion to
    v f(v), so a reflection law R(v) returns its flux average. For the Gaussian
    R = exp(-v^2/2u^2) that is u^2/(u^2+sigma^2), one half at u = sigma, where the
    average over the distribution would be 0.71. The ratio of fluxes is the
    coefficient that sets the floating potential (Hobbs and Wesson, Plasma Phys.
    9, 85, 1967). What comes back, comes back at restitution times its speed."""
    sigma, length, n = 1e6, 1e-2, 100_000
    domain = Domain(length=length, cells=64, dt_over_dx_c=50.0, particle_bc="absorbing",
                    field_bc="absorbing", restitution=0.5)
    steps = int(round(0.1 * length / sigma / domain.dt))          # a tenth of a transit: nothing hits twice
    _, v0 = quiet_start(n, length, vth=(np.sqrt(2) * sigma, 0, 0))

    def law(speed):
        return jnp.exp(-speed ** 2 / (2 * sigma ** 2))

    electrons = Species.electrons(n=n, density=1e6, vth=(np.sqrt(2) * sigma, 0, 0), quiet=True, reflection=law)
    out = Simulation(domain, [electrons], Solver()).run(steps, seed=0, store_every=steps)
    w = np.asarray(out.weight[-1])
    w0 = w.max()                                     # the weight of a particle that met no wall
    hit = w < w0
    assert 0.05 < hit.mean() < 0.11                  # 2 sigma t / (sqrt(2 pi) L) of them reach a wall
    assert abs(w[hit].sum() / (w0 * hit.sum()) - 0.5) < 5e-3
    back = hit & (w > 0)
    # to within the ~0.03 m/s the plasma's own field adds over the run
    assert np.allclose(np.abs(np.asarray(out.v[-1, back, 0])), 0.5 * np.abs(v0[back, 0]), rtol=0, atol=1.0)


def test_a_thermal_wall_re_emits_the_half_maxwellian_flux():
    """What reaches a thermal wall comes back as if from a Maxwellian reservoir
    behind it at the species' temperature: the normal velocity from the flux
    distribution (v/sigma^2) exp(-v^2/2 sigma^2), with <v> = sigma sqrt(pi/2) and
    <v^2> = 2 sigma^2, and the tangential ones from the Maxwellian itself. That is
    the source boundary of a bounded-plasma simulation (Schwager and Birdsall,
    Phys. Fluids B 2, 1057, 1990), and what keeps the tail of the plasma filled."""
    n, length, sigma = 40_000, 1e-2, 0.02 * c
    x, v = np.zeros((n, 3)), np.zeros((n, 3))
    x[:, 0], v[:, 0] = np.linspace(-0.45, 0.45, n) * length, -0.5 * c     # everything heads for the left wall
    vth = np.sqrt(2) * sigma
    electrons = Species.electrons(n=n, density=1e6, vth=(vth, vth, vth)).replace(x=x, v=v)
    domain = Domain(length=length, cells=64, particle_bc=("thermal", "reflective"), field_bc="reflective")
    u = np.asarray(Simulation(domain, [electrons], Solver()).run(300, seed=0, store_every=300).v[-1])
    assert (u[:, 0] > 0).all()
    assert abs(u[:, 0].mean() / (sigma * np.sqrt(np.pi / 2)) - 1) < 0.02
    assert abs((u[:, 0] ** 2).mean() / (2 * sigma ** 2) - 1) < 0.03
    assert abs((u[:, 1:] ** 2).mean() / sigma ** 2 - 1) < 0.03


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
    qB/m. The relativistic Boris rotation reproduces that frequency and, after a
    whole orbit, still has the speed it started with, because a magnetic field does
    no work."""
    B0, speed = 0.05, 0.9 * c
    gamma = 1 / np.sqrt(1 - (speed / c) ** 2)
    omega_c = e_charge * B0 / (gamma * mass_electron)
    steps = 400
    dt = 2 * np.pi / omega_c / steps                       # one relativistic orbit
    field_E, field_B = jnp.zeros((1, 3)), jnp.array([[0.0, 0.0, B0]])
    charge_to_mass = jnp.array([[e_charge / mass_electron]])

    def orbit(push, start):
        def step(w, _):
            return push(w, field_E, field_B, charge_to_mass, dt), None
        return jax.lax.scan(step, jnp.array([[start, 0.0, 0.0]]), None, length=steps)[0]

    u = orbit(boris_relativistic, gamma * speed)           # the relativistic pusher advances u = gamma v
    assert abs(float(jnp.linalg.norm(u)) / (gamma * speed) - 1) < 1e-12
    # after one relativistic period the velocity is back where it started
    assert abs(float(jnp.arctan2(u[0, 1], u[0, 0]))) < 2 * np.pi / steps

    # the non-relativistic pusher turns gamma times too fast and so overshoots
    v = orbit(boris, speed)
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
    assert float(np.asarray(diagnostics(out)["momentum_error"]).max()) < 1e-4
    T = np.asarray(temperatures(out)["electrons"])
    start, end = T[0], T[-1]
    assert end[2] > start[2] and end[0] < start[0]          # the cold axis heats
    assert abs(end[2] / end[0] - 1) < abs(start[2] / start[0] - 1)


@pytest.mark.parametrize("particle_bc, field_bc, reflection", [
    ("periodic", "periodic", 0.0), ("reflective", "reflective", 0.0), ("absorbing", "absorbing", 0.0),
    ("absorbing", "absorbing", 0.5), (("thermal", "absorbing"), ("reflective", "absorbing"), 0.0)])
@pytest.mark.parametrize("algorithm, filter_passes", [("explicit", 0), ("explicit", 2), ("implicit", 0)])
def test_gauss_law_holds_at_every_wall_with_and_without_filtering(particle_bc, field_bc, reflection, algorithm,
                                                                  filter_passes):
    """The discrete Gauss law is exact for every wall type -- including a wall that
    returns part of each electron and a thermal wall that re-emits them -- because
    the current is derived from the same charge density the field is checked against.
    That holds for both schemes: the implicit one takes the continuity current of the
    deposits at the two ends of every sub-step, where it once took the transpose of the
    gather and missed the Gauss law by 0.15 to 1.7 of e n/eps0 at these walls.

    Three things have to line up for that, and each was wrong once. The density
    at the half step has to be the one the first half of the step ended on, or
    the charge a wall absorbs disappears between the halves with no current to
    carry it. The initial field has to be built from the density the loop starts
    from, which at a wall is not where the particles were placed. And the filter
    has to be applied to the density before the current is taken from it.

    Two absorbing walls are conductors short-circuited to each other, so the far one
    also stays at the potential of the near one.
    """
    e = Species.electrons(n=2000, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0), quiet=True,
                          reflection=reflection)
    i = Species.ions(n=2000, density=1e17, electrons=e, quiet=True)
    domain = Domain(length=1e-2, cells=32, dt_over_dx_c=1.0, particle_bc=particle_bc, field_bc=field_bc)
    out = Simulation(domain, [e, i], Solver(algorithm, filter_passes=filter_passes,
                                            filter_strides=(1, 2))).run(120, seed=0)
    d = diagnostics(out)
    assert float(np.asarray(d["gauss_residual"]).max()) < 1e-10
    # relative to e n/eps0, the density of one sign of charge, and not to the net density; at the first
    # step, before a wall has taken any ion
    kicked = gauss_residual(out.replace(E=out.E.at[:, 10, 0].add(1.0)))
    assert float(kicked[0]) == pytest.approx(1 / out.dx / (1e17 * e_charge / epsilon_0), rel=1e-6)
    if field_bc == "absorbing":
        # short-circuited conductors: the right wall stays at the potential of the left one
        phi = np.abs(np.asarray(d["potential"]))
        assert float(phi[:, -1].max()) < 1e-9 * float(phi.max())


@pytest.mark.parametrize("wall, conserving", [("periodic", True), ("reflective", True),
                                              ("absorbing", False)])
def test_the_filter_moves_the_sources_around_without_inventing_any(wall, conserving):
    """Smoothing redistributes charge; it must not create it. A reflective wall
    mirrors the stencil back into the box, so the total is untouched; only an
    absorbing wall, which is supposed to let charge leave, may lose any."""
    rho = np.zeros(24)
    rho[[0, 3, 20, 23]] = 1.0                    # deliberately loaded against both walls
    code = {"periodic": (0, 0), "reflective": (1, 1), "absorbing": (2, 2)}[wall]
    total = float(jnp.sum(smooth(jnp.asarray(rho), 2, 0.5, (1, 2, 4), code)))
    if conserving:
        assert abs(total / rho.sum() - 1) < 1e-12
    else:
        assert total < 0.95 * rho.sum()


def test_electrostatic_solvers_agree_and_both_satisfy_gauss(quiet_two_stream):
    """`field_solver="gauss"` recomputes E_x from the charge density instead of
    advancing it with the current. On an electrostatic problem the two must
    agree to the discretisation error, and both must satisfy the discrete Gauss
    law: one by construction, the other because the current conserves charge."""
    ampere = quiet_two_stream
    gauss = two_stream("explicit", 300, n=4000, drift=5e7, quiet=True, field_solver="gauss")
    for out in (ampere, gauss):
        assert float(np.asarray(diagnostics(out)["gauss_residual"]).max()) < 1e-10
    field = np.asarray(ampere.E[:, :, 0])
    assert np.abs(np.asarray(gauss.E[:, :, 0]) - field).max() < 1e-3 * np.abs(field).max()


def test_relativistic_run_conserves_the_energy_the_pusher_conserves():
    """With `relativistic=True` the kinetic energy is sum (gamma - 1) m c^2, and
    the diagnostic has to follow the solver: reporting the Newtonian energy for a
    relativistic run would show a spurious drift where there is none."""
    def run(relativistic, steps):
        e = Species.electrons(n=2000, density=1e17, vth=(0.3 * c, 0, 0), quiet=True,
                              perturbation_amplitude=1e-5, perturbation_mode=1)
        i = Species.ions(n=2000, density=1e17, electrons=e, quiet=True)
        return Simulation(Domain(length=0.05, cells=32, dt_over_dx_c=1.0), [e, i],
                          Solver(relativistic=relativistic)).run(steps, seed=0)

    out = run(True, 200)
    total = np.asarray(diagnostics(out)["total"])
    assert float(np.abs(np.asarray(out.v)).max()) > 0.5 * c      # relativity matters here
    assert float(np.max(np.abs(total / total[0] - 1))) < 1e-3
    gamma = 1 / np.sqrt(1 - np.sum(np.asarray(out.v[0]) ** 2, axis=-1) / c ** 2)
    expected = float(np.sum((gamma - 1) * np.asarray(out.mass * out.weight[0]) * c ** 2))
    assert abs(float(diagnostics(out)["kinetic"][0]) / expected - 1) < 1e-12

    # a Newtonian run reports (1/2) m v^2, which is lower; two steps are enough to read it
    # (the frequency diagnostic needs more than one stored sample)
    newtonian = run(False, 2)
    half_mv2 = float(np.sum(0.5 * np.asarray(newtonian.mass * newtonian.weight[0])
                            * np.sum(np.asarray(newtonian.v[0]) ** 2, axis=-1)))
    assert abs(float(diagnostics(newtonian)["kinetic"][0]) / half_mv2 - 1) < 1e-12
    assert half_mv2 < expected


def test_an_external_magnetic_field_magnetises_the_plasma():
    """A uniform external B_x is the one field the code cannot generate itself,
    because the curl has no x component in one dimension. Particles in it gyrate
    at Omega_c = qB/m in the y-z plane at constant speed, since a magnetic field
    does no work."""
    B0, cells, length, n = 5e-4, 32, 1.0, 2000
    omega_c = e_charge * B0 / mass_electron
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


def test_a_floating_wall_holds_the_sheath_drop_of_hobbs_and_wesson():
    """The edge of a plasma, between a thermal wall that re-emits a Maxwellian as the
    plasma behind it would (Schwager and Birdsall, Phys. Fluids B 2, 1057, 1990) and
    a floating conductor. The conductor charges until it holds back all but as many
    electrons as ions arrive. From where the ions reach the Bohm speed
    c_s = sqrt(T_e/m_i) the potential then falls to the wall by
    (T_e/e)[(1/2) ln(m_i/2 pi m_e) + ln(1 - R)] when the wall returns the fraction R
    of the electrons (Hobbs and Wesson, Plasma Phys. 9, 85, 1967). Checked for a wall
    that keeps everything and one that returns half, together with the structure: a
    positive layer against the conductor and a neutral plasma away from it.

    The thermal wall keeps the electrons reaching the conductor Maxwellian, which is
    what the formula needs; between two absorbing walls the tail is stripped within a
    few transits. The mass ratio is 100 to fit the ion transit into a test.

    The drop is converged in cell size here, and what limits the comparison is where the
    edge is put: the potential still falls by about 0.1 T_e/e per Debye length at the Bohm
    point, which these bins, two Debye lengths wide, place to within about one. Both drops
    are held to 0.15 T_e/e; they come out 0.06 and 0.002 away, and a wall that returned no
    electron in the second case would miss by ln 2 = 0.69.
    """
    T_e, density, mass_ratio, n, cells, steps = 1.0, 1e16, 100.0, 20000, 60, 1500
    sigma = np.sqrt(T_e * e_charge / mass_electron)
    omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
    debye, c_s = sigma / omega_pe, sigma / np.sqrt(mass_ratio)
    length = 30 * debye
    domain = Domain(length=length, cells=cells, dt_over_dx_c=(0.2 / omega_pe) * c / (length / cells),
                    particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
    x, v = quiet_start(n, length, vth=(np.sqrt(2) * sigma, 0, 0))
    v_th_i = np.sqrt(2) * sigma / np.sqrt(40 * mass_ratio)
    x_i, v_i = quiet_start(n, length, vth=(v_th_i, 0, 0))
    ions = Species("ions", n, 1.0, mass_ratio * mass_electron, density, (v_th_i, 0, 0)).replace(x=x_i, v=v_i)
    bins = np.linspace(-length / 2, length / 2, cells // 4 + 1)
    faces = np.asarray(domain.grid) + domain.dx / 2
    theory = 0.5 * np.log(mass_ratio / (2 * np.pi))
    for reflection, expected in ((0.0, theory), (0.5, theory - np.log(2))):
        electrons = Species.electrons(n=n, density=density, vth=(np.sqrt(2) * sigma, 0, 0),
                                      reflection=(0.0, reflection)).replace(x=x, v=v)
        out = Simulation(domain, [electrons, ions], Solver(filter_passes=4)).run(steps, seed=0, store_every=50)
        late = slice(15, None)
        phi = np.asarray(potential(out))[late] / T_e
        position, speed, weight = (np.asarray(a)[late, n:] for a in (out.x[..., 0], out.v[..., 0], out.weight))
        flow = (np.histogram(position, bins, weights=weight * speed)[0]
                / np.maximum(np.histogram(position, bins, weights=weight)[0], 1e-300))
        assert (flow >= c_s).any()
        # where the flow crosses c_s, between the first bin that reaches it and the one before; a bin
        # centre alone would move the edge by half a bin, where the potential falls 0.1 T_e/e per lambda_D
        k, centres = int(np.argmax(flow >= c_s)), 0.5 * (bins[:-1] + bins[1:])
        edge = np.interp(c_s, flow[k - 1:k + 1], centres[k - 1:k + 1]) if k > 0 else centres[0]
        drop = np.interp(edge, faces, phi.mean(axis=0) - phi[:, -1].mean())
        assert abs(drop - expected) < 0.15, (reflection, drop, expected)
        rho = np.asarray(out.rho)[late].mean(axis=0) / (density * e_charge)
        assert rho[-3:].mean() > 0.02 and abs(rho[: cells // 2].mean()) < 0.01
