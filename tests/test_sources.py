"""The maintained source, the wall ledger, the electrode closure and the electrostatic
model. Each test states the closed form it checks against, or the invariant that has to
hold whatever the numbers are."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import (Domain, Impacts, Simulation, Solver, Source, Species, bohm_edge, epsilon_0,
                       gauss_residual, mass_electron, potential, elementary_charge as e_charge,
                       speed_of_light as c)
from jaxincell._core import apply_particle_bc, deposit
from jaxincell._sources import _flux_cdf, _flux_quantile, crossing_flux, sample_crossing
from jaxincell.sheath import densities, floating_potential, hobbs_wesson, source_density

SIGMA = np.sqrt(1.0 * e_charge / mass_electron)          # electron spread at T_e = 1 eV
DENSITY = 1e16
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
DEBYE = SIGMA / OMEGA_PE


def box(cells=32, length=None, steps_per_plasma_period=10.0, **domain):
    """A box a few Debye lengths across, stepped at a tenth of the plasma period."""
    length = 10 * DEBYE if length is None else length
    dt = 1.0 / (steps_per_plasma_period * OMEGA_PE)
    return Domain(length=length, cells=cells, dt_over_dx_c=dt * c / (length / cells), **domain)


def maxwellian_source(emit, density=DENSITY, **kwargs):
    return Source(density=density, vth=(np.sqrt(2) * SIGMA,) * 3, emit=emit, **kwargs)


# --- the crossing distribution -----------------------------------------------------------

def test_the_crossing_flux_is_the_closed_form_for_a_maxwellian_and_a_beam():
    """Gamma = n sigma / sqrt(2 pi) at rest, n u for a cold beam, and in between the
    drifting form n [u Phi(u/sigma) + sigma phi(u/sigma)], which reduces to both."""
    rest = maxwellian_source(4)
    assert float(crossing_flux(rest)) == pytest.approx(DENSITY * SIGMA / np.sqrt(2 * np.pi), rel=1e-12)
    beam = Source(density=DENSITY, vth=0.0, drift=(3e5, 0, 0), emit=4)
    assert float(crossing_flux(beam)) == pytest.approx(DENSITY * 3e5, rel=1e-12)
    # a drift the plane does not select on leaves the flux alone
    tangential = maxwellian_source(4).replace(drift=(0.0, 7e5, 0.0))
    assert float(crossing_flux(tangential)) == pytest.approx(float(crossing_flux(rest)), rel=1e-12)


def test_the_sampler_draws_the_flux_and_not_the_velocity_density():
    """A plane is crossed by fast particles more often, so the normal speed is Rayleigh:
    <v_n> = sigma sqrt(pi/2), <v_n^2> = 2 sigma^2, twice the variance of the Maxwellian it
    came from. The tangential components are not selected on and keep the spread their own
    component of vth asks for, which is not the normal one."""
    isotropic = maxwellian_source(8).replace(vth=(np.sqrt(2) * SIGMA,) * 3)
    v = np.asarray(sample_crossing(random.PRNGKey(0), isotropic, 200000, 1.0))
    assert np.all(v[:, 0] > 0)
    assert v[:, 0].mean() / SIGMA == pytest.approx(np.sqrt(np.pi / 2), rel=3e-3)
    assert (v[:, 0] ** 2).mean() / SIGMA ** 2 == pytest.approx(2.0, rel=5e-3)
    assert (v[:, 1] ** 2).mean() / SIGMA ** 2 == pytest.approx(1.0, rel=5e-3)
    assert abs(v[:, 1].mean()) / SIGMA < 0.01
    # the right wall sends the same distribution the other way
    left = np.asarray(sample_crossing(random.PRNGKey(0), isotropic, 5000, -1.0))
    assert np.all(left[:, 0] < 0)


def test_each_tangential_component_gets_the_spread_it_was_given():
    """vth has three components and all three are used. A reservoir spread only along the
    normal emits particles with no tangential motion at all, which in a magnetised sheath is
    a different physical inflow from an isotropic one -- and was what every source in this
    repository silently was, because the sampler used vth[0] for all three."""
    source = maxwellian_source(8).replace(vth=(np.sqrt(2) * SIGMA, 0.7 * np.sqrt(2) * SIGMA, 0.0),
                                          drift=(0.0, 1e4, -2e4))
    v = np.asarray(sample_crossing(random.PRNGKey(3), source, 100000, 1.0))
    assert v[:, 1].std() / SIGMA == pytest.approx(0.7, rel=1e-2)
    assert v[:, 2].std() == 0.0
    assert v[:, 1].mean() == pytest.approx(1e4, abs=0.02 * SIGMA)      # the tangential drift rides along
    assert np.allclose(v[:, 2], -2e4)


def test_a_reservoir_that_drifts_towards_the_plane_is_sampled_from_its_own_flux():
    """p(v) ∝ v exp[-(v-u)^2/2 sigma^2] on v > 0 has no elementary inverse, so it is sampled
    by inverting its distribution function. Shifting a Rayleigh sample by u is a different
    distribution; the two agree only at u = 0. Checked against quadrature of p itself, and at
    u = 0 against the closed form the Rayleigh branch uses."""
    sigma = SIGMA
    for ratio in (-1.0, 0.5, 2.0):
        source = Source(density=DENSITY, vth=(np.sqrt(2) * sigma,) * 3, drift=(ratio * sigma, 0, 0),
                        emit=4, model="drifting")
        v = np.asarray(sample_crossing(random.PRNGKey(5), source, 100000, 1.0))[:, 0]
        grid = np.linspace(0, max(ratio, 0) * sigma + 14 * sigma, 200001)
        weight = grid * np.exp(-(grid - ratio * sigma) ** 2 / (2 * sigma ** 2))
        norm = np.trapezoid(weight, grid)
        assert np.all(v > 0)
        assert v.mean() == pytest.approx(np.trapezoid(grid * weight, grid) / norm, rel=5e-3)
        assert (v ** 2).mean() == pytest.approx(np.trapezoid(grid ** 2 * weight, grid) / norm, rel=8e-3)
        # Gamma = n [u Phi(u/sigma) + sigma phi(u/sigma)] is the same integral, in closed form
        assert float(crossing_flux(source)) == pytest.approx(        # 1e-6 is the quadrature's
            DENSITY * norm / (np.sqrt(2 * np.pi) * sigma), rel=1e-6)   # error, not the formula's
    # and a shifted Rayleigh is not it: at u = 2 sigma the means differ by more than a spread
    shifted = SIGMA * np.sqrt(np.pi / 2) + 2 * sigma
    assert abs(shifted - v.mean()) > 0.2 * sigma


def test_the_quantile_of_the_drifting_flux_is_inverted_and_differentiated_exactly():
    """The sampler inverts F(t, a) = p by bisection, which has derivative zero: a fixed
    number of comparisons would report no sensitivity to the drift at all. The rule
    differentiates the equation instead. Value against F, derivative against a central
    difference of the bisection itself."""
    p = jnp.linspace(0.005, 0.995, 41)
    for a in (-2.0, 0.0, 0.7, 3.0):
        assert np.allclose(np.asarray(_flux_cdf(_flux_quantile(p, a), a)), np.asarray(p), atol=1e-6, rtol=0)
        step = 1e-4
        difference = float((_flux_quantile(0.37, a + step) - _flux_quantile(0.37, a - step)) / (2 * step))
        assert float(jax.grad(lambda drift: _flux_quantile(0.37, drift))(a)) == pytest.approx(
            difference, rel=1e-6)
    # at rest the quantile is the Rayleigh closed form the fast path uses
    assert np.allclose(np.asarray(_flux_quantile(p, 0.0)), np.asarray(jnp.sqrt(-2 * jnp.log(1 - p))),
                       rtol=1e-9, atol=0)


def test_a_cold_beam_carries_its_drift_and_one_pointing_outwards_is_refused():
    beam = Source(density=DENSITY, vth=0.0, drift=(3e5, 1e5, 0), emit=4)
    v = np.asarray(sample_crossing(random.PRNGKey(0), beam, 16, 1.0))
    assert np.allclose(v[:, 0], 3e5, rtol=1e-12, atol=0) and np.allclose(v[:, 1], 1e5, rtol=1e-12, atol=0)
    assert np.all(v[:, 2] == 0.0)
    assert float(crossing_flux(beam)) == pytest.approx(DENSITY * 3e5, rel=1e-12)
    with pytest.raises(ValueError, match="needs a normal drift towards the box"):
        Source(density=DENSITY, vth=0.0, drift=(-3e5, 0, 0), emit=4)
    right = Source(density=DENSITY, vth=0.0, drift=(-3e5, 0, 0), emit=4, side="right")
    assert float(crossing_flux(right)) == pytest.approx(DENSITY * 3e5, rel=1e-12)
    assert np.allclose(np.asarray(sample_crossing(random.PRNGKey(0), right, 16, -1.0))[:, 0], -3e5,
                       rtol=1e-12, atol=0)


def test_which_crossing_distribution_a_source_is_gets_decided_once_and_not_read_from_a_tracer():
    """`model` selects a branch, so it cannot be a traced leaf: inside jit every `vth` is a
    tracer, `vth != 0` is not a Python bool, and a Maxwellian source would sample a beam at
    rest and emit particles that never move. The same holds for the normal drift, which picks
    between the Rayleigh closed form and the drifting quantile. Both are fixed when the Source
    is built, and a Source built from a tracer has to say which it is."""
    warm = maxwellian_source(4)
    assert warm.model == "maxwellian" and warm.beam is False
    assert Source(density=DENSITY, vth=0.0, drift=(1e5, 0, 0), emit=4).model == "beam"
    assert Source(density=DENSITY, vth=(1e6, 0, 0), drift=(1e5, 0, 0), emit=4).model == "drifting"

    def speeds(source):
        return sample_crossing(random.PRNGKey(0), source, 64, 1.0)[:, 0]

    inside = jax.jit(speeds)(warm)
    assert float(jnp.std(inside)) > 0.1 * SIGMA                       # a beam would have none
    assert np.allclose(np.asarray(inside), np.asarray(speeds(warm)))
    with pytest.raises(ValueError, match="traced vth"):
        jax.jit(lambda vth: Source(density=DENSITY, vth=(vth, 0, 0), emit=4).model)(1e6)
    with pytest.raises(ValueError, match="traced normal drift"):
        jax.jit(lambda u: Source(density=DENSITY, vth=(1e6, 0, 0), drift=(u, 0, 0), emit=4).model)(1e5)


# --- what the source puts in --------------------------------------------------------------

def test_the_emitted_weight_is_exactly_the_prescribed_flux_and_differentiable_in_it():
    """N_emit particles of weight Gamma dt / N_emit put in exactly Gamma dt, whatever
    N_emit is, and its derivatives with respect to the reservoir density and thermal speed
    are the derivatives of Gamma: d/dn = sigma/sqrt(2 pi) dt and d/dvth = n/(2 sqrt(pi)) dt.
    A count would have neither, being an integer."""
    domain = box(particle_bc="absorbing", field_bc=("open", "absorbing"))
    steps = 5

    def emitted(density, vth):
        source = Source(density=density, vth=(vth, 0, 0), emit=7, model="maxwellian")
        species = Species("electrons", 400, -1.0, mass_electron, 0.0, source=source)
        out = Simulation(domain, [species], Solver(model="electrostatic")).run(steps, store_particles=False)
        return out.wall.injected[-1, 0, 0]

    vth = np.sqrt(2) * SIGMA
    value, grad = jax.value_and_grad(emitted, argnums=(0, 1))(DENSITY, vth)
    assert float(value) == pytest.approx(DENSITY * SIGMA / np.sqrt(2 * np.pi) * domain.dt * steps, rel=1e-12)
    assert float(grad[0]) == pytest.approx(SIGMA / np.sqrt(2 * np.pi) * domain.dt * steps, rel=1e-10)
    assert float(grad[1]) == pytest.approx(DENSITY / (2 * np.sqrt(np.pi)) * domain.dt * steps, rel=1e-10)


def flux_energy(vth, mass):
    """Mean and variance of the kinetic energy of one draw from a half-Maxwellian flux.

    The normal square is exponential, :math:`v_n^2 = 2\\sigma_x^2 E`, so it has mean
    :math:`2\\sigma_x^2` and variance :math:`4\\sigma_x^4`; each tangential square is
    :math:`\\sigma^2\\chi^2_1`, mean :math:`\\sigma^2` and variance :math:`2\\sigma^4`. The
    variance is what the scatter of a measured mean is, so a test can ask whether a difference
    is real instead of guessing a tolerance."""
    sigma = np.asarray(vth) / np.sqrt(2)
    mean = 0.5 * mass * (2 * sigma[0] ** 2 + sigma[1] ** 2 + sigma[2] ** 2)
    variance = 0.25 * mass ** 2 * (4 * sigma[0] ** 4 + 2 * sigma[1] ** 4 + 2 * sigma[2] ** 4)
    return mean, variance


def test_the_ledger_carries_the_energy_and_momentum_a_source_puts_in():
    """A budget for an open box needs what came in as well as what went out. The reservoir
    emits N particles of weight Gamma dt / N whose mean energy is the flux-Maxwellian's
    m(2 sigma_x^2 + sigma_y^2 + sigma_z^2)/2 and whose mean normal momentum is
    m sigma_x sqrt(pi/2), both per unit weight and both closed forms.

    A mean over a finite sample is compared within four standard errors of that sample, worked
    out from the variance of the same distribution rather than guessed: at 2400 draws the
    scatter alone is 1.6 %, which a tolerance of 2 % would call a defect one time in a hundred
    -- and, under the absolute tolerance `pytest.approx` keeps by default, would never call
    anything at all."""
    domain = box(cells=16, particle_bc="absorbing", field_bc=("open", "absorbing"))
    vth = (np.sqrt(2) * SIGMA, 0.6 * np.sqrt(2) * SIGMA, 0.0)
    species = Species("electrons", 20000, -1.0, mass_electron, 0.0,
                      source=Source(density=DENSITY, vth=vth, emit=60))
    steps, emit = 200, 60
    out = Simulation(domain, [species], Solver(model="electrostatic")).run(steps, store_every=steps)
    wall = jax.tree.map(lambda a: a[-1], out.wall)
    weight = float(wall.injected[0, 0])
    sigma = np.asarray(vth) / np.sqrt(2)
    assert weight == pytest.approx(DENSITY * SIGMA / np.sqrt(2 * np.pi) * domain.dt * steps, rel=1e-12, abs=0)
    mean, variance = flux_energy(vth, mass_electron)
    assert abs(float(wall.energy_injected[0, 0]) / weight - mean) < 4 * np.sqrt(variance / (steps * emit))
    # <v_n> = sigma sqrt(pi/2) with variance (2 - pi/2) sigma^2
    momentum = mass_electron * sigma[0] * np.sqrt(np.pi / 2)
    scatter = mass_electron * sigma[0] * np.sqrt((2 - np.pi / 2) / (steps * emit))
    assert abs(float(wall.momentum_injected[0, 0, 0]) / weight - momentum) < 4 * scatter
    assert abs(float(wall.momentum_injected[0, 0, 1]) / weight) < 4 * mass_electron * sigma[1] / np.sqrt(
        steps * emit)
    assert float(wall.momentum_injected[0, 0, 2]) == 0.0
    # nothing was injected through the wall the source is not on
    assert float(wall.injected[0, 1]) == 0.0 and float(wall.energy_injected[0, 1]) == 0.0


def test_an_injected_half_maxwellian_fills_the_box_to_half_the_reservoir_density():
    """Free streaming from a reservoir gives the density of the half-space behind the
    plane, n/2, uniformly: the flux Gamma = n sigma / sqrt(2 pi) divided by the mean
    inverse speed sqrt(pi/2)/sigma of the crossing distribution. The self-field is made
    negligible by a tiny reservoir density, so this tests the sampler, the entry times and
    the free flight and nothing else."""
    tiny = 1e-10 * DENSITY
    domain = box(cells=64, particle_bc="absorbing", field_bc="reflective")
    species = Species("electrons", 20000, -1.0, mass_electron, 0.0, source=maxwellian_source(20, density=tiny))
    out = Simulation(domain, [species], Solver(model="electrostatic")).run(1500, store_every=250, moments=True)
    assert float(jnp.max(jnp.abs(out.E))) < 1e-4                      # the field really is out of the way
    window = (out.moments[-1] - out.moments[-2]) / 250
    interior = np.asarray(window[0, 0, 4:-4]) / tiny
    assert interior.mean() == pytest.approx(0.5, rel=0.06)
    assert interior.std() < 0.08


def test_an_injected_particle_enters_on_a_trajectory_and_not_on_a_straight_line():
    """An emitted particle enters part-way through a step and is put in the arrays as if it
    had always been there. Streaming it freely to the end of that interval and then giving it
    the whole step's push leaves an error of order (q/m)|E| dt / v in its entry velocity, and
    a run has no way to see it. Both are checked against a closed form: a cold beam falling
    through a prescribed uniform field has n(x) = Gamma / v(x) with v^2 = v_0^2 + 2(q/m)E x,
    and one turning in a prescribed uniform B moves on a circle of radius v_0/Omega, giving
    n(x) = 2 Gamma / (v_0 sqrt(1 - (x/r)^2)) out to that radius."""
    length, cells, v0 = 1e-2, 64, 1e6
    mass = 1e4 * mass_electron            # heavy and tenuous, so its own field is nothing
    over_mass, reservoir = e_charge / mass, 1e6

    def profile(steps, external_E=None, external_B=None, dt=1.0):
        domain = Domain(length=length, cells=cells, time_step=dt, particle_bc="absorbing",
                        field_bc=("open", "absorbing"))
        beam = Species("beam", 80000, 1.0, mass, 0.0,
                       source=Source(density=reservoir, vth=0.0, drift=(v0, 0, 0), emit=20))
        out = Simulation(domain, [beam], Solver(model="electrostatic"),
                         external_E=external_E, external_B=external_B).run(
            steps, store_every=steps // 4, store_particles=False, moments=True)
        assert out.problems == ()
        return np.asarray(out.moments[-1] - out.moments[-2])[0, 0] / (steps // 4)

    x = np.asarray(Domain(length=length, cells=cells).grid) + length / 2
    field = 0.5 * v0 ** 2 / (over_mass * length)                    # doubles the kinetic energy
    n = profile(600, external_E=jnp.zeros((cells, 3)).at[:, 0].set(field), dt=length / v0 / 200)
    exact = reservoir * v0 / np.sqrt(v0 ** 2 + 2 * over_mass * field * x)
    assert np.abs(n[3:-3] / exact[3:-3] - 1).max() < 2e-4           # free streaming gives 1.1e-3

    radius = length / 3.0
    strength = v0 / (over_mass * radius)
    n = profile(480, external_B=jnp.zeros((cells, 3)).at[:, 2].set(strength),
                dt=2 * np.pi / (over_mass * strength) / 80)         # Omega dt = 0.079
    turning = (np.arange(cells) >= 3) & (x < 0.7 * radius)      # past the plane's cloud truncation
    exact = 2 * reservoir / np.sqrt(np.maximum(1 - (x / radius) ** 2, 1e-30))
    assert np.abs(n[turning] / exact[turning] - 1).mean() < 5e-3    # free streaming gives 4.0e-2


def test_the_pool_is_capacity_and_the_source_refills_the_slots_the_walls_empty():
    """An empty start is a full set of dead slots, not an empty array: the shapes never
    change. Dead slots hold no weight and no charge-to-mass ratio, deposit nothing and feel
    nothing, and the source takes the emptiest ones. The population settles where the flux
    and the residence time put it, well inside the capacity, and `overflow` stays zero."""
    domain = box(cells=32, particle_bc="absorbing", field_bc="reflective")
    species = Species("electrons", 6000, -1.0, mass_electron, 0.0,
                      source=maxwellian_source(20, density=1e-10 * DENSITY))
    sim = Simulation(domain, [species], Solver(model="electrostatic"))
    start, _ = sim.initial_state(random.PRNGKey(0))
    assert float(jnp.sum(start.w)) == 0.0 and float(jnp.sum(jnp.abs(start.qm))) == 0.0
    assert float(jnp.max(start.x[:, 0])) < -domain.length / 2       # all parked beyond the wall
    out = sim.run(1500, store_every=500, store_particles=False)
    live = np.asarray(out.state.w) > 0
    assert 1000 < live.sum() < 5000
    assert float(out.wall.overflow[-1]) == 0.0
    assert float(out.wall.overflow[-1]) == 0.0
    assert float(out.wall.arrived[-1, 0, 1]) > 0                     # and they are leaving at the far wall


def test_a_pool_too_small_says_so_instead_of_pretending():
    """With no room left the source overwrites the lightest live particle rather than
    skipping the emission or changing the flux, and reports the largest weight it
    destroyed. Silence here would be a density quietly set by the array size."""
    domain = box(cells=16, particle_bc="absorbing", field_bc="reflective")
    species = Species("electrons", 60, -1.0, mass_electron, 0.0,
                      source=maxwellian_source(20, density=1e-10 * DENSITY))
    out = Simulation(domain, [species], Solver(model="electrostatic")).run(200, store_particles=False)
    assert float(out.overflow[-1]) > 0.0
    # and saying so is not enough: a run that overwrote live particles has to be refusable in
    # one line, or every script has to remember to look
    assert len(out.problems) == 1 and "overwrote live particles" in out.problems[0]
    with pytest.raises(RuntimeError, match="Species.n is a capacity"):
        out.validate()
    # the report is a running maximum, so a run cannot look valid because it recovered later
    assert np.all(np.diff(np.asarray(out.overflow)) >= 0)
    roomy = Simulation(domain, [species.replace(n=6000)], Solver(model="electrostatic")).run(
        200, store_particles=False)
    assert roomy.problems == () and roomy.validate() is roomy


def test_active_separates_the_initial_population_from_the_capacity():
    """`n` is how many slots there are and `active` how many start filled; the filled ones
    spread over the whole box and carry density * length / active, so the physical density
    does not depend on the headroom left for a source."""
    domain = box(cells=16)
    for active in (200, 1000):
        species = Species.electrons(n=1000, active=active, density=DENSITY, vth=(SIGMA, 0, 0))
        ions = Species.ions(n=1000, active=active, density=DENSITY, mass_ratio=1e9, vth=0.0)
        state, _ = Simulation(domain, [species, ions], Solver()).initial_state(random.PRNGKey(0))
        w, x = np.asarray(state.w)[:1000], np.asarray(state.x)[:1000, 0]
        assert (w > 0).sum() == active
        assert w.sum() == pytest.approx(DENSITY * domain.length, rel=1e-12)
        assert x[w > 0].max() - x[w > 0].min() == pytest.approx(domain.length, rel=0.02)
    with pytest.raises(ValueError, match="active must be between"):
        Species.electrons(n=10, active=11, density=DENSITY)


def test_an_empty_start_is_safe_under_jit_grad_and_vmap():
    """`active=0` is how a source-driven run naturally begins: an empty box that the
    reservoir fills. Every slot is then dead, and the initial spacing and weight divide by
    `active`. Both divisions sit in the untaken branch of a `where`, so a forward run
    survives them -- but the cotangent of `where(False, inf, 0)` is NaN, and an empty start
    is exactly the configuration an optimisation over a source would use."""
    domain = box(cells=16, particle_bc="absorbing", field_bc=("open", "absorbing"))

    def field_energy(species_density, source_density):
        species = Species("electrons", 200, -1.0, mass_electron, species_density, (SIGMA, 0, 0), active=0,
                          source=maxwellian_source(4, density=source_density))
        return jnp.sum(Simulation(domain, [species], Solver(model="electrostatic")).run(8).E ** 2)

    state, _ = Simulation(domain, [Species("electrons", 200, -1.0, mass_electron, DENSITY, (SIGMA, 0, 0),
                                           active=0, source=maxwellian_source(4))],
                          Solver(model="electrostatic")).initial_state(random.PRNGKey(0))
    assert float(jnp.sum(state.w)) == 0.0                      # nothing is alive at t = 0
    assert np.all(np.isfinite(np.asarray(state.x)))
    energy = float(field_energy(DENSITY, DENSITY))
    assert np.isfinite(energy) and energy > 0                  # the source has filled some of the box
    # an empty start does not depend on the species density, and says so with a zero
    assert float(jax.grad(field_energy, argnums=0)(DENSITY, DENSITY)) == 0.0
    slope = float(jax.grad(field_energy, argnums=1)(DENSITY, DENSITY))
    assert np.isfinite(slope) and slope != 0.0
    seeds = jax.vmap(lambda seed: jnp.sum(Simulation(
        domain, [Species("electrons", 200, -1.0, mass_electron, DENSITY, (SIGMA, 0, 0), active=0,
                         source=maxwellian_source(4))],
        Solver(model="electrostatic")).run(8, seed=seed).E ** 2))(jnp.arange(3))
    assert np.all(np.isfinite(np.asarray(seeds))) and np.all(np.asarray(seeds) > 0)


# --- what the walls take out ----------------------------------------------------------------

def test_the_wall_ledger_counts_one_impact_exactly():
    """One particle of known weight and speed hits a wall that returns the fraction R of
    it. The wall then holds (1 - R) w and gives back R w, and the energy it kept is the
    difference between what arrived and what left, which a coefficient of restitution e
    reduces by the factor e^2 on the way out."""
    w, speed, R, restitution = 3.0, 2e6, 0.25, 0.8
    x = jnp.array([[0.6, 0.0, 0.0]])
    v = jnp.array([[speed, 0.0, 0.0]])
    reflection = (jnp.zeros(1), jnp.full(1, R))
    _, v_out, w_out, _, (arrived, kept, _, _) = apply_particle_bc(
        x, v, jnp.full(1, w), jnp.ones(1), (1.0, 1.0, 1.0), (2, 2), (1.0, restitution), reflection, 0.1)
    assert float(arrived[1, 0]) == pytest.approx(w) and float(arrived[0, 0]) == 0.0
    assert float(kept[1, 0]) == pytest.approx((1 - R) * w)
    assert float(w_out[0]) == pytest.approx(R * w)
    assert float(v_out[0, 0]) == pytest.approx(-restitution * speed)
    mass = mass_electron
    energy_in = 0.5 * mass * speed ** 2 * w
    energy_out = 0.5 * mass * (restitution * speed) ** 2 * (R * w)
    assert float(arrived[1, 0]) * 0.5 * mass * speed ** 2 == pytest.approx(energy_in, rel=1e-12, abs=0)
    assert float((arrived - kept)[1, 0]) * 0.5 * mass * (restitution * speed) ** 2 == pytest.approx(
        energy_out, rel=1e-12, abs=0)


def test_the_impact_spectrum_is_the_crossing_distribution_and_sums_to_the_fluence():
    """An impact spectrum is a record of crossings, not a snapshot of who is near the wall.
    A steady free-streaming reservoir delivers its own crossing distribution to the far wall,
    so the spectrum has two closed forms to meet: a mean energy of 2 T_e for an isotropic
    reservoir, and the cosine law, p(theta) = sin 2theta, for the incidence. Summing the
    accumulator over its bins has to give back the fluence exactly, or an entry has been
    counted twice or dropped."""
    domain = box(cells=32, particle_bc="absorbing", field_bc="reflective")
    bins = Impacts(energy_max=8 * e_charge, energy_bins=16, angle_bins=9)
    species = Species("electrons", 20000, -1.0, mass_electron, 0.0,
                      source=maxwellian_source(20, density=1e-10 * DENSITY))
    out = Simulation(domain, [species], Solver(model="electrostatic"), impacts=bins).run(
        1200, store_every=300, store_particles=False)
    assert float(jnp.max(jnp.abs(out.E))) < 1e-4                  # the field is out of the way
    spectrum = np.asarray(out.wall.spectrum)
    window = spectrum[-1] - spectrum[-2]                          # the last quarter of the run
    arrived = np.asarray(out.wall.arrived)
    assert window[0, 1].sum() == pytest.approx(float(arrived[-1, 0, 1] - arrived[-2, 0, 1]), rel=1e-12)
    assert window[0, 0].sum() == 0.0                              # nothing turns round to reach the source plane
    right = window[0, 1]
    assert right[-1].sum() / right.sum() < 0.01                   # 8 T_e is above almost everything
    centres = (np.arange(16) + 0.5) * float(bins.energy_max) / 16
    mean = (right[:-1].sum(axis=1) * centres).sum() / right[:-1].sum()
    assert mean / e_charge == pytest.approx(2.0, rel=0.02)        # m(2 sigma_x^2 + sigma_y^2 + sigma_z^2)/2
    edges = np.arange(10) * (np.pi / 2 / 9)
    assert np.allclose(right.sum(axis=0) / right.sum(), np.diff(-0.5 * np.cos(2 * edges)), atol=0.01)
    theta = 0.5 * (edges[:-1] + edges[1:])
    assert np.degrees((right.sum(axis=0) / right.sum() * theta).sum()) == pytest.approx(45.0, abs=1.0)


def test_a_spectrum_whose_range_is_too_small_says_so_in_its_overflow_bin():
    """Everything above `energy_max` goes to one extra bin rather than into the last resolved
    one, so a range that was chosen too small is visible instead of piling up on the end."""
    domain = box(cells=16, particle_bc="absorbing", field_bc="reflective")
    species = Species("electrons", 8000, -1.0, mass_electron, 0.0,
                      source=maxwellian_source(20, density=1e-10 * DENSITY))
    narrow = Impacts(energy_max=0.5 * e_charge, energy_bins=4)
    wide = Impacts(energy_max=40 * e_charge, energy_bins=4)
    fractions = []
    for bins in (narrow, wide):
        out = Simulation(domain, [species], Solver(model="electrostatic"), impacts=bins).run(
            600, store_every=600, store_particles=False)
        spectrum = np.asarray(out.wall.spectrum)[-1, 0, 1]
        assert spectrum.sum() == pytest.approx(float(out.wall.arrived[-1, 0, 1]), rel=1e-12)
        fractions.append(spectrum[-1].sum() / spectrum.sum())
    assert fractions[0] > 0.5 and fractions[1] < 1e-3


def test_the_wall_energy_of_a_relativistic_impact_is_the_relativistic_one():
    """m v^2/2 is not the kinetic energy of a particle at 0.9 c; (gamma - 1) m c^2 is, and
    it is three times larger. The ledger takes it from the carried momentum as
    m|u|^2/(gamma + 1), which is the same number written so that a slow particle does not
    lose it to the difference of two large ones."""
    speed, density = 0.9 * c, 1e-3            # a density low enough that the self-field is nothing
    gamma = 1 / np.sqrt(1 - 0.81)
    domain = Domain(length=1.0, cells=8, time_step=0.4 / c, particle_bc="absorbing", field_bc="reflective")
    species = Species("electrons", 1, -1.0, mass_electron, density,
                      x=np.array([[0.4, 0.0, 0.0]]), v=np.array([[speed, 0.0, 0.0]]))
    sim = Simulation(domain, [species], Solver(model="electrostatic", relativistic=True))
    wall = jax.tree.map(lambda a: a[-1], sim.run(3, store_every=3).wall)
    arrived = float(wall.arrived[0, 1])
    assert arrived == pytest.approx(density * domain.length, rel=1e-12)
    assert float(wall.energy_in[0, 1]) / arrived == pytest.approx(
        (gamma - 1) * mass_electron * c ** 2, rel=1e-9, abs=0)
    assert float(wall.energy_in[0, 1]) / arrived / (0.5 * mass_electron * speed ** 2) == pytest.approx(
        (gamma - 1) / (0.5 * 0.81), rel=1e-9)              # 3.19 times the Newtonian value
    assert float(wall.momentum[0, 1, 0]) / arrived == pytest.approx(
        gamma * mass_electron * speed, rel=1e-9, abs=0)
    # and the same expression is the Newtonian one when nothing is relativistic
    slow = Simulation(domain.replace(time_step=0.4 / c), [species.replace(v=np.array([[1e5, 0.0, 0.0]]))],
                      Solver(model="electrostatic"))
    state, extra = slow.initial_state(random.PRNGKey(0))
    assert float(slow._kinetic(extra[0], jnp.array([[1e5, 0.0, 0.0]]))[0]) == pytest.approx(
        0.5 * mass_electron * 1e10, rel=1e-12, abs=0)


def test_the_wall_charge_and_energy_add_up_over_a_run():
    """Everything a wall took is what arrived minus what went back, species by species and
    wall by wall, and the charge on the collector is the sum of the collected weights times
    the charges. Nothing is lost between the two."""
    domain = box(cells=32, particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
    electrons = Species.electrons(n=4000, density=DENSITY, vth=(np.sqrt(2) * SIGMA, 0, 0), reflection=(0.0, 0.3))
    ions = Species.ions(n=4000, density=DENSITY, mass_ratio=400.0, electrons=electrons)
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(200, store_every=50)
    wall = jax.tree.map(lambda a: a[-1], out.wall)
    arrived, collected = np.asarray(wall.arrived), np.asarray(wall.collected)
    assert np.all(collected <= arrived + 1e-9) and np.all(collected >= -1e-9)
    assert collected[0, 1] == pytest.approx(0.7 * arrived[0, 1], rel=1e-9)     # keeps 1 - R of every impact
    assert collected[1, 1] == pytest.approx(arrived[1, 1], rel=1e-12)          # ions are not reflected
    # the collector returns the fraction R of each impact at the same speed, so it never
    # gives back more than it received
    energy_in, energy_out = np.asarray(wall.energy_in), np.asarray(wall.energy_out)
    assert np.all(energy_out[:, 1] <= energy_in[:, 1] + 1e-30)
    charge = np.asarray(wall.charge([s.charge_si for s in (electrons, ions)]))
    assert charge[1] == pytest.approx(e_charge * (collected[1, 1] - collected[0, 1]), rel=1e-12)


def test_a_thermal_wall_is_a_heat_bath_and_the_ledger_says_so():
    """The energy a wall returns is the energy of the particle it re-emitted, which at a
    thermal wall is a fresh draw from its own half-Maxwellian flux and not the bounce that
    preceded the redraw. Recorded before the redraw -- as it was -- restitution one makes
    energy_in and energy_out identical and the wall appears to exchange exactly nothing,
    which is not a measurement. What it actually returns, per unit weight, is
    m(2 sigma_x^2 + sigma_y^2 + sigma_z^2)/2, whatever arrived."""
    domain = box(cells=32, particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
    n, vth = 60000, (np.sqrt(2) * SIGMA, SIGMA, 0.0)
    electrons = Species.electrons(n=n, density=DENSITY, vth=vth)
    out = Simulation(domain, [electrons], Solver(model="electrostatic")).run(300, store_every=300)
    wall = jax.tree.map(lambda a: a[-1], out.wall)
    arrived, out_energy = float(wall.arrived[0, 0]), float(wall.energy_out[0, 0])
    expected, variance = flux_energy(vth, mass_electron)
    impacts = arrived / (DENSITY * float(domain.length) / n)       # how many draws the mean is over
    assert impacts > 1000 and float(wall.collected[0, 0]) == 0.0   # a thermal wall keeps nothing
    assert abs(out_energy / arrived - expected) < 4 * np.sqrt(variance / impacts)
    assert abs(out_energy / float(wall.energy_in[0, 0]) - 1) > 0.02   # not the specular energy
    # and it pushes: what arrived less what left, in the direction the wall is driven
    assert float(wall.momentum[0, 0, 0]) < 0


def test_a_reflecting_wall_would_hold_a_particle_for_ever_without_a_floor():
    """Returning the fraction R of each impact leaves R^k after k of them, which never
    reaches zero, so the slot is never free for a source to refill. Below `min_weight`
    times the emitted weight the wall keeps the remainder; the charge goes on the ledger,
    so nothing is lost, and the slot comes back."""
    domain = box(cells=16, particle_bc="absorbing", field_bc="reflective")
    source = maxwellian_source(10, density=1e-10 * DENSITY)
    weights, cost = {}, {}
    for name, floor in (("floor", 1e-2), ("none", 0.0)):
        species = Species("electrons", 12000, -1.0, mass_electron, 0.0, reflection=0.5,
                          source=source.replace(min_weight=floor))
        out = Simulation(domain, [species], Solver(model="electrostatic")).run(600, store_particles=False)
        live = np.asarray(out.state.w)
        weights[name] = (live > 0).sum()
        assert float(out.overflow[-1]) == 0.0
        total = np.asarray(out.wall.collected)[-1].sum() + live.sum()
        assert total == pytest.approx(float(np.asarray(out.wall.injected)[-1].sum()), rel=1e-9)
        cost[name] = float(np.asarray(out.wall.truncated)[-1].sum() / np.asarray(out.wall.collected)[-1].sum())
    assert weights["floor"] < weights["none"]
    # what the cutoff cost is on the ledger, not assumed to be nothing
    assert cost["none"] == 0.0
    assert 1e-4 < cost["floor"] < 5e-2


def test_the_cutoff_budget_falls_with_the_cutoff():
    """`min_weight` truncates the last part of an orbit, and how much is a choice. The weight a
    wall takes for that reason rather than by its reflection law is on the ledger, and it falls
    with the cutoff, so a run can be refined until the budget is below whatever it is being
    compared against instead of hoping that it is."""
    domain = box(cells=16, particle_bc="absorbing", field_bc="reflective")
    source = maxwellian_source(10, density=1e-10 * DENSITY)
    budget = []
    for floor in (1e-1, 1e-2, 1e-3):
        species = Species("electrons", 20000, -1.0, mass_electron, 0.0, reflection=0.5,
                          source=source.replace(min_weight=floor))
        out = Simulation(domain, [species], Solver(model="electrostatic")).run(600, store_particles=False)
        assert out.problems == ()
        budget.append(float(np.asarray(out.wall.truncated)[-1].sum()
                            / np.asarray(out.wall.collected)[-1].sum()))
    assert budget[0] > budget[1] > budget[2] > 0.0
    assert budget[2] < 0.1 * budget[0]


# --- the electrical boundary -----------------------------------------------------------------

def test_the_collector_field_is_the_charge_it_holds_and_the_clouds_reaching_past_it():
    """E_x at the conductor face is -sigma_w/eps_0, and sigma_w is the charge the ledger says
    the collector has taken **plus** the part of the live clouds that reaches past it. A cloud
    is one and a half cells wide and so crosses the wall before its centre does; left out of
    both the volume and the surface, that part is simply gone."""
    domain = box(cells=48, particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", 4000, -1.0, mass_electron, 0.0,
                        source=maxwellian_source(10, density=1e-3 * DENSITY))
    ions = Species("ions", 4000, 1.0, 400 * mass_electron, 0.0,
                   source=Source(density=1e-3 * DENSITY, vth=0.0, drift=(0.2 * SIGMA, 0, 0), emit=10))
    sim = Simulation(domain, [electrons, ions], Solver(model="electrostatic"))
    out = sim.run(400, store_every=100)
    collected = np.asarray(out.wall.collected)[:, :, 1] @ np.array([-e_charge, e_charge])
    # from the integer-time positions the last deposit used, not the half-step ones the
    # leapfrog carries in the state
    overlap = float(sim._overlap_charge(out.x[-1], out.weight[-1]))
    assert np.asarray(out.E[:, -1, 0])[-1] == pytest.approx(-(collected[-1] + overlap) / epsilon_0, rel=1e-10)
    assert abs(overlap / collected[-1]) > 1e-4        # and it is not a rounding-sized correction


def test_a_sheet_crossing_the_collector_takes_its_whole_charge_with_it():
    """The deposited part and the part beyond the wall partition a particle exactly, at every
    sub-cell offset, and the field inside does not notice the crossing. Without the second
    part the two together swing between a half and one and a half of a particle, and the field
    at an interior face jumps by half a particle's worth the moment the centre crosses."""
    length, cells, dt = 1.0, 20, 1e-9
    dx = length / cells
    box_only = Domain(length=length, cells=cells, time_step=dt, particle_bc="absorbing",
                      field_bc=("open", "absorbing"))
    sheet = Species("sheet", 1, 1.0, 1e-10, 1.0 / e_charge,        # q w = 1 C/m^2, heavy and slow
                    x=np.array([[-length / 2 + 0.2, 0.0, 0.0]]), v=np.array([[0.04 * dx / dt, 0.0, 0.0]]))
    sim = Simulation(box_only, [sheet], Solver(model="electrostatic"))
    # the two parts of one particle, at forty sub-cell offsets through the wall
    for offset in np.linspace(-3.0, 0.0, 40):
        x = jnp.array([[length / 2 + offset * dx, 0.0, 0.0]])
        w = jnp.array([1.0 / e_charge])
        volume = float(jnp.sum(deposit(x[:, 0], jnp.array([e_charge]) * w, box_only.grid[0], dx,
                                       cells, box_only.particle_bc)) * dx)
        assert volume + float(sim._overlap_charge(x, w)) == pytest.approx(1.0, rel=1e-12)
    # and the field of the sheet at an interior face, as it approaches and crosses
    # the leftmost face, which the sheet starts to the right of and never returns past: every
    # charge in the box is between it and the collector, so its field is -1/eps_0 throughout
    out = sim.run(460, store_every=1)
    weight, field = np.asarray(out.weight)[:, 0], np.asarray(out.E)[:, 0, 0]
    crossing = int(np.argmax(weight == 0))
    assert 0 < crossing < len(weight) - 1                       # it really did cross, inside the run
    assert field[0] == pytest.approx(-1.0 / epsilon_0, rel=1e-12)
    assert np.ptp(field) < 1e-9 * abs(field[0])                 # half a particle would be 50 %
    assert field[crossing] == pytest.approx(field[crossing - 1], rel=1e-12)


def test_the_current_a_floating_collector_closes_on_is_the_real_one():
    """Ampere's law makes the total current uniform across a one-dimensional box, so
    J + eps0 dE/dt is the current in the external circuit. A floating collector is connected
    to nothing, so it vanishes -- at every face, not only at the wall. The continuity current
    is closed on the rate at which the electrode's charge changes, which is what makes that
    true; anchored at zero, as it was, the residual is six tenths of the current's own size
    and J is an internal transport measured from the source plane rather than a current."""
    domain = box(cells=48, particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", 12000, -1.0, mass_electron, DENSITY, (np.sqrt(2) * SIGMA,) * 3,
                        active=3000, quiet=True, source=maxwellian_source(12, density=1.1149 * DENSITY))
    ions = Species("ions", 12000, 1.0, 1836 * mass_electron, DENSITY, 0.0, (0.2 * SIGMA, 0, 0),
                   active=3000, quiet=True,
                   source=Source(density=DENSITY, vth=0.0, drift=(0.2 * SIGMA, 0, 0), emit=12))
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(
        200, store_every=1, store_particles=False)
    assert out.problems == ()
    J, E = np.asarray(out.J)[:, :, 0], np.asarray(out.E)[:, :, 0]
    residual = J[1:] + epsilon_0 * (E[1:] - E[:-1]) / float(domain.dt)
    assert np.abs(residual).max() < 1e-12 * np.abs(J[1:]).max()


def test_the_implicit_scheme_refuses_the_open_plane():
    """It carries no surface charge, so there is nothing to close the continuity current on."""
    with pytest.raises(ValueError, match="carries no surface charge"):
        Simulation(box(field_bc=("open", "absorbing"), particle_bc="absorbing"),
                   [Species.electrons(n=64, density=DENSITY, vth=(SIGMA, 0, 0))],
                   Solver(algorithm="implicit"))


def test_the_electrode_closure_and_a_symmetry_plane_agree_when_nothing_crosses():
    """With a wall that returns every particle on the left, the charge the collector has
    taken is all the charge the box has lost, so closing the field on that charge and
    closing it on E_x = 0 at a symmetry plane are the same problem. They agree to the
    charge the deposit truncates at the walls, which falls with the cell size."""
    errors = []
    for cells in (32, 64, 128):
        fields = []
        for field_bc in (("reflective", "absorbing"), ("open", "absorbing")):
            domain = box(cells=cells, particle_bc=("reflective", "absorbing"), field_bc=field_bc)
            electrons = Species.electrons(n=8000, density=DENSITY, vth=(np.sqrt(2) * SIGMA, 0, 0), quiet=True)
            ions = Species.ions(n=8000, density=DENSITY, mass_ratio=400.0, vth=0.0, quiet=True)
            out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(300, store_particles=False)
            fields.append(np.asarray(out.E[-1, :, 0]))
        errors.append(np.max(np.abs(fields[0] - fields[1])) / np.max(np.abs(fields[0])))
    assert errors[-1] < 0.02 and errors[-1] < errors[0]


def test_an_open_plane_needs_a_collector_opposite():
    with pytest.raises(ValueError, match="open"):
        box(field_bc="open")
    with pytest.raises(ValueError, match="open"):
        box(field_bc=("absorbing", "open"))


# --- the electrostatic model -------------------------------------------------------------------

def test_the_electrostatic_model_leaves_the_transverse_fields_alone_but_keeps_the_motion():
    """An electrostatic run solves dE_x/dx = rho/eps_0 and nothing else: no transverse
    self-field is evolved and no transverse current is deposited. All three velocity
    components remain, and an external magnetic field turns them as it always did, at the
    gyro-frequency."""
    field = 0.05
    B = jnp.zeros((16, 3)).at[:, 2].set(field)
    omega_c = e_charge * field / mass_electron
    electrons = Species.electrons(n=200, density=1e6, vth=0.0, drift=(0.0, 1e5, 0.0), quiet=True)
    domain = box(cells=16, steps_per_plasma_period=200.0)
    out = Simulation(domain, [electrons], Solver(model="electrostatic"), external_B=B).run(60, store_every=1)
    assert float(jnp.max(jnp.abs(out.B))) == 0.0
    assert float(jnp.max(jnp.abs(out.E[:, :, 1:]))) == 0.0
    assert float(jnp.max(jnp.abs(out.J[:, :, 1:]))) == 0.0
    vy = np.asarray(out.v[:, 0, 1])
    assert np.allclose(vy, 1e5 * np.cos(omega_c * np.asarray(out.t)), atol=2e3)


def test_the_two_field_models_agree_between_walls_and_differ_by_the_mean_field_in_a_ring():
    """A Langmuir oscillation along the grid has no transverse field, so the two models
    solve the same problem. Between walls they agree to round-off, and not merely to the
    truncation error of the step: Ampere's law advanced with the continuity current keeps
    the discrete Gauss law exactly, and the wall fixes the same constant of integration
    for both, so the two routes to E_x reach the same field.

    In a periodic box they do not, and the difference is a degree of freedom rather than
    an error: a ring has no wall to fix the constant of integration, so Ampere's law
    carries the mean field the net particle current drives, while the Gauss solve sets the
    mean to zero. Here the quiet start's own noise builds a mean field of a few volts per
    metre over eight plasma periods. Neither is wrong; they are different conventions, and
    a comparison between them has to be of the fluctuating field."""
    electrons = Species.electrons(n=4000, density=DENSITY, vth=(0.3 * SIGMA, 0, 0), quiet=True,
                                  perturbation_amplitude=1e-6, perturbation_mode=1)
    ions = Species.ions(n=1000, density=DENSITY, mass_ratio=1e9, vth=0.0, quiet=True)

    def final_field(model, steps_per_period, **domain):
        out = Simulation(box(cells=32, steps_per_plasma_period=steps_per_period, **domain),
                         [electrons, ions], Solver(model=model)).run(int(8 * steps_per_period))
        assert float(gauss_residual(out)[-1]) < 1e-12
        return np.asarray(out.E[-1, :, 0])

    for steps_per_period in (10.0, 40.0):
        walled = [final_field(model, steps_per_period, particle_bc="reflective", field_bc="reflective")
                  for model in ("electromagnetic", "electrostatic")]
        assert np.allclose(walled[0], walled[1], rtol=0, atol=1e-12 * np.abs(walled[0]).max())

    ring = [final_field(model, 10.0) for model in ("electromagnetic", "electrostatic")]
    assert abs(ring[1].mean()) < 1e-12 * np.abs(ring[1]).max()
    assert abs(ring[0].mean()) > 0.01 * np.abs(ring[0]).max()


def test_the_implicit_scheme_refuses_the_electrostatic_model():
    with pytest.raises(ValueError, match="electrostatic"):
        Simulation(Domain(), [Species.electrons(n=10, density=1e10)],
                   Solver(algorithm="implicit", model="electrostatic"))


@pytest.mark.parametrize("solver, message", [
    (Solver(algorithm="implicit"), "algorithm='explicit'"),
    (Solver(model="electromagnetic"), "electrostatic"),
])
def test_unsupported_source_combinations_are_refused_rather_than_half_done(solver, message):
    # a reflective field wall, so that the implicit case reaches the source's own objection
    # and not the open plane's, which is tested on its own above
    species = Species("electrons", 100, -1.0, mass_electron, 0.0, source=maxwellian_source(4))
    with pytest.raises(ValueError, match=message):
        Simulation(box(particle_bc="absorbing", field_bc="reflective"), [species], solver)


def test_a_source_needs_walls_and_a_wall_that_takes_particles_back():
    species = Species("electrons", 100, -1.0, mass_electron, 0.0, source=maxwellian_source(4))
    with pytest.raises(ValueError, match="periodic box"):
        Simulation(box(), [species], Solver(model="electrostatic"))
    with pytest.raises(ValueError, match="absorbing"):
        Simulation(box(particle_bc=("reflective", "absorbing"), field_bc=("open", "absorbing")),
                   [species], Solver(model="electrostatic"))


# --- time, restart and the streaming moments ------------------------------------------------------

def test_the_clock_is_absolute_and_a_run_split_in_two_is_the_run_taken_whole():
    """A continued run carries the time, the particles, the fields, the random key, the
    source and the wall ledger, so splitting a run changes nothing but where the output
    is cut."""
    domain = box(cells=32, particle_bc="absorbing", field_bc=("open", "absorbing"))
    species = Species("electrons", 2000, -1.0, mass_electron, 0.0,
                      source=maxwellian_source(8, density=1e-4 * DENSITY))
    sim = Simulation(domain, [species], Solver(model="electrostatic"))
    whole = sim.run(120, store_every=40, moments=True)
    first = sim.run(40, store_every=40, moments=True)
    rest = sim.run(80, store_every=40, moments=True, state=first.state)
    assert float(first.t[-1]) == pytest.approx(40 * domain.dt, rel=1e-12)
    assert np.allclose(np.asarray(rest.t), np.asarray(whole.t[1:]), rtol=1e-12, atol=0)
    # the step count is absolute too, so a cumulative window is a difference of counts and
    # never the length of an array
    assert list(np.asarray(whole.steps)) == [40, 80, 120]
    assert list(np.asarray(first.steps)) == [40] and list(np.asarray(rest.steps)) == [80, 120]
    assert np.allclose(np.asarray(rest.E[-1]), np.asarray(whole.E[-1]), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(rest.state.w), np.asarray(whole.state.w), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(rest.wall.injected[-1]), np.asarray(whole.wall.injected[-1]),
                       rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(rest.moments[-1]), np.asarray(whole.moments[-1]), rtol=1e-10, atol=0)


def test_the_streaming_moments_are_the_deposit_and_need_no_particle_history():
    """The running sums are the same profiles a deposit of the stored particles gives, and
    they are there when the particles are not, which is what makes a mean over every step
    of a long window affordable."""
    domain = box(cells=32, particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
    electrons = Species.electrons(n=2000, density=DENSITY, vth=(np.sqrt(2) * SIGMA, 0, 0), quiet=True)
    ions = Species.ions(n=2000, density=DENSITY, mass_ratio=400.0, electrons=electrons, quiet=True)
    sim = Simulation(domain, [electrons, ions], Solver(model="electrostatic"))
    stored = sim.run(2, store_every=1, moments=True)
    direct = sim.moments(stored.x[0], stored.v[0], stored.weight[0])
    assert np.allclose(np.asarray(stored.moments[0]), np.asarray(direct), rtol=1e-10)
    light = sim.run(2, store_every=1, store_particles=False, moments=True)
    assert light.x is None and np.allclose(np.asarray(light.moments[-1]), np.asarray(stored.moments[-1]))
    assert sim.run(2).moments is None


# --- the sheath edge and the analytic reference ------------------------------------------------------

def test_the_sheath_edge_is_interpolated_and_says_when_there_is_none():
    position = jnp.linspace(0.0, 10.0, 11)
    edge, count = bohm_edge(position, position, 5.5)
    assert float(edge) == pytest.approx(5.5) and int(count) == 1
    edge, count = bohm_edge(position, 0.2 * position, 5.0)
    assert np.isnan(float(edge)) and int(count) == 0
    flow = jnp.array([0.0, 1.0, 6.0, 1.0, 0.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
    edge, count = bohm_edge(position, flow, 5.0)
    assert int(count) == 2 and float(edge) == pytest.approx(1.8)
    assert np.isnan(float(bohm_edge(position, jnp.zeros(11), 1.0)[0]))


def test_the_kinetic_reference_solves_its_own_equations():
    """The wall potential is the root of exp(phi)/[1 + erf(sqrt(-phi))] = sqrt(pi/2) v_0,
    the source amplitude makes the source plane neutral, and the densities are the ones
    energy conservation gives: n_e = n_i = 1 where phi = 0, and only the outgoing
    electrons are left at the wall."""
    for beam_speed in (0.1, 0.2, 0.3):
        phi = float(floating_potential(beam_speed))
        assert np.exp(phi) / (1 + float(jax.scipy.special.erf(np.sqrt(-phi)))) == pytest.approx(
            np.sqrt(np.pi / 2) * beam_speed, rel=1e-12)
        n_e, n_i = densities(np.array([0.0, phi]), phi, beam_speed, 1836.0)
        assert n_e[0] == pytest.approx(1.0, rel=1e-12) and n_i[0] == pytest.approx(1.0, rel=1e-12)
        assert n_e[1] == pytest.approx(0.5 * float(source_density(phi)) * np.exp(phi), rel=1e-12)
    assert float(floating_potential(0.2)) == pytest.approx(-0.7992615777938566, abs=1e-14)
    assert float(source_density(floating_potential(0.2))) == pytest.approx(1.114897194662092, rel=1e-14)
    with pytest.raises(ValueError, match="beam_speed"):
        floating_potential(0.5)
    assert float(hobbs_wesson(1836.0)) == pytest.approx(0.5 * np.log(1836.0 / (2 * np.pi)), rel=1e-12)
    assert float(hobbs_wesson(400.0, 0.5)) == pytest.approx(float(hobbs_wesson(400.0)) - np.log(2), rel=1e-12)


def test_a_maintained_sheath_reaches_the_kinetic_floating_potential():
    """The whole model end to end: a reservoir of electrons at rest and a cold ion beam at
    Mach 8.6 enter through an open plane, a floating conductor collects what reaches it,
    and the potential it settles at is the one equal particle currents demand. The plasma
    is quasineutral away from the wall and the electrons are pushed out of the sheath while
    the beam is not, which is the positive charge that holds the drop up."""
    beam_speed, mass_ratio, cells, capacity, emit, steps = 0.2, 1836.0, 100, 60000, 60, 6000
    phi_wall = float(floating_potential(beam_speed))
    domain = box(cells=cells, particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", capacity, -1.0, mass_electron, DENSITY, (np.sqrt(2) * SIGMA, 0, 0),
                        active=capacity // 4, quiet=True,
                        source=maxwellian_source(emit, density=float(source_density(phi_wall)) * DENSITY))
    ions = Species("ions", capacity, 1.0, mass_ratio * mass_electron, DENSITY, 0.0, (beam_speed * SIGMA, 0, 0),
                   active=capacity // 4, quiet=True,
                   source=Source(density=DENSITY, vth=0.0, drift=(beam_speed * SIGMA, 0, 0), emit=emit))
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(
        steps, seed=0, store_every=steps // 60, store_particles=False, moments=True)
    late = slice(30, None)
    measured = float(jnp.mean(potential(out)[late, -1]))
    assert measured == pytest.approx(phi_wall, abs=0.08)
    assert float(out.wall.overflow[-1]) == 0.0
    window = np.asarray(out.moments[-1] - out.moments[30]) / (30 * steps // 60)
    n_e, n_i = window[0, 0] / DENSITY, window[1, 0] / DENSITY
    assert np.allclose(n_e[cells // 4:cells // 2], 1.0, atol=0.06)
    assert np.allclose(n_i[cells // 4:cells // 2], 1.0, atol=0.06)
    assert n_e[-2] < 0.75 * n_i[-2]                   # the sheath is a layer of positive charge
    # the collector draws no net current once it floats
    late_charge = np.asarray(out.wall.collected)[-1, :, 1] - np.asarray(out.wall.collected)[30, :, 1]
    assert late_charge[0] == pytest.approx(late_charge[1], rel=0.05)


def test_only_some_species_need_a_source():
    """A species without one keeps the population it started with while the source
    maintains the other: electrons supplied through the plane against a fixed neutralising
    background of heavy ions."""
    domain = box(cells=32, particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", 4000, -1.0, mass_electron, 0.0,
                        source=maxwellian_source(10, density=1e-4 * DENSITY))
    ions = Species.ions(n=500, density=1e-4 * DENSITY, mass_ratio=1e9, vth=0.0, quiet=True)
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(300, store_particles=False)
    injected = np.asarray(out.wall.injected)[-1]
    assert injected[0].sum() > 0 and injected[1].sum() == 0.0
    assert float(jnp.sum(out.state.w[4000:])) == pytest.approx(1e-4 * DENSITY * domain.length, rel=1e-12)


def test_an_ensemble_of_reservoirs_is_one_vmap_and_one_compilation():
    """Every physical number in a Source is a pytree leaf, so a scan over reservoir
    densities is a vmap of the run rather than a loop of them, and the `None` template a
    tree operation builds to say which axes to map is not mistaken for a configuration."""
    domain = box(cells=16, particle_bc="absorbing", field_bc=("open", "absorbing"))

    def emitted(density):
        source = maxwellian_source(6, density=density)
        species = Species("electrons", 300, -1.0, mass_electron, 0.0, source=source)
        return Simulation(domain, [species], Solver(model="electrostatic")).run(
            20, store_particles=False).wall.injected[-1, 0, 0]

    densities = jnp.array([1e-6, 2e-6, 4e-6]) * DENSITY
    together = jax.vmap(emitted)(densities)
    assert np.allclose(np.asarray(together), [float(emitted(d)) for d in densities], rtol=1e-12)
    assert np.allclose(np.asarray(together) / np.asarray(together)[0], [1.0, 2.0, 4.0], rtol=1e-12)

    # the same ensemble through a Source built once, mapped over its density alone
    axes = jax.tree.map(lambda _: None, maxwellian_source(6)).replace(density=0)
    assert axes.emit == 6 and axes.vth == (None, None, None) and axes.density == 0
    stacked = maxwellian_source(6).replace(density=densities)

    def from_source(source):
        species = Species("electrons", 300, -1.0, mass_electron, 0.0, source=source)
        return Simulation(domain, [species], Solver(model="electrostatic")).run(
            20, store_particles=False).wall.injected[-1, 0, 0]

    assert np.allclose(np.asarray(jax.vmap(from_source, in_axes=(axes,))(stacked)),
                       np.asarray(together), rtol=1e-12)
