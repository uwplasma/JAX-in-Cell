"""The maintained source, the wall ledger, the electrode closure and the electrostatic
model. Each test states the closed form it checks against, or the invariant that has to
hold whatever the numbers are."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import (Domain, Simulation, Solver, Source, Species, bohm_edge, epsilon_0, gauss_residual,
                       mass_electron, potential, elementary_charge as e_charge, speed_of_light as c)
from jaxincell._core import apply_particle_bc
from jaxincell._sources import crossing_flux, sample_crossing
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
    return Source(density=density, vth=(np.sqrt(2) * SIGMA, 0, 0), emit=emit, **kwargs)


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
    <v_n> = sigma sqrt(pi/2), <v_n^2> = 2 sigma^2, twice the variance of the Maxwellian
    it came from, while the tangential components keep <v_t^2> = sigma^2."""
    v = np.asarray(sample_crossing(random.PRNGKey(0), maxwellian_source(8), 200000, 1.0))
    assert np.all(v[:, 0] > 0)
    assert v[:, 0].mean() / SIGMA == pytest.approx(np.sqrt(np.pi / 2), rel=3e-3)
    assert (v[:, 0] ** 2).mean() / SIGMA ** 2 == pytest.approx(2.0, rel=5e-3)
    assert (v[:, 1] ** 2).mean() / SIGMA ** 2 == pytest.approx(1.0, rel=5e-3)
    assert abs(v[:, 1].mean()) / SIGMA < 0.01
    # the right wall sends the same distribution the other way
    left = np.asarray(sample_crossing(random.PRNGKey(0), maxwellian_source(8), 5000, -1.0))
    assert np.all(left[:, 0] < 0)


def test_a_cold_source_is_a_beam_and_a_warm_one_with_a_normal_drift_is_refused():
    beam = Source(density=DENSITY, vth=0.0, drift=(3e5, 1e5, 0), emit=4)
    v = np.asarray(sample_crossing(random.PRNGKey(0), beam, 16, 1.0))
    assert np.allclose(v[:, 0], 3e5) and np.allclose(v[:, 1], 1e5) and np.allclose(v[:, 2], 0.0)
    with pytest.raises(ValueError, match="Maxwellian at rest or a cold beam"):
        Source(density=DENSITY, vth=(1e6, 0, 0), drift=(1e5, 0, 0), emit=4)


def test_whether_a_source_is_a_beam_is_decided_once_and_not_read_from_a_tracer():
    """`beam` selects a branch, so it cannot be a traced leaf: inside jit every `vth` is a
    tracer, `vth != 0` is not a Python bool, and a Maxwellian source would sample a beam at
    rest and emit particles that never move. It is fixed when the Source is built, and a
    Source built from a tracer has to say which it is."""
    warm = maxwellian_source(4)
    assert warm.beam is False and Source(density=DENSITY, vth=0.0, drift=(1e5, 0, 0), emit=4).beam is True

    def speeds(source):
        return sample_crossing(random.PRNGKey(0), source, 64, 1.0)[:, 0]

    inside = jax.jit(speeds)(warm)
    assert float(jnp.std(inside)) > 0.1 * SIGMA                       # a beam would have none
    assert np.allclose(np.asarray(inside), np.asarray(speeds(warm)))
    with pytest.raises(ValueError, match="traced vth"):
        jax.jit(lambda vth: Source(density=DENSITY, vth=(vth, 0, 0), emit=4).beam)(1e6)


# --- what the source puts in --------------------------------------------------------------

def test_the_emitted_weight_is_exactly_the_prescribed_flux_and_differentiable_in_it():
    """N_emit particles of weight Gamma dt / N_emit put in exactly Gamma dt, whatever
    N_emit is, and its derivatives with respect to the reservoir density and thermal speed
    are the derivatives of Gamma: d/dn = sigma/sqrt(2 pi) dt and d/dvth = n/(2 sqrt(pi)) dt.
    A count would have neither, being an integer."""
    domain = box(particle_bc="absorbing", field_bc=("open", "absorbing"))
    steps = 5

    def emitted(density, vth):
        source = Source(density=density, vth=(vth, 0, 0), emit=7, beam=False)
        species = Species("electrons", 400, -1.0, mass_electron, 0.0, source=source)
        out = Simulation(domain, [species], Solver(model="electrostatic")).run(steps, store_particles=False)
        return out.wall.injected[-1, 0, 0]

    vth = np.sqrt(2) * SIGMA
    value, grad = jax.value_and_grad(emitted, argnums=(0, 1))(DENSITY, vth)
    assert float(value) == pytest.approx(DENSITY * SIGMA / np.sqrt(2 * np.pi) * domain.dt * steps, rel=1e-12)
    assert float(grad[0]) == pytest.approx(SIGMA / np.sqrt(2 * np.pi) * domain.dt * steps, rel=1e-10)
    assert float(grad[1]) == pytest.approx(DENSITY / (2 * np.sqrt(np.pi)) * domain.dt * steps, rel=1e-10)


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
    assert float(out.wall.overflow[-1]) > 0.0


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
    _, v_out, w_out, _, (arrived, kept) = apply_particle_bc(
        x, v, jnp.full(1, w), jnp.ones(1), (1.0, 1.0, 1.0), (2, 2), (1.0, restitution), reflection, 0.1)
    assert float(arrived[1, 0]) == pytest.approx(w) and float(arrived[0, 0]) == 0.0
    assert float(kept[1, 0]) == pytest.approx((1 - R) * w)
    assert float(w_out[0]) == pytest.approx(R * w)
    assert float(v_out[0, 0]) == pytest.approx(-restitution * speed)
    mass = mass_electron
    energy_in = 0.5 * mass * speed ** 2 * w
    energy_out = 0.5 * mass * (restitution * speed) ** 2 * (R * w)
    assert float(arrived[1, 0]) * 0.5 * mass * speed ** 2 == pytest.approx(energy_in)
    assert float((arrived - kept)[1, 0]) * 0.5 * mass * (restitution * speed) ** 2 == pytest.approx(energy_out)


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
    assert np.all(np.asarray(wall.energy_out) <= np.asarray(wall.energy_in) + 1e-30)
    charge = np.asarray(wall.charge([s.charge_si for s in (electrons, ions)]))
    assert charge[1] == pytest.approx(e_charge * (collected[1, 1] - collected[0, 1]), rel=1e-12)


def test_a_reflecting_wall_would_hold_a_particle_for_ever_without_a_floor():
    """Returning the fraction R of each impact leaves R^k after k of them, which never
    reaches zero, so the slot is never free for a source to refill. Below `min_weight`
    times the emitted weight the wall keeps the remainder; the charge goes on the ledger,
    so nothing is lost, and the slot comes back."""
    domain = box(cells=16, particle_bc="absorbing", field_bc="reflective")
    source = maxwellian_source(10, density=1e-10 * DENSITY)
    weights = {}
    for name, floor in (("floor", 1e-2), ("none", 0.0)):
        species = Species("electrons", 12000, -1.0, mass_electron, 0.0, reflection=0.5,
                          source=source.replace(min_weight=floor))
        out = Simulation(domain, [species], Solver(model="electrostatic")).run(600, store_particles=False)
        live = np.asarray(out.state.w)
        weights[name] = (live > 0).sum()
        assert float(out.wall.overflow[-1]) == 0.0
        total = np.asarray(out.wall.collected)[-1].sum() + live.sum()
        assert total == pytest.approx(float(np.asarray(out.wall.injected)[-1].sum()), rel=1e-9)
    assert weights["floor"] < weights["none"]


# --- the electrical boundary -----------------------------------------------------------------

def test_the_collector_field_is_the_charge_it_holds():
    """E_x at the conductor face is -sigma_w/eps_0, from the charge the ledger says it has
    collected, and that is what closes the Gauss solve when the plane opposite is an open
    source plane and cannot impose anything."""
    domain = box(cells=48, particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", 4000, -1.0, mass_electron, 0.0,
                        source=maxwellian_source(10, density=1e-3 * DENSITY))
    ions = Species("ions", 4000, 1.0, 400 * mass_electron, 0.0,
                   source=Source(density=1e-3 * DENSITY, vth=0.0, drift=(0.2 * SIGMA, 0, 0), emit=10))
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic")).run(400, store_every=100)
    sigma_w = np.asarray(out.wall.collected)[:, :, 1] @ np.array([-e_charge, e_charge])
    assert np.allclose(np.asarray(out.E[:, -1, 0]), -sigma_w / epsilon_0, rtol=1e-10)


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
    (Solver(algorithm="implicit"), "explicit"),
    (Solver(model="electromagnetic"), "electrostatic"),
])
def test_unsupported_source_combinations_are_refused_rather_than_half_done(solver, message):
    species = Species("electrons", 100, -1.0, mass_electron, 0.0, source=maxwellian_source(4))
    with pytest.raises(ValueError, match=message):
        Simulation(box(particle_bc="absorbing", field_bc=("open", "absorbing")), [species], solver)


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
    assert np.allclose(np.asarray(rest.t), np.asarray(whole.t[1:]), rtol=1e-12)
    assert np.allclose(np.asarray(rest.E[-1]), np.asarray(whole.E[-1]), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(rest.state.w), np.asarray(whole.state.w), rtol=1e-12, atol=0)
    assert np.allclose(np.asarray(rest.wall.injected[-1]), np.asarray(whole.wall.injected[-1]), rtol=1e-12)
    assert np.allclose(np.asarray(rest.moments[-1]), np.asarray(whole.moments[-1]), rtol=1e-10)


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
