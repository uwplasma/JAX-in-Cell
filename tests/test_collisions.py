"""The binary collision operator: who collides with whom, what each collision
conserves, which density and Coulomb logarithm set the rate, and that gradients
pass through it."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import Collisions, Domain, Simulation, Solver, Species, _collisions, epsilon_0
from jaxincell._collisions import collide, coulomb_logarithm
from jaxincell import elementary_charge as e_charge, mass_electron
from jaxincell import speed_of_light as c_light


def _cells(per_cell, rng):
    """Positions in a unit box of ``per_cell[c]`` particles in cell ``c``, in random order."""
    cell = rng.permutation(np.repeat(np.arange(len(per_cell)), per_cell))
    x = np.zeros((cell.size, 3))
    x[:, 0] = (cell + 0.5) / len(per_cell) - 0.5
    return cell, jnp.asarray(x)


def _per_cell(cell, n_cells, values):
    total = np.zeros((n_cells,) + values.shape[1:])
    np.add.at(total, cell, values)
    return total


def test_every_particle_is_paired_inside_its_own_cell_however_many_cells():
    """One electron and one positron in each of 50 000 cells, 100 000 particles in all.
    The pairing once built an int32 key cell * (N + 1) + rank, which overflows once
    (cells + 1)(N + 1) passes 2**31 and then paired particles across cells or not at
    all. Every particle has to collide, with the partner in its own cell, so that the
    momentum and energy of every cell are conserved separately."""
    n_cells = 50_000
    rng = np.random.default_rng(0)
    cell_a, x_a = _cells(np.ones(n_cells, int), rng)
    cell_b, x_b = _cells(np.ones(n_cells, int), rng)
    cell, x = np.concatenate([cell_a, cell_b]), jnp.concatenate([x_a, x_b])
    v = jnp.asarray(1e6 * rng.standard_normal((2 * n_cells, 3)))
    mass = np.full(2 * n_cells, mass_electron)
    charge = jnp.asarray(np.repeat([-e_charge, e_charge], n_cells))
    new = collide(random.PRNGKey(1), x, v, jnp.ones(2 * n_cells), jnp.asarray(mass), charge,
                  ((0, n_cells), (n_cells, n_cells)), ((0, 1),), 1e8, 1.0, 1 / n_cells, 1.0, n_cells)
    v, new = np.asarray(v), np.asarray(new)
    assert (new != v).any(axis=1).all()
    momentum = [_per_cell(cell, n_cells, mass[:, None] * u) for u in (v, new)]
    energy = [_per_cell(cell, n_cells, mass * (u ** 2).sum(axis=1)) for u in (v, new)]
    assert np.abs(momentum[1] - momentum[0]).max() < 1e-12 * np.abs(momentum[0]).max()
    assert np.abs(energy[1] / energy[0] - 1).max() < 1e-12


def _self_colliding(per_cell, seed=0):
    """Electrons in eight cells, ``per_cell`` in six of them, one alone in the seventh,
    and ``per_cell`` in the eighth with one of them already collected by a wall."""
    rng = np.random.default_rng(seed)
    cell, x = _cells([per_cell] * 6 + [1, per_cell], rng)
    weight = np.ones(cell.size)
    weight[np.flatnonzero(cell == 7)[0]] = 0.0
    v = jnp.asarray(1e6 * rng.standard_normal((cell.size, 3)))
    arrays = (jnp.asarray(weight), jnp.full(cell.size, mass_electron), jnp.full(cell.size, -e_charge))
    return cell, x, v, arrays, ((0, cell.size),), ((0, 0),), 1e8, 1.0, 1 / 8, 1.0, 8


@pytest.mark.parametrize("per_cell", [2, 3, 4, 10, 100])
def test_self_collisions_pair_every_particle_once_inside_its_cell(per_cell, monkeypatch):
    """Takizuka and Abe pair a species with itself inside each cell: 1-2, 3-4, ...,
    and for an odd number the last three collide 1-2, 2-3, 3-1 with half the
    variance each. Every live particle therefore takes part in one collision, or in
    two half collisions, and a lone or collected particle in none."""
    collisions = []
    scatter = _collisions._scatter

    def record(key, v, i, j, active, density, fraction, *args):
        collisions.append((np.asarray(i), np.asarray(j), np.asarray(active),
                           np.broadcast_to(np.asarray(fraction), np.shape(i))))
        return scatter(key, v, i, j, active, density, fraction, *args)

    monkeypatch.setattr(_collisions, "_scatter", record)
    cell, x, v, arrays, *rest = _self_colliding(per_cell)
    collide(random.PRNGKey(2), x, v, *arrays, *rest)
    live = np.asarray(arrays[0]) > 0
    count, share = np.zeros(cell.size, int), np.zeros(cell.size)
    for i, j, active, fraction in collisions:
        assert (cell[i[active]] == cell[j[active]]).all() and (i[active] != j[active]).all()
        for side in (i, j):
            np.add.at(count, side[active], 1)
            np.add.at(share, side[active], fraction[active])
    in_cell = np.bincount(cell[live], minlength=8)[cell]
    collides = live & (in_cell > 1)
    assert (count[~collides] == 0).all()
    assert np.array_equal(share[collides], np.ones(collides.sum()))
    for c in range(8):
        twice = (count[cell == c] == 2).sum()
        assert twice == (3 if in_cell[cell == c][0] % 2 == 1 and in_cell[cell == c][0] > 1 else 0)


@pytest.mark.parametrize("per_cell", [2, 3, 4, 10, 100])
def test_self_collisions_conserve_momentum_and_energy_to_round_off(per_cell):
    """Through large angles, one step leaves the total momentum and energy of the
    species unchanged to round-off, and moves every particle that has a partner.
    Sharing a partner between two simultaneous collisions would break the energy."""
    cell, x, v, arrays, *rest = _self_colliding(per_cell, seed=1)
    new = np.asarray(collide(random.PRNGKey(3), x, v, *arrays, *rest))
    v, mass = np.asarray(v), np.asarray(arrays[1])
    live = np.asarray(arrays[0]) > 0
    moved = (new != v).any(axis=1)
    assert moved[live & (np.bincount(cell[live], minlength=8)[cell] > 1)].all()
    assert not moved[~live].any()
    p, p_new = (mass[:, None] * v).sum(axis=0), (mass[:, None] * new).sum(axis=0)
    assert np.abs(p_new - p).max() < 1e-12 * np.abs(mass[:, None] * v).sum()
    assert abs((mass * (new ** 2).sum(axis=1)).sum() / (mass * (v ** 2).sum(axis=1)).sum() - 1) < 1e-12


def test_species_too_small_to_pair_are_left_alone():
    """A species of one particle has nobody to collide with; one of two collides with
    its only partner and conserves the pair's momentum and energy."""
    v = jnp.asarray(np.random.default_rng(4).standard_normal((3, 3)) * 1e6)
    new = np.asarray(collide(random.PRNGKey(4), jnp.zeros((3, 3)), v, jnp.ones(3), jnp.full(3, mass_electron),
                             jnp.full(3, -e_charge), ((0, 1), (1, 2)), ((0, 0), (1, 1)), 1e8, 1.0, 1.0, 1.0, 1))
    v = np.asarray(v)
    assert np.array_equal(new[0], v[0]) and (new[1:] != v[1:]).all()
    assert np.allclose(new[1:].sum(axis=0), v[1:].sum(axis=0), rtol=0, atol=1e-9)
    assert abs((new[1:] ** 2).sum() / (v[1:] ** 2).sum() - 1) < 1e-12


@pytest.mark.parametrize("mass_ratio", [1.0, 10.0])
def test_unequal_cell_counts_conserve_energy_and_momentum(mass_ratio):
    """A particle reused by simultaneous pairs used to receive their summed kicks,
    creating kinetic energy even with equal weights. Disjoint pairs must conserve
    each cell through large angles, for either longer species and empty cells."""
    rng = np.random.default_rng(9)
    cell_a, x_a = _cells([1, 9, 2, 18, 0, 1], rng)
    cell_b, x_b = _cells([9, 1, 18, 2, 1, 0], rng)
    n_a, n_b = len(cell_a), len(cell_b)
    cell, x = np.r_[cell_a, cell_b], jnp.concatenate([x_a, x_b])
    v = jnp.asarray(rng.normal(0.0, 1e6, (n_a + n_b, 3)))
    mass = np.r_[np.full(n_a, mass_electron), np.full(n_b, mass_ratio * mass_electron)]
    weight = np.full(n_a + n_b, 1e18)
    weight[np.flatnonzero(cell == 2)[0]] = 0.0
    charge = jnp.asarray(np.r_[np.full(n_a, -e_charge), np.full(n_b, e_charge)])
    new = np.asarray(jax.jit(lambda v: collide(
        random.PRNGKey(10), x, v, jnp.asarray(weight), jnp.asarray(mass), charge,
        ((0, n_a), (n_a, n_b)), ((0, 1),), 10.0, 1e-8, 1 / 6, 1.0, 6))(v))
    wm = weight * mass
    momentum = [_per_cell(cell, 6, wm[:, None] * u) for u in (np.asarray(v), new)]
    energy = [_per_cell(cell, 6, wm * (u ** 2).sum(axis=1)) for u in (np.asarray(v), new)]
    assert np.max(np.abs(momentum[1] - momentum[0])) < 1e-12 * np.max(np.abs(momentum[0]))
    assert np.max(np.abs(energy[1] / energy[0] - 1)) < 1e-12
    alone = (cell == 4) | (cell == 5) | (weight == 0)
    assert np.array_equal(new[alone], np.asarray(v)[alone])


@pytest.mark.parametrize("dt_factor", [1.0, 0.5])
def test_minority_beam_slows_and_diffuses_at_the_full_background_density(dt_factor):
    """A 1:9 beam must see the full background, even though only a random ninth of
    it is paired. Both initial rates follow the Fokker-Planck small-angle limit."""
    n_a, n_b, u, density, coulomb_log = 8192, 9 * 8192, 1e6, 1e20, 10.0
    v = jnp.zeros((n_a + n_b, 3)).at[:n_a, 0].set(u)
    nu_0 = e_charge ** 4 * density * coulomb_log / (4 * np.pi * epsilon_0 ** 2 * mass_electron ** 2 * u ** 3)
    dt = dt_factor * 1e-4 / nu_0
    new = np.asarray(jax.jit(lambda v: collide(
        random.PRNGKey(11), jnp.zeros_like(v), v, jnp.full(n_a + n_b, density / n_b),
        jnp.full(n_a + n_b, mass_electron), jnp.full(n_a + n_b, -e_charge),
        ((0, n_a), (n_a, n_b)), ((0, 1),), coulomb_log, dt, 1.0, 1.0, 1))(v))
    slowing = (u - new[:n_a, 0].mean()) / (u * dt)
    diffusion = (new[:n_a, 1:] ** 2).sum(axis=1).mean() / (u ** 2 * dt)
    assert abs(slowing / (2 * nu_0) - 1) < 0.05
    assert abs(diffusion / (2 * nu_0) - 1) < 0.05


@pytest.mark.parametrize("n_a, w_a, n_b, w_b", [
    (40_000, 1.0, 40_000, 1.0), (40_000, 1.0, 10_000, 4.0), (40_000, 1.0, 10_000, 0.25), (10_000, 0.25, 40_000, 1.0)])
def test_each_species_scatters_off_the_density_of_the_other(n_a, w_a, n_b, w_b):
    """A beam crossing a background at rest, with unequal numbers and weights. In one
    step each beam particle must gain <|dv|^2> = (m_b/M)^2 4 u^2 <delta^2>(n_b) and each
    background particle (m_a/M)^2 4 u^2 <delta^2>(n_a), whichever list is longer and
    whichever is heavier, as if it had collided once with the whole density of the
    other species. The Nanbu-Yonemura acceptance alone gets this wrong when the
    shorter list carries the smaller weight, unless the pair density compensates for
    both sampled counts and weight acceptance."""
    u, coulomb_log, dt = 1e6, 10.0, 1.0
    n = n_a + n_b
    v = np.zeros((n, 3))
    v[:n_a, 0] = u
    weight = jnp.asarray(np.concatenate([np.full(n_a, w_a), np.full(n_b, w_b)]))
    mass = jnp.full(n, mass_electron)
    charge = jnp.full(n, -e_charge)

    def variance(density):
        reduced_mass = mass_electron / 2
        return (e_charge ** 2 / epsilon_0 / reduced_mass) ** 2 * density * coulomb_log * dt / (8 * np.pi * u ** 3)

    # Sampling/acceptance enlarges the variance of an individual pair. Keep that
    # variance small, rather than only the smaller species' effective variance.
    pair_density = max(n_a, n_b) * max(w_a, w_b)
    scale = 1e-3 / variance(pair_density)
    pair_s2 = variance(pair_density) * scale
    new = np.asarray(collide(random.PRNGKey(5), jnp.zeros((n, 3)), jnp.asarray(v), weight, mass, charge,
                             ((0, n_a), (n_a, n_b)), ((0, 1),), coulomb_log, dt * scale, 1.0, 1.0, 1))
    kick = ((new - v) ** 2).sum(axis=1)
    for block, density in ((slice(0, n_a), n_b * w_b), (slice(n_a, n), n_a * w_a)):
        s2 = variance(density) * scale
        expected = u ** 2 * s2 / (1 + 3 * pair_s2)     # Gaussian angular average through second order
        samples = kick[block] / expected
        sem = samples.std(ddof=1) / np.sqrt(samples.size)
        # Independent angular/acceptance draws; unmatched zeros make this estimate
        # conservative for sampling without replacement. The small finite-angle
        # bias at pair_s2=.001 is below the 1% allowance, separate from seed noise.
        assert abs(samples.mean() - 1) < 4 * sem + 0.01


@pytest.mark.parametrize("within", [True, False])
@pytest.mark.parametrize("dt_factor", [1.0, 0.25])
def test_velocity_correlated_weights_reproduce_each_populations_fast_beam_rate(within, dt_factor):
    """Higginson et al. (2020), sections 2.2 and 3: global weight normalization
    fails when weights and velocities differ within a group, even as dt -> 0.
    Both stopping and diffusion must see physical partner density, not macro weight."""
    na, nb = (1024, 19 * 1024) if within else (4096, 8192)
    n, u, density = na + nb, 1e6, 1e20
    v = jnp.zeros((n, 3)).at[:na, 0].set(u)
    w = np.r_[np.full(na, 9.0 if within else 1.0), np.ones(nb)]
    if not within:
        w[na + nb // 2:] = 9.0
    w *= density / w[na:].sum()
    nu = e_charge ** 4 * density * 10 / (4 * np.pi * epsilon_0 ** 2 * mass_electron ** 2 * u ** 3)
    dt = dt_factor * 1e-5 / nu
    blocks, pairs = (((0, n),), ((0, 0),)) if within else (((0, na), (na, nb)), ((0, 1),))
    new = np.asarray(jax.jit(jax.vmap(lambda key: collide(
        key, jnp.zeros_like(v), v, jnp.asarray(w), jnp.full(n, mass_electron), jnp.full(n, -e_charge),
        blocks, pairs, 10.0, dt, 1.0, 1.0, 1)))(random.split(random.PRNGKey(17), 64)))

    def correct(ratio):
        # Independent keys: six standard errors plus a small finite-step allowance.
        assert abs(ratio.mean() - 1) < 6 * ratio.std(ddof=1) / np.sqrt(len(ratio)) + 0.002

    correct((u - new[:, :na, 0].mean(axis=1)) / (2 * nu * u * dt))
    correct((new[:, :na, 1:] ** 2).sum(axis=2).mean(axis=1) / (2 * nu * u ** 2 * dt))
    if not within:
        expected = 2 * nu * w[:na].sum() / density * u ** 2 * dt
        for part in (slice(na, na + nb // 2), slice(na + nb // 2, n)):
            correct((new[:, part] ** 2).sum(axis=2).mean(axis=1) / expected)


@pytest.mark.parametrize("mass_ratio", [1.0, 1836.15])
def test_unequal_weights_conserve_energy_and_momentum_in_expectation(mass_ratio):
    v = jnp.array([[1e6, 0.0, 0.0], [0.0, 4e5, 0.0]])
    w, m = jnp.array([1e18, 9e18]), jnp.array([mass_electron, mass_ratio * mass_electron])
    new = np.asarray(jax.jit(jax.vmap(lambda key: collide(
        key, jnp.zeros_like(v), v, w, m, jnp.array([-e_charge, e_charge]),
        ((0, 1), (1, 1)), ((0, 1),), 10.0, 1e-9, 1.0, 1.0, 1)))(random.split(random.PRNGKey(18), 8192)))
    wm = np.asarray(w * m)
    momentum_change = np.sum(wm[None, :, None] * (new - np.asarray(v)[None]), axis=1)
    energy_change = np.sum(wm[None, :] * np.sum(new ** 2 - np.asarray(v)[None] ** 2, axis=2), axis=1)
    for change in (momentum_change, energy_change):
        assert np.all(np.abs(change.mean(axis=0)) < 6 * change.std(axis=0, ddof=1) / np.sqrt(len(change)) + 1e-20)


@pytest.mark.parametrize("dt, logarithm, charge", [(0.0, 10.0, -e_charge), (1e-9, 0.0, -e_charge), (1e-9, 10.0, 0.0)])
def test_zero_scattering_is_exactly_inactive(dt, logarithm, charge):
    v = jnp.array([[1e6, 0.0, 0.0], [0.0, 0.0, 0.0]])
    new = collide(random.PRNGKey(19), jnp.zeros_like(v), v, jnp.ones(2), jnp.full(2, mass_electron),
                  jnp.full(2, charge), ((0, 2),), ((0, 0),), logarithm, dt, 1.0, 1.0, 1)
    assert np.array_equal(new, v)


def test_realized_arbitrary_weight_velocity_gradient_matches_finite_difference():
    rng = np.random.default_rng(20)
    v, direction = jnp.asarray(rng.normal(size=(13, 3))), jnp.asarray(rng.normal(size=(13, 3)))
    w = jnp.asarray(np.resize([1e18, 3e18, 9e18], 13))
    m = jnp.asarray(np.r_[np.full(8, mass_electron), np.full(5, 5 * mass_electron)])

    def energy(v):
        new = collide(random.PRNGKey(20), jnp.zeros_like(v), 1e6 * v, w, m, jnp.full(13, -e_charge),
                      ((0, 8), (8, 5)), ((0, 0), (0, 1), (1, 1)), 10.0, 1e-11, 1.0, 1.0, 1)
        return jnp.sum((w / 1e18)[:, None] * (new / 1e6) ** 2)

    derivative = jnp.sum(jax.grad(energy)(v) * direction)
    h = 1e-5
    finite = (energy(v + h * direction) - energy(v - h * direction)) / (2 * h)
    assert np.isfinite(derivative) and np.isclose(derivative, finite, rtol=1e-6, atol=1e-7)


def test_coulomb_logarithm_is_floored_and_survives_zero_temperature():
    """The formulary goes negative in cold, dense plasma (-0.03 at 1e20 m^-3 and 10 meV)
    and divides by zero at T = 0. Both give the floor of 2, with finite gradients."""
    assert float(coulomb_logarithm(1e20, 0.01)) == 2.0
    assert float(coulomb_logarithm(1e20, 1e-3)) == 2.0
    assert float(coulomb_logarithm(1e20, 0.0)) == 2.0
    assert abs(float(coulomb_logarithm(1e20, 0.01, floor=-np.inf)) + 0.026) < 1e-3
    grads = jax.grad(lambda n, t: coulomb_logarithm(n, t), argnums=(0, 1))(1e20, 0.0)
    assert all(np.isfinite(float(g)) for g in grads)


def test_default_coulomb_logarithm_comes_from_the_electrons_wherever_they_are_listed():
    """`Collisions()` takes ln(Lambda) at the density and temperature of the lightest
    negatively charged species, not of whichever species comes first; the temperature
    is m v_th^2 / 2 of the largest thermal-speed component."""
    electrons = Species.electrons(n=400, density=1e20, vth=(2e6, 2e6, 1e6), sampling="low_noise")
    ions = Species.ions(n=400, density=1e20, electrons=electrons, temperature_ratio=0.01, sampling="low_noise")
    domain = Domain(length=1e-4, cells=4, dt_over_dx_c=1.0)
    ln_lambda = float(coulomb_logarithm(1e20, mass_electron * 2e6 ** 2 / 2 / e_charge))
    explicit = Simulation(domain, [ions, electrons], Solver(), Collisions(coulomb_log=ln_lambda)).run(5, seed=0)
    default = Simulation(domain, [ions, electrons], Solver(), Collisions()).run(5, seed=0)
    assert np.allclose(np.asarray(default.v), np.asarray(explicit.v), rtol=1e-9, atol=0)


def test_gradients_through_collisions_stay_finite_for_identical_velocities():
    """Particles with identical velocities, or differing only along z, leave the
    scattering frame undefined and the variance unbounded. The operator is
    differentiated through, so the gradient has to stay finite there, and the
    collision of identical particles has to change nothing."""
    n = 6
    v = np.full((2 * n, 3), 1e5)
    v[n:, 2] += 1e3                                   # the second species differs only along z
    x, mass = jnp.zeros((2 * n, 3)), jnp.full(2 * n, mass_electron)
    charge = jnp.asarray(np.repeat([-e_charge, e_charge], n))

    def run(v, weight, coulomb_log, pairs=((0, 0), (0, 1), (1, 1))):
        return collide(random.PRNGKey(6), x, v, weight, mass, charge, ((0, n), (n, n)), pairs, coulomb_log,
                       1e-9, 1.0, 1.0, 1)

    grads = jax.grad(lambda *a: run(*a).sum(), argnums=(0, 1, 2))(jnp.asarray(v), jnp.ones(2 * n), 10.0)
    assert all(np.isfinite(np.asarray(g)).all() for g in grads)
    assert np.array_equal(np.asarray(run(jnp.asarray(v), jnp.ones(2 * n), 10.0, ((0, 0), (1, 1)))), v)


def _oscillators_energy_error(omega_dt, collisions, n=1000, periods=16.0, heavy=25.0):
    """Electrons and a heavier negative species in one harmonic well E_x = k (x - x0), all
    inside one cell so that every collision between the two conserves the pair's energy
    exactly. Returns the largest relative error of sum(m v^2/2 + e k (x - x0)^2/2) at the
    integer times, where the velocity lives."""
    length, omega = 0.5, 1e9
    k = omega ** 2 * mass_electron / e_charge
    rng = np.random.default_rng(0)
    x, v = np.zeros((n, 3)), rng.normal(0.0, 5e5, (n, 3))
    x0, x[:, 0] = length / 128, length / 128 + rng.uniform(-1e-3, 1e-3, n)
    species = [Species("electrons", n, -1.0, mass_electron, 1e10, x=x, v=v),
               Species("heavy", n, -1.0, heavy * mass_electron, 1e10, x=x, v=v / np.sqrt(heavy))]
    domain = Domain(length=length, cells=64, time_step=omega_dt / omega)
    well = np.zeros((64, 3))
    well[:, 0] = k * (np.asarray(domain.grid) + domain.dx / 2 - x0)
    pairs = Collisions(coulomb_log=1e11, pairs=(("electrons", "heavy"),)) if collisions else None
    steps = 40 * int(round(2 * np.pi * periods / omega_dt / 40))
    out = Simulation(domain, species, Solver(model="electrostatic"), pairs, external_E=well).run(
        steps, store_every=steps // 40)
    mass = np.r_[np.full(n, mass_electron), np.full(n, heavy * mass_electron)]
    energy = np.sum(np.asarray(out.weight) * (0.5 * mass * np.sum(np.asarray(out.v) ** 2, axis=-1)
                                              + 0.5 * e_charge * k * (np.asarray(out.x)[:, :, 0] - x0) ** 2), axis=1)
    return float(np.max(np.abs(energy / energy[0] - 1)))


def test_collisions_at_the_integer_time_keep_the_leapfrog_energy_error():
    """The explicit step kicks u^n -> u^{n+1} in the field at x^{n+1/2}, so the velocity lives at
    integer times. Scattering there, at x^{n+1} = x^{n+1/2} + dt v^{n+1}/2, leaves the kinetic plus
    potential energy of every pair unchanged, and a strongly collisional run (nu/omega ~ 0.1,
    all collisions exact) keeps the leapfrog's own second-order error: at most the collisionless
    one and falling by more than three when the step halves (1.0e-3, 2.0e-4, 3.9e-5 at omega dt =
    0.4, 0.2, 0.1). Scattering at x^{n+1/2}, as the code once did, moved the integer-time position
    by dt dv/2 and multiplied the energy of the same run by 23, 7.8 and 2.5
    (docs/numerics/collisions.md)."""
    coarse, fine = (_oscillators_energy_error(omega_dt, True) for omega_dt in (0.4, 0.2))
    assert coarse < 1.5 * _oscillators_energy_error(0.4, False)
    assert coarse > 3 * fine


@pytest.mark.parametrize("gamma_dt", [1e-3, 1e-4])
def test_maxwellian_temperature_difference_has_coupled_relaxation_rate(gamma_dt):
    """The NRL temperature-transfer rate applies to each species: for equal
    densities the evolving difference decays at 4 nu_inter/3, not 2 nu_inter/3.
    Independent Maxwellian realizations include both phase-space and collision noise."""
    from jaxincell import boltzmann_constant as kb
    n, density, ta, tb, logarithm = 4096, 1e20, 3000.0, 300.0, 10.0
    nu = 16 * np.sqrt(np.pi) * e_charge ** 4 * density * logarithm / (
        (4 * np.pi * epsilon_0) ** 2 * mass_electron ** 2
        * (2 * kb * (ta + tb) / mass_electron) ** 1.5)
    gamma, dt = 4 * nu / 3, gamma_dt / (4 * nu / 3)

    def realization(key):
        phase, collision = random.split(key)
        sigma = jnp.repeat(jnp.sqrt(kb * jnp.array([ta, tb]) / mass_electron), n)
        v = random.normal(phase, (2 * n, 3)) * sigma[:, None]
        new = collide(collision, jnp.zeros_like(v), v, jnp.full(2 * n, density / n),
                      jnp.full(2 * n, mass_electron), jnp.full(2 * n, -e_charge),
                      ((0, n), (n, n)), ((0, 1),), logarithm, dt, 1.0, 1.0, 1)

        def difference(v):
            t = mass_electron / kb * jnp.var(v.reshape(2, n, 3), axis=1).mean(axis=1)
            return t[0] - t[1]
        return (difference(v) - difference(new)) / (dt * gamma * (ta - tb))
    ratios = np.asarray(jax.jit(jax.vmap(realization))(random.split(random.PRNGKey(22), 256)))
    assert abs(ratios.mean() - 1) < 6 * ratios.std(ddof=1) / np.sqrt(len(ratios)) + 0.02


@pytest.mark.parametrize("value", [-1., np.nan, np.inf, [1., 2.]])
def test_fixed_coulomb_log_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="finite nonnegative scalar"):
        Collisions(coulomb_log=value)


def test_nanbu_distribution_has_the_prescribed_angular_moments():
    """Integrate the inverse CDF independently of its random-number sampling.
    The first moment fixes the physical rate, and the second tests the shape;
    very small scattering and the isotropic limit must both remain resolved."""
    from scipy.optimize import brentq
    from scipy.special import roots_legendre

    nodes, weights = roots_legendre(512)
    s = np.array([1e-10, 1e-6, .001, .049, .05, .2, 1., 3., 10., 40.])
    sine, loss = _collisions._nanbu_angle(jnp.asarray(s[:, None]), jnp.asarray((nodes + 1) / 2))
    first = np.asarray(loss) @ (weights / 2)
    second = np.asarray(sine ** 2) @ (weights / 2)
    expected_second = []
    for value in s:
        target = np.exp(-value)

        def langevin(a):
            return a / 3 - a ** 3 / 45 + 2 * a ** 5 / 945 if a < 1e-3 else 1 / np.tanh(a) - 1 / a

        a = (1 / -np.expm1(-value) if value < .05 else
             brentq(lambda a: langevin(a) - target, 0., 2 / -np.expm1(-value), xtol=1e-30))
        expected_second.append(2 * target / a)
    np.testing.assert_allclose(first / -np.expm1(-s), 1., rtol=3e-6, atol=0.)
    np.testing.assert_allclose(second / expected_second, 1., rtol=3e-6, atol=0.)
    assert np.isfinite(sine).all() and (np.asarray(loss) >= 0).all() and (np.asarray(loss) <= 2).all()


def _kinetic_relativistic(u, mass):
    gamma = np.sqrt(1 + np.sum(u * u, axis=-1) / c_light ** 2)
    return mass * np.sum(u * u, axis=-1) / (gamma + 1)


@pytest.mark.parametrize("dtype,tolerance", [(jnp.float64, 2e-12), (jnp.float32, 2e-5)])
@pytest.mark.parametrize("mass_ratio", [1., 1836.15267343])
def test_relativistic_cells_conserve_kinetic_energy_and_momentum(dtype, tolerance, mass_ratio):
    """Exercise self/triplet/cross-species pairs, sparse cells and removed slots.
    Compare kinetic energy, so ion rest mass cannot hide an electron-energy error.
    Large angles stress kinematics; this is not a large-timestep accuracy claim."""
    rng = np.random.default_rng(39)
    ca, xa = _cells([2, 3, 1, 5, 0], rng)
    cb, xb = _cells([1, 2, 5, 3, 1], rng)
    na, nb = len(ca), len(cb)
    cell, x = np.r_[ca, cb], jnp.concatenate([xa, xb]).astype(dtype)
    u = jnp.asarray(c_light * rng.normal(size=(na + nb, 3)) * 10 ** rng.uniform(-3, 2, (na + nb, 1)), dtype)
    # Co-propagating relativistic pairs must not subtract two gamma^2 values.
    first, second = np.flatnonzero(ca == 0)  # this cell's two electrons necessarily pair
    u = u.at[first].set(jnp.array([1000., 0., 0.], dtype) * c_light)
    u = u.at[second].set(jnp.array([1001., .01, 0.], dtype) * c_light)
    mass = jnp.asarray(np.r_[np.full(na, mass_electron), np.full(nb, mass_ratio * mass_electron)], dtype)
    weight = jnp.full(na + nb, 1e20, dtype).at[-1].set(0.)
    new = np.asarray(jax.jit(lambda u: collide(
        random.PRNGKey(39), x, u, weight, mass, jnp.full(na + nb, -e_charge, dtype),
        ((0, na), (na, nb)), ((0, 0), (0, 1), (1, 1)), 15., 1e-3, .2, 1., 5,
        relativistic=True))(u))
    old, mass, weight = np.asarray(u, dtype=float), np.asarray(mass, dtype=float), np.asarray(weight, dtype=float)
    assert new.dtype == dtype and np.isfinite(new).all()
    assert np.array_equal(new[-1], old[-1])
    momenta = [_per_cell(cell, 5, weight[:, None] * mass[:, None] * p) for p in (old, new.astype(float))]
    energies = [_per_cell(cell, 5, weight * _kinetic_relativistic(p, mass)) for p in (old, new.astype(float))]
    momentum_scale = _per_cell(cell, 5, weight * mass * np.linalg.norm(old, axis=1))
    assert np.max(np.linalg.norm(momenta[1] - momenta[0], axis=1) / np.maximum(momentum_scale, 1e-100)) < tolerance
    occupied = energies[0] > 0
    assert np.max(np.abs(energies[1][occupied] / energies[0][occupied] - 1)) < tolerance


def _cold_relativistic_rate(gamma, mass_ratio, density, logarithm=15.):
    """Independent invariant form of Perez Eq. 9 for an electron beam and a
    stationary target. nu is s/dt, the first angular-moment relaxation rate."""
    invariant_mass2 = 1 + mass_ratio ** 2 + 2 * mass_ratio * gamma
    return ((e_charge ** 2 / epsilon_0 / mass_electron) ** 2 * density * logarithm
            / (4 * np.pi * c_light ** 3) * invariant_mass2 * gamma
            / (mass_ratio ** 2 * (gamma ** 2 - 1) ** 1.5))


@pytest.mark.parametrize("gamma,mass_ratio", [(1.00001, 1.), (5., 1.), (5., 1836.15267343)])
def test_relativistic_weighted_beam_rates_see_each_physical_partner_density(gamma, mass_ratio):
    """Check stopping and transverse diffusion against the independent invariant
    rate, with unequal counts and both directions of the weight inequality.
    The low-energy case also checks the nonrelativistic limit."""
    na, nb, density = 4096, 3 * 4096, 1e20
    u0 = c_light * np.sqrt(gamma ** 2 - 1)
    u = jnp.zeros((na + nb, 3)).at[:na, 0].set(u0)
    mass = jnp.repeat(jnp.array([mass_electron, mass_ratio * mass_electron]), jnp.array([na, nb]))
    rate = _cold_relativistic_rate(gamma, mass_ratio, density)
    if gamma < 1.001:
        reduced_mass = mass_electron * mass_ratio / (1 + mass_ratio)
        classical_rate = ((e_charge ** 2 / epsilon_0 / reduced_mass) ** 2 * density * 15.
                          / (4 * np.pi * (u0 / gamma) ** 3))
        np.testing.assert_allclose(rate, classical_rate, rtol=3e-5, atol=0.)
        rate = classical_rate  # independently check the measured nonrelativistic limit
    invariant_mass2 = 1 + mass_ratio ** 2 + 2 * mass_ratio * gamma
    slowing = mass_ratio * (gamma + mass_ratio) / invariant_mass2
    diffusion = 2 * mass_ratio ** 2 / invariant_mass2
    run = jax.jit(lambda w, dt: collide(
        random.PRNGKey(40), jnp.zeros_like(u), u, w, mass, jnp.full(na + nb, -e_charge),
        ((0, na), (na, nb)), ((0, 1),), 15., dt, 1., 1., 1, relativistic=True))
    for ratio in (.25, 1., 4.):
        weight = jnp.repeat(jnp.array([ratio, 1.]) * density / nb, jnp.array([na, nb]))
        dt = 1e-3 / (rate * max(ratio, 1.))  # keep the sampled pair's s small
        new = np.asarray(run(weight, dt))
        observables = [(u0 - new[:na, 0]) / (slowing * rate * dt * u0),
                       np.sum(new[:na, 1:] ** 2, axis=1) / (diffusion * rate * dt * u0 ** 2)]
        other_density = na * ratio / nb
        observables += [new[na:, 0] / (slowing / mass_ratio * rate * other_density * dt * u0),
                        np.sum(new[na:, 1:] ** 2, axis=1)
                        / (diffusion / mass_ratio ** 2 * rate * other_density * dt * u0 ** 2)]
        for samples in observables:
            assert abs(samples.mean() - 1) < 5 * samples.std(ddof=1) / np.sqrt(samples.size) + .003


@pytest.mark.parametrize("mass_ratio", [1., 1836.15267343])
def test_relativistic_unequal_weights_conserve_moments_in_expectation(mass_ratio):
    u = c_light * jnp.array([[3., .1, -.2], [-.1, .4, .05]])
    weight, mass = jnp.array([1e18, 7e18]), jnp.array([mass_electron, mass_ratio * mass_electron])
    keys = random.split(random.PRNGKey(41), 8192)
    new = np.asarray(jax.jit(jax.vmap(lambda key: collide(
        key, jnp.zeros_like(u), u, weight, mass, jnp.array([-e_charge, e_charge]),
        ((0, 1), (1, 1)), ((0, 1),), 15., .02, 1., 1., 1, relativistic=True)))(keys))
    momentum = np.sum(np.asarray(weight * mass)[None, :, None] * (new - np.asarray(u)), axis=1)
    change = _kinetic_relativistic(new, np.asarray(mass)) - _kinetic_relativistic(np.asarray(u), np.asarray(mass))
    energy = np.sum(np.asarray(weight)[None, :] * change, axis=1)
    for delta in (momentum, energy):
        assert np.all(np.abs(delta.mean(axis=0)) < 5 * delta.std(axis=0, ddof=1) / np.sqrt(len(delta)))


def test_relativistic_velocity_correlated_weights_keep_the_pair_local_rate():
    """A single species contains heavy fast markers and light resting markers.
    Self-pairing must use each pair's weight, not a global average. Comoving
    beam-beam and background-background draws give exactly zero deflection."""
    na, nb, density, gamma = 256, 3840, 1e20, 5.
    n = na + nb
    u0 = c_light * np.sqrt(gamma ** 2 - 1)
    u = jnp.zeros((n, 3)).at[:na, 0].set(u0)
    weight = jnp.repeat(jnp.array([9., 1.]) * density / nb, jnp.array([na, nb]))
    rate = _cold_relativistic_rate(gamma, 1., density)
    dt = 1e-3 / (rate * (n - 1) * 9 / nb)
    keys = random.split(random.PRNGKey(47), 32)
    new = np.asarray(jax.jit(jax.vmap(lambda key: collide(
        key, jnp.zeros_like(u), u, weight, jnp.full(n, mass_electron), jnp.full(n, -e_charge),
        ((0, n),), ((0, 0),), 15., dt, 1., 1., 1, relativistic=True)))(keys))
    stopping = (u0 - new[:, :na, 0].mean(axis=1)) / (.5 * rate * dt * u0)
    diffusion = np.sum(new[:, :na, 1:] ** 2, axis=2).mean(axis=1) / (rate * dt * u0 ** 2 / (1 + gamma))
    for ratios in (stopping, diffusion):
        assert abs(ratios.mean() - 1) < 5 * ratios.std(ddof=1) / np.sqrt(len(keys)) + .003


@pytest.mark.parametrize("dt,logarithm,charge", [(0., 15., -e_charge), (1., 0., -e_charge), (1., 15., 0.)])
def test_relativistic_zero_rates_are_identity_with_finite_gradients(dt, logarithm, charge):
    u = c_light * jnp.array([[3., .1, .2], [0., 0., 0.]])

    def run(u, weight):
        return collide(random.PRNGKey(42), jnp.zeros_like(u), u, weight, jnp.full(2, mass_electron),
                       jnp.full(2, charge), ((0, 2),), ((0, 0),), logarithm, dt, 1., 1., 1, relativistic=True)

    assert np.array_equal(run(u, jnp.ones(2)), u)
    gradients = jax.grad(lambda u, w: jnp.sum(run(u, w) / c_light), argnums=(0, 1))(u, jnp.ones(2))
    assert all(np.isfinite(g).all() for g in gradients)


def test_relativistic_comoving_and_axial_pairs_have_finite_state_gradients():
    initial = c_light * jnp.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 10.], [0., 0., 11.]])

    def run(u):
        return collide(random.PRNGKey(43), jnp.zeros_like(u), u, jnp.ones(4), jnp.full(4, mass_electron),
                       jnp.full(4, -e_charge), ((0, 2), (2, 2)), ((0, 0), (1, 1)), 15., 1., 1., 1., 1,
                       relativistic=True)
    assert np.array_equal(run(initial)[:2], initial[:2])
    assert np.isfinite(jax.jacrev(run)(initial)).all()


def test_relativistic_azimuth_frame_has_correct_derivatives_at_the_axes():
    """A separate constant frame at exactly p || z gives a finite but wrong
    derivative: the scattering plane must tilt with the incident momentum."""
    vectors = jnp.asarray([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                           [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    direction = jnp.asarray(np.random.default_rng(49).normal(size=vectors.shape))

    def observable(p):
        change = _collisions._rotate(random.PRNGKey(49), p, jnp.ones(6), jnp.full(6, .3), relativistic=True)
        return jnp.sum((p + change) ** 2 * jnp.array([1., 2., 3.]))

    derivative = jnp.sum(jax.grad(observable)(vectors) * direction)
    h = 1e-5
    finite = (observable(vectors + h * direction) - observable(vectors - h * direction)) / (2 * h)
    assert np.isclose(derivative, finite, rtol=1e-7, atol=0.)


def test_relativistic_angular_underflow_does_not_create_nan_gradients():
    """A positive rate can fall below float32 range. It must not pass 0*inf
    from the derivative of sqrt(s) into an otherwise unchanged trajectory."""
    u = jnp.array([[3., .2, -.1], [0., 0., 0.]], jnp.float32) * c_light

    def run(u, dt):
        return collide(random.PRNGKey(48), jnp.zeros_like(u), u, jnp.ones(2, jnp.float32),
                       jnp.full(2, mass_electron, jnp.float32), jnp.full(2, -e_charge, jnp.float32),
                       ((0, 2),), ((0, 0),), 15., dt, 1., 1., 1, relativistic=True)

    dt = jnp.asarray(1e-30, jnp.float32)
    assert np.array_equal(run(u, dt), u)
    gradients = jax.grad(lambda u, dt: jnp.sum(run(u, dt) / c_light), argnums=(0, 1))(u, dt)
    assert all(np.isfinite(g).all() for g in gradients)


def test_relativistic_density_gradient_matches_ensemble_mean_and_common_seed_difference():
    """An ensemble gradient is checked against an analytic angular moment, not
    just against differentiating the same fixed random realization twice."""
    n, gamma, density, s = 128, 5., 1e20, .2
    eta = np.sqrt(gamma ** 2 - 1)
    u = jnp.zeros((2 * n, 3)).at[:n, 0].set(c_light * eta)
    dt = s / _cold_relativistic_rate(gamma, 1., density)

    def objective(scale, key):
        new = collide(key, jnp.zeros_like(u), u, jnp.full(2 * n, scale * density / n),
                      jnp.full(2 * n, mass_electron), jnp.full(2 * n, -e_charge),
                      ((0, n), (n, n)), ((0, 1),), 15., dt, 1., 1., 1, relativistic=True)
        return jnp.mean(new[:n, 0] / c_light)

    keys = random.split(random.PRNGKey(44), 64)
    gradients = np.asarray(jax.jit(jax.vmap(jax.grad(objective), in_axes=(None, 0)))(1., keys))
    mean = jax.jit(lambda scale: jnp.mean(jax.vmap(objective, in_axes=(None, 0))(scale, keys)))
    finite = float((mean(1.0001) - mean(.9999)) / .0002)
    assert np.isclose(gradients.mean(), finite, rtol=2e-6, atol=0.)
    expected = -.5 * eta * s * np.exp(-s)
    assert abs(gradients.mean() - expected) < 5 * gradients.std(ddof=1) / np.sqrt(len(keys))


@jax.jit
def _relax_relativistic_box(u, weight, dt):
    n = u.shape[0]

    def step(state, _):
        u, key = state
        key, draw = random.split(key)
        u = collide(draw, jnp.zeros_like(u), u, jnp.full(n, weight), jnp.full(n, mass_electron),
                    jnp.full(n, -e_charge), ((0, n),), ((0, 0),), 15., dt, 1., 1., 1, relativistic=True)
        return (u, key), None
    return jax.lax.scan(step, (u, random.PRNGKey(45)), None, length=128)[0][0]


@pytest.mark.parametrize("temperature,drift", [(.1, 0.), (1., 0.), (1., .6)])
def test_maxwell_juttner_is_stationary_under_relativistic_self_collisions(temperature, drift):
    """Use the d^3p measure, including p^2 in the radial CDF. Check quantiles and
    inverse gamma, which pair-energy conservation alone cannot preserve."""
    n, density = 4096, 1e20
    radial = np.linspace(0., max(20 * np.sqrt(temperature), 40 * temperature), 20001)
    pdf = radial ** 2 * np.exp(-(np.sqrt(1 + radial ** 2) - 1) / temperature)
    cdf = np.r_[0., np.cumsum((pdf[:-1] + pdf[1:]) / 2)]
    cdf /= cdf[-1]
    rng = np.random.default_rng(46)
    size = 3 * n if drift else n
    uniform = rng.random(size) if drift else rng.permutation((np.arange(n) + .5) / n)
    magnitude = np.interp(uniform, cdf, radial)
    direction = rng.normal(size=(size, 3))
    eta = magnitude[:, None] * direction / np.linalg.norm(direction, axis=1)[:, None]
    boost = 1 / np.sqrt(1 - drift ** 2)
    if drift:
        # A laboratory time slice has d^3p_lab=(gamma_lab/gamma_rest)d^3p_rest.
        # Bias the rest-frame samples by 1+beta*v_x/c before boosting; a blind
        # boost of uniformly sampled rest-frame momenta is a different measure.
        gamma_rest = np.sqrt(1 + np.sum(eta * eta, axis=1))
        accepted = rng.random(size) < (1 + drift * eta[:, 0] / gamma_rest) / (1 + drift)
        eta, gamma_rest, magnitude = eta[accepted][:n], gamma_rest[accepted][:n], magnitude[accepted][:n]
        assert len(eta) == n
        eta[:, 0] = boost * (eta[:, 0] + drift * gamma_rest)
    u = c_light * eta
    dt = .01 / _cold_relativistic_rate(1 + 3 * temperature, 1., density)
    new = np.asarray(_relax_relativistic_box(jnp.asarray(u), density / n, dt))
    rest = new / c_light
    gamma_lab = np.sqrt(1 + np.sum(rest * rest, axis=1))
    rest[:, 0] = boost * (rest[:, 0] - drift * gamma_lab)
    radius = np.linalg.norm(rest, axis=1)
    for probability in (.25, .5, .9):
        measured = np.mean(radius < np.interp(probability, cdf, radial))
        assert abs(measured - probability) < 5 * np.sqrt(probability * (1 - probability) / n) + .003
    initial_inverse = 1 / np.sqrt(1 + magnitude ** 2)
    final_inverse = 1 / np.sqrt(1 + radius ** 2)
    assert abs(final_inverse.mean() - initial_inverse.mean()) < 5 * final_inverse.std() / np.sqrt(n) + .001
    assert np.linalg.norm(new - u) / np.linalg.norm(u) > .1
    before, after = (_kinetic_relativistic(p, mass_electron).sum() for p in (u, new))
    assert abs(after / before - 1) < 1e-12


def _juttner_mean_gamma(theta):
    """Exact <gamma> of f ~ exp(-gamma/theta) d^3p: 3 theta + K1(1/theta)/K2(1/theta)."""
    from scipy.special import kve
    return 3 * theta + kve(1, 1 / theta) / kve(2, 1 / theta)


def test_collisions_drive_a_nonequilibrium_gas_to_the_juttner_distribution():
    """Stationarity does not say the operator relaxes TO Maxwell-Juttner. A uniform ball of
    momenta (|p| <= 2 mc, isotropic) has the equipartition temperature <p.v>/3 about 3% above
    the Juttner temperature that carries its mean gamma. Collisions conserve <gamma>, so
    the relativistic fixed point is determined: <p.v>/3 must reach theta with
    <gamma> = 3 theta + K1/K2, and the gamma histogram must pass a Kolmogorov test."""
    from scipy.integrate import cumulative_trapezoid
    from scipy.optimize import brentq
    from scipy.stats import kstest
    n, density = 20000, 1e20
    rng = np.random.default_rng(0)
    direction = rng.normal(size=(n, 3))
    eta = 2 * rng.random(n)[:, None] ** (1 / 3) * direction / np.linalg.norm(direction, axis=1)[:, None]

    def stats(eta):
        gamma = np.sqrt(1 + np.sum(eta ** 2, axis=1))
        return gamma.mean(), np.mean(np.sum(eta ** 2, axis=1) / gamma) / 3
    gamma0, temperature0 = stats(eta)
    theta = brentq(lambda t: _juttner_mean_gamma(t) - gamma0, 1e-2, 50)
    assert temperature0 > 1.025 * theta
    dt = .05 * 4 * np.pi * epsilon_0 ** 2 * mass_electron ** 2 * c_light ** 3 / (density * e_charge ** 4 * 15.)
    u = c_light * jnp.asarray(eta)

    @jax.jit
    def relax(u, key):
        def step(u, k):
            return collide(k, jnp.zeros_like(u), u, jnp.full(n, density / n), jnp.full(n, mass_electron),
                           jnp.full(n, -e_charge), ((0, n),), ((0, 0),), 15., dt, 1., 1., 1, relativistic=True), None
        return jax.lax.scan(step, u, random.split(key, 100))[0]
    for key in range(6):
        u = relax(u, random.PRNGKey(key))
    gamma1, temperature1 = stats(np.asarray(u) / c_light)
    assert abs(gamma1 / gamma0 - 1) < 1e-12
    assert abs(temperature1 / theta - 1) < 4e-3 < (temperature0 / theta - 1) / 6
    gamma = np.sort(np.sqrt(1 + np.sum((np.asarray(u) / c_light) ** 2, axis=1)))
    grid = np.linspace(1, gamma[-1] + 1, 20001)
    pdf = grid * np.sqrt(grid ** 2 - 1) * np.exp(-grid / theta)
    cdf = cumulative_trapezoid(pdf, grid, initial=0)
    assert kstest(gamma, lambda g: np.interp(g, grid, cdf / cdf[-1])).pvalue > .01


def test_electron_ion_energy_exchange_matches_the_spitzer_rate_at_a_large_temperature_ratio():
    """Spitzer's equipartition frequency, nu = 8 sqrt(2 pi)/3 n e^4 lnL sqrt(m_e m_i)/
    ((4 pi eps0)^2 (m_e T_i + m_i T_e)^(3/2)), is exact for any mass ratio and unequal
    temperatures; here m_i = 25 m_e and T_e/T_i = 4. The ion heating of a short step must equal
    nu (T_e - T_i) dt to the sampling error plus the 3% the finite step costs at dt = tau_e/400."""
    n, density, logarithm, te, ti, ratio = 100000, 1e20, 10., 2000. * e_charge, 500. * e_charge, 25.
    mi = ratio * mass_electron
    nu = (8 * np.sqrt(2 * np.pi) / 3 * density * e_charge ** 4 * logarithm * np.sqrt(mass_electron * mi)
          / ((4 * np.pi * epsilon_0) ** 2 * (mass_electron * ti + mi * te) ** 1.5))
    tau_e = (6 * np.sqrt(2) * np.pi ** 1.5 * epsilon_0 ** 2 * np.sqrt(mass_electron) * te ** 1.5
             / (logarithm * e_charge ** 4 * density))
    dt = tau_e / 400
    rng = np.random.default_rng(1)
    half = rng.normal(size=(n // 2, 3)) * np.sqrt(np.repeat([te / mass_electron, ti / mi], [1, 0])[0])
    ions = rng.normal(size=(n // 2, 3)) * np.sqrt(ti / mi)
    u = jnp.asarray(np.concatenate((half, -half, ions, -ions)))
    mass = jnp.repeat(jnp.array([mass_electron, mi]), n)

    def temperatures(u):
        return mass_electron * jnp.sum(u[:n] ** 2) / (3 * n), mi * jnp.sum(u[n:] ** 2) / (3 * n)
    start = temperatures(u)

    def heating(key):
        new = collide(key, jnp.zeros_like(u), u, jnp.full(2 * n, density / n), mass,
                      jnp.repeat(jnp.array([-e_charge, e_charge]), n), ((0, n), (n, n)), ((0, 1),),
                      logarithm, dt, 1., 1., 1, relativistic=True)
        return temperatures(new)[1] - start[1]
    measured = np.asarray(jax.jit(jax.vmap(heating))(random.split(random.PRNGKey(3), 16)))
    expected = nu * (start[0] - start[1]) * dt
    assert abs(measured.mean() / expected - .97) < 5 * measured.std(ddof=1) / np.sqrt(16) / expected + .02


def _cold_target_defect(ratio, restore, replicates=512, pairs=2048):
    """Weighted energy defect of one stage, in units of the expected energy lost by the incident
    light electrons, for gamma = 5 on targets at rest of ``ratio`` times their weight."""
    gamma, density = 5., 1e20
    u = jnp.zeros((2 * pairs, 3)).at[:pairs, 0].set(c_light * np.sqrt(gamma ** 2 - 1))
    weight = jnp.repeat(jnp.array([1., ratio]) * density / (ratio * pairs), pairs)
    mass, charge = jnp.full(2 * pairs, mass_electron), jnp.full(2 * pairs, -e_charge)
    s = .002
    dt = s / _cold_relativistic_rate(gamma, 1., density)

    def realization(key):
        new = collide(key, jnp.zeros_like(u), u, weight, mass, charge, ((0, pairs), (pairs, pairs)), ((0, 1),),
                      15., dt, 1., 1., 1, relativistic=True, restore=restore)

        def kinetic(v):
            return mass_electron * c_light ** 2 * (jnp.sqrt(1 + jnp.sum(v * v, axis=1) / c_light ** 2) - 1)
        change = weight * (kinetic(new) - kinetic(u))
        return jnp.sum(change), jnp.sum(change[:pairs])
    defect, light = jax.vmap(realization)(random.split(random.PRNGKey(5), replicates))
    expected = -weight[0] * pairs * mass_electron * c_light ** 2 * (gamma - 1) / 2 * (1 - np.exp(-s))
    return np.asarray(defect) / -expected, np.asarray(light) / expected  # expected < 0


def test_unequal_weight_energy_defect_has_the_higginson_variance_and_restoration_removes_it():
    """A light marker is always updated and its heavy partner with probability 1/R, so the defect
    summed over M pairs, in units of the light population's expected loss, has mean zero and
    variance 2 (R - 1)/M (cold target: Q has <Q^2> = 2 <Q>^2 at small s). With restoration it
    is zero to round-off while the energy the light species loses is unchanged."""
    ratio, pairs = 64., 2048
    defect, light = (np.asarray(a) for a in _cold_target_defect(ratio, False))
    variance = np.var(defect, ddof=1)
    standard_error = np.std((defect - defect.mean()) ** 2) / np.sqrt(len(defect))
    assert abs(defect.mean()) < 5 * defect.std(ddof=1) / np.sqrt(len(defect))
    assert abs(variance - 2 * (ratio - 1) / pairs) < 5 * standard_error
    fixed, light_fixed = (np.asarray(a) for a in _cold_target_defect(ratio, True))
    assert np.abs(fixed).max() < 1e-10 < variance ** .5 / 1e6
    np.testing.assert_array_equal(light_fixed, light)
    assert abs(light.mean() - 1) < 5 * light.std(ddof=1) / np.sqrt(len(light)) + .02


def _restore_cells(rng, per_cell=(20, 30, 20, 20), light_per_cell=(6, 7, 6, 6)):
    """Light electrons and heavy markers in four cells, one heavy marker collected by a wall,
    with the heavy block listed first or second."""
    n_cells = len(per_cell)
    cell = np.concatenate((np.repeat(np.arange(n_cells), light_per_cell), np.repeat(np.arange(n_cells), per_cell)))
    x = jnp.zeros((cell.size, 3)).at[:, 0].set((cell + .5) / n_cells - .5)
    nl = sum(light_per_cell)
    u = jnp.asarray(np.concatenate((c_light * .5 * rng.normal(size=(nl, 3)),
                                    c_light * .05 * rng.normal(size=(cell.size - nl, 3)))))
    weight = np.concatenate((np.full(nl, 1e16), np.full(cell.size - nl, 2e18)))
    weight[nl] = 0.
    mass = np.concatenate((np.full(nl, mass_electron), np.full(cell.size - nl, 100 * mass_electron)))
    charge = np.concatenate((np.full(nl, -e_charge), np.full(cell.size - nl, e_charge)))
    return n_cells, cell, x, u, *(jnp.asarray(a) for a in (weight, mass, charge)), nl


@pytest.mark.parametrize("relativistic", [False, True])
@pytest.mark.parametrize("heavy_first", [False, True])
def test_restoration_returns_every_cell_to_its_momentum_and_energy_and_spares_the_light_species(
        relativistic, heavy_first):
    n_cells, cell, x, u, weight, mass, charge, nl = _restore_cells(np.random.default_rng(0))
    blocks, pairs = ((0, nl), (nl, cell.size - nl)), ((0, 1),)
    if heavy_first:
        order = np.concatenate((np.arange(nl, cell.size), np.arange(nl)))
        x, u, weight, mass, charge, cell = x[order], u[order], weight[order], mass[order], charge[order], cell[order]
        blocks, pairs = ((0, cell.size - nl), (cell.size - nl, nl)), ((1, 0),)
    light = np.asarray(mass) == mass_electron

    def run(restore):
        return np.asarray(collide(random.PRNGKey(3), x, u, weight, mass, charge, blocks, pairs, 10., 1e-9,
                                  1 / n_cells, 1., n_cells, relativistic=relativistic, restore=restore))
    raw, fixed = run(False), run(True)

    def moments(v):
        energy = (_kinetic_relativistic(v, np.asarray(mass)) if relativistic
                  else np.asarray(mass) * np.sum(v * v, axis=1) / 2)
        wm = np.asarray(weight * mass)
        return (_per_cell(cell, n_cells, wm[:, None] * v), _per_cell(cell, n_cells, np.asarray(weight) * energy))
    p0, e0 = moments(np.asarray(u))
    for v, tolerance in ((raw, None), (fixed, 1e-13)):
        p, e = moments(v)
        error = max(np.abs(e - e0).max() / e0.max(), np.abs(p - p0).max() / np.abs(p0).max())
        assert (error > 1e-8) if tolerance is None else (error < tolerance)
    np.testing.assert_array_equal(fixed[light], raw[light])
    assert np.all(fixed[np.asarray(weight) == 0] == raw[np.asarray(weight) == 0])
    assert np.abs(fixed[~light] - raw[~light]).max() > 0


def test_restoration_is_the_identity_between_equal_weights_and_finite_where_it_cannot_act():
    """Equal weights have no defect, so nothing is changed; a cell with a single heavy marker has no
    spread to scale, and one whose heavy markers cannot give up the energy to be removed
    stops at their common velocity: momentum is still exact and nothing is NaN."""
    u = jnp.asarray(c_light * np.array([[.9, 0, 0], [-.3, .1, 0], [.02, 0, 0], [0, .02, 0], [.001, 0, 0]]))
    mass = jnp.array([mass_electron, mass_electron, mass_electron, 50. * mass_electron, 50. * mass_electron])
    charge = jnp.array([-e_charge, -e_charge, -e_charge, e_charge, e_charge])
    x = jnp.zeros((5, 3))
    for weight in (jnp.full(5, 1e18), jnp.array([1., 1., 1e4, 1e4, 1e4]) * 1e18):
        args = (random.PRNGKey(1), x, u, weight, mass, charge, ((0, 2), (2, 3)), ((0, 1),), 10., 1e-9, 1., 1., 1)
        raw, fixed = (np.asarray(collide(*args, relativistic=True, restore=r)) for r in (False, True))
        assert np.isfinite(fixed).all()
        if weight[0] == weight[-1]:
            np.testing.assert_array_equal(fixed, raw)
    one = jnp.array([[.9, 0, 0], [-.3, .1, 0], [.002, 0, 0]]) * c_light
    mass, charge = jnp.array([1., 1., 50.]) * mass_electron, jnp.array([-1., -1., 1.]) * e_charge
    weight = jnp.array([1., 1., 1e4]) * 1e18
    for relativistic in (False, True):
        fixed = np.asarray(collide(random.PRNGKey(2), jnp.zeros((3, 3)), one, weight, mass, charge,
                                   ((0, 2), (2, 1)), ((0, 1),), 10., 1e-9, 1., 1., 1,
                                   relativistic=relativistic, restore=True))
        assert np.isfinite(fixed).all()
        p0 = np.sum(np.asarray(weight * mass)[:, None] * np.asarray(one), axis=0)
        p1 = np.sum(np.asarray(weight * mass)[:, None] * fixed, axis=0)
        assert np.abs(p1 - p0).max() < 1e-12 * np.abs(p0).max()


def test_restoration_gradients_stay_finite():
    n_cells, cell, x, u, weight, mass, charge, nl = _restore_cells(np.random.default_rng(1))

    def energy(scale, relativistic):
        v = collide(random.PRNGKey(4), x, u * scale, weight, mass, charge, ((0, nl), (nl, cell.size - nl)),
                    ((0, 1),), 10., 1e-9, 1 / n_cells, 1., n_cells, relativistic=relativistic, restore=True)
        return jnp.sum(weight[:, None] * v[:, 0] ** 2 * mass[:, None])
    for relativistic in (False, True):
        assert np.isfinite(float(jax.grad(energy)(1.0, relativistic)))


def test_collisions_validate_and_apply_the_restoration_option():
    with pytest.raises(ValueError, match="restore"):
        Collisions(coulomb_log=10., restore=1)
    assert Collisions(coulomb_log=10., restore=np.True_).restore is True
    electrons = Species.electrons(n=256, density=1e20, vth=2e6, sampling="low_noise")
    ions = Species.ions(n=128, density=1e20, electrons=electrons, temperature_ratio=1., sampling="low_noise")
    domain = Domain(length=1e-4, cells=4, dt_over_dx_c=1.0)

    def ledger(restore):
        out = Simulation(domain, [ions, electrons], Solver(), Collisions(coulomb_log=10., restore=restore)).run(
            20, seed=0)
        return float(np.sum(np.asarray(out.work.collision_work)[-1])), float(np.sum(np.asarray(out.work.kinetic)[-1]))
    raw, kinetic = ledger(False)
    fixed, _ = ledger(True)
    assert abs(raw) > 1e-9 * kinetic > 100 * abs(fixed)


def test_ultrarelativistic_drag_on_cold_electrons_is_the_connor_hastie_critical_force():
    """Connor and Hastie (1975): an electron with v -> c loses energy to cold background electrons at
    e E_c c, E_c = n e^3 lnL/(4 pi eps0^2 m c^2). At n = 1e18, lnL = 15, E_c = 7.649e-4 V/m, the
    field of the 5 mV/m runaway baseline being 6.5 E_c. The kernel's stopping of gamma = 100 electrons,
    which sets the collisional drag that the baseline's reduced model assumes, must equal that force."""
    pairs, density, logarithm, gamma = 8192, 1e18, 15., 100.
    critical = density * e_charge ** 3 * logarithm / (4 * np.pi * epsilon_0 ** 2 * mass_electron * c_light ** 2)
    assert abs(critical / 7.6486e-4 - 1) < 1e-4
    u = jnp.zeros((2 * pairs, 3)).at[:pairs, 0].set(c_light * np.sqrt(gamma ** 2 - 1))
    mass, charge = jnp.full(2 * pairs, mass_electron), jnp.full(2 * pairs, -e_charge)
    dt = 1e-3 / _cold_relativistic_rate(gamma, 1., density, logarithm)

    def lost(key):
        new = collide(key, jnp.zeros_like(u), u, jnp.full(2 * pairs, density / pairs), mass, charge,
                      ((0, pairs), (pairs, pairs)), ((0, 1),), logarithm, dt, 1., 1., 1, relativistic=True)
        gamma_new = jnp.sqrt(1 + jnp.sum(new[:pairs] ** 2, axis=1) / c_light ** 2)
        return (gamma - gamma_new) * mass_electron * c_light ** 2 / dt
    power = np.asarray(jax.vmap(lost)(random.split(random.PRNGKey(8), 16))).mean(axis=0)
    speed = c_light * np.sqrt(1 - 1 / gamma ** 2)
    expected = e_charge * critical * speed
    assert abs(power.mean() / expected - 1) < 5 * power.std(ddof=1) / np.sqrt(power.size) / expected + 2e-3
