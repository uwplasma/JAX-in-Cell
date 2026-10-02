"""The binary collision operator: who collides with whom, what each collision
conserves, which density and Coulomb logarithm set the rate, and that gradients
pass through it."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from jaxincell import Simulation, _collisions, epsilon_0
from jaxincell._collisions import collide, coulomb_logarithm
from jaxincell import elementary_charge as e_charge, mass_electron


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



@pytest.mark.parametrize("unsupported", [{"relativistic": True}, {"time_evolution_algorithm": 1}])
def test_collisions_reject_unsupported_integrators(unsupported):
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=2)
    p["solver_parameters"].update(collisions=True, **unsupported)
    with pytest.raises(AssertionError, match="Coulomb collisions"):
        Simulation(p)


def test_collisions_reject_unvalidated_wall_coupling():
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=2)
    p["solver_parameters"]["collisions"] = True
    p["domain_parameters"].update(particle_BC_left=1, particle_BC_right=1)
    with pytest.raises(AssertionError, match="periodic particle boundaries"):
        Simulation(p)


def test_optional_collisions_leave_zero_rate_boris_run_unchanged_to_roundoff():
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=3)
    reference = Simulation(p).run()
    p["solver_parameters"].update(collisions=True, coulomb_logarithm=0.0)
    inactive = Simulation(p).run()
    # Rebuilding the integer-time half drift introduces only roundoff in positions.
    np.testing.assert_allclose(inactive["positions"], reference["positions"], rtol=0, atol=1e-17)
    np.testing.assert_allclose(inactive["velocities"], reference["velocities"], rtol=1e-14, atol=1e-10)


@pytest.mark.parametrize("thermal_speed_over_c", [1e-4, .01])
def test_automatic_coulomb_log_uses_physical_electron_density_and_temperature(thermal_speed_over_c):
    from tests.test_simulation import small_simulation_parameters
    from jaxincell import speed_of_light
    p = small_simulation_parameters(total_steps=2, number_grid_points=4, number_pseudoparticles=12)
    for group in p["species_parameters"].values():
        for population in group.values():
            population.update(weight=1e15, vth_over_c_x=thermal_speed_over_c,
                              vth_over_c_y=thermal_speed_over_c, vth_over_c_z=thermal_speed_over_c)
    p["solver_parameters"].update(collisions=True, coulomb_logarithm=None)
    automatic = Simulation(p).run()
    density_cm3 = 12e15 / .01 * 1e-6
    temperature_ev = mass_electron * (thermal_speed_over_c * speed_of_light)**2 / (2 * e_charge)
    expected = (23 - .5*np.log(density_cm3) + 1.5*np.log(temperature_ev) if temperature_ev < 10 else
                24 - .5*np.log(density_cm3) + np.log(temperature_ev))
    p["solver_parameters"]["coulomb_logarithm"] = max(expected, 2.)
    prescribed = Simulation(p).run()
    np.testing.assert_allclose(automatic["velocities"], prescribed["velocities"], rtol=2e-14, atol=1e-9)
    np.testing.assert_allclose(automatic["positions"], prescribed["positions"], rtol=2e-14, atol=1e-18)


def test_boris_collisions_use_integer_positions_and_explicit_species_blocks():
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=1, number_pseudoparticles=21)
    p["solver_parameters"].update(collisions=True, coulomb_logarithm=1e19)
    sim = Simulation(p)
    colliding = sim.run()
    p["solver_parameters"]["collisions"] = False
    reference = Simulation(p).run()
    x, v = reference["positions"][0], reference["velocities"][0]
    ids = reference["species_integer_index"]
    expected = collide(random.fold_in(random.PRNGKey(1701), 0), x, v,
                       reference["weights"].reshape(-1), reference["mass_integer_lookup"][ids],
                       reference["charge_integer_lookup"][ids], ((0, 21), (21, 21)),
                       ((0, 0), (0, 1), (1, 1)), 1e19, reference["dt"], reference["dx"], 0.01, 8)
    np.testing.assert_allclose(colliding["positions"][0], x, rtol=0, atol=1e-17)
    np.testing.assert_allclose(colliding["velocities"][0], expected, rtol=2e-14, atol=1e-8)
    assert not np.allclose(expected, v)


def test_temperature_components_use_physical_weights_and_remove_drift():
    from tests.test_simulation import small_simulation_parameters
    from jaxincell import diagnostics, boltzmann_constant
    p = small_simulation_parameters(total_steps=3)
    output = Simulation(p).run()
    v = np.asarray(output["velocities"])
    q, m, w = (np.asarray(output[k]).reshape(-1) for k in ("charges", "masses", "weights"))
    diagnostics(output)
    for name, mask in (("electrons", q < 0), ("ions", q >= 0)):
        mean = np.average(v[:, mask], weights=w[mask], axis=1)
        expected = np.sum(m[mask][None, :, None] * (v[:, mask] - mean[:, None]) ** 2, axis=1)
        expected /= boltzmann_constant * w[mask].sum()
        np.testing.assert_allclose(output["temperature_" + name], expected, rtol=1e-12)


@pytest.mark.parametrize("section", ["domain", "solver"])
def test_parameter_updates_cannot_enable_unvalidated_wall_collisions(section):
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=2)
    p["solver_parameters"]["collisions"] = section == "domain"
    p["domain_parameters"].update(particle_BC_left=int(section == "solver"), particle_BC_right=int(section == "solver"))
    sim = Simulation(p)
    with pytest.raises(AssertionError, match="periodic particle boundaries"):
        if section == "domain":
            sim.domain_parameters = {**sim.domain_parameters, "particle_BC_left": 1, "particle_BC_right": 1}
        else:
            sim.solver_parameters = {**sim.solver_parameters, "collisions": True}


@pytest.mark.parametrize("section", ["constructor", "source", "solver"])
@pytest.mark.filterwarnings("ignore:source_term_active.*:UserWarning")
def test_sources_cannot_enable_unvalidated_collision_coupling(section):
    from tests.test_simulation import small_simulation_parameters
    p = small_simulation_parameters(total_steps=1)
    p["solver_parameters"].update(collisions=section != "solver", field_solver=2)
    p["source_parameters"] = {"source_term_active": int(section != "source")}
    if section == "constructor":
        with pytest.raises(ValueError, match="collisions with particle sources"):
            Simulation(p)
    else:
        sim = Simulation(p)
        with pytest.raises(ValueError, match="collisions with particle sources"):
            if section == "source":
                sim.source_parameters = {"source_term_active": 1}
            else:
                sim.solver_parameters = {**sim.solver_parameters, "collisions": True}


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
