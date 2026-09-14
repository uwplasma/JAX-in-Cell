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


@pytest.mark.parametrize("n_a, w_a, n_b, w_b", [
    (40_000, 1.0, 40_000, 1.0), (40_000, 1.0, 10_000, 4.0), (40_000, 1.0, 10_000, 0.25), (10_000, 0.25, 40_000, 1.0)])
def test_each_species_scatters_off_the_density_of_the_other(n_a, w_a, n_b, w_b):
    """A beam crossing a background at rest, with unequal numbers and weights. In one
    step each beam particle must gain <|dv|^2> = (m_b/M)^2 4 u^2 <delta^2>(n_b) and each
    background particle (m_a/M)^2 4 u^2 <delta^2>(n_a), whichever list is longer and
    whichever is heavier, as if it had collided once with the whole density of the
    other species. The Nanbu-Yonemura acceptance alone gets this wrong when the
    shorter list carries the smaller weight, unless the variance uses n_a n_b / n_ab."""
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

    scale = 1e-3 / variance(min(n_a * w_a, n_b * w_b))            # the smaller variance is 1e-3
    new = np.asarray(collide(random.PRNGKey(5), jnp.zeros((n, 3)), jnp.asarray(v), weight, mass, charge,
                             ((0, n_a), (n_a, n_b)), ((0, 1),), coulomb_log, dt * scale, 1.0, 1.0, 1))
    kick = ((new - v) ** 2).sum(axis=1)
    for block, density in ((slice(0, n_a), n_b * w_b), (slice(n_a, n), n_a * w_a)):
        s2 = variance(density) * scale
        expected = 0.25 * 4 * u ** 2 * s2 / (1 + 3 * s2)          # <delta^2 / (1 + delta^2)> to second order
        assert abs(kick[block].mean() / expected - 1) < 0.08


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
    electrons = Species.electrons(n=400, density=1e20, vth=(2e6, 2e6, 1e6), quiet=True)
    ions = Species.ions(n=400, density=1e20, electrons=electrons, temperature_ratio=0.01, quiet=True)
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
