"""Binary Coulomb collisions, after Takizuka and Abe (J. Comput. Phys. 25, 205, 1977).

Inside every cell the particles are paired at random and the relative velocity of
each pair is rotated through an angle whose variance carries the collision frequency,
which conserves the pair's momentum and energy. The pairing within a species and
between two, the Nanbu-Yonemura acceptance for unequal weights, the pair density of
Perez et al. and the variance are derived on the Collisions page of the numerical
methods."""
import jax.numpy as jnp
from jax import lax, random

from ._constants import epsilon_0

__all__ = ["coulomb_logarithm", "collide"]


def coulomb_logarithm(density, temperature_ev, charge_number=1.0, floor=2.0):
    """Electron-ion Coulomb logarithm of the NRL formulary, from the electron
    density (:math:`\\mathrm{m^{-3}}`) and temperature (eV).

    The formulary's expressions turn negative in cold, dense plasma, where the
    weak-coupling picture behind them fails. The result is kept above ``floor``,
    the minimum of Lee and More (Phys. Fluids 27, 1273, 1984) that particle codes
    commonly adopt, and density and temperature are floored at the smallest
    positive float, so that zero gives the floor instead of an error or NaN."""
    tiny = jnp.finfo(jnp.result_type(float)).tiny
    log_n_cm3 = jnp.log(jnp.maximum(density * 1e-6, tiny))
    log_t = jnp.log(jnp.maximum(temperature_ev, tiny))
    cold = 23.0 - 0.5 * log_n_cm3 - jnp.log(charge_number) + 1.5 * log_t
    hot = 24.0 - 0.5 * log_n_cm3 + log_t
    return jnp.maximum(jnp.where(temperature_ev < 10 * charge_number ** 2, cold, hot), floor)


def _shuffle_by_cell(key, cell, n_cells):
    """Sort particles by cell, in random order inside each cell, with one sort.

    ``cell`` runs from 0 to ``n_cells``, the last value marking removed particles.
    Returns the order, the sorted cells, the first sorted index of every cell and
    the population of every cell, with the removed ones counted as none."""
    bits = random.bits(key, cell.shape)
    sorted_cell, _, order = lax.sort((cell, bits, jnp.arange(cell.shape[0])), num_keys=2)
    first = jnp.searchsorted(sorted_cell, jnp.arange(n_cells + 1, dtype=cell.dtype))
    count = jnp.diff(first, append=cell.shape[0]).at[n_cells].set(0)
    return order, sorted_cell, first, count


def _rotate(key, u, u_mag, variance):
    """Change of the relative velocities ``u`` when rotated through the
    Takizuka-Abe angle. ``u_mag`` is the floored magnitude used for the variance."""
    k_delta, k_phi = random.split(key)
    delta = random.normal(k_delta, u_mag.shape, u.dtype) * jnp.sqrt(variance)
    sin_t = 2 * delta / (1 + delta ** 2)
    one_minus_cos = 2 * delta ** 2 / (1 + delta ** 2)
    phi = random.uniform(k_phi, u_mag.shape, u.dtype, maxval=2 * jnp.pi)
    cos_p, sin_p = jnp.cos(phi), jnp.sin(phi)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    perp2 = ux ** 2 + uy ** 2
    # a relative velocity along z leaves the transverse frame undefined; rotate in x-z instead.
    # The square roots only see positive arguments, so the gradient stays finite at u = 0.
    along_z = perp2 <= 1e-24 * (perp2 + uz ** 2)
    u_perp = jnp.sqrt(jnp.where(along_z, 1.0, perp2))
    u_abs = jnp.sqrt(jnp.where(along_z, 1.0, perp2 + uz ** 2))
    du_x = jnp.where(along_z, jnp.abs(uz) * sin_t * cos_p,
                     (ux * uz * cos_p - uy * u_abs * sin_p) / u_perp * sin_t) - ux * one_minus_cos
    du_y = jnp.where(along_z, jnp.abs(uz) * sin_t * sin_p,
                     (uy * uz * cos_p + ux * u_abs * sin_p) / u_perp * sin_t) - uy * one_minus_cos
    du_z = -jnp.where(along_z, 0.0, u_perp) * sin_t * cos_p - uz * one_minus_cos
    return jnp.stack([du_x, du_y, du_z], axis=1)


def _scatter(key, v, i, j, active, density, fraction, weight, mass, charge, coulomb_log, dt):
    """Scatter particle ``i[k]`` against ``j[k]`` wherever ``active[k]``, with
    ``fraction[k]`` of the variance of a full collision at ``density[k]``."""
    m_i, m_j = mass[i], mass[j]
    q_i, q_j = charge[i], charge[j]
    # Grouped so that no intermediate leaves the range of single precision: the product of
    # two particle masses, or of the permittivity and a mass, is far below 1e-38.
    m_r = m_i / (1 + m_i / m_j)
    u = v[i] - v[j]
    u_mag = jnp.sqrt(jnp.maximum(jnp.sum(u ** 2, axis=1), 1e-20))
    variance = (q_i / epsilon_0 * (q_j / m_r)) ** 2 * fraction * density * coulomb_log * dt / (8 * jnp.pi * u_mag ** 3)
    # Past 1e30 the angle is within 1e-15 of pi anyway; the cap keeps single precision finite. The
    # floor keeps the derivative of the square root finite where the variance vanishes, as it does
    # in the slots that hold no collision.
    variance = jnp.clip(variance, jnp.finfo(u.dtype).tiny, 1e30)
    k_rotate, k_accept = random.split(key)
    du = _rotate(k_rotate, u, u_mag, variance)
    w_i, w_j = weight[i], weight[j]
    accept = random.uniform(k_accept, active.shape, u.dtype) * jnp.maximum(w_i, w_j)
    take_i, take_j = active & (accept < w_j), active & (accept < w_i)
    v = v.at[i].add(jnp.where(take_i[:, None], (m_j / (m_i + m_j))[:, None] * du, 0.0))
    return v.at[j].add(jnp.where(take_j[:, None], -(m_i / (m_i + m_j))[:, None] * du, 0.0))


def _within(key, v, start, n, cell, n_cells, dx, *args):
    """Takizuka-Abe collisions of one species with itself."""
    k_sort, k_pairs, k_second, k_third = random.split(key, 4)
    weight = args[0]
    order, sorted_cell, first, count = _shuffle_by_cell(k_sort, cell, n_cells)
    s = start + order
    half = count // 2
    triplet = (count % 2 == 1) & (count > 1)
    # pairs (0, 1), (2, 3), ... of every cell, packed into n // 2 slots
    slot = jnp.arange(n // 2)
    pair_cell = jnp.repeat(jnp.arange(n_cells + 1), half, total_repeat_length=n // 2)
    rank = slot - (jnp.cumsum(half) - half)[pair_cell]
    p = jnp.clip(first[pair_cell] + 2 * rank, 0, n - 2)
    pairs_i, pairs_j, pairs_active = s[p], s[p + 1], slot < half.sum()
    # in a cell with an odd number the last pair opens the triplet
    pairs_fraction = jnp.where(triplet[pair_cell] & (rank == half[pair_cell] - 1), 0.5, 1.0)
    last = jnp.clip(first + count - 1, 0, n - 1)
    t_1, t_2, t_3 = s[jnp.clip(last - 2, 0, n - 1)], s[jnp.clip(last - 1, 0, n - 1)], s[last]
    # densities of the cell: n_a, and n_aa = 2 sum over its collisions of fraction * min(w_i, w_j)
    w_pairs = jnp.where(pairs_active, pairs_fraction * jnp.minimum(weight[pairs_i], weight[pairs_j]), 0.0)
    w_1, w_2, w_3 = weight[t_1], weight[t_2], weight[t_3]
    w_triplets = jnp.where(triplet, 0.5 * (jnp.minimum(w_2, w_3) + jnp.minimum(w_3, w_1)), 0.0)
    cells = jnp.concatenate([sorted_cell, pair_cell, jnp.arange(n_cells + 1)])
    columns = jnp.zeros((n_cells + 1, 2)).at[cells].add(jnp.stack([
        jnp.concatenate([weight[s], jnp.zeros(n // 2 + n_cells + 1)]),
        2 * jnp.concatenate([jnp.zeros(n), w_pairs, w_triplets])], axis=1)) / dx
    has_pairs = columns[:, 1] > 0
    density = columns[:, 0] * (columns[:, 0] / jnp.where(has_pairs, columns[:, 1], 1.0))
    v = _scatter(k_pairs, v, pairs_i, pairs_j, pairs_active, density[pair_cell], pairs_fraction, *args)
    if n > 2:                   # the triplet's second and third collisions see the velocities left by the first
        v = _scatter(k_second, v, t_2, t_3, triplet, density, 0.5, *args)
        v = _scatter(k_third, v, t_3, t_1, triplet, density, 0.5, *args)
    return v


def _between(key, v, block_a, block_b, cell_a, cell_b, n_cells, dx, *args):
    """Collisions between two species, the longer list driving cell by cell."""
    (start_a, n_a), (start_b, n_b) = block_a, block_b
    weight = args[0]
    k_a, k_b, k_scatter = random.split(key, 3)
    order_a, sorted_a, first_a, count_a = _shuffle_by_cell(k_a, cell_a, n_cells)
    order_b, sorted_b, first_b, count_b = _shuffle_by_cell(k_b, cell_b, n_cells)
    ia, ib = start_a + order_a, start_b + order_b
    rank_a, rank_b = jnp.arange(n_a) - first_a[sorted_a], jnp.arange(n_b) - first_b[sorted_b]
    # every particle of a takes the partner of its rank in b, cycled; the particles of b left
    # over in cells where b is the longer list take theirs from a
    on_b = count_b[sorted_a]
    partner_a = ib[jnp.minimum(first_b[sorted_a] + rank_a % jnp.maximum(on_b, 1), n_b - 1)]
    on_a = count_a[sorted_b]
    partner_b = ia[jnp.minimum(first_a[sorted_b] + rank_b % jnp.maximum(on_a, 1), n_a - 1)]
    i, j = jnp.concatenate([ia, ib]), jnp.concatenate([partner_a, partner_b])
    active = jnp.concatenate([on_b > 0, (on_a > 0) & (rank_b >= on_a)])
    cells = jnp.concatenate([sorted_a, sorted_b])
    # densities of the cell: n_a, n_b and n_ab = sum over its pairs of min(w_i, w_j)
    zeros_a, zeros_b = jnp.zeros(n_a), jnp.zeros(n_b)
    columns = jnp.zeros((n_cells + 1, 3)).at[cells].add(jnp.stack([
        jnp.concatenate([weight[ia], zeros_b]), jnp.concatenate([zeros_a, weight[ib]]),
        jnp.where(active, jnp.minimum(weight[i], weight[j]), 0.0)], axis=1)) / dx
    has_pairs = columns[:, 2] > 0
    density = columns[:, 0] * (columns[:, 1] / jnp.where(has_pairs, columns[:, 2], 1.0))
    return _scatter(k_scatter, v, i, j, active, density[cells], 1.0, *args)


def collide(key, x, v, weight, mass, charge, blocks, pairs, coulomb_log, dt, dx, length, n_cells):
    """Apply one collision step to the velocities.

    Args:
        key: PRNG key.
        x, v: Positions and velocities, ``(N, 3)``.
        weight, mass, charge: Per-particle pseudo-particle weight, physical mass
            and physical charge, ``(N,)``.
        blocks: ``((start, count), ...)`` of each species in the particle arrays; static.
        pairs: ``((a, b), ...)`` species indices to collide; static.
        coulomb_log: Coulomb logarithm (scalar).
        dt, dx, length, n_cells: Time step, cell size, box length, cell count.
    """
    # A particle a wall has collected has no weight left. It goes to a cell of its own
    # past the last one, where it neither collides nor takes a partner from one that can.
    inside = jnp.clip(((x[:, 0] + length / 2) / dx).astype(jnp.int32), 0, n_cells - 1)
    cell = jnp.where(weight > 0, inside, n_cells)
    args = (weight, mass, charge, coulomb_log, dt)
    for a, b in pairs:
        key, sub = random.split(key)
        (start_a, n_a), (start_b, n_b) = blocks[a], blocks[b]
        if a == b:
            if n_a > 1:
                v = _within(sub, v, start_a, n_a, cell[start_a:start_a + n_a], n_cells, dx, *args)
        else:
            v = _between(sub, v, blocks[a], blocks[b], cell[start_a:start_a + n_a],
                         cell[start_b:start_b + n_b], n_cells, dx, *args)
    return v
