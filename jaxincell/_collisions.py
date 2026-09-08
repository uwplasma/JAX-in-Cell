"""Binary Coulomb collisions, after Takizuka and Abe (J. Comput. Phys. 25, 205, 1977).

Inside every cell the particles of the two species of a pair are matched at
random and each pair is scattered through an angle whose variance carries the
collision frequency. Only the relative velocity is rotated and the resulting
change is shared in inverse proportion to the masses, so each pair conserves
momentum and energy exactly, whatever the time step.

Two details make the scheme work away from the ideal case of two equally
numerous, equally weighted species:

* When a cell holds unequal numbers of the two species, the shorter list is
  cycled so that every particle of the longer one still collides once; the
  longer list drives the pairing, cell by cell.
* The change is applied to each partner with the probability of the ratio of
  the partner's weight to the larger of the two, which gives the right exchange
  on average for unequal pseudo-particle weights and exactly compensates the
  cycling above (Nanbu and Yonemura, J. Comput. Phys. 145, 639, 1998).

The variance of one step follows from matching the perpendicular velocity
diffusion of the Fokker-Planck operator,
:math:`\\langle\\delta^2\\rangle = q_a^2 q_b^2 n \\ln\\Lambda\\,\\Delta t
/ (8\\pi\\epsilon_0^2 m_{ab}^2 u^3)`, with :math:`\\tan(\\Theta/2)=\\delta`.
"""
import jax.numpy as jnp
from jax import random

from ._constants import epsilon_0

__all__ = ["coulomb_logarithm", "collide"]


def coulomb_logarithm(density, temperature_ev, charge_number=1.0):
    """Electron-ion Coulomb logarithm of the NRL formulary, from the electron
    density (:math:`\\mathrm{m^{-3}}`) and temperature (eV)."""
    n_cm3 = density * 1e-6
    cold = 23.0 - jnp.log(jnp.sqrt(n_cm3) * charge_number * temperature_ev ** -1.5)
    hot = 24.0 - jnp.log(jnp.sqrt(n_cm3) / temperature_ev)
    return jnp.where(temperature_ev < 10 * charge_number ** 2, cold, hot)


def _by_cell(cell, n_cells):
    """Sort particles by cell. Returns the permutation, the sorted cells, the
    rank of each particle inside its cell and the population of each cell."""
    order = jnp.argsort(cell)
    sorted_cell = cell[order]
    count = jnp.zeros(n_cells, jnp.int32).at[sorted_cell].add(1)
    first = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(count)[:-1]])
    return order, sorted_cell, jnp.arange(cell.shape[0]) - first[sorted_cell], count


def _rotate(key, u, variance):
    """Rotate the relative velocities ``u`` through the Takizuka-Abe angle."""
    k_delta, k_phi = random.split(key)
    delta = random.normal(k_delta, (u.shape[0],)) * jnp.sqrt(variance)
    sin_t = 2 * delta / (1 + delta ** 2)
    cos_t = (1 - delta ** 2) / (1 + delta ** 2)
    phi = random.uniform(k_phi, (u.shape[0],), maxval=2 * jnp.pi)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    u_perp = jnp.sqrt(ux ** 2 + uy ** 2)
    u_mag = jnp.sqrt(u_perp ** 2 + uz ** 2)
    safe = jnp.maximum(u_perp, 1e-30)
    du_x = (ux / safe) * uz * sin_t * jnp.cos(phi) - (uy / safe) * u_mag * sin_t * jnp.sin(phi) - ux * (1 - cos_t)
    du_y = (uy / safe) * uz * sin_t * jnp.cos(phi) + (ux / safe) * u_mag * sin_t * jnp.sin(phi) - uy * (1 - cos_t)
    du_z = -u_perp * sin_t * jnp.cos(phi) - uz * (1 - cos_t)
    # a relative velocity along z leaves the transverse frame undefined; rotate in x-z instead
    along_z = u_perp < 1e-12 * u_mag
    du_x = jnp.where(along_z, u_mag * sin_t * jnp.cos(phi), du_x)
    du_y = jnp.where(along_z, u_mag * sin_t * jnp.sin(phi), du_y)
    return jnp.stack([du_x, du_y, du_z], axis=1)


def _scatter(key, v, driver, cell, rank, other, other_key, other_count, drives, density,
             weight, mass, charge, coulomb_log, dt, stride):
    """Scatter every particle of ``driver`` against its partner in ``other``.

    The partner is the one of the same rank inside the cell, cycling through the
    partners when there are fewer of them. Cells where ``drives`` is false are
    left to the call that runs the other way round.
    """
    slot = cell * stride + rank % jnp.maximum(other_count[cell], 1)
    j = jnp.clip(jnp.searchsorted(other_key, slot), 0, other_key.shape[0] - 1)
    partner = other[j]
    paired = drives[cell] & (other_count[cell] > 0) & (other_key[j] == slot)
    m_a, m_b = mass[driver], mass[partner]
    q_a, q_b = charge[driver], charge[partner]
    m_r = m_a * m_b / (m_a + m_b)
    u = v[driver] - v[partner]
    u_mag = jnp.maximum(jnp.linalg.norm(u, axis=1), 1e-30)
    variance = (q_a * q_b) ** 2 * density[cell] * coulomb_log * dt / (8 * jnp.pi * epsilon_0 ** 2 * m_r ** 2 * u_mag ** 3)
    k_rotate, k_a, k_b = random.split(key, 3)
    du = _rotate(k_rotate, u, variance)
    w_a, w_b = weight[driver], weight[partner]
    w_max = jnp.maximum(w_a, w_b)
    take_a = paired & (random.uniform(k_a, paired.shape) < w_b / w_max)
    take_b = paired & (random.uniform(k_b, paired.shape) < w_a / w_max)
    v = v.at[driver].add(jnp.where(take_a[:, None], (m_b / (m_a + m_b))[:, None] * du, 0.0))
    return v.at[partner].add(jnp.where(take_b[:, None], -(m_a / (m_a + m_b))[:, None] * du, 0.0))


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
    cell_of = lambda i: jnp.clip(((x[i, 0] + length / 2) / dx).astype(jnp.int32), 0, n_cells - 1)
    for a, b in pairs:
        key, k_a, k_b, k_1, k_2 = random.split(key, 5)
        start_a, n_a = blocks[a]
        start_b, n_b = blocks[b]
        if a == b:
            # a species collides with itself: split it in two and pair the halves
            perm = start_a + random.permutation(k_a, n_a)
            ia, ib = perm[: n_a // 2], perm[n_a // 2: 2 * (n_a // 2)]
        else:
            ia = start_a + random.permutation(k_a, n_a)
            ib = start_b + random.permutation(k_b, n_b)
        order_a, cell_a, rank_a, count_a = _by_cell(cell_of(ia), n_cells)
        order_b, cell_b, rank_b, count_b = _by_cell(cell_of(ib), n_cells)
        ia, ib = ia[order_a], ib[order_b]
        stride = max(ia.shape[0], ib.shape[0]) + 1
        key_a, key_b = cell_a * stride + rank_a, cell_b * stride + rank_b
        density_a = jnp.zeros(n_cells).at[cell_a].add(weight[ia]) / dx
        density_b = jnp.zeros(n_cells).at[cell_b].add(weight[ib]) / dx
        # a particle of a self-colliding species sees the whole species; otherwise the
        # sparser of the two partners sets the rate (Takizuka and Abe, section 2)
        density = density_a + density_b if a == b else jnp.minimum(density_a, density_b)
        args = (density, weight, mass, charge, coulomb_log, dt, stride)
        v = _scatter(k_1, v, ia, cell_a, rank_a, ib, key_b, count_b, count_a >= count_b, *args)
        v = _scatter(k_2, v, ib, cell_b, rank_b, ia, key_a, count_a, count_b > count_a, *args)
    return v
