"""Post-processing of an :class:`Output`: energies, momentum, Gauss-law residual,
temperatures and the dominant frequency. Everything is a plain function of the
stored arrays and can be recomputed at will."""
import jax.numpy as jnp

from ._config import epsilon_0, mu_0, speed_of_light as c, elementary_charge

__all__ = ["bohm_edge", "charge_balance", "diagnostics", "dominant_frequency", "energies",
           "gauss_residual", "moment_profiles", "potential", "temperatures"]


def _blocks(out):
    """Name and index slice of each species; the particles of a species are one
    contiguous block, in the order of ``out.names``."""
    start = 0
    for name, count in zip(out.names, out.counts):
        yield name, slice(start, start + count)
        start += count


def energies(out):
    """Field and kinetic energies per unit area (J/m^2) at every stored step, the momentum, and
    the relative errors of the two: ``energy_error``, :math:`|W(t) - W(0)|/W(0)`, and
    ``momentum_error``, :math:`|\\mathbf P(t) - \\mathbf P(0)|/\\sum_p |\\mathbf p_p(0)|`, with ``0``
    the first stored step. The total momentum can vanish, as it does for two counter-streaming
    beams, while the sum of the magnitudes of the particle momenta does not."""
    field_E = 0.5 * epsilon_0 * jnp.sum(out.E ** 2, axis=(1, 2)) * out.dx
    field_B = 0.5 / mu_0 * jnp.sum(out.B ** 2, axis=(1, 2)) * out.dx
    result = {"electric": field_E, "magnetic": field_B}
    if out.v is not None:
        v2 = jnp.sum(out.v ** 2, axis=-1)
        mass = out.mass[None, :] * out.weight          # of each pseudo-particle, at each stored step
        if out.relativistic:                       # the quantity the relativistic pusher conserves,
            root = jnp.sqrt(1 - v2 / c ** 2)       # (gamma - 1) m c^2, written to keep its digits at low speed
            gamma, kinetic_p = 1 / root, v2 / (root * (1 + root)) * mass
        else:                                      # and the one the Boris pusher conserves
            gamma = jnp.ones_like(v2)
            kinetic_p = 0.5 * mass * v2
        result["kinetic"] = jnp.sum(kinetic_p, axis=1)
        for name, block in _blocks(out):
            result[f"kinetic_{name}"] = jnp.sum(kinetic_p[:, block], axis=1)
        result["total"] = field_E + field_B + result["kinetic"]
        p = (gamma * mass)[..., None] * out.v              # the momentum the pusher conserves, gamma m v
        result["momentum"] = jnp.sum(p, axis=1)
        result["energy_error"] = jnp.abs(result["total"] - result["total"][0]) / _nonzero(result["total"][0])
        result["momentum_error"] = (jnp.linalg.norm(result["momentum"] - result["momentum"][0], axis=1)
                                    / _nonzero(jnp.sum(jnp.linalg.norm(p[0], axis=1))))
    return result


def _nonzero(scale):
    return jnp.maximum(scale, jnp.finfo(scale.dtype).tiny)


def gauss_residual(out):
    """Relative violation of the discrete Gauss law at every stored step, the error in the
    conservation of charge: :math:`\\max_i |(E_{i+1/2} - E_{i-1/2})/\\Delta x - \\rho_i/\\epsilon_0|`
    over :math:`e n/\\epsilon_0`, where :math:`e n` is the mean density of the charge of one sign,
    the larger of the positive and the negative, from the weights of that step (of the final
    state when the particles were not stored). The net density :math:`\\rho` is no scale: in a
    neutral plasma it is the particle noise, which a quiet start makes as small as it likes.

    In a periodic box the field beyond the left wall is the field at the far end,
    and the law is checked in every cell. At a wall it is not: :math:`E_{-1/2}` is
    the field at the electrode, set by the charge that has collected on it, and the
    output does not carry it. The equation for the first cell then *defines* that
    field rather than testing anything, so the residual is taken over the
    remaining cells, which are still one independent check short of the number of
    stored values. :func:`charge_balance` is the independent check this is not: it compares the
    deposit with the wall ledger, covers every cell and both walls, and passes or fails on its
    own."""
    E = out.E[:, :, 0]
    if out.field_bc[0] == 0:
        div = (E - jnp.roll(E, 1, axis=1)) / out.dx
        rhs = out.rho / epsilon_0
    else:
        div = (E[:, 1:] - E[:, :-1]) / out.dx
        rhs = out.rho[:, 1:] / epsilon_0
    charge = out.charge * (out.state.w if out.weight is None else out.weight)
    one_sign = jnp.maximum(jnp.sum(jnp.maximum(charge, 0), axis=-1), jnp.sum(jnp.maximum(-charge, 0), axis=-1))
    return jnp.max(jnp.abs(div - rhs), axis=1) / _nonzero(one_sign / (out.length * epsilon_0))


def charge_balance(out):
    """Charge the box has gained that nothing accounts for, at every stored step, as a fraction
    of the charge of one sign it holds.

    Everything the box holds is the charge on the grid plus the charge on the two walls, and
    everything that has entered or left it is what the sources injected:

    .. math::

        \\Delta\\Big[\\Delta x\\sum_i \\rho_i + \\sigma_L + \\sigma_R\\Big]
        = \\sum_s q_s W_{s,\\rm injected},

    with the difference taken from the first stored step. A wall's own charge is on the ledger,
    so a particle it takes moves from the first term to the second and the total does not
    notice; one that a cloud has reached past is on the wall too, which is the only way the
    two can balance while it is half in and half out.

    This is **not** what :func:`gauss_residual` measures. That one asks whether the field solve
    inverted the charge density it was given, and it cannot ask it of the first cell, whose
    equation defines the wall field the output does not store. This one asks whether the charge
    density and the wall ledger -- two different passes over the particles, one a deposit and
    one a boundary law -- agree about how much charge exists, and it covers every cell and both
    walls. A run can pass either and fail the other.

    A periodic box has no walls to hold charge and no sources, so the sum is constant and this
    is the drift of the deposit alone.
    """
    held = out.dx * jnp.sum(out.rho, axis=1) + jnp.sum(out.sigma, axis=1)
    per_species = jnp.stack([out.charge[block][0] for _, block in _blocks(out)])   # one particle's
    put_in = jnp.sum(per_species[None, :, None] * out.wall.injected, axis=(1, 2))
    charge = out.charge * (out.state.w if out.weight is None else out.weight)
    one_sign = jnp.maximum(jnp.sum(jnp.maximum(charge, 0), axis=-1), jnp.sum(jnp.maximum(-charge, 0), axis=-1))
    return jnp.abs((held - held[0]) - (put_in - put_in[0])) / _nonzero(one_sign)


def moment_profiles(out, start=0, stop=-1):
    """Density, mean velocity, pressure tensor and temperature tensor of each species, averaged
    over the window that ends at stored step ``stop`` and begins at ``start``.

    The window is half-open, :math:`(t_a, t_b]`: a running sum is stored after the chunk it
    ends, so what separates two of them is ``Output.steps[stop] - Output.steps[start]`` and not
    the number of stored entries between. Counting entries is off by one and reads a constant
    profile back low.

    What comes back depends on how many moments the run kept. ``run(moments="density")`` gives
    the density; ``"flux"`` adds the mean velocity; ``"full"`` adds both tensors,

    .. math::

        P_{ij} = m\\,[\\langle n v_iv_j\\rangle - n u_iu_j], \\qquad T_{ij} = P_{ij}/n,

    with the temperature in joules -- divide by the elementary charge for electronvolts. Both
    are ``(species, 3, 3, cells)`` and symmetric by construction, because the six independent
    second moments are what was deposited.

    Empty cells have no velocity and no temperature; they come back as zero rather than as the
    division that made them.
    """
    if out.moments is None:
        raise ValueError("this run kept no moments: pass moments='density', 'flux' or 'full' to run()")
    span = int(out.steps[stop]) - int(out.steps[start])
    if span <= 0:
        raise ValueError(f"a window has to span at least one step: stored steps {start} and {stop} are "
                         f"{int(out.steps[start])} and {int(out.steps[stop])}, which is {span}")
    window = (out.moments[stop] - out.moments[start]) / span
    density = window[:, 0]
    live = density > 0
    result = {"density": density}
    if window.shape[1] < 4:
        return result
    safe = jnp.where(live, density, 1.0)
    velocity = jnp.where(live[:, None], window[:, 1:4] / safe[:, None], 0.0)
    result["velocity"] = velocity
    if window.shape[1] < 10:
        return result
    mass = jnp.stack([out.mass[block][0] for _, block in _blocks(out)])
    order = {(0, 0): 4, (1, 1): 5, (2, 2): 6, (0, 1): 7, (0, 2): 8, (1, 2): 9}
    second = jnp.stack([jnp.stack([window[:, order[min(i, j), max(i, j)]] for j in range(3)])
                        for i in range(3)])                       # (3, 3, species, cells)
    second = jnp.moveaxis(second, 2, 0)                           # (species, 3, 3, cells)
    drift = velocity[:, :, None, :] * velocity[:, None, :, :]
    pressure = mass[:, None, None, None] * (second - density[:, None, None, :] * drift)
    result["pressure"] = jnp.where(live[:, None, None, :], pressure, 0.0)
    result["temperature"] = jnp.where(live[:, None, None, :], pressure / safe[:, None, None, :], 0.0)
    return result


def bohm_edge(position, flow, speed):
    """Where a flow profile crosses a given speed, and whether that is a sheath edge.

    The edge of a sheath is where the ions reach the Bohm speed, so it is read off a
    measured flow profile rather than computed. Returns the interpolated position of the
    first upward crossing of ``speed`` and the number of crossings there are.

    A bin centre is not good enough: the potential still falls by about a tenth of
    :math:`T_e/e` per Debye length at the Bohm point, so a crossing placed half a bin out
    moves the sheath drop by several per cent. This interpolates linearly between the two
    samples that bracket the crossing.

    **No crossing is an answer.** Then the position is NaN and the count is zero, which
    says the run has no sheath edge by this measure -- too short, too noisy, or a flow
    already supersonic at the source. Returning the first sample instead would invent an
    edge and a potential drop to go with it. More than one crossing means the profile is
    not monotonic and the first one may not be the edge; look at the profile.

    Args:
        position: Sample positions, increasing.
        flow: Mean flow speed at each, along the same axis.
        speed: The speed to cross, such as :math:`c_s=\\sqrt{T_e/m_i}`.

    Returns:
        tuple: the crossing position (NaN if there is none) and the number of crossings.
    """
    position, flow = jnp.asarray(position), jnp.asarray(flow)
    below, above = flow[:-1] < speed, flow[1:] >= speed
    crossing = below & above
    count = jnp.sum(crossing)
    i = jnp.argmax(crossing)
    gap = flow[i + 1] - flow[i]
    fraction = jnp.where(gap == 0, 0.0, (speed - flow[i]) / jnp.where(gap == 0, 1.0, gap))
    return jnp.where(count > 0, position[i] + fraction * (position[i + 1] - position[i]), jnp.nan), count


def potential(out, centres=False):
    """Electrostatic potential at the cell faces, the trapezoidal integral of the
    longitudinal field from the left wall,
    :math:`\\phi_{i+1/2} = -\\Delta x\\sum_{j\\le i} (E_{x,j-1/2} + E_{x,j+1/2})/2`.

    The integral needs the field at the left wall face, which the output does not store:
    in a periodic box it is the field at the far end; at a reflective wall, a symmetry
    plane, it is zero; at an absorbing wall it is the field of the charge the conductor
    has collected, which the Gauss law of the first cell gives,
    :math:`E_{x,1/2} - \\Delta x\\,\\rho_0/\\epsilon_0`. The field solver closes two
    absorbing walls with the same rule, so between them the last entry, the potential of
    the right wall, is zero to round-off and the bulk floats above both. A periodic box
    has no wall, so the mean is set to zero instead.

    ``centres=True`` gives it on ``Output.grid`` instead, the mean of the two faces bounding
    each cell, which is where a density or a deposited moment lives. The face to the left of
    the first cell is the far end of a periodic box, and at a wall it is the wall face itself,
    where the integral starts and which is therefore the zero of the gauge: that cell's value
    is then half the first stored face and not the face itself -- half a cell out and a factor
    of two in the one place a sheath profile is steepest."""
    E = out.E[:, :, 0]
    if out.field_bc[0] == 0:
        left = E[:, -1]
    elif out.field_bc[0] == 1:
        left = jnp.zeros_like(E[:, 0])
    else:
        left = E[:, 0] - out.dx * out.rho[:, 0] / epsilon_0
    faces = jnp.concatenate([left[:, None], E], axis=1)
    phi = -out.dx * jnp.cumsum(0.5 * (faces[:, :-1] + faces[:, 1:]), axis=1)
    if out.field_bc[0] == 0:
        phi = phi - jnp.mean(phi, axis=1, keepdims=True)         # the gauge, fixed on the faces
    if not centres:
        return phi
    # the face to the left of the first cell: the far end of a periodic box, and otherwise the
    # wall face, which is where the integral above started and so is the zero of the gauge
    before = phi[:, -1:] if out.field_bc[0] == 0 else jnp.zeros_like(phi[:, :1])
    return 0.5 * (jnp.concatenate([before, phi[:, :-1]], axis=1) + phi)


def temperatures(out):
    """Temperature per species and component in eV, from the velocity variance
    about the mean velocity, :math:`k_B T = m\\,\\mathrm{var}(v)`. Both are
    weighted by the pseudo-particle weights, so what the walls have collected
    no longer counts."""
    result = {}
    for name, block in _blocks(out):
        v, w = out.v[:, block], out.weight[:, block, None]
        total = jnp.maximum(jnp.sum(w, axis=1), jnp.finfo(v.dtype).tiny)
        mean = jnp.sum(w * v, axis=1) / total
        var = jnp.sum(w * (v - mean[:, None]) ** 2, axis=1) / total
        result[name] = out.mass[block.start] * var / elementary_charge
    return result


def dominant_frequency(out):
    """Angular frequency of the strongest peak of :math:`E_x` at the box centre,
    other than the mean, sampled at the stored steps. NaN when fewer than two
    steps were stored, since one sample has no frequency."""
    if out.t.shape[0] < 2:
        return jnp.asarray(jnp.nan)
    signal = out.E[:, out.E.shape[1] // 2, 0]
    spectrum = jnp.abs(jnp.fft.rfft(signal - jnp.mean(signal)))
    freqs = 2 * jnp.pi * jnp.fft.rfftfreq(signal.shape[0], d=out.t[1] - out.t[0])
    return freqs[jnp.argmax(spectrum[1:]) + 1]


def diagnostics(out):
    """All of the above in one dictionary."""
    result = energies(out)
    result["gauss_residual"] = gauss_residual(out)
    result["charge_balance"] = charge_balance(out)
    result["potential"] = potential(out)
    result["dominant_frequency"] = dominant_frequency(out)
    if out.v is not None:
        result["temperatures"] = temperatures(out)
    return result
