"""Post-processing of an :class:`Output`: energies, momentum, Gauss-law residual,
temperatures and the dominant frequency. Everything is a plain function of the
stored arrays and can be recomputed at will."""
import jax.numpy as jnp

from ._constants import epsilon_0, mu_0, speed_of_light as c, elementary_charge

__all__ = ["diagnostics", "dominant_frequency", "energies", "gauss_residual", "potential", "temperatures"]


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
    stored values."""
    E = out.E[:, :, 0]
    if out.field_bc[0] == 0:
        div = (E - jnp.roll(E, 1, axis=1)) / out.dx
        rhs = out.rho / epsilon_0
    else:
        div = (E[:, 1:] - E[:, :-1]) / out.dx
        rhs = out.rho[:, 1:] / epsilon_0
    charge = out.charge * (out.state[4] if out.weight is None else out.weight)
    one_sign = jnp.maximum(jnp.sum(jnp.maximum(charge, 0), axis=-1), jnp.sum(jnp.maximum(-charge, 0), axis=-1))
    return jnp.max(jnp.abs(div - rhs), axis=1) / _nonzero(one_sign / (out.length * epsilon_0))


def potential(out):
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
    has no wall, so the mean is set to zero instead."""
    E = out.E[:, :, 0]
    if out.field_bc[0] == 0:
        left = E[:, -1]
    elif out.field_bc[0] == 1:
        left = jnp.zeros_like(E[:, 0])
    else:
        left = E[:, 0] - out.dx * out.rho[:, 0] / epsilon_0
    faces = jnp.concatenate([left[:, None], E], axis=1)
    phi = -out.dx * jnp.cumsum(0.5 * (faces[:, :-1] + faces[:, 1:]), axis=1)
    return phi - jnp.mean(phi, axis=1, keepdims=True) if out.field_bc[0] == 0 else phi


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
    result["potential"] = potential(out)
    result["dominant_frequency"] = dominant_frequency(out)
    if out.v is not None:
        result["temperatures"] = temperatures(out)
    return result
