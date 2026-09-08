"""Post-processing of an :class:`Output`: energies, momentum, Gauss-law residual,
temperatures and the dominant frequency. Everything is a plain function of the
stored arrays and can be recomputed at will."""
import jax.numpy as jnp

from ._constants import epsilon_0, mu_0, speed_of_light as c, elementary_charge

__all__ = ["diagnostics", "energies", "gauss_residual", "temperatures"]


def energies(out):
    """Field and kinetic energies per unit area (J/m^2) at every stored step."""
    field_E = 0.5 * epsilon_0 * jnp.sum(out.E ** 2, axis=(1, 2)) * out.dx
    field_B = 0.5 / mu_0 * jnp.sum(out.B ** 2, axis=(1, 2)) * out.dx
    result = {"electric": field_E, "magnetic": field_B}
    if out.v is not None:
        v2 = jnp.sum(out.v ** 2, axis=-1)
        if out.relativistic:                       # the quantity the relativistic pusher conserves
            gamma = 1 / jnp.sqrt(1 - v2 / c ** 2)
            kinetic_p = (gamma - 1) * out.mass[None, :] * c ** 2
        else:                                      # and the one the Boris pusher conserves
            gamma = jnp.ones_like(v2)
            kinetic_p = 0.5 * out.mass[None, :] * v2
        result["kinetic"] = jnp.sum(kinetic_p, axis=1)
        for i, name in enumerate(out.names):
            result[f"kinetic_{name}"] = jnp.sum(jnp.where(out.species[None, :] == i, kinetic_p, 0.0), axis=1)
        result["total"] = field_E + field_B + result["kinetic"]
        result["momentum"] = jnp.sum(gamma[..., None] * out.mass[None, :, None] * out.v, axis=1)
    return result


def gauss_residual(out):
    """Relative violation of the discrete Gauss law at every stored step,
    :math:`\\max_i |(E_{i+1/2} - E_{i-1/2})/\\Delta x - \\rho_i/\\epsilon_0| / \\max_i |\\rho_i/\\epsilon_0|`.

    The field beyond the left wall, :math:`E_{-1/2}`, is the one the solver used:
    the far end of the box when the wall is periodic, and zero otherwise, since a
    wall carries no field from the other side. Taking it as periodic regardless
    reports a violation in the first cell that the solver never committed."""
    E = out.E[:, :, 0]
    ghost = E[:, -1] if out.field_bc[0] == 0 else jnp.zeros_like(E[:, -1])
    div = (E - jnp.concatenate([ghost[:, None], E[:, :-1]], axis=1)) / out.dx
    rhs = out.rho / epsilon_0
    return jnp.max(jnp.abs(div - rhs), axis=1) / jnp.maximum(jnp.max(jnp.abs(rhs), axis=1), 1e-300)


def temperatures(out):
    """Temperature per species and component in eV, from the velocity variance
    about the mean velocity: :math:`k_B T = m\\,\\mathrm{var}(v)`."""
    result = {}
    for i, name in enumerate(out.names):
        sel = out.species == i
        v = out.v[:, sel]
        m = out.mass[sel] / out.weight[sel]
        var = jnp.var(v, axis=1)
        result[name] = m[0] * var / elementary_charge
    return result


def dominant_frequency(out):
    """Angular frequency of the strongest peak of :math:`E_x` at the box centre."""
    signal = out.E[:, out.E.shape[1] // 2, 0]
    signal = signal - jnp.mean(signal)
    spectrum = jnp.abs(jnp.fft.rfft(signal))
    dt = out.t[1] - out.t[0] if out.t.shape[0] > 1 else out.dt
    freqs = 2 * jnp.pi * jnp.fft.rfftfreq(signal.shape[0], d=dt)
    return freqs[jnp.argmax(spectrum[1:]) + 1]


def diagnostics(out):
    """All of the above in one dictionary."""
    result = energies(out)
    result["gauss_residual"] = gauss_residual(out)
    result["dominant_frequency"] = dominant_frequency(out)
    if out.v is not None:
        result["temperatures"] = temperatures(out)
    return result
