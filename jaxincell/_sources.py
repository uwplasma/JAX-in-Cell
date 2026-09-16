"""A maintained plasma supply: the inflow of one species through one wall.

A thermal wall returns what reaches it and so cannot replace what another wall
collects; a sheath between a thermal wall and a collector drains. A source
supplies an inflow that does not depend on the outflow, which is what keeps a
source-to-collector sheath stationary.

The model is a **prescribed reservoir**. Behind the source plane sits a
half-space of plasma with a given distribution :math:`f_{\\rm in}(\\mathbf v)`;
particles that reach the plane from inside leave and are not returned. What
enters is the flux the reservoir sends across the plane,

.. math::

    \\Gamma = \\int_{v_n>0} v_n f_{\\rm in}(\\mathbf v)\\,d^3v,
    \\qquad
    p_{\\rm cross}(\\mathbf v) = v_n f_{\\rm in}(\\mathbf v)/\\Gamma,

which is the velocity density weighted by the normal speed, not the velocity
density itself: fast particles cross more often. For a Maxwellian at rest of
component spread :math:`\\sigma` that makes the normal speed Rayleigh
distributed, :math:`v_n = \\sigma\\sqrt{-2\\ln U}`, and
:math:`\\Gamma = n\\sigma/\\sqrt{2\\pi}`.

Each step emits a fixed number of particles carrying a continuous weight

.. math:: w = \\Gamma\\,\\Delta t/N_{\\rm emit},

so the emitted weight is exactly :math:`\\Gamma\\Delta t` and is a differentiable
function of the reservoir's density and temperature. A count that changed with
the flux would not be: a particle number is an integer, and
:math:`\\lfloor\\Gamma\\Delta t/w\\rfloor` has derivative zero almost everywhere.
"""
import jax.numpy as jnp
from jax import lax, random

__all__ = ["crossing_flux", "sample_crossing", "inject", "check_sources"]


def crossing_flux(source):
    """Number flux :math:`\\Gamma` through the source plane, :math:`\\mathrm{m^{-2}s^{-1}}`.

    With :math:`\\sigma = v_{th}/\\sqrt2` the component spread and :math:`u` the normal
    drift, a Maxwellian sends :math:`\\Gamma = n[u\\Phi(u/\\sigma) + \\sigma\\varphi(u/\\sigma)]`
    across the plane, with :math:`\\Phi` and :math:`\\varphi` the standard normal
    distribution function and density. It reduces to :math:`n\\sigma/\\sqrt{2\\pi}` at rest
    and to :math:`nu` for a cold beam, the two cases :func:`sample_crossing` draws from.
    """
    sigma, u = source.sigma, jnp.abs(source.drift[0])
    scaled = u / jnp.where(sigma > 0, sigma, 1.0)
    warm = sigma * jnp.exp(-scaled ** 2 / 2) / jnp.sqrt(2 * jnp.pi) + u * _normal_cdf(scaled)
    return source.density * jnp.where(sigma > 0, warm, u)


def _normal_cdf(z):
    return 0.5 * (1 + lax.erf(z / jnp.sqrt(2.0)))


def sample_crossing(key, source, n, inward):
    """``n`` velocities drawn from the crossing distribution, ``(n, 3)``.

    ``inward`` is +1 at the left wall and -1 at the right. Two distributions are
    supported and :class:`~jaxincell.Source` rejects anything else at construction:

    * a **Maxwellian at rest**, normal speed :math:`\\sigma\\sqrt{-2\\ln U}` (Rayleigh)
      and Gaussian tangential components;
    * a **cold beam**, every particle at the drift velocity.

    Adding a drift to a Rayleigh sample is not the drifting crossing distribution,
    whose density is proportional to :math:`v\\exp[-(v-u)^2/2\\sigma^2]` on :math:`v>0`;
    the two agree only at :math:`u = 0`.
    """
    sigma, drift = source.sigma, jnp.asarray(source.drift)
    tangential = jnp.array([0.0, 1.0, 1.0]) * drift          # the drift the plane does not select on
    if source.beam:
        v = jnp.broadcast_to(tangential, (n, 3))
        normal = jnp.broadcast_to(jnp.abs(drift[0]), (n,))
    else:
        k_normal, k_tangential = random.split(key)
        uniform = random.uniform(k_normal, (n,), minval=jnp.finfo(float).tiny)
        v = sigma * random.normal(k_tangential, (n, 3)) + tangential
        normal = sigma * jnp.sqrt(-2 * jnp.log(uniform))
    return v.at[:, 0].set(inward * normal)


def inject(key, source, block, x, v, w, qm, charge_over_mass, dt, length):
    """Emit one step's worth of the reservoir's flux into the dead slots of one
    species block, and return the new arrays and what was emitted.

    A slot is dead when its weight has reached zero: :func:`~jaxincell._core.apply_particle_bc`
    parks such a particle beyond the wall with no weight and no charge-to-mass ratio, where
    it neither deposits, nor feels a force, nor collides. The ``emit`` slots of smallest
    weight are the ones refilled, which are dead slots whenever any are free; the largest
    weight among them is returned as ``overflow``, positive exactly when a live particle
    was overwritten because the pool was full. Finding them costs one partial sort of the
    block, not a search per slot.

    Entry times are a quiet quadrature of the interval that ends where the leapfrog's
    carried position stands, :math:`s_k = (k + 1/2)/N_{\\rm emit}`, and each particle then
    streams freely for the remaining :math:`(1 - s_k)\\Delta t` at its entry velocity,
    feeling no force until the next push. It is emitted at the top of a step, so the
    deposit of that step already counts it and no charge appears between the two halves
    of the step with no current to account for it.

    Returns:
        tuple: the updated ``x, v, w, qm``, the weight each emitted particle carries, and
        ``overflow``.
    """
    start, n = block
    emit = source.emit
    inward = 1.0 if source.side == "left" else -1.0
    weight = crossing_flux(source) * dt / emit
    velocity = sample_crossing(key, source, emit, inward)
    flight = (1.0 - (jnp.arange(emit) + 0.5) / emit) * dt
    wall = -inward * length / 2                                  # the left wall is at -L/2 and sends +x
    entry = wall + velocity[:, 0] * flight                       # the plane plus the residual flight
    # the tangential coordinates are ignorable and periodic; start on the plane
    position = jnp.stack([entry, jnp.zeros(emit), jnp.zeros(emit)], axis=1)
    # dead slots first: the emit smallest weights in the block
    slots = start + lax.top_k(-lax.dynamic_slice(w, (start,), (n,)), emit)[1]
    overflow = jnp.max(w[slots])
    return (x.at[slots].set(position), v.at[slots].set(velocity), w.at[slots].set(weight),
            qm.at[slots].set(charge_over_mass), weight, overflow)


def check_sources(species, solver, domain):
    """Reject the source combinations that are not implemented, before a run starts."""
    sources = [s.source for s in species if s.source is not None]
    if not sources:
        return
    if solver.algorithm != "explicit":
        raise ValueError("a Source needs algorithm='explicit': the implicit scheme would emit a new population "
                         "inside every Picard iteration, which its fixed-point argument does not allow.")
    if solver.model != "electrostatic":
        raise ValueError("a Source needs Solver(model='electrostatic'): an injected particle appears inside the "
                         "box without a trajectory through the wall, so the continuity current that Ampere's law "
                         "integrates would miss it. The electrostatic solve takes E_x from the charge density and "
                         "is exact with a source.")
    if 0 in domain.particle_bc or 0 in domain.field_bc:
        raise ValueError("a Source needs walls: a periodic box has no plane to supply plasma through.")
    for s in species:
        if s.source is not None and domain.particle_bc[0 if s.source.side == "left" else 1] != 2:
            raise ValueError(f"the {s.source.side} wall must be particle_bc='absorbing' for a Source on it: "
                             "a reservoir takes back whatever reaches it.")
