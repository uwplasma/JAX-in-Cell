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
import jax
import jax.numpy as jnp
from jax import lax, random

__all__ = ["crossing_flux", "sample_crossing", "inject", "check_sources"]

_BISECTIONS = 60   # halvings of the quantile bracket: 2^-60 of it is below double precision


def _inward(source):
    """+1 for a source on the left wall, -1 for one on the right: the direction a
    particle entering through that plane travels."""
    return 1.0 if source.side == "left" else -1.0


def _normal_cdf(z):
    return 0.5 * (1 + lax.erf(z / jnp.sqrt(2.0)))


def _normal_pdf(z):
    return jnp.exp(-z ** 2 / 2) / jnp.sqrt(2 * jnp.pi)


def crossing_flux(source):
    """Number flux :math:`\\Gamma` through the source plane, :math:`\\mathrm{m^{-2}s^{-1}}`.

    With :math:`\\sigma` the normal component spread and :math:`u` the **inward** normal
    drift, a Maxwellian sends :math:`\\Gamma = n[u\\Phi(u/\\sigma) + \\sigma\\varphi(u/\\sigma)]`
    across the plane, with :math:`\\Phi` and :math:`\\varphi` the standard normal
    distribution function and density. It reduces to :math:`n\\sigma/\\sqrt{2\\pi}` at rest
    and to :math:`nu` for a cold beam.

    The sign of :math:`u` is physical and is taken as given: a reservoir drifting away from
    the plane, :math:`u<0`, still sends the tail of its distribution across, and
    :math:`|u|` in its place would turn that trickle into a full beam.
    """
    if source.model == "sampled":
        return source.density * jnp.mean(_crossing_speed(source))
    sigma, u = source.sigma[0], _inward(source) * source.drift[0]
    if source.model == "beam":
        return source.density * u
    return source.density * (sigma * _normal_pdf(u / sigma) + u * _normal_cdf(u / sigma))


def _crossing_speed(source):
    """The inward normal speed of each stored sample, zero for the ones going the other way.

    A reservoir given as samples is the distribution behind the plane, so the flux it sends
    across is :math:`n\\langle v_n\\rangle_+` over that distribution, and the samples that
    cross are drawn in proportion to the same weight."""
    return jnp.maximum(_inward(source) * jnp.asarray(source.samples)[:, 0], 0.0)


def _flux_norm(a):
    """:math:`\\Gamma/(n\\sigma) = a\\Phi(a) + \\varphi(a)` at inward drift :math:`a\\sigma`."""
    return a * _normal_cdf(a) + _normal_pdf(a)


def _flux_cdf(t, a):
    """Probability that a particle crossing the plane does so with normal speed below
    :math:`t\\sigma`, for a Maxwellian of unit spread with inward normal drift :math:`a`:
    the integral of :math:`t\\exp[-(t-a)^2/2]` from 0, over :func:`_flux_norm`."""
    return ((a * (_normal_cdf(t - a) - _normal_cdf(-a)) + _normal_pdf(a) - _normal_pdf(t - a))
            / _flux_norm(a))


@jax.custom_jvp
def _flux_quantile(p, a):
    """The normal speed, in units of :math:`\\sigma`, at which :func:`_flux_cdf` equals ``p``.

    Bisection on :math:`[0, \\max(a,0) + 14]`, which brackets the root for every
    :math:`p<1`: the distribution sits within a few spreads of :math:`\\max(a,0)`, and
    fourteen of them leave nothing a float can resolve above. The bisection itself is
    **not** what is differentiated -- a comparison has derivative zero, so a fixed number of
    halvings would report a sensitivity of zero. The rule below differentiates the equation
    :math:`F(t,a)=p` instead, which is the implicit function theorem and is exact.
    """
    lo = jnp.zeros_like(p * a)
    hi = jnp.broadcast_to(jnp.maximum(a, 0.0) + 14.0, lo.shape) + 0.0 * lo

    def halve(bounds, _):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        low = _flux_cdf(mid, a) < p
        return (jnp.where(low, mid, lo), jnp.where(low, hi, mid)), None

    (lo, hi), _ = lax.scan(halve, (lo, hi), None, length=_BISECTIONS)
    return 0.5 * (lo + hi)


@_flux_quantile.defjvp
def _flux_quantile_jvp(primals, tangents):
    """:math:`F(t,a)=p` differentiated: :math:`F_t\\,dt + F_a\\,da = dp`, with
    :math:`F_t = t\\varphi(t-a)/D`, :math:`D = a\\Phi(a)+\\varphi(a)` and
    :math:`F_a = [\\Phi(t-a)-\\Phi(-a) - t\\varphi(t-a) - F\\Phi(a)]/D`."""
    p, a = primals
    dp, da = tangents
    t = _flux_quantile(p, a)
    norm = _flux_norm(a)
    density = t * _normal_pdf(t - a)
    f_a = (_normal_cdf(t - a) - _normal_cdf(-a) - density - _flux_cdf(t, a) * _normal_cdf(a)) / norm
    return t, (dp - f_a * da) * norm / density


def sample_crossing(key, source, n, inward):
    """``n`` velocities drawn from the crossing distribution, ``(n, 3)``.

    ``inward`` is +1 at the left wall and -1 at the right. The tangential components come
    from their own Maxwellians, at the spreads ``vth[1]`` and ``vth[2]`` give, plus the
    tangential drift, which the plane does not select on. The normal component comes from
    the flux distribution, :math:`p(v) \\propto v f_{\\rm in}(v)` on :math:`v>0`, in one of
    three forms, chosen once by :class:`~jaxincell.Source`:

    * ``model="beam"``: a cold reservoir, every particle at the drift velocity;
    * ``model="maxwellian"``: no normal drift, so :math:`p \\propto v e^{-v^2/2\\sigma^2}` is
      Rayleigh and :math:`v = \\sigma\\sqrt{-2\\ln U}` in closed form;
    * ``model="drifting"``: :math:`p \\propto v e^{-(v-u)^2/2\\sigma^2}`, which has no
      elementary inverse and is sampled by inverting its distribution function.

    Shifting a Rayleigh sample by the drift is none of these: the two agree only at
    :math:`u=0`.

    ``model="sampled"`` is the fourth and is not a form at all: the reservoir is given as
    velocities, and one of them is drawn whole, in proportion to its inward normal
    component. The three components keep whatever correlation they were given, which is
    the point -- a distribution that is Maxwellian in neither the normal nor the
    tangential direction, such as the entrance condition of a magnetised presheath, is
    not a product of three one-dimensional draws.
    """
    if source.model == "sampled":
        weights = _crossing_speed(source)
        drawn = random.choice(key, weights.shape[0], (n,), p=weights / jnp.sum(weights))
        return jnp.asarray(source.samples)[drawn]
    sigma, drift = source.sigma, jnp.asarray(source.drift)
    tangential = jnp.array([0.0, 1.0, 1.0]) * drift          # the drift the plane does not select on
    if source.model == "beam":
        v = jnp.broadcast_to(tangential, (n, 3))
        normal = jnp.broadcast_to(inward * drift[0], (n,))
    else:
        k_normal, k_tangential = random.split(key)
        uniform = random.uniform(k_normal, (n,), minval=jnp.finfo(jnp.result_type(float)).tiny)
        v = sigma * random.normal(k_tangential, (n, 3)) + tangential
        if source.model == "maxwellian":
            normal = sigma[0] * jnp.sqrt(-2 * jnp.log(uniform))
        else:
            normal = sigma[0] * _flux_quantile(1.0 - uniform, inward * drift[0] / sigma[0])
    return v.at[:, 0].set(inward * normal)


def inject(key, source, block, x, v, w, qm, charge_over_mass, dt, length, field, push):
    """Emit one step's worth of the reservoir's flux into the dead slots of one
    species block, and return the new arrays and what was emitted.

    A slot is dead when its weight has reached zero: :func:`~jaxincell._core.apply_particle_bc`
    parks such a particle beyond the wall with no weight and no charge-to-mass ratio, where
    it neither deposits, nor feels a force, nor collides. The ``emit`` slots of smallest
    weight are the ones refilled, which are dead slots whenever any are free; the largest
    weight among them is returned as ``overflow``, positive exactly when a live particle
    was overwritten because the pool was full. Finding them costs one partial sort of the
    block, not a search per slot.

    **The staggering.** The loop carries the position at half-integer times and the velocity
    at integer ones. A particle emitted on this step enters the plane at
    :math:`\\tau_k = t^n + (s_k - \\tfrac12)\\Delta t`, with
    :math:`s_k = (k + 1/2)/N_{\\rm emit}` a quiet quadrature of the interval
    :math:`(t^{n-1/2}, t^{n+1/2}]` that ends where the carried position stands. It is put in
    the arrays as if it had always been there: at the position it reaches by
    :math:`t^{n+1/2}`, and with the velocity it *would* have had at :math:`t^n`, so that the
    one full-step push the rest of the loop applies leaves it with the right velocity at
    :math:`t^{n+1}`. Both come from the field at the plane, ``field``, held constant over the
    entry: the position to second order in the flight, and the velocity from ``push`` run over
    the signed interval :math:`(\\tfrac12 - s_k)\\Delta t`, which is exact for a uniform field
    and is what makes a magnetised entry keep its gyro-phase. Streaming freely instead leaves
    an error of order :math:`(q/m)|E|\\Delta t/v` in the entry velocity of every particle.

    It is emitted at the top of a step, so the deposit of that step already counts it and no
    charge appears between the two halves of the step with no current to account for it.

    Returns:
        tuple: the updated ``x, v, w, qm``, the weight each emitted particle carries, the
        ``(emit, 3)`` velocities it crossed the plane with -- what the reservoir put in, not
        the back-dated value the arrays hold -- and ``overflow``.
    """
    start, n = block
    emit = source.emit
    inward = 1.0 if source.side == "left" else -1.0
    weight = crossing_flux(source) * dt / emit
    velocity = sample_crossing(key, source, emit, inward)
    fraction = (jnp.arange(emit) + 0.5) / emit
    flight = (1.0 - fraction) * dt
    wall = -inward * length / 2                                  # the left wall is at -L/2 and sends +x
    at_plane = jnp.broadcast_to(field, (emit, 6))
    acceleration = charge_over_mass * (at_plane[:, :3] + jnp.cross(velocity, at_plane[:, 3:]))
    entry = wall + velocity[:, 0] * flight + 0.5 * acceleration[:, 0] * flight ** 2
    carried = push(velocity, at_plane, jnp.full((emit,), charge_over_mass),
                   ((0.5 - fraction) * dt)[:, None])
    # the tangential coordinates are ignorable and periodic; start on the plane
    position = jnp.stack([entry, jnp.zeros(emit), jnp.zeros(emit)], axis=1)
    # dead slots first: the emit smallest weights in the block
    slots = start + lax.top_k(-lax.dynamic_slice(w, (start,), (n,)), emit)[1]
    overflow = jnp.max(w[slots])
    return (x.at[slots].set(position), v.at[slots].set(carried), w.at[slots].set(weight),
            qm.at[slots].set(charge_over_mass), weight, velocity, overflow)


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
        if s.source is None:
            continue
        if domain.particle_bc[0 if s.source.side == "left" else 1] != 2:
            raise ValueError(f"the {s.source.side} wall must be particle_bc='absorbing' for a Source on it: "
                             "a reservoir takes back whatever reaches it.")
        if s.source.emit > s.n:
            raise ValueError(f"species {s.name!r} emits {s.source.emit} particles a step into {s.n} slots. "
                             "Species.n is the capacity of the pool, and it has to hold every particle alive "
                             "at once: at least emit times the longest residence time in steps.")
