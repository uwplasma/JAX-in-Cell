"""Numerical kernels of the particle-in-cell cycle.

Staggering (Yee, one dimension): the charge density and the magnetic field live
at cell centres :math:`x_i`, the electric field and the current density at cell
faces :math:`x_{i+1/2}`. Every array has one entry per cell; entry ``i`` of a
face quantity refers to :math:`x_{i+1/2}`.

Boundary codes: 0 periodic, 1 reflective, 2 absorbing, 3 thermal, 4 open, given
per wall as a ``(left, right)`` pair. They are static, so the branches below are
resolved when the program is traced and cost nothing at run time. An absorbing
wall may send back part of each particle, as a fraction of its weight. A thermal
wall is a reflective one here; the simulation then redraws the velocities. An
open wall is the electrical condition of a plane a ``Source`` supplies through:
nothing beyond it, as at a conductor, but no condition on the field either, whose
constant then comes from the charge on the electrode opposite.
"""
import jax.numpy as jnp

from ._config import epsilon_0, speed_of_light as c

__all__ = ["s2_weights", "map_indices", "deposit", "PARITY", "PARK", "to_centres", "with_ghosts", "gather",
           "current_from_continuity", "to_faces", "wall_faces_E", "curl_E", "curl_B", "half_step_fields",
           "E_x_from_rho", "boris", "boris_relativistic", "apply_particle_bc", "wrap_positions", "smooth"]


# --- shape function and boundary mapping ---------------------------------------

def s2_weights(x, x0, dx):
    """Quadratic-spline (triangular-shaped-cloud) weights of positions ``x`` on a
    grid whose first point is ``x0``. Returns the three stencil indices and
    weights, shapes ``(N, 3)``."""
    s = (x - x0) / dx
    k = jnp.floor(s + 0.5).astype(jnp.int32)
    d = s - k
    w = jnp.stack([0.5 * (0.5 - d) ** 2, 0.75 - d ** 2, 0.5 * (0.5 + d) ** 2], axis=-1)
    return k[:, None] + jnp.array([-1, 0, 1], dtype=jnp.int32), w


def map_indices(idx, n, bc):
    """Bring stencil indices outside ``[0, n)`` back according to the wall codes:
    periodic wraps, reflective clamps to the boundary cell (zero-gradient
    extrapolation), absorbing drops the contribution."""
    if bc[0] == 0:
        return idx % n, jnp.ones_like(idx, dtype=bool)
    keep = jnp.ones_like(idx, dtype=bool)
    if bc[0] in (2, 4):
        keep = keep & (idx >= 0)
    if bc[1] in (2, 4):
        keep = keep & (idx < n)
    return jnp.clip(idx, 0, n - 1), keep


def deposit(x, q, x0, dx, n, bc):
    """Density on the grid from particle positions ``x`` and amounts ``q``
    (charge, or charge times a velocity component): :math:`\\sum_p q_p S_2(x_i - x_p)`."""
    idx, w = s2_weights(x, x0, dx)
    idx, keep = map_indices(idx, n, bc)
    return jnp.zeros(n).at[idx].add(jnp.where(keep, w, 0.0) * (q / dx)[:, None])


# Sign of (E_x, E_y, E_z, B_x, B_y, B_z) under the reflection x -> -x: E is a vector, B a pseudovector.
PARITY = (-1.0, 1.0, 1.0, 1.0, -1.0, -1.0)


def to_centres(E, left, right):
    """Face values averaged to the centres, :math:`(E_{i-1/2} + E_{i+1/2})/2`. ``left`` is
    the value at the left wall face :math:`E_{-1/2}`, which the grid does not store, and
    ``right`` stands in for the stored value at the last face."""
    faces = jnp.concatenate([left[None], E[:-1], right[None]])
    return 0.5 * (faces[:-1] + faces[1:])


def with_ghosts(F, bc, parity=None):
    """A centred field ``(n, C)`` with one centre added beyond each wall, ``(n + 2, C)``,
    which is as far as the cloud of a particle inside the box reaches.

    Beyond a periodic wall is the far end of the box. A reflective wall is a symmetry
    plane, so the centre beyond it is the mirror image of the centre inside, with each
    component's sign under the reflection, ``parity``: the box then gathers exactly as
    the doubled box of the image method would, and a particle feels its image and not
    itself. An absorbing wall is a conductor with nothing beyond it, so the value there is
    zero, which is also what the deposit does with the part of a cloud outside the box.
    Beyond an open plane is more of the same plasma, so the field continues unchanged;
    zero there would be the field of a conductor that is not present, and a particle
    within half a cloud of the plane would feel its image.
    Without ``parity`` the field is a prescribed one and simply continues beyond a wall."""
    def ghost(code, edge, far):
        if code == 0:
            return far
        if parity is None or code == 4:
            return edge
        return parity * edge if code == 1 else jnp.zeros_like(edge)

    return jnp.concatenate([ghost(bc[0], F[0], F[-1])[None], F, ghost(bc[1], F[-1], F[0])[None]])


def gather(F, x, x0, dx):
    """Interpolate a centred field with its ghost centres, ``(n + 2, C)`` from
    :func:`with_ghosts`, to positions ``x`` with the spline of the deposit, giving
    ``(N, C)``; ``x0`` is the first centre inside the box. Indices are clipped only for
    particles parked beyond a wall, whose force is zero anyway."""
    idx, w = s2_weights(x, x0 - dx, dx)
    return jnp.einsum("nk,nkc->nc", w, F[jnp.clip(idx, 0, F.shape[0] - 1)])


# --- sources -----------------------------------------------------------------------

def _integrate_from_walls(s, dx, bc, wall_value=0.0):
    """The face quantity :math:`F` with :math:`(F_{i+1/2} - F_{i-1/2})/\\Delta x = s_i` for a
    centred source :math:`s`. That fixes :math:`F` up to one constant, the value
    :math:`F_{-1/2}` at the left wall face (which the grid does not store), and the walls
    fix the constant. The Gauss solve (:math:`F = E_x`, :math:`s = \\rho/\\epsilon_0`) and
    the continuity current (:math:`F = J_x`, :math:`s = -\\partial_t\\rho`) share this
    function, so that the field Ampere's law advances keeps the closure the initial
    Gauss solve imposed. Every rule is its own mirror image, so a charge distribution
    and its reflection give reflected fields.

    * **Periodic**: a solution exists only for a source with zero sum, so the mean
      source is removed (a uniform neutralising background) and :math:`F` is given
      zero mean.
    * **Reflective at both walls**: a box between two symmetry planes is half of a
      periodic box twice as long, holding the charge and its mirror image. The mean
      source is removed as in a periodic box, and :math:`F` then vanishes at both
      walls, as :math:`E_x` must at a symmetry plane.
    * **Reflective at one wall**: :math:`F` vanishes at that wall and the sum runs from
      it; the absorbing wall opposite floats.
    * **Absorbing at both walls**: conductors short-circuited to each other, so the
      potential difference across the box vanishes. With the potential at the centres
      and each conductor half a cell beyond the last centre, that difference is the
      trapezoidal sum over the :math:`N_x + 1` faces from wall to wall,
      :math:`\\tfrac12 F_{-1/2} + \\sum_{i=0}^{N_x-2} F_{i+1/2} + \\tfrac12 F_{N_x-1/2} = 0`.
    * **Open on the left, absorbing on the right**: the plane a source supplies through
      imposes nothing, and the constant comes from the electrode opposite instead.
      ``wall_value`` is what :math:`F` takes at the right wall face; for the Gauss solve
      that is :math:`-\\sigma_w/\\epsilon_0`, the field of the charge the collector holds.
      A symmetry plane would give the same answer when nothing crosses it and the box
      starts neutral; with a source it is the ledger of the collector, not the source
      plane, that closes the problem.
    """
    if bc[0] == bc[1] != 2:
        s = s - jnp.mean(s)
    F = dx * jnp.cumsum(s)                                      # F_{i+1/2} - F_{-1/2}
    if bc == (0, 0):
        return F - jnp.mean(F)
    if bc == (2, 1):
        return F - F[-1]
    if bc == (2, 2):
        return F - (jnp.sum(F) - 0.5 * F[-1]) / F.shape[0]
    if bc == (4, 2):
        return F + (wall_value - F[-1])
    return F


def current_from_continuity(rho_old, rho_new, dt, dx, wall_current, bc):
    """Longitudinal current at the faces that satisfies the discrete continuity
    equation :math:`(\\rho_i^{new} - \\rho_i^{old})/\\Delta t + (J_{i+1/2} - J_{i-1/2})/\\Delta x = 0`
    exactly, with the wall closures of :func:`_integrate_from_walls`. A periodic box has
    no wall to fix the constant, which is instead the mean current the particles carry,
    :math:`\\langle J\\rangle = L^{-1}\\sum_p q_p v_{x,p}`; between two absorbing walls the
    current the closure removes is the one in the external circuit.

    With an open source plane the current across that plane is not tracked, and
    ``wall_current`` is taken as zero: :math:`J_x` is then the internal transport measured
    from the source plane rather than an absolute current density, which is a diagnostic
    and nothing more. An electrostatic run, which is the only kind a source is allowed in,
    takes :math:`E_x` from the charge density and never uses it."""
    J = _integrate_from_walls(-(rho_new - rho_old) / dt, dx, bc, wall_current)
    return J + wall_current if bc[0] == 0 else J


def to_faces(f, bc):
    """Average a cell-centred quantity to the faces, :math:`f_{i+1/2} = (f_i + f_{i+1})/2`."""
    if bc[1] == 0:
        right = f[0]
    elif bc[1] == 1:
        right = f[-1]
    else:                                 # a conductor, or an open plane
        right = jnp.zeros_like(f[-1])
    return 0.5 * (f + jnp.concatenate([f[1:], right[None]]))


# --- Maxwell ---------------------------------------------------------------------------

def _left_ghost_E(E, B, bc):
    """Value of E at the face beyond the left wall, :math:`E_{-1/2}`."""
    if bc[0] == 0:
        return E[-1]
    if bc[0] == 1:
        return E[0]
    # first-order radiating (Mur) condition: the outgoing plane-wave relations
    return jnp.array([0.0, -2 * c * B[0, 2] - E[0, 1], 2 * c * B[0, 1] - E[0, 2]])


def _right_ghost_B(B, E, bc):
    """Value of B at the centre beyond the right wall, :math:`B_{N_x}`."""
    if bc[1] == 0:
        return B[0]
    if bc[1] == 1:
        return B[-1]
    return jnp.array([0.0, -(2 / c) * E[-1, 2] - B[-1, 1], (2 / c) * E[-1, 1] - B[-1, 2]])


def wall_faces_E(E, B, rho, dx, bc):
    """E at the two wall faces, as :func:`to_centres` takes them.

    The grid does not store the left wall face. There :math:`E_x` is the field at the
    far end of a periodic box; zero at a reflective wall, the symmetry plane; and at an
    absorbing wall the field of the charge the conductor has collected, which the Gauss
    law of the first cell gives, :math:`E_{1/2} - \\Delta x\\,\\rho_0/\\epsilon_0`. The
    transverse components are the ghost values the curl uses. On the right the last
    stored face is the wall face, with :math:`E_x` held at zero at a reflective wall,
    where the closure of :func:`E_x_from_rho` already puts it."""
    left = _left_ghost_E(E, B, bc)
    if bc[0] == 1:
        left = left.at[0].set(0.0)
    elif bc[0] in (2, 4):                 # a conductor, or an open plane: the Gauss law of the first cell
        left = left.at[0].set(E[0, 0] - dx * rho[0] / epsilon_0)
    return left, (E[-1].at[0].set(0.0) if bc[1] == 1 else E[-1])


def curl_E(E, B, dx, bc):
    """:math:`\\nabla\\times\\mathbf E` at the cell centres from E at the faces."""
    Eg = jnp.concatenate([_left_ghost_E(E, B, bc)[None], E])
    d = (Eg[1:] - Eg[:-1]) / dx
    return jnp.stack([jnp.zeros_like(d[:, 0]), -d[:, 2], d[:, 1]], axis=1)


def curl_B(B, E, dx, bc):
    """:math:`\\nabla\\times\\mathbf B` at the faces from B at the centres."""
    Bg = jnp.concatenate([B, _right_ghost_B(B, E, bc)[None]])
    d = (Bg[1:] - Bg[:-1]) / dx
    return jnp.stack([jnp.zeros_like(d[:, 0]), -d[:, 2], d[:, 1]], axis=1)


def half_step_fields(E, B, J, dt2, dx, bc, electric_first):
    """Advance Maxwell's equations by ``dt2``. The two orderings compose into the
    symmetric split E-B-B-E of one full step."""
    if electric_first:
        E = E + dt2 * (c ** 2 * curl_B(B, E, dx, bc) - J / epsilon_0)
        B = B - dt2 * curl_E(E, B, dx, bc)
    else:
        B = B - dt2 * curl_E(E, B, dx, bc)
        E = E + dt2 * (c ** 2 * curl_B(B, E, dx, bc) - J / epsilon_0)
    return E, B


def E_x_from_rho(rho, dx, bc, wall_field=0.0):
    """Solve the discrete Gauss law :math:`(E_{i+1/2} - E_{i-1/2})/\\Delta x = \\rho_i/\\epsilon_0`
    for the longitudinal field at the faces, with the wall closures of
    :func:`_integrate_from_walls`; ``wall_field`` is the field at the collector face,
    :math:`-\\sigma_w/\\epsilon_0`, which closes a box with an open source plane. In a
    periodic box this is exactly the field the finite-difference symbol
    :math:`(1 - e^{-ik\\Delta x})/\\Delta x` gives in Fourier space, without the complex
    arithmetic."""
    return _integrate_from_walls(rho / epsilon_0, dx, bc, wall_field)


# --- particles --------------------------------------------------------------------------

def boris(v, E, B, qm, dt):
    """One Boris step for the velocity: half electric kick, exact rotation about
    B by :math:`2\\arctan(|q B \\Delta t / 2m|)`, half electric kick."""
    v = v + qm * E * (dt / 2)
    b = qm * B * (dt / 2)
    v_prime = v + jnp.cross(v, b)
    v = v + 2 * jnp.cross(v_prime, b) / (1 + jnp.sum(b * b, axis=1, keepdims=True))
    return v + qm * E * (dt / 2)


def boris_relativistic(u, E, B, qm, dt):
    """The same three sub-steps applied to the momentum per unit mass
    :math:`\\mathbf u = \\gamma\\mathbf v`, which a relativistic run carries in place of
    the velocity; takes and returns :math:`\\mathbf u`. Working per unit mass keeps every
    intermediate within single precision, which :math:`(m_e c)^2 \\approx 10^{-43}` is not.
    Carrying :math:`\\mathbf u` keeps :math:`\\gamma = \\sqrt{1 + u^2/c^2}` accurate at any
    energy, where :math:`1/\\sqrt{1 - v^2/c^2}` loses a fraction :math:`\\gamma^2\\epsilon`
    at each conversion: converting every step, a single-precision particle started at
    :math:`\\gamma = 1000` ended at 1423 after a thousand steps with no field."""
    u = u + qm * E * (dt / 2)
    gamma = jnp.sqrt(1 + jnp.sum(u * u, axis=1, keepdims=True) / c ** 2)
    t = qm * B * (dt / 2) / gamma
    u_prime = u + jnp.cross(u, t)
    u = u + 2 * jnp.cross(u_prime, t) / (1 + jnp.sum(t * t, axis=1, keepdims=True))
    return u + qm * E * (dt / 2)


PARK = 1.5      # cells beyond a wall where an absorbed particle is parked; see apply_particle_bc


def apply_particle_bc(x, v, w, qm, box, bc, restitution, reflection, dx, floor=0.0):
    """Bring particles that left the box back according to the wall codes, and
    report what each wall received.

    A reflective wall mirrors the position and multiplies the normal velocity by
    ``-restitution``. An absorbing wall sends back the fraction ``reflection`` of
    the weight ``w`` of each particle that reaches it, mirrored and bounced the
    same way, and collects the rest. A particle with no weight left is stopped,
    has its charge-to-mass ratio zeroed and is parked outside the grid, where it
    stays: one and a half cells beyond the wall, the half-width of the quadratic
    spline, so that its cloud lies wholly beyond the wall on the centred grid and on
    the staggered one alike. ``restitution`` is a ``(left, right)`` pair and
    ``reflection`` a pair of per-particle arrays. The ignorable coordinates are
    always periodic.

    ``floor`` is a weight, per particle or one for all, at or below which a wall keeps
    what is left of a particle instead of reflecting it again. Without it a wall
    returning the fraction :math:`R` of every impact holds a particle for ever, its
    weight falling as :math:`R^k`, and its slot is never free for a
    :class:`~jaxincell.Source` to refill. The remainder goes to the wall, so the
    ledger below stays exact.

    Returns:
        tuple: ``x, v, w, qm`` and ``(arrived, kept)``, two ``(2, N)`` arrays, side 0
        the left wall and side 1 the right: the weight of each particle that reached
        that wall on this step, zero where it did not, and the part of it the wall
        kept. The caller has the velocity before and after the bounce, so the charge,
        energy and momentum a wall received follow from these two arrays alone.
    """
    L, Ly, Lz = box
    x = x.at[:, 1].set((x[:, 1] + Ly / 2) % Ly - Ly / 2)
    x = x.at[:, 2].set((x[:, 2] + Lz / 2) % Lz - Lz / 2)
    xx, vx = x[:, 0], v[:, 0]
    out = jnp.zeros_like(xx, dtype=bool)
    arrived, kept = [], []
    for code, beyond, mirror, park, e, r in (
            (bc[0], xx < -L / 2, -L - xx, -L / 2 - PARK * dx, restitution[0], reflection[0]),
            (bc[1], xx > L / 2, L - xx, L / 2 + PARK * dx, restitution[1], reflection[1])):
        arrived.append(jnp.where(beyond, w, 0.0))
        kept.append(jnp.zeros_like(w))
        if code == 0:
            xx = jnp.where(beyond, (xx + L / 2) % L - L / 2, xx)
            continue
        if code == 2:
            returned = jnp.where(beyond, w * r, 0.0)
            spent = beyond & (returned <= floor)          # too little left to follow: the wall takes it
            returned = jnp.where(spent, 0.0, returned)
            kept[-1] = arrived[-1] - returned
            w = jnp.where(beyond, returned, w)
            lost = beyond & (w <= 0)
            xx, out, beyond = jnp.where(lost, park, xx), out | lost, beyond & ~lost
        xx = jnp.where(beyond, mirror, xx)
        vx = jnp.where(beyond, -e * vx, vx)
    x = x.at[:, 0].set(xx)
    v = v.at[:, 0].set(vx)
    if 2 in bc:
        v = jnp.where(out[:, None], 0.0, v)
        qm = jnp.where(out, 0.0, qm)
    return x, v, w, qm, (jnp.stack(arrived), jnp.stack(kept))


def wrap_positions(x, w, box, bc, dx):
    """The position map of :func:`apply_particle_bc` alone, for the reconstructed
    integer-time positions: wrapped at a periodic wall, mirrored at a reflective one, and
    at an absorbing one mirrored if the particle still has weight and parked if it has none.

    It repeats the position rules rather than calling :func:`apply_particle_bc` with dummy
    velocities, because XLA does not remove that dummy work: the full map cost 9-15 ns per
    particle and this one 4-5 (N = 1e5, CPU). A test holds the two to identical positions
    for every pair of walls."""
    L, Ly, Lz = box
    periods = jnp.array([Ly, Lz])
    yz = (x[:, 1:] + periods / 2) % periods - periods / 2
    xx = x[:, 0]
    for code, beyond, mirror, park in ((bc[0], xx < -L / 2, -L - xx, -L / 2 - PARK * dx),
                                       (bc[1], xx > L / 2, L - xx, L / 2 + PARK * dx)):
        if code == 0:
            xx = jnp.where(beyond, (xx + L / 2) % L - L / 2, xx)
            continue
        if code == 2:
            xx = jnp.where(beyond & (w <= 0), park, xx)
            beyond = beyond & (w > 0)
        xx = jnp.where(beyond, mirror, xx)
    return jnp.concatenate([xx[:, None], yz], axis=1)


# --- digital filter --------------------------------------------------------------------

def _shift(f, s, bc):
    """``f`` shifted by ``s`` cells with the wall treatment of the sources.

    A reflective wall mirrors the stencil back into the box, which is what keeps
    the filter conservative: what a cell would have sent through the wall stays
    on this side, so smoothing redistributes the source without creating or
    destroying any of it. An absorbing wall drops it, which is what letting it
    leave means. Clamping instead of mirroring is a zero-gradient extrapolation;
    that is right for a field but it invents source at a reflective wall, by
    several per cent of the total for a stride-two stencil.
    """
    if bc[0] == 0:
        return jnp.roll(f, s, axis=0)
    n = f.shape[0]
    idx = jnp.arange(n) - s
    outside_left, outside_right = idx < 0, idx >= n
    if bc[0] == 1:
        idx = jnp.where(outside_left, -idx - 1, idx)
    if bc[1] == 1:
        idx = jnp.where(outside_right, 2 * n - idx - 1, idx)
    g = f[jnp.clip(idx, 0, n - 1)]
    expand = (slice(None),) + (None,) * (f.ndim - 1)
    if bc[0] in (2, 4):
        g = jnp.where(outside_left[expand], 0.0, g)
    if bc[1] in (2, 4):
        g = jnp.where(outside_right[expand], 0.0, g)
    return g


def smooth(f, passes, alpha, strides, bc):
    """Compensated binomial filter along the grid axis (Birdsall and Langdon,
    appendix C). For each stride: ``passes`` three-point passes with weight
    ``alpha``, then one compensation pass with weight :math:`1 + p(1-\\alpha)`
    that cancels the long-wavelength attenuation."""
    if passes == 0:
        return f

    def one(f, s, a):
        return a * f + 0.5 * (1 - a) * (_shift(f, s, bc) + _shift(f, -s, bc))

    for s in strides:
        for _ in range(passes):
            f = one(f, s, alpha)
        f = one(f, s, 1 + passes * (1 - alpha))
    return f
