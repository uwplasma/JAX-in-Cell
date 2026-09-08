"""Numerical kernels of the particle-in-cell cycle.

Staggering (Yee, one dimension): the charge density and the magnetic field live
at cell centres :math:`x_i`, the electric field and the current density at cell
faces :math:`x_{i+1/2}`. Every array has one entry per cell; entry ``i`` of a
face quantity refers to :math:`x_{i+1/2}`.

Boundary codes: 0 periodic, 1 reflective, 2 absorbing, given per wall as a
``(left, right)`` pair. They are static, so the branches below are resolved
when the program is traced and cost nothing at run time.
"""
import jax.numpy as jnp

from ._constants import epsilon_0, speed_of_light as c

__all__ = ["s2_weights", "map_indices", "deposit", "gather", "current_from_continuity",
           "to_faces", "curl_E", "curl_B", "half_step_fields", "E_x_from_rho", "boris",
           "boris_relativistic", "apply_particle_bc", "wrap_positions", "smooth"]


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
    if bc[0] == 2:
        keep = keep & (idx >= 0)
    if bc[1] == 2:
        keep = keep & (idx < n)
    return jnp.clip(idx, 0, n - 1), keep


def deposit(x, q, x0, dx, n, bc):
    """Density on the grid from particle positions ``x`` and amounts ``q``
    (charge, or charge times a velocity component): :math:`\\sum_p q_p S_2(x_i - x_p)`."""
    idx, w = s2_weights(x, x0, dx)
    idx, keep = map_indices(idx, n, bc)
    return jnp.zeros(n).at[idx].add(jnp.where(keep, w, 0.0) * (q / dx)[:, None])


def gather(field, x, x0, dx, bc):
    """Interpolate a grid field of shape ``(n, C)`` to positions ``x`` with the
    same spline, giving ``(N, C)``."""
    idx, w = s2_weights(x, x0, dx)
    idx, keep = map_indices(idx, field.shape[0], bc)
    return jnp.einsum("nk,nkc->nc", jnp.where(keep, w, 0.0), field[idx])


# --- sources -----------------------------------------------------------------------

def current_from_continuity(rho_old, rho_new, dt, dx, mean_current, bc):
    """Longitudinal current at the faces that satisfies the discrete continuity
    equation :math:`(\\rho_i^{new} - \\rho_i^{old})/\\Delta t + (J_{i+1/2} - J_{i-1/2})/\\Delta x = 0`
    exactly. In a periodic box the integration constant is fixed by the mean
    current of the particles; at a wall the current through the wall is zero."""
    J = -dx * jnp.cumsum((rho_new - rho_old) / dt)
    if bc[0] == 0:
        J = J - jnp.mean(J) + mean_current
    return J


def to_faces(f, bc):
    """Average a cell-centred quantity to the faces, :math:`f_{i+1/2} = (f_i + f_{i+1})/2`."""
    if bc[1] == 0:
        right = f[0]
    elif bc[1] == 1:
        right = f[-1]
    else:
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


def E_x_from_rho(rho, dx, bc):
    """Solve the discrete Gauss law :math:`(E_{i+1/2} - E_{i-1/2})/\\Delta x = \\rho_i/\\epsilon_0`
    for the longitudinal field at the faces. Periodic: by FFT with the
    finite-difference symbol :math:`(1 - e^{-ik\\Delta x})/\\Delta x` and zero mean.
    Otherwise: integrated from the left wall, where :math:`E_{-1/2} = 0`."""
    if bc[0] == 0:
        n = rho.shape[0]
        k = 2 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
        symbol = (1 - jnp.exp(-1j * k * dx)) / dx
        symbol = symbol.at[0].set(1.0)
        E_hat = jnp.fft.fft(rho) / epsilon_0 / symbol
        return jnp.fft.ifft(E_hat.at[0].set(0.0)).real
    return dx / epsilon_0 * jnp.cumsum(rho)


# --- particles --------------------------------------------------------------------------

def boris(v, E, B, qm, dt):
    """One Boris step for the velocity: half electric kick, exact rotation about
    B by :math:`2\\arctan(|q B \\Delta t / 2m|)`, half electric kick."""
    v = v + qm * E * (dt / 2)
    b = qm * B * (dt / 2)
    v_prime = v + jnp.cross(v, b)
    v = v + 2 * jnp.cross(v_prime, b) / (1 + jnp.sum(b * b, axis=1, keepdims=True))
    return v + qm * E * (dt / 2)


def boris_relativistic(v, E, B, q, m, dt):
    """The same three sub-steps applied to the momentum :math:`p = \\gamma m v`."""
    gamma = 1 / jnp.sqrt(jnp.maximum(1 - jnp.sum(v * v, axis=1, keepdims=True) / c ** 2, 1e-15))
    p = gamma * m * v + q * E * (dt / 2)
    gamma = jnp.sqrt(1 + jnp.sum(p * p, axis=1, keepdims=True) / (m * c) ** 2)
    t = q * B * (dt / 2) / (m * gamma)
    p_prime = p + jnp.cross(p, t)
    p = p + 2 * jnp.cross(p_prime, t) / (1 + jnp.sum(t * t, axis=1, keepdims=True))
    p = p + q * E * (dt / 2)
    gamma = jnp.sqrt(1 + jnp.sum(p * p, axis=1, keepdims=True) / (m * c) ** 2)
    return p / (gamma * m)


def apply_particle_bc(x, v, q, qm, box, bc, restitution, dx):
    """Bring particles that left the box back according to the wall codes.
    Reflective walls mirror the position and multiply the normal velocity by
    ``-restitution``; absorbing walls zero the charge, the charge-to-mass ratio
    and the velocity and park the particle outside the grid. The ignorable
    coordinates are always periodic."""
    L, Ly, Lz = box
    x = x.at[:, 1].set((x[:, 1] + Ly / 2) % Ly - Ly / 2)
    x = x.at[:, 2].set((x[:, 2] + Lz / 2) % Lz - Lz / 2)
    xx, vx = x[:, 0], v[:, 0]
    out = jnp.zeros_like(xx, dtype=bool)
    for code, beyond, mirror, park in ((bc[0], xx < -L / 2, -L - xx, -L / 2 - 1.5 * dx),
                                       (bc[1], xx > L / 2, L - xx, L / 2 + 3.0 * dx)):
        if code == 0:
            xx = jnp.where(beyond, (xx + L / 2) % L - L / 2, xx)
        elif code == 1:
            xx = jnp.where(beyond, mirror, xx)
            vx = jnp.where(beyond, -restitution * vx, vx)
        else:
            xx = jnp.where(beyond, park, xx)
            out = out | beyond
    x = x.at[:, 0].set(xx)
    v = v.at[:, 0].set(vx)
    if 2 in bc:
        v = jnp.where(out[:, None], 0.0, v)
        q = jnp.where(out, 0.0, q)
        qm = jnp.where(out, 0.0, qm)
    return x, v, q, qm


def wrap_positions(x, box, bc, dx):
    """The position map of :func:`apply_particle_bc` alone, for half-step positions."""
    x, _, _, _ = apply_particle_bc(x, jnp.zeros_like(x), jnp.zeros(x.shape[0]),
                                   jnp.zeros(x.shape[0]), box, bc, 1.0, dx)
    return x


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
    if bc[0] == 2:
        g = jnp.where(outside_left[expand], 0.0, g)
    if bc[1] == 2:
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
