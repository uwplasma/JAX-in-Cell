from jax import vmap, jit
import jax.numpy as jnp
from ._boundary_conditions import field_2_ghost_cells
from ._constants import speed_of_light as c
from ._metric_tensor import metric_bundle

__all__ = ['fields_to_particles_grid', 'rotation', 'boris_step', 'boris_step_relativistic',
           'u0_from_v_metric', 'gravity_kick_half',
           'gamma_from_v', 'p_from_v', 'gamma_from_p', 'v_from_p', 'relativistic_rotation']

@jit
def u0_from_v_metric(v, g):
    """
    For static diagonal metric with g0i=0:
      normalization: g00 (u0)^2 + gij u^i u^j = -c^2
      with u^i = v^i * u0  (since v^i = dx^i/dt = u^i/u^0).
      => (u0)^2 ( g00 + gij v^i v^j ) = -c^2
    """
    g00 = g[0,0]
    gij = g[1:,1:]                   # 3x3
    vv = jnp.dot(v, jnp.dot(gij, v)) # scalar
    denom = g00 + vv                 # <0 for physically sensible metrics used here
    u0 = c / jnp.sqrt(-denom)
    return u0

@jit
def gravity_kick_half(dt_half, t, x_vec, p, m, metric_cfg):
    """
    Half-step geodesic update on the *contravariant spatial* momentum p^i
    (we store and evolve 3-vector p consistent with p^i = m u^i = m v^i u^0).

    Inputs:
      x_vec: (3,) Cartesian position; we only use x_vec[0] as your code is 1D in space.
      metric_cfg: dict with keys { 'kind', 'params' } from parameters.
    """
    x = x_vec[0]
    mb = metric_bundle(t, x, metric_cfg["kind"], c, **metric_cfg.get("params", {}))
    g  = mb["g"]; Gamma = mb["Gamma"]

    # Build 4-vectors using current p (contravariant spatial) and metric normalization
    v  = p / (jnp.sqrt(m*m*c*c + jnp.dot(p,p)) / m)  # flat-space estimate; next line refines via metric
    u0 = u0_from_v_metric(v, g)                      # metric-correct u^0
    ui = v * u0                                      # u^i

    # Assemble u^α and p^β with indices: (0..3)
    u4 = jnp.concatenate([jnp.array([u0]), ui])  # u^0, u^i
    p4 = jnp.concatenate((jnp.atleast_1d(m*u0*c), p))  # p^0 = m u^0 c; spatial p^i as stored

    # dp^i/dt = - Γ^i_{αβ} u^α p^β, integrate for dt_half
    def dp_i(i):
        # G = Γ^i_{αβ}
        G = Gamma[i+1, :, :]   # shape (4,4)
        inner = jnp.tensordot(G, p4, axes=([1],[0]))    # Γ^i_{αβ} p^β
        return - jnp.tensordot(u4, inner, axes=([0],[0]))  # u^α Γ^i_{αβ} p^β
    dp = jnp.array([dp_i(0), dp_i(1), dp_i(2)])
    return p + dt_half * dp

@jit
def fields_to_particles_grid(x_n, field, dx, grid, grid_start, field_BC_left, field_BC_right):
    """
    This function retrieves the electric or magnetic field values at particle positions 
    using a field interpolation scheme. The function first adds ghost cells to the field 
    array to handle boundary conditions, then interpolates the field based on the 
    particle's position in the grid.

    Args:
        x_n (array): The position of particles at time step n, shape (N,).
        field (array): The field values at each grid point, shape (G,).
        dx (float): The spatial grid spacing.
        grid (array): The grid positions where the field is defined, shape (G,).
        grid_start (float): The starting position of the grid (usually the left boundary).
        field_BC_left  (int): Boundary condition for the left side of the particle grid.
        field_BC_right (int): Boundary condition for the right side of the particle grid.

    Returns:
        array: The interpolated field values at the particle positions, shape (N,).
    """
    # Add ghost cells for the boundaries using provided boundary conditions
    ghost_L2, ghost_L1, ghost_R = field_2_ghost_cells(field_BC_left, field_BC_right, field)
    # single left pad (L2, L1), single right pad (R)
    left_pad  = jnp.stack([ghost_L2, ghost_L1], axis=0)       # (2, 3)
    right_pad = ghost_R[None, ...]                             # (1, 3)
    field = jnp.concatenate([left_pad, field, right_pad], axis=0)

    x = x_n[0]
    
    # Adjust the grid to accommodate particles at the first half grid cell (staggered grid)
    #If using a staggered grid, particles at first half cell will be out of grid, so add extra cell
    grid_left = (grid[0] - dx)[None]                           # (1,)
    grid = jnp.concatenate([grid_left, grid], axis=0)

    # because we added one cell on the left, effective start shifts by -dx
    i = ((x - (grid_start - dx) + dx) // dx).astype(int)
    
    # Interpolate the field at the particle position using a quadratic interpolation
    fields_n = (
        0.5 * field[i]   * (0.5 + (grid[i]   - x) / dx) ** 2 +
              field[i+1] * (0.75 - ((grid[i] - x) / dx) ** 2) +
        0.5 * field[i+2] * (0.5 - (grid[i]   - x) / dx) ** 2
    )
    return fields_n

@jit
def rotation(dt, B, vsub, q_m):
    """
    This function implements the Boris algorithm to rotate the particle velocity vector 
    in the magnetic field for one time step. This step is part of the numerical solution 
    of the Lorentz force equation.

    Args:
        dt (float): Time step for the simulation.
        B (array): Magnetic field at the particle's position, shape (3,).
        vsub (array): The particle's velocity before the rotation, shape (3,).
        q_m (array): The charge-to-mass ratio of the particle, shape (3,).

    Returns:
        array: The updated velocity after the rotation, shape (3,).
    """
    # First part of the Boris algorithm: calculate intermediate velocity
    Rvec = vsub + 0.5 * dt * q_m * jnp.cross(vsub, B)
    
    # Magnetic field vector term for the rotation step
    Bvec = 0.5 * q_m * dt * B
    
    # Apply the Boris rotation step to the velocity vector
    vplus = (jnp.cross(Rvec, Bvec) + jnp.dot(Rvec, Bvec) * Bvec + Rvec) / (1 + jnp.dot(Bvec, Bvec))
    
    return vplus

@jit
def boris_step(dt, xs_nplushalf, ps_n, q_ms, ms, E_fields_at_x, B_fields_at_x):
    """
    This function performs one step of the Boris algorithm for particle motion. 
    The particle velocity is updated using the electric and magnetic fields at its position, 
    and the particle position is updated using the new velocity.

    Args:
        dt (float): Time step for the simulation.
        xs_nplushalf (array): The particle positions at the half-time step n+1/2, shape (N, 3).
        ps_n (array): The particle momenta at time step n, shape (N, 3).
        q_ms (array): The charge-to-mass ratio of each particle, shape (N, 1).
        ms (array): The mass of each particle, shape (N, 1).
        E_fields_at_x (array): The interpolated electric field values at the particle positions, shape (N, 3).
        B_fields_at_x (array): The magnetic field values at the particle positions, shape (N, 3).

    Returns:
        tuple: A tuple containing:
            - xs_nplus3_2 (array): The updated particle positions at time step n+3/2, shape (N, 3).
            - ps_nplus1 (array): The updated particle momenta at time step n+1, shape (N, 3).
    """
    # Convert momentum to velocity
    vs_n = ps_n / ms
    
    # First half step update for velocity due to electric field
    vs_n_int = vs_n + (q_ms) * E_fields_at_x * dt / 2
    
    # Apply the Boris rotation step for the magnetic field
    vs_n_rot = vmap(lambda B_n, v_n, q_m: rotation(dt, B_n, v_n, q_m))(B_fields_at_x, vs_n_int, q_ms[:, 0])
    
    # Second half step update for velocity due to electric field
    vs_nplus1 = vs_n_rot + (q_ms) * E_fields_at_x * dt / 2
    
    # Update the particle positions using the new velocities
    xs_nplus3_2 = xs_nplushalf + dt * vs_nplus1

    # Convert velocity to momentum
    ps_nplus1 = vs_nplus1 * ms
    
    return xs_nplus3_2, ps_nplus1
    # vs_nplus1 = vs_n + (q_ms) * E_fields_at_x * dt
    # xs_nplus1 = xs_nplushalf + dt * vs_nplus1
    # return xs_nplus1, vs_nplus1

@jit
def gamma_from_v(v):
    beta2 = jnp.sum((v / c) ** 2, axis=-1)              # (...)
    return 1.0 / jnp.sqrt(1.0 - beta2)                  # (...)

@jit
def p_from_v(v, m):
    g = gamma_from_v(v)[..., None]                      # (...,1)
    return g * m * v                               # (...,3)

@jit
def gamma_from_p(p, m):
    u2 = jnp.sum(p * p, axis=-1, keepdims=True) / (m * m * c * c)  # (...,1)
    return jnp.sqrt(1.0 + u2)[..., 0]                   # (...)

@jit
def v_from_p(p, m):
    g = gamma_from_p(p, m)[..., None]                   # (...,1)
    return p / (g * m)                                  # (...,3)

@jit
def v_from_p_metric(p, m, g):
    """
    Coordinate velocity v^i from spatial contravariant momentum p^i = m u^i,
    using metric g_{μν} (diagonal, zero shift).
    """
    gij = g[1:, 1:]
    g00 = g[0, 0]
    # u^i = p^i / m
    ui = p / m
    # (u0)^2 = (-c^2 - g_ij u^i u^j) / g00   (note g00 < 0)
    ui_g_ui = jnp.dot(ui, jnp.dot(gij, ui))
    u0 = jnp.sqrt((-c*c - ui_g_ui) / g00)
    return ui / u0

@jit
def p_from_v_metric(v, m, g):
    """
    Spatial momentum p^i = m u^i from coordinate velocity v^i using metric g_{μν}.
    """
    u0 = u0_from_v_metric(v, g)
    ui = v * u0
    return m * ui

def v_from_p_at_positions(p_arr, m_arr, x_arr, t, metric_cfg):
    def one(p, m, x):
        mb = metric_bundle(t, x[0], metric_cfg["kind"], c, **metric_cfg.get("params", {}))
        g  = mb["g"]
        return v_from_p_metric(p, m, g)
    return vmap(one)(p_arr, m_arr, x_arr)

@jit
def relativistic_rotation(dt, B, p_minus, q, m):
    """
    Rotate momentum vector in magnetic field (relativistic Boris step).
    """
    # gamma_minus from p_minus
    gamma_minus = gamma_from_p(p_minus, m)  # (...)

    # t vector (rotation vector)
    t = (q * dt) / (2 * m * gamma_minus) * B
    p_dot_t = jnp.dot(p_minus, t)
    p_cross_t = jnp.cross(p_minus, t)
    t_squared = jnp.dot(t, t)

    p_plus = (p_minus*(1-t_squared) + 2*(p_dot_t * t + p_cross_t)) / (1 + t_squared)

    return p_plus

@jit
def boris_step_relativistic(dt, xs_nplushalf, ps_n, q_s, m_s, E_fields_at_x,
                            B_fields_at_x, metric_cfg=None, t_cur=None):
    """
    Relativistic Boris pusher for N particles.
    Momentum-based update, with momentum p = gamma*m*v.

    Args:
        dt: Time step
        xs_nplushalf: Particle positions at t = n + 1/2, shape (N, 3)
        ps_n: Momentum at time t = n, shape (N, 3)
        q_s: Charges, shape (N,)
        m_s: Masses, shape (N,)
        E_fields_at_x: Electric fields at particle positions, shape (N, 3)
        B_fields_at_x: Magnetic fields at particle positions, shape (N, 3)
        c: Speed of light (default = 1.0 for normalized units)
    Returns:
        xs_nplus3_2: Updated positions at t = n + 3/2, shape (N, 3)
        ps_nplus1: Updated momentum at t = n + 1, shape (N, 3)
    """

    def single_particle_step(x, p_n, q, m, E, B):
        if (metric_cfg is not None) and (t_cur is not None):
            # first half gravity at midpoint time t+dt/2 and *current* position
            p_half = gravity_kick_half(0.5*dt, t_cur + 0.5*dt, x, p_n, m, metric_cfg)
        else:
            p_half = p_n
        # Build local scalings for diagonal gamma_ii
        x0 = x[0]
        mb = metric_bundle(t_cur, x0, metric_cfg["kind"], c, **metric_cfg.get("params", {}))
        g  = mb["g"]

        v_n = v_from_p_metric(p_half, m, g)

        # build tetrad scalings
        alpha = jnp.sqrt(-g[0,0]) / c
        S = jnp.array([jnp.sqrt(g[1,1]), jnp.sqrt(g[2,2]), jnp.sqrt(g[3,3])])

        # map EM fields and momentum to local orthonormal frame
        E_loc = S * E
        B_loc = S * B
        p_half_loc = S * p_half

        # local timestep
        dt_loc = alpha * dt

        # local velocity from local momentum (flat relation in orthonormal frame)
        v_loc = v_from_p(p_half_loc, m)

        # --- Vay update in local frame (Phys. Plasmas 15, 056701) ---
        p_star = p_half_loc + q * dt_loc * (E_loc + 0.5 * jnp.cross(v_loc, B_loc))
        tau    = (q * dt_loc / (2 * m)) * B_loc
        gamma_prime = jnp.sqrt(1.0 + jnp.sum(p_star * p_star) / (m*m*c*c))
        sigma  = 0.5 * (gamma_prime**2 - jnp.sum(tau * tau))
        w      = jnp.dot(p_star, tau)
        gamma_plus = jnp.sqrt(sigma + jnp.sqrt(sigma**2 + jnp.sum(tau * tau) + w*w))
        tvec   = tau / gamma_plus
        p_loc_plus = (p_star + jnp.dot(p_star, tvec) * tvec + jnp.cross(p_star, tvec)) / (1.0 + jnp.dot(tvec, tvec))

        # map updated momentum back to coordinates
        p_nplus1 = p_loc_plus / S

        # # Standard relativistic Boris integrator, Birdsall & Langdon 2004
        # # Half electric field acceleration
        # p_minus = p_half + q * E * dt / 2
        # # Magnetic rotation
        # p_plus = relativistic_rotation(dt, B, p_minus, q, m)
        # # Second half electric field acceleration
        # p_nplus1 = p_plus + q * E * dt / 2

        # coordinate velocity for advection (metric-aware)
        v_nplus1 = v_from_p_metric(p_nplus1, m, g)

        # Update position using new velocity
        x_nplus3_2 = x + dt * v_nplus1

        if (metric_cfg is not None) and (t_cur is not None):
            # second half gravity at midpoint time around end state: also t+dt/2,
            # but evaluated at x_np3_2 for symmetry
            p_nplus1 = gravity_kick_half(0.5*dt, t_cur + 0.5*dt, x_nplus3_2, p_nplus1, m, metric_cfg)
        else:
            p_nplus1 = p_nplus1

        return x_nplus3_2, p_nplus1

    # Vectorize over particles
    xs_nplus3_2, ps_nplus1 = vmap(single_particle_step)(
        xs_nplushalf, ps_n, q_s, m_s, E_fields_at_x, B_fields_at_x
    )

    return xs_nplus3_2, ps_nplus1