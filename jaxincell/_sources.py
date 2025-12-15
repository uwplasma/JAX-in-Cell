import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
from ._filters import filter_scalar_field, filter_vector_field

__all__ = ['get_S2_weights_and_indices_periodic_CN', 'calculate_charge_density', 'current_density', 'current_density_periodic_CN']

@jit
def Jx_from_continuity_periodic_fft(rho_prev, rho_next, dx, dt):
    """
    Periodic solve for Jx enforcing discrete continuity with backward-diff div:
        (Jx - roll(Jx,1))/dx = -(rho_next - rho_prev)/dt

    Gauge: set k=0 mode of Jx to 0 (mean current is undetermined by continuity).
    """
    n = rho_prev.shape[0]
    drho_dt = (rho_next - rho_prev) / dt

    # Compatibility for periodic solve: mean(drho_dt) must be ~0; project it out robustly.
    drho_dt = drho_dt - jnp.mean(drho_dt)

    drho_k = jnp.fft.fft(drho_dt)
    kx = jnp.fft.fftfreq(n, d=dx) * 2.0 * jnp.pi

    # Backward-difference symbol Db(k) = (1 - e^{-ik dx})/dx
    Db = (1.0 - jnp.exp(-1j * kx * dx)) / dx
    Db = Db.at[0].set(1.0 + 0.0j)      # avoid divide-by-zero at k=0

    Jx_k = -drho_k / Db
    Jx_k = Jx_k.at[0].set(0.0 + 0.0j)  # choose zero-mean Jx gauge

    Jx = jnp.fft.ifft(Jx_k).real
    return Jx

@jit
def get_S2_weights_and_indices_periodic_CN(x, dx, grid_start, grid_size):
    """
    Calculates weights and indices for Quadratic Spline (S2).
    Applies Periodic Wrapping to indices immediately.
    """
    # 1. Normalize position
    x_norm = (x - grid_start) / dx
    
    # 2. Nearest node index k
    k = jnp.floor(x_norm + 0.5).astype(jnp.int32)
    
    # 3. Raw Indices [k-1, k, k+1]
    # We apply modulo (%) here to handle wrapping automatically.
    # e.g., if k=0, k-1=-1, which wraps to grid_size-1.
    indices = jnp.array([k - 1, k, k + 1]) % grid_size
    
    # 4. Distance d from node k
    d = x_norm - k  # d in [-0.5, 0.5]
    
    # 5. Continuous S2 Weights
    # Center (at k):
    w_cen = 0.75 - d**2
    
    # Left (at k-1): Distance is (1 + d)
    w_left = 0.5 * (0.5 - d)**2
    
    # Right (at k+1): Distance is (1 - d)
    w_right = 0.5 * (0.5 + d)**2
    
    weights = jnp.array([w_left, w_cen, w_right])
    
    return indices, weights

@jit
def get_S2_weights_and_indices(x, dx, grid_start, grid_size, bc_left, bc_right):
    """
    Quadratic spline (S2) indices+weights for a 1D grid with BC support.

    Returns:
      indices: (3,) int32
      weights: (3,) float
      active:  (3,) bool mask (False => this stencil point is inactive)

    bc convention:
      0 periodic: wrap indices
      1 reflective: clamp indices (Neumann-like extension)
      2 absorbing: outside domain -> inactive (for gather); deposition should renormalize
    """
    x_norm = (x - grid_start) / dx
    k = jnp.floor(x_norm + 0.5).astype(jnp.int32)
    d = x_norm - k

    w_c = 0.75 - d**2
    w_l = 0.5 * (0.5 - d)**2
    w_r = 0.5 * (0.5 + d)**2
    weights = jnp.array([w_l, w_c, w_r])

    raw = jnp.array([k - 1, k, k + 1], dtype=jnp.int32)

    periodic = (bc_left == 0) & (bc_right == 0)

    def _periodic(_):
        idx = raw % grid_size
        active = jnp.ones((3,), dtype=bool)
        return idx, weights, active

    def _nonperiodic(_):
        idx = jnp.clip(raw, 0, grid_size - 1)

        # per-stencil-point out-of-range flags
        out_left  = raw < 0
        out_right = raw >= grid_size

        # reflective: keep active, clamp does the reflection-like behavior
        # absorbing: drop only the out-of-range stencil points
        active = jnp.ones((3,), dtype=bool)
        active = jnp.where((bc_left == 2),  active & (~out_left),  active)
        active = jnp.where((bc_right == 2), active & (~out_right), active)

        return idx, weights, active

    return lax.cond(periodic, _periodic, _nonperiodic, operand=None)

@partial(jit, static_argnames=('grid_size',))
def deposit_S2_scalar(xs, qs, dx, grid_start, grid_size, bc_left, bc_right):
    """
    Fully vectorized S2 scalar deposition:
      rho[idx] += (q/dx) * w
    xs: (N,1), qs: (N,1)
    """
    x = xs[:, 0]
    q = qs[:, 0]

    idx, w, active = vmap(
        get_S2_weights_and_indices,
        in_axes=(0, None, None, None, None, None)
    )(x, dx, grid_start, grid_size, bc_left, bc_right)    # idx,w,active: (N,3)

    # apply active mask
    w = jnp.where(active, w, 0.0)

    # optional per-particle renorm for absorbing
    renorm = (bc_left == 2) | (bc_right == 2)

    def _renorm_all(w_in):
        s = jnp.sum(w_in, axis=1, keepdims=True)          # (N,1)
        return jnp.where(s > 0, w_in / s, w_in)

    w = lax.cond(renorm, _renorm_all, lambda w_in: w_in, w)

    contrib = (q / dx)[:, None] * w                       # (N,3)

    idx_f = idx.reshape(-1)                               # (N*3,)
    c_f   = contrib.reshape(-1)                           # (N*3,)

    rho = jnp.zeros((grid_size,), dtype=contrib.dtype).at[idx_f].add(c_f)
    return rho


@partial(jit, static_argnames=('grid_size',))
def deposit_S2_vector(xs, qs, vs, dx, grid_start, grid_size, bc_left, bc_right):
    """
    Fully vectorized S2 vector deposition:
      J[idx] += (q/dx) * w * v
    xs: (N,1), qs: (N,1), vs: (N,3)
    """
    x = xs[:, 0]
    q = qs[:, 0]

    idx, w, active = vmap(
        get_S2_weights_and_indices,
        in_axes=(0, None, None, None, None, None)
    )(x, dx, grid_start, grid_size, bc_left, bc_right)    # (N,3)

    w = jnp.where(active, w, 0.0)

    renorm = (bc_left == 2) | (bc_right == 2)

    def _renorm_all(w_in):
        s = jnp.sum(w_in, axis=1, keepdims=True)
        return jnp.where(s > 0, w_in / s, w_in)

    w = lax.cond(renorm, _renorm_all, lambda w_in: w_in, w)

    factor  = (q / dx)[:, None] * w                       # (N,3)
    contrib = factor[:, :, None] * vs[:, None, :]         # (N,3,3)

    idx_f = idx.reshape(-1)                               # (N*3,)
    c_f   = contrib.reshape(-1, 3)                        # (N*3,3)

    J = jnp.zeros((grid_size, 3), dtype=contrib.dtype).at[idx_f].add(c_f)
    return J


@jit
def calculate_charge_density(xs_n, qs, dx, grid, particle_BC_left, particle_BC_right,
                             filter_passes=5, filter_alpha=0.5, filter_strides=(1, 2, 4),
                             field_BC_left=0, field_BC_right=0):
    """
    Computes the total charge density on the grid by summing contributions from all particles.

    Args:
        xs_n (array): Particle positions at the current timestep, shape (N, 1).
        qs (array): Particle charges, shape (N, 1).
        dx (float): The grid spacing.
        grid (array): The grid points.
        particle_BC_left (int): Left particle boundary condition type (0: periodic, 1: reflective, 2: absorbing).
        particle_BC_right (int): Right particle boundary condition type (0: periodic, 1: reflective, 2: absorbing).
        filter_passes (int): Number of digital filter passes to apply (default: 5). Internally capped at 17.
        filter_alpha (float): Filter strength parameter (default: 0.5). Controls the weight of the center point in the 3-point filter.
        filter_strides (tuple): Tuple of stride values for multi-scale filtering (default: (1, 2, 4)).
        field_BC_left (int): Left boundary condition for filtering (default: 0: periodic, 1: reflective, 2: absorbing).
        field_BC_right (int): Right boundary condition for filtering (default: 0: periodic, 1: reflective, 2: absorbing).

    Returns:
        array: Total charge density on the grid.
    """
    grid_size = grid.shape[0]
    grid_start = grid[0]
    
    # Sum the contributions across all particles
    total_chargedens = deposit_S2_scalar(
        xs_n, qs, dx, grid_start, grid_size,
        particle_BC_left, particle_BC_right
    )

    # Apply digital filtering to the total charge density
    total_chargedens = filter_scalar_field(
        total_chargedens,
        passes=filter_passes,
        alpha=filter_alpha,
        strides=filter_strides,
        bc_left=field_BC_left,
        bc_right=field_BC_right,
    )
    return total_chargedens

@jit
def current_density(xs_nminushalf, xs_n, xs_nplushalf,
                    vs_n, qs, dx, dt, grid, grid_start,
                    particle_BC_left, particle_BC_right,
                    filter_passes=5, filter_alpha=0.5, filter_strides=(1, 2, 4),
                    field_BC_left=0, field_BC_right=0):
    """
    Computes the current density `j` on the grid from particle motion.

    Args:
        xs_nminushalf (array): Particle positions at the half timestep before the current one, shape (N, 1).
        xs_n (array): Particle positions at the current timestep, shape (N, 1).
        xs_nplushalf (array): Particle positions at the half timestep after the current one, shape (N, 1).
        vs_n (array): Particle velocities at the current timestep, shape (N, 3).
        qs (array): Particle charges, shape (N, 1).
        dx (float): The grid spacing.
        dt (float): The time step size.
        grid (array): The grid points.
        grid_start (float): The starting position of the grid.
        particle_BC_left (int): Left particle boundary condition type (0: periodic, 1: reflective, 2: absorbing).
        particle_BC_right (int): Right particle boundary condition type (0: periodic, 1: reflective, 2: absorbing).
        filter_passes (int): Number of digital filter passes to apply (default: 5). Internally capped at 17.
        filter_alpha (float): Filter strength parameter (default: 0.5). Controls the weight of the center point in the 3-point filter.
        filter_strides (tuple): Tuple of stride values for multi-scale filtering (default: (1, 2, 4)).
        field_BC_left (int): Left boundary condition for filtering (default: 0: periodic, 1: reflective, 2: absorbing).
        field_BC_right (int): Right boundary condition for filtering (default: 0: periodic, 1: reflective, 2: absorbing).
    Returns:
        array: Current density on the grid, shape (G, 3), where G is the number of grid points.
    """
    grid_size = grid.shape[0]
    rho_grid_start = grid[0]      # centers (where rho lives)
    J_grid_start   = grid_start   # edges (where E and J live)

    # rho lives on cell centers (same indexing as `grid`);
    # Ex/Jx live on right-edges (grid_start passed in as E_grid_start).
    rho_prev = deposit_S2_scalar(
        xs_nminushalf, qs, dx, rho_grid_start, grid_size,
        particle_BC_left, particle_BC_right
    )
    rho_next = deposit_S2_scalar(
        xs_nplushalf, qs, dx, rho_grid_start, grid_size,
        particle_BC_left, particle_BC_right
    )

    periodic = (particle_BC_left == 0) & (particle_BC_right == 0)

    # 2) Transverse + a "reference" Jx from ordinary deposition (for DC/current mean)
    J_xyz = deposit_S2_vector(
        xs_n, qs, vs_n, dx, J_grid_start, grid_size,
        particle_BC_left, particle_BC_right
    )

    # 3) Optional continuity-preserving smoothing (see next section):
    #    Smooth rho_prev & rho_next with the SAME linear filter, then compute Jx from them.
    def _maybe_filter_rhos(r0, r1):
        r0f = filter_scalar_field(r0, passes=filter_passes, alpha=filter_alpha, strides=filter_strides,
                                 bc_left=field_BC_left, bc_right=field_BC_right)
        r1f = filter_scalar_field(r1, passes=filter_passes, alpha=filter_alpha, strides=filter_strides,
                                 bc_left=field_BC_left, bc_right=field_BC_right)
        return r0f, r1f

    rho_prev, rho_next = lax.cond(
        filter_passes > 0,
        lambda _: _maybe_filter_rhos(rho_prev, rho_next),
        lambda _: (rho_prev, rho_next),
        operand=None
    )

    # 4) Enforce continuity for Jx
    def _Jx_periodic(_):
        Jx_fluct = Jx_from_continuity_periodic_fft(rho_prev, rho_next, dx, dt)
        Jx_mean = jnp.mean(J_xyz[:, 0])  # preserve DC current from standard deposition
        return Jx_fluct + Jx_mean

    def _Jx_nonperiodic(_):
        # Correct nonperiodic backward-div integration:
        # If ghost-left current is 0, then J[0] = -dx*drho_dt[0], J[i] = -dx*sum_{j<=i} drho_dt[j]
        drho_dt = (rho_next - rho_prev) / dt
        return -dx * jnp.cumsum(drho_dt)

    Jx = lax.cond(periodic, _Jx_periodic, _Jx_nonperiodic, operand=None)

    # 5) Assemble final J; only filter Jy/Jz (filtering Jx would break continuity)
    current_density = J_xyz.at[:, 0].set(Jx)

    def _filter_transverse(J):
        J_yz = filter_vector_field(
            J[:, 1:],
            passes=filter_passes, alpha=filter_alpha, strides=filter_strides,
            bc_left=field_BC_left, bc_right=field_BC_right,
        )
        return J.at[:, 1:].set(J_yz)

    current_density = lax.cond(
        filter_passes > 0,
        lambda _: _filter_transverse(current_density),
        lambda _: current_density,
        operand=None
    )
    return current_density


@partial(jit, static_argnames=('grid_size',))
def current_density_periodic_CN(xs_n, vs_n, qs, dx, grid_start, grid_size):
    x = xs_n[:, 0]
    q = qs[:, 0]

    x_norm = (x - grid_start) / dx
    k = jnp.floor(x_norm + 0.5).astype(jnp.int32)
    d = x_norm - k

    w_c = 0.75 - d**2
    w_l = 0.5 * (0.5 - d)**2
    w_r = 0.5 * (0.5 + d)**2
    w = jnp.stack([w_l, w_c, w_r], axis=1)  # (N,3)

    idx = jnp.stack([k-1, k, k+1], axis=1) % grid_size  # (N,3)

    # contrib: (N,3,3) = weights*(q/dx)*v
    factor = (q / dx)[:, None] * w                       # (N,3)
    contrib = factor[:, :, None] * vs_n[:, None, :]      # (N,3,3)

    idx_f = idx.reshape(-1)                 # (N*3,)
    contrib_f = contrib.reshape(-1, 3)      # (N*3,3)

    J = jnp.zeros((grid_size, 3), dtype=xs_n.dtype).at[idx_f].add(contrib_f)
    return J