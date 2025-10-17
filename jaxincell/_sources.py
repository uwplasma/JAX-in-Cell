import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
from jax.lax import dynamic_update_slice
from ._filters import filter_scalar_field
from ._fields import _geom_at_grid
from ._metric_tensor import metric_bundle

__all__ = ['charge_density_BCs', 'single_particle_charge_density', 'calculate_charge_density', 'current_density']

@jit
def charge_density_BCs(particle_BC_left, particle_BC_right, position, dx, grid, charge):
    """
    Compute the charge contribution to the boundary points based on particle positions and boundary conditions.

    Args:
        BC_left (int): Boundary condition for the left edge (0: periodic, 1: reflective, 2: absorbing).
        BC_right (int): Boundary condition for the right edge (0: periodic, 1: reflective, 2: absorbing).
        position (float): Position of the particle.
        dx (float): Grid spacing.
        grid (array-like): Grid points as a 1D array.
        charge (float): Charge of the particle.

    Returns:
        tuple: Charge contributions to the left and right boundaries.
    """
    # Compute charges outside the grid boundaries
    extra_charge_left = (charge / dx) * jnp.where(
        jnp.abs(position - grid[0]) <= dx / 2,
        0.5 * (0.5 + (grid[0] - position) / dx) ** 2,
        0
    )
    extra_charge_right = (charge / dx) * jnp.where(
        jnp.abs(position - grid[-1]) <= dx / 2,
        0.5 * (0.5 + (position - grid[-1]) / dx) ** 2,
        0
    )

    # Apply boundary conditions
    charge_left = jnp.select(
        [particle_BC_left == 0, particle_BC_left == 1, particle_BC_left == 2],
        [extra_charge_right, extra_charge_left, 0]
    )
    charge_right = jnp.select(
        [particle_BC_right == 0, particle_BC_right == 1, particle_BC_right == 2],
        [extra_charge_left, extra_charge_right, 0]
    )

    return charge_left, charge_right

@jit
def single_particle_charge_density(x, q, dx, grid, particle_BC_left, particle_BC_right):
    """
    Computes the charge density contribution of a single particle to the grid using a 
    quadratic particle shape function.

    Args:
        x (float): The particle position.
        q (float): The particle charge.
        dx (float): The grid spacing.
        grid (array): The grid points.
        BC_left (int): Left boundary condition type.
        BC_right (int): Right boundary condition type.

    Returns:
        array: The charge density contribution on the grid.
    """
    # Compute charge density using a quadratic shape function
    grid_noBCs =  (q/dx)*jnp.where(abs(x-grid)<=dx/2,3/4-(x-grid)**2/(dx**2),
                         jnp.where((dx/2<abs(x-grid))&(abs(x-grid)<=3*dx/2),
                                    0.5*(3/2-abs(x-grid)/dx)**2,
                                    jnp.zeros(len(grid))))

    # Handle boundary conditions
    chargedens_for_L, chargedens_for_R = charge_density_BCs(particle_BC_left, particle_BC_right, x, dx, grid, q)
    grid_BCs = grid_noBCs.at[ 0].set(chargedens_for_L + grid_noBCs[ 0])
    grid_BCs = grid_BCs  .at[-1].set(chargedens_for_R + grid_BCs  [-1])
    return grid_BCs

@jit
def calculate_charge_density(xs_n, qs, dx, grid, particle_BC_left, particle_BC_right, filter_passes=5, filter_alpha=0.5, filter_strides=(1, 2, 4)):
    """
    Computes the total charge density on the grid by summing contributions from all particles.

    Args:
        xs_n (array): Particle positions at the current timestep, shape (N, 1).
        qs (array): Particle charges, shape (N, 1).
        dx (float): The grid spacing.
        grid (array): The grid points.
        BC_left (int): Left boundary condition type.
        BC_right (int): Right boundary condition type.

    Returns:
        array: Total charge density on the grid.
    """
    # Vectorize over particles
    chargedens_contrib = vmap(single_particle_charge_density, in_axes=(0, 0, None, None, None, None))
    
    # Compute charge density for all particles
    chargedens = chargedens_contrib(xs_n[:, 0], qs[:, 0], dx, grid, particle_BC_left, particle_BC_right)

    # Sum the contributions across all particles
    total_chargedens = jnp.sum(chargedens, axis=0)

    total_chargedens = filter_scalar_field(total_chargedens, passes=filter_passes, alpha=filter_alpha, strides=filter_strides) 

    return total_chargedens

@jit
def current_density(xs_nminushalf, xs_n, xs_nplushalf, vs_n, qs,
                    dx, dt, grid, grid_start,
                    particle_BC_left, particle_BC_right,
                    t, metric_cfg):
    """
    Computes the current density `j` on the grid from particle motion.
    GR-aware current deposition for diagonal, zero-shift metrics.
    Enforces the densitized continuity equation:
        ∂t(√γ ρ) + ∂x(√γ J^x) = 0

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
        BC_left (int): Left boundary condition type.
        BC_right (int): Right boundary condition type.

    Returns:
        array: Current density on the grid, shape (G, 3), where G is the number of grid points.

    Metric-aware current deposition (diagonal, zero shift).
    Enforces: ∂t(√γ ρ) + ∂x(√γ α J^x) = 0   on the grid.
    Returns coordinate components J^i (G,3).
    """
    # Geometry at grid
    alphas, S = _geom_at_grid(t, grid, metric_cfg)   # alphas: (G,), S: (G,3)
    S1, S2, S3 = S[:,0], S[:,1], S[:,2]
    sqrt_gamma = S1 * S2 * S3                        # (G,)
    alpha = alphas                                   # (G,)

    def compute_current(i):
        x_minus = xs_nminushalf[i, 0]
        x_plus  = xs_nplushalf [i, 0]
        x_n     = xs_n        [i, 0]
        q       = qs[i, 0]

        # coordinate charge densities at half times
        rho_minus = single_particle_charge_density(x_minus, q, dx, grid, particle_BC_left, particle_BC_right)
        rho_plus  = single_particle_charge_density(x_plus,  q, dx, grid, particle_BC_left, particle_BC_right)

        # densitized charges
        rho_den_minus = sqrt_gamma * rho_minus
        rho_den_plus  = sqrt_gamma * rho_plus

        # time difference (coord time)
        diff_rho_den = (rho_den_plus - rho_den_minus) / dt   # (G,)

        # choose a local stencil reference around the departure cell (same as before)
        cell_no = ((x_minus - grid_start) // dx).astype(int)

        # build short 6-cell cumulative integral to satisfy ∂x F = -∂t(√γ ρ)
        # Here F := √γ α J^x  (THIS is the crucial change: include α inside the flux)
        rhs_short = jnp.roll(diff_rho_den, 3 - cell_no)[:6]  # (6,)
        F_short   = jnp.cumsum(-rhs_short * dx)              # (6,)

        # place back on full grid and unroll
        F_full = jnp.zeros_like(diff_rho_den)
        F_full = dynamic_update_slice(F_full, F_short, (0,))
        F_full = jnp.roll(F_full, cell_no - 3)               # (G,)

        # densitized flux → coordinate current: Jx = F / (√γ α)
        Jx = F_full / (sqrt_gamma * alpha + 1e-30)           # small eps to guard div-by-zero

        # Transverse currents from coordinate ρ at t^n and v^y,v^z (coordinate)
        rho_avg = 0.5 * (rho_minus + rho_plus)
        Jy = rho_avg * vs_n[i, 1]
        Jz = rho_avg * vs_n[i, 2]

        return Jx, Jy, Jz

    Jx_all, Jy_all, Jz_all = vmap(compute_current)(jnp.arange(len(xs_nminushalf)))
    Jx = jnp.sum(Jx_all, axis=0)
    Jy = jnp.sum(Jy_all, axis=0)
    Jz = jnp.sum(Jz_all, axis=0)
    return jnp.stack([Jx, Jy, Jz], axis=1)    # shape (G,3)