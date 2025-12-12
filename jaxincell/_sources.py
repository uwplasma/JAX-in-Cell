import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import dynamic_update_slice
from functools import partial
from ._filters import filter_scalar_field, filter_vector_field

__all__ = ['get_S2_weights_and_indices_periodic_CN','charge_density_BCs', 'single_particle_charge_density', 'calculate_charge_density', 'current_density', 'current_density_periodic_CN']

@jit
def get_S2_weights_and_indices_periodic_CN(x, dx, grid_start, grid_size):
    """
    Calculates weights and indices for Quadratic Spline (S2).
    Applies Periodic Wrapping to indices immediately.
    """
    # 1. Normalize position
    x_norm = (x - grid_start) / dx
    
    # 2. Nearest node index k
    k = jnp.round(x_norm).astype(int)
    
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
def charge_density_BCs(particle_BC_left, particle_BC_right, position, dx, grid, charge):
    """
    Compute the charge contribution to the boundary points based on particle positions and boundary conditions.

    Args:
        particle_BC_left (int): Boundary condition for the left edge (0: periodic, 1: reflective, 2: absorbing).
        particle_BC_right (int): Boundary condition for the right edge (0: periodic, 1: reflective, 2: absorbing).
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
        particle_BC_left (int): Left boundary condition type (0: periodic, 1: reflective, 2: absorbing).
        particle_BC_right (int): Right boundary condition type (0: periodic, 1: reflective, 2: absorbing).

    Returns:
        array: The charge density contribution on the grid.
    """
    # Compute charge density using a quadratic shape function
    grid_noBCs =  (q/dx)*jnp.where(jnp.abs(x-grid)<=dx/2,3/4-(x-grid)**2/(dx**2),
                         jnp.where((dx/2<jnp.abs(x-grid))&(jnp.abs(x-grid)<=3*dx/2),
                                    0.5*(3/2-jnp.abs(x-grid)/dx)**2,
                                    jnp.zeros(len(grid))))

    # Handle boundary conditions
    chargedens_for_L, chargedens_for_R = charge_density_BCs(particle_BC_left, particle_BC_right, x, dx, grid, q)
    grid_BCs = grid_noBCs.at[ 0].set(chargedens_for_L + grid_noBCs[ 0])
    grid_BCs = grid_BCs  .at[-1].set(chargedens_for_R + grid_BCs  [-1])
    return grid_BCs

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
    # Vectorize over particles
    chargedens_contrib = vmap(single_particle_charge_density, in_axes=(0, 0, None, None, None, None))
    
    # Compute charge density for all particles
    chargedens = chargedens_contrib(xs_n[:, 0], qs[:, 0], dx, grid, particle_BC_left, particle_BC_right)

    # Sum the contributions across all particles
    total_chargedens = jnp.sum(chargedens, axis=0)

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
    def compute_current(i):
        # Compute x-component of the current density
        x_nminushalf = xs_nminushalf[i, 0]
        x_nplushalf = xs_nplushalf[i, 0]
        q = qs[i, 0]
        cell_no = ((x_nminushalf - grid_start) // dx).astype(int)

        # Compute the charge density difference over time
        diff_chargedens_1particle_whole = (
            single_particle_charge_density(x_nplushalf, q, dx, grid, particle_BC_left, particle_BC_right) -
            single_particle_charge_density(x_nminushalf, q, dx, grid, particle_BC_left, particle_BC_right)
        ) / dt

        # Sweep only cells -3 to 2 relative to particle's initial position.
        diff_chargedens_1particle_short = jnp.roll(diff_chargedens_1particle_whole, 3 - cell_no)[:6]
        j_grid_short = jnp.cumsum(-diff_chargedens_1particle_short * dx)

        # Copy 6-cell grid back onto proper grid
        j_grid_x = jnp.zeros(len(grid))
        j_grid_x = dynamic_update_slice(j_grid_x, j_grid_short, (0,))

        # Roll back to its correct position on grid
        j_grid_x = jnp.roll(j_grid_x, cell_no - 3)

        # Compute y- and z-components of the current density
        x_n = xs_n[i, 0]
        vy_n = vs_n[i, 1]
        vz_n = vs_n[i, 2]
        chargedens = single_particle_charge_density(x_n, q, dx, grid, particle_BC_left, particle_BC_right)

        j_grid_y = chargedens * vy_n
        j_grid_z = chargedens * vz_n

        return j_grid_x, j_grid_y, j_grid_z  # Each output has shape (grid_size,)
 
    current_dens_x, current_dens_y, current_dens_z = vmap(compute_current)(jnp.arange(len(xs_nminushalf)))
    current_dens_x = jnp.sum(current_dens_x, axis=0)
    current_dens_y = jnp.sum(current_dens_y, axis=0)
    current_dens_z = jnp.sum(current_dens_z, axis=0)

    current_density = jnp.stack([current_dens_x, current_dens_y, current_dens_z], axis=0).T

    # Apply digital filtering to the current density
    current_density = filter_vector_field(
        current_density,
        passes=filter_passes,
        alpha=filter_alpha,
        strides=filter_strides,
        bc_left=field_BC_left,
        bc_right=field_BC_right,
    )
    
    return current_density


@partial(jit, static_argnames=('grid_size',))
def current_density_periodic_CN(xs_n, vs_n, qs, dx, grid_start, grid_size):
    """
    Deposits Current J using Periodic BCs.
    Note: We removed xs_nminushalf/plus half arguments as we use the midpoint 
    approximation (xs_n, vs_n) consistent with CN.
    """
    
    def compute_single_particle_J(i):
        x = xs_n[i, 0]
        q = qs[i, 0]
        
        # Variational Current: J = (q/dx) * v * S(x)
        vx = vs_n[i, 0] * (q / dx)
        vy = vs_n[i, 1] * (q / dx)
        vz = vs_n[i, 2] * (q / dx)
        
        # Get wrapped indices and weights
        indices, weights = get_S2_weights_and_indices_periodic_CN(x, dx, grid_start, grid_size)
        
        # Calculate contributions
        jx_contrib = weights * vx
        jy_contrib = weights * vy
        jz_contrib = weights * vz
        
        return indices, jx_contrib, jy_contrib, jz_contrib

    # Vectorize over particles
    batch_indices, batch_Jx, batch_Jy, batch_Jz = vmap(compute_single_particle_J)(jnp.arange(len(xs_n)))
    
    # Flatten
    batch_indices = batch_indices.flatten()
    batch_Jx = batch_Jx.flatten()
    batch_Jy = batch_Jy.flatten()
    batch_Jz = batch_Jz.flatten()
    
    # Accumulate onto grid
    # .at[indices].add() handles collisions (multiple particles in same cell) correctly
    J_x = jnp.zeros(grid_size).at[batch_indices].add(batch_Jx)
    J_y = jnp.zeros(grid_size).at[batch_indices].add(batch_Jy)
    J_z = jnp.zeros(grid_size).at[batch_indices].add(batch_Jz)
    
    return jnp.stack([J_x, J_y, J_z], axis=1)