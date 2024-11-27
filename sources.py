import jax.numpy as jnp
from jax import jit
from jax.lax import fori_loop

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
    grid_BCs = grid_noBCs.at[0].set(chargedens_for_L + grid_noBCs[0])
    grid_BCs = grid_BCs.at[-1].set(chargedens_for_R + grid_BCs[-1])
    return grid_BCs

@jit
def calculate_charge_density(xs_n, qs, dx, grid, particle_BC_left, particle_BC_right):
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
    chargedens = jnp.zeros(len(grid))

    def chargedens_update(i, chargedens):
        chargedens += single_particle_charge_density(xs_n[i, 0], qs[i, 0], dx, grid, particle_BC_left, particle_BC_right)
        return chargedens

    chargedens = fori_loop(0, len(xs_n), chargedens_update, chargedens)
    return chargedens

@jit
def current_density(xs_nminushalf, xs_n, xs_nplushalf, vs_n, qs, dx, dt, grid, grid_start, particle_BC_left, particle_BC_right):
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
        BC_left (int): Left boundary condition type.
        BC_right (int): Right boundary condition type.

    Returns:
        array: Current density on the grid, shape (G, 3), where G is the number of grid points.
    """
    current_dens_x = jnp.zeros(len(grid))
    current_dens_y = jnp.zeros(len(grid))
    current_dens_z = jnp.zeros(len(grid))

    def current_update_x(i, jx):
        x_nminushalf = xs_nminushalf[i, 0]
        x_nplushalf = xs_nplushalf[i, 0]
        q = qs[i, 0]
        cell_no = ((x_nminushalf - grid_start) // dx).astype(int)

        # Compute the charge density difference over time
        diff_chargedens_1particle_whole = (
            single_particle_charge_density(x_nplushalf,  q, dx, grid, particle_BC_left, particle_BC_right) -
            single_particle_charge_density(x_nminushalf, q, dx, grid, particle_BC_left, particle_BC_right)
        ) / dt

        #Sweep only cells -3 to 2 relative to particle's initial position. 
        #To do so, roll grid to front, perform current calculations, and roll back.
        #Roll grid so that particle's initial position is on 4th cell, and select
        #first 6 cells. See readme for diagram.
        #Note 1st cell should always be 0.
        diff_chargedens_1particle_short = jnp.roll(diff_chargedens_1particle_whole,3-cell_no)[:6]
        
        def iterate_short_grid(k,j_grid):
            j_grid = j_grid.at[k+1].set(-diff_chargedens_1particle_short[k+1]*dx+j_grid[k])
            return j_grid
        j_grid_short = jnp.zeros(6)
        j_grid_short = fori_loop(0,6,iterate_short_grid,j_grid_short)

        #Copy 6-cell grid back onto proper grid
        def short_grid_to_grid(n,j_grid):
            j_grid = j_grid.at[n].set(j_grid_short[n])
            return j_grid
        j_grid = jnp.zeros(len(grid))
        j_grid = fori_loop(0,6,short_grid_to_grid,j_grid)
        
        #Roll back to its correct position on grid
        j_grid = jnp.roll(j_grid,cell_no-3)

        jx += j_grid
        return jx

    current_dens_x = fori_loop(0, len(xs_nminushalf), current_update_x, current_dens_x)

    # Update current densities for y and z directions
    def current_update_y(i, jy):
        x_n = xs_n[i, 0]
        q = qs[i, 0]
        vy_n = vs_n[i, 1]
        chargedens = single_particle_charge_density(x_n, q, dx, grid, particle_BC_left, particle_BC_right)
        jy += chargedens * vy_n
        return jy

    current_dens_y = fori_loop(0, len(xs_nminushalf), current_update_y, current_dens_y)

    def current_update_z(i, jz):
        x_n = xs_n[i, 0]
        q = qs[i, 0]
        vz_n = vs_n[i, 2]
        chargedens = single_particle_charge_density(x_n, q, dx, grid, particle_BC_left, particle_BC_right)
        jz += chargedens * vz_n
        return jz

    current_dens_z = fori_loop(0, len(xs_nminushalf), current_update_z, current_dens_z)

    return jnp.transpose(jnp.array([current_dens_x, current_dens_y, current_dens_z]))