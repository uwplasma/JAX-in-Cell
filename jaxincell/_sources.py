import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import dynamic_update_slice
import jax
from jax.debug import print as jprint
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
    grid_noBCs =  (q/dx)*jnp.where(abs(x-grid)<=dx/2,   3/4-(x-grid)**2/(dx**2),
                                   
                         jnp.where(    (dx/2<abs(x-grid))&(abs(x-grid)<=3*dx/2),
                                   

                                    0.5*(   3/2-abs(x-grid)/dx   )**2,

                                    jnp.zeros(len(grid))     )          )

    # Handle boundary conditions
    chargedens_for_L, chargedens_for_R = charge_density_BCs(particle_BC_left, particle_BC_right, x, dx, grid, q)
    grid_BCs = grid_noBCs.at[ 0].set(chargedens_for_L + grid_noBCs[ 0])
    grid_BCs = grid_BCs  .at[-1].set(chargedens_for_R + grid_BCs  [-1])
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
    # Vectorize over particles
    chargedens_contrib = vmap(single_particle_charge_density, in_axes=(0, 0, None, None, None, None))
    
    # Compute charge density for all particles
    chargedens = chargedens_contrib(xs_n[:, 0], qs[:, 0], dx, grid, particle_BC_left, particle_BC_right)

    # Sum the contributions across all particles
    total_chargedens = jnp.sum(chargedens, axis=0)

    return total_chargedens

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

    return jnp.stack([current_dens_x, current_dens_y, current_dens_z], axis=0).T















# @jit
def J_from_rhov(xs_n, vs_n, qs, grid):

    """
    Compute the current density from the charge density and particle velocities.

    Args:
        particles (list): List of particle species, each with methods to get charge, subcell position, resolution, and index.
        rho (ndarray): Charge density array.
        J (tuple): Current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
        constants (dict): Dictionary containing physical constants.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
    """

    C = 2.99792458e8  # speed of light



    x_wind = grid[-1] - grid[0]
    y_wind = 1
    z_wind = 1
    # get the grid parameters

    dx = grid[1] - grid[0]
    dy = 1
    dz = 1
    # get the grid spacing
    # get the world parameters

    # new_rho = compute_rho(particles, rho, world)
    # old_rho = rho
    # compute the charge density from the particles

    # rho = 0.5 * (new_rho + old_rho)
    # average the charge density to ensure stability

    # Jx, Jy, Jz = J
    # # unpack the values of J
    # Jx = Jx.at[:, :, :].set(0)
    # Jy = Jy.at[:, :, :].set(0)
    # Jz = Jz.at[:, :, :].set(0)
    # # initialize the current arrays as 0
    # J = (Jx, Jy, Jz)
    # initialize the current density as a tuple

    Jx = jnp.zeros((len(grid), 1, 1))
    Jy = jnp.zeros((len(grid), 1, 1))
    Jz = jnp.zeros((len(grid), 1, 1))

    # for species in particles:
    #     N_particles = species.get_number_of_particles()
    #     charge = species.get_charge()
    #     # get the number of particles and their charge
    #     particle_x, particle_y, particle_z = species.get_position()
    #     # get the position of the particles in the species

    #     vx, vy, vz = species.get_velocity()
    #     # get the velocity of the particles in the species

    #     shape_factor = species.get_shape()

    J = (Jx, Jy, Jz)

    N_particles = xs_n.shape[0]

    # print(    xs_n.shape  )

    def add_to_J(particle, J):
        x = xs_n[particle, 0] - dx/2
        y = xs_n[particle, 1] - dy/2
        z = xs_n[particle, 2] - dz/2
        # get the position of the particle
        vx_particle = vs_n[particle, 0]
        vy_particle = vs_n[particle, 1]
        vz_particle = vs_n[particle, 2]
        # get the velocity of the particle
        charge = qs[particle, 0]

        # print(x, y, z, vx_particle, vy_particle, vz_particle, charge)



        J = J_second_order_weighting(charge, x, y, z, vx_particle, vy_particle, vz_particle, J, grid)
        

        return J
    # add the particle species to the charge density array
    J = jax.lax.fori_loop(0, N_particles, add_to_J, J)

    Jx, Jy, Jz = J

    J = jnp.stack([Jx, Jy, Jz], axis=0).T
    J = jnp.squeeze(J)

    return J


# @jit
def J_second_order_weighting(q, x, y, z, vx, vy, vz, J, grid):
    """
    Distribute the current of a particle to the surrounding grid points using second-order weighting.

    Args:
        q (float): Charge of the particle.
        x, y, z (float): Position of the particle.
        vx, vy, vz (float): Velocity components of the particle.
        J (tuple): Current density arrays (Jx, Jy, Jz).
        rho (ndarray): Charge density array (for shape).
        dx, dy, dz (float): Grid spacing.
        x_wind, y_wind, z_wind (float): Window offsets.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz).
    """

    x_wind = grid[-1] - grid[0]
    y_wind = 1
    z_wind = 1
    # get the grid parameters

    dx = grid[1] - grid[0]
    dy = 1
    dz = 1
    # get the grid spacing

    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    # print(Jx.shape)
    # Ny = 1
    # Nz = 1

    x0 = jnp.floor((x + x_wind / 2) / dx).astype(int)
    y0 = jnp.floor((y + y_wind / 2) / dy).astype(int)
    z0 = jnp.floor((z + z_wind / 2) / dz).astype(int)

    deltax = x - jnp.floor(x / dx) * dx
    deltay = y - jnp.floor(y / dy) * dy
    deltaz = z - jnp.floor(z / dz) * dz

    x1 = wrap_around(x0 + 1, Nx)
    y1 = wrap_around(y0 + 1, Ny)
    z1 = wrap_around(z0 + 1, Nz)

    x_minus1 = x0 - 1
    y_minus1 = y0 - 1
    z_minus1 = z0 - 1

    # print(x_minus1, x0, x1, y_minus1, y0, y1, z_minus1, z0, z1)

    dv = dx * dy * dz

    # Weighting factors
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2

#   0.5*(   3/2-abs(x-grid)/dx   )**2



    Sx1 = jax.lax.cond(
        deltax <= dx/2,
        lambda _: (1/2) * ((1/2) - (deltax/dx))**2,
        # lambda _: (1/2) * (  3/2 - abs(deltax/dx)   )**2,
        lambda _: jnp.array(0.0, dtype=deltax.dtype),
        operand=None
    )
    Sy1 = jax.lax.cond(
        deltay <= dy/2,
        lambda _: (1/2) * ((1/2) - (deltay/dy))**2,
        # lambda _: (1/2) * (  3/2 - abs(deltay/dy)   )**2,
        lambda _: jnp.array(0.0, dtype=deltay.dtype),
        operand=None
    )
    Sz1 = jax.lax.cond(
        deltaz <= dz/2,
        lambda _: (1/2) * ((1/2) - (deltaz/dz))**2,
        # lambda _: (1/2) * (  3/2 - abs(deltaz/dz)   )**2,
        lambda _: jnp.array(0.0, dtype=deltaz.dtype),
        operand=None
    )

    Sx_minus1 = jax.lax.cond(
        deltax <= dx/2,
        lambda _: (1/2) * ((1/2) + (deltax/dx))**2,
        # lambda _: (1/2) * (  3/2 - abs(deltax/dx)   )**2,
        lambda _: jnp.array(0.0, dtype=deltax.dtype),
        operand=None
    )
    Sy_minus1 = jax.lax.cond(
        deltay <= dy/2,
        lambda _: (1/2) * ((1/2) + (deltay/dy))**2,
        # lambda _: (1/2) * (  3/2 - abs(deltay/dy)   )**2,
        lambda _: jnp.array(0.0, dtype=deltay.dtype),
        operand=None
    )
    Sz_minus1 = jax.lax.cond(
        deltaz <= dz/2,
        lambda _: (1/2) * ((1/2) + (deltaz/dz))**2,
        # lambda _: (1/2) * (  3/2 - abs(deltaz/dz)   )**2,
        lambda _: jnp.array(0.0, dtype=deltaz.dtype),
        operand=None
    )

    # Jx distribution
    Jx = Jx.at[x0, y0, z0].add((q * vx / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x1, y0, z0].add((q * vx / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x0, y1, z0].add((q * vx / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jx = Jx.at[x0, y0, z1].add((q * vx / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jx = Jx.at[x1, y1, z0].add((q * vx / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jx = Jx.at[x1, y0, z1].add((q * vx / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jx = Jx.at[x0, y1, z1].add((q * vx / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jx = Jx.at[x1, y1, z1].add((q * vx / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    Jx = Jx.at[x_minus1, y0, z0].add((q * vx / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    Jx = Jx.at[x0, y_minus1, z0].add((q * vx / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    Jx = Jx.at[x0, y0, z_minus1].add((q * vx / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    Jx = Jx.at[x_minus1, y_minus1, z0].add((q * vx / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    Jx = Jx.at[x_minus1, y0, z_minus1].add((q * vx / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    Jx = Jx.at[x0, y_minus1, z_minus1].add((q * vx / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    Jx = Jx.at[x_minus1, y_minus1, z_minus1].add((q * vx / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')

    # Jy distribution
    Jy = Jy.at[x0, y0, z0].add((q * vy / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x1, y0, z0].add((q * vy / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x0, y1, z0].add((q * vy / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jy = Jy.at[x0, y0, z1].add((q * vy / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jy = Jy.at[x1, y1, z0].add((q * vy / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jy = Jy.at[x1, y0, z1].add((q * vy / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jy = Jy.at[x0, y1, z1].add((q * vy / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jy = Jy.at[x1, y1, z1].add((q * vy / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    Jy = Jy.at[x_minus1, y0, z0].add((q * vy / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    Jy = Jy.at[x0, y_minus1, z0].add((q * vy / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    Jy = Jy.at[x0, y0, z_minus1].add((q * vy / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    Jy = Jy.at[x_minus1, y_minus1, z0].add((q * vy / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    Jy = Jy.at[x_minus1, y0, z_minus1].add((q * vy / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    Jy = Jy.at[x0, y_minus1, z_minus1].add((q * vy / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    Jy = Jy.at[x_minus1, y_minus1, z_minus1].add((q * vy / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')

    # Jz distribution
    Jz = Jz.at[x0, y0, z0].add((q * vz / dv) * Sx0 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x1, y0, z0].add((q * vz / dv) * Sx1 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x0, y1, z0].add((q * vz / dv) * Sx0 * Sy1 * Sz0, mode='drop')
    Jz = Jz.at[x0, y0, z1].add((q * vz / dv) * Sx0 * Sy0 * Sz1, mode='drop')
    Jz = Jz.at[x1, y1, z0].add((q * vz / dv) * Sx1 * Sy1 * Sz0, mode='drop')
    Jz = Jz.at[x1, y0, z1].add((q * vz / dv) * Sx1 * Sy0 * Sz1, mode='drop')
    Jz = Jz.at[x0, y1, z1].add((q * vz / dv) * Sx0 * Sy1 * Sz1, mode='drop')
    Jz = Jz.at[x1, y1, z1].add((q * vz / dv) * Sx1 * Sy1 * Sz1, mode='drop')
    Jz = Jz.at[x_minus1, y0, z0].add((q * vz / dv) * Sx_minus1 * Sy0 * Sz0, mode='drop')
    Jz = Jz.at[x0, y_minus1, z0].add((q * vz / dv) * Sx0 * Sy_minus1 * Sz0, mode='drop')
    Jz = Jz.at[x0, y0, z_minus1].add((q * vz / dv) * Sx0 * Sy0 * Sz_minus1, mode='drop')
    Jz = Jz.at[x_minus1, y_minus1, z0].add((q * vz / dv) * Sx_minus1 * Sy_minus1 * Sz0, mode='drop')
    Jz = Jz.at[x_minus1, y0, z_minus1].add((q * vz / dv) * Sx_minus1 * Sy0 * Sz_minus1, mode='drop')
    Jz = Jz.at[x0, y_minus1, z_minus1].add((q * vz / dv) * Sx0 * Sy_minus1 * Sz_minus1, mode='drop')
    Jz = Jz.at[x_minus1, y_minus1, z_minus1].add((q * vz / dv) * Sx_minus1 * Sy_minus1 * Sz_minus1, mode='drop')

    return (Jx, Jy, Jz)



@jit
def wrap_around(ix, size):
    """Wrap around index to ensure it is within bounds."""
    return jax.lax.cond(
        ix > size - 1,
        lambda _: ix - size,
        lambda _: ix,
        operand=None
    )