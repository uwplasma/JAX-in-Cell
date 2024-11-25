import jax.numpy as jnp
from jax import jit, vmap

@jit
def charge_density_BCs(BC_left, BC_right, position, dx, grid, charge):
    """
    Calculate the charge distribution at the boundaries of a grid for particles.
    Boundary condition indicators: 0 is periodic, 1 is reflective, 2 is absorbing.
    Parameters:
    BC_left  (int): Boundary condition indicator for the left boundary.
    BC_right (int): Boundary condition indicator for the right boundary.
    position (array-like): Positions of the particles.
    dx (float): Grid spacing.
    grid (array-like): Grid points.
    charge (float): Charge of the particles.
    Returns:
    tuple: A tuple containing:
        - charge_left  (array-like): Charge to be added to the left boundary.
        - charge_right (array-like): Charge to be added to the right boundary.
    """
    assert BC_left  in [0,1,2], "Invalid left boundary condition."
    assert BC_right in [0,1,2], "Invalid right boundary condition."
    # Charge density outside the grid boundaries
    extra_charge_left  = (charge/dx)*jnp.where(abs(position-grid[ 0])<=dx/2,        # find particles outside the left boundary
                                              0.5*(0.5+(grid[0]-position)/dx)**2,  # calculate the charge of those particles
                                              0)
    extra_charge_right = (charge/dx)*jnp.where(abs(grid[-1]-position)<=dx/2,        # find particles outside the right boundary
                                              0.5*(0.5+(position-grid[-1])/dx)**2, # calculate the charge of those particles
                                              0)
    # Apply boundary conditions to those charges
    charge_left = jnp.select(
        [BC_left == 0, BC_left == 1, BC_left == 2],
        [extra_charge_right, extra_charge_left, 0]
    )
    charge_right = jnp.select(
        [BC_right == 0, BC_right == 1, BC_right == 2],
        [extra_charge_left, extra_charge_right, 0]
    )
    return charge_left, charge_right

@jit
def single_particle_charge_density(position,charge,dx,grid,BC_left,BC_right):
    """
    Pseudoparticles have a number density Np=n*L/Omega, where n is the number density,
    L is the length of the domain, and Omega is the number of real particles per
    pseudoparticle per area. The pseudoparticles have a triangular shape function
    of width 2*dx, and the charge is distributed over the two nearest grid points.
    """
    # Charge density on the grid
    dist = abs(position - grid)
    charge_density = (charge/dx)*jnp.where(dist <= dx / 2,                                   # if x is inside dx/2 of the grid point
                                           3 / 4 - (position - grid) ** 2 / dx ** 2,  # then the charge is distributed according to rho=(q/dx)*(3/4-(x-grid)^2/dx^2)
                                           jnp.where((dx / 2 < dist) & (dist <= 3 * dx / 2), # if x is between dx/2 to 3*dx/2 of the grid point
                                                      0.5 * (3 / 2 - dist / dx) ** 2, # then the charge is distributed according to rho=(q/dx)*(3/4-(1/2)*|x-grid|/dx)^2)
                                                      0 * grid                        # otherwise the charge is                     rho=0
                                                    )
                                           )
    # Boundary conditions
    charge_density_left, charge_density_right = charge_density_BCs(BC_left, BC_right, position, dx, grid, charge)
    charge_density = charge_density.at[ 0].add(charge_density_left )
    charge_density = charge_density.at[-1].add(charge_density_right)
    return charge_density

@jit
def calculate_charge_density(particle_positions,particle_charges,dx,grid,BC_left,BC_right):
    """
    Calculate the charge density on a grid given particle positions and charges.
    Args:
        particle_positions (array-like): An array of shape (N, 1) containing the positions of N particles.
        particle_charges (array-like): An array of shape (N, 1) containing the charges of N particles.
        dx (float): The grid spacing.
        grid (array-like): The grid on which to calculate the charge density.
        BC_left  (float): The boundary condition on the left side of the grid.
        BC_right (float): The boundary condition on the right side of the grid.
    Returns:
        array-like: The charge density on the grid.
    """

    def compute_density(pos, charge):
        return single_particle_charge_density(pos, charge, dx, grid, BC_left, BC_right)
    # Use vmap to vectorize over all particles
    all_densities = vmap(compute_density)(particle_positions[:, 0], particle_charges[:, 0])
    # Sum across particles
    return jnp.sum(all_densities, axis=0)

@jit
def find_j(xs_nminushalf, xs_n, xs_nplushalf, vs_n, qs, dx, dt, grid, grid_start, BC_left, BC_right):
    num_particles = xs_nminushalf.shape[0]
    grid_len = len(grid)

    # Precompute single particle charge densities for positions
    def compute_density_pair(x_start, x_end, q):
        rho_start = single_particle_charge_density(x_start, q, dx, grid, BC_left, BC_right)
        rho_end = single_particle_charge_density(x_end, q, dx, grid, BC_left, BC_right)
        return (rho_end - rho_start) / dt

    diff_chargedens = vmap(compute_density_pair)(xs_nminushalf[:, 0], xs_nplushalf[:, 0], qs[:, 0])

    # Calculate j_x
    def compute_current_x(i):
        cell_no = ((xs_nminushalf[i, 0] - grid_start) // dx).astype(int)
        short_density = jnp.roll(diff_chargedens[i], 3 - cell_no)[:6]

        # Build the current for the short grid
        j_grid_short = jnp.zeros(6)
        j_grid_short = j_grid_short.at[1:].set(
            -jnp.cumsum(short_density[1:] * dx) + j_grid_short[0]
        )

        # Map short grid to full grid
        j_grid = jnp.zeros(grid_len)
        j_grid = j_grid.at[cell_no - 3 : cell_no + 3].set(j_grid_short)

        return jnp.roll(j_grid, cell_no - 3)

    current_dens_x = jnp.sum(vmap(compute_current_x)(jnp.arange(num_particles)), axis=0)

    # Calculate j_y and j_z using rho * v
    def compute_current_yz(i):
        rho = single_particle_charge_density(xs_n[i, 0], qs[i, 0], dx, grid, BC_left, BC_right)
        vy = vs_n[i, 1]
        vz = vs_n[i, 2]
        return rho * vy, rho * vz

    jy, jz = vmap(compute_current_yz)(jnp.arange(num_particles))
    current_dens_y = jnp.sum(jy, axis=0)
    current_dens_z = jnp.sum(jz, axis=0)

    # Combine current densities
    return jnp.transpose(jnp.array([current_dens_x, current_dens_y, current_dens_z]))
