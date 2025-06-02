from jax import vmap, jit
import jax.numpy as jnp
from ._boundary_conditions import field_2_ghost_cells
from ._constants import speed_of_light as c

__all__ = ['fields_to_particles_grid', 'rotation', 'boris_step', 'boris_step_relativistic']

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
    ghost_cell_L1, ghost_cell_L2, ghost_cell_R = field_2_ghost_cells(field_BC_left,field_BC_right,field)
    field = jnp.insert(field,0,ghost_cell_L2,axis=0)
    field = jnp.insert(field,0,ghost_cell_L1,axis=0)
    field = jnp.append(field,jnp.array([ghost_cell_R]),axis=0)
    x = x_n[0]
    
    # Adjust the grid to accommodate particles at the first half grid cell (staggered grid)
    #If using a staggered grid, particles at first half cell will be out of grid, so add extra cell
    grid = jnp.insert(grid,0,grid[0]-dx,axis=0) 
    
    # Calculate the index of the field grid corresponding to the particle position
    i = ((x-grid_start+dx)//dx).astype(int) #new grid_start = grid_start-dx due to extra cell
    
    # Interpolate the field at the particle position using a quadratic interpolation
    fields_n = 0.5*field[i]*(0.5+(grid[i]-x)/dx)**2 + field[i+1]*(0.75-(grid[i]-x)**2/dx**2) + 0.5*field[i+2]*(0.5-(grid[i]-x)/dx)**2
    
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
def boris_step(dt, xs_nplushalf, vs_n, q_ms, E_fields_at_x, B_fields_at_x):
    """
    This function performs one step of the Boris algorithm for particle motion. 
    The particle velocity is updated using the electric and magnetic fields at its position, 
    and the particle position is updated using the new velocity.

    Args:
        dt (float): Time step for the simulation.
        xs_nplushalf (array): The particle positions at the half-time step n+1/2, shape (N, 3).
        vs_n (array): The particle velocities at time step n, shape (N, 3).
        q_ms (array): The charge-to-mass ratio of each particle, shape (N, 1).
        E_fields_at_x (array): The interpolated electric field values at the particle positions, shape (N, 3).
        B_fields_at_x (array): The magnetic field values at the particle positions, shape (N, 3).

    Returns:
        tuple: A tuple containing:
            - xs_nplus3_2 (array): The updated particle positions at time step n+3/2, shape (N, 3).
            - vs_nplus1 (array): The updated particle velocities at time step n+1, shape (N, 3).
    """
    # First half step update for velocity due to electric field
    vs_n_int = vs_n + (q_ms) * E_fields_at_x * dt / 2
    
    # Apply the Boris rotation step for the magnetic field
    vs_n_rot = vmap(lambda B_n, v_n, q_m: rotation(dt, B_n, v_n, q_m))(B_fields_at_x, vs_n_int, q_ms[:, 0])
    
    # Second half step update for velocity due to electric field
    vs_nplus1 = vs_n_rot + (q_ms) * E_fields_at_x * dt / 2
    
    # Update the particle positions using the new velocities
    xs_nplus3_2 = xs_nplushalf + dt * vs_nplus1
    
    return xs_nplus3_2, vs_nplus1
    # vs_nplus1 = vs_n + (q_ms) * E_fields_at_x * dt
    # xs_nplus1 = xs_nplushalf + dt * vs_nplus1
    # return xs_nplus1, vs_nplus1

@jit
def relativistic_rotation(dt, B, p_minus, q, m):
    """
    Rotate momentum vector in magnetic field (relativistic Boris step).
    """
    # gamma_minus from p_minus
    gamma_minus = jnp.sqrt(1 + jnp.sum(p_minus ** 2) / (m ** 2 * c ** 2))

    # t vector (rotation vector)
    t = (q * dt) / (2 * m * gamma_minus) * B
    p_dot_t = jnp.dot(p_minus, t)
    p_cross_t = jnp.cross(p_minus, t)
    t_squared = jnp.dot(t, t)

    p_plus = (p_minus*(1-t_squared) + 2*(p_dot_t * t + p_cross_t)) / (1 + t_squared)

    return p_plus

@jit
def boris_step_relativistic(dt, xs_nplushalf, vs_n, q_s, m_s, E_fields_at_x, B_fields_at_x):
    """
    Relativistic Boris pusher for N particles.
    
    Args:
        dt: Time step
        xs_nplushalf: Particle positions at t = n + 1/2, shape (N, 3)
        vs_n: Velocities at time t = n, shape (N, 3)
        q_s: Charges, shape (N,)
        m_s: Masses, shape (N,)
        E_fields_at_x: Electric fields at particle positions, shape (N, 3)
        B_fields_at_x: Magnetic fields at particle positions, shape (N, 3)
        c: Speed of light (default = 1.0 for normalized units)
    Returns:
        xs_nplus3_2: Updated positions at t = n + 3/2, shape (N, 3)
        vs_nplus1: Updated velocities at t = n + 1, shape (N, 3)
    """

    def single_particle_step(x, v, q, m, E, B):
        # Compute initial momentum
        gamma_n = 1/jnp.sqrt(1.0 - jnp.sum((v / c) ** 2))

        p_n = gamma_n * m * v

        # Half electric field acceleration
        p_minus = p_n + q * E * dt / 2

        # Magnetic rotation
        p_plus = relativistic_rotation(dt, B, p_minus, q, m)

        # Second half electric field acceleration
        p_nplus1 = p_plus + q * E * dt / 2

        # Compute new gamma
        gamma_nplus1 = jnp.sqrt(1.0 + jnp.sum((p_nplus1 / (m * c)) ** 2))

        # Recover new velocity
        v_nplus1 = p_nplus1 / (gamma_nplus1 * m)

        # Update position using new velocity
        x_nplus3_2 = x + dt * v_nplus1

        return x_nplus3_2, v_nplus1

    # Vectorize over particles
    xs_nplus3_2, vs_nplus1 = vmap(single_particle_step)(
        xs_nplushalf, vs_n, q_s, m_s, E_fields_at_x, B_fields_at_x
    )

    return xs_nplus3_2, vs_nplus1