from jax import jit, vmap
import jax.numpy as jnp
from ._constants import speed_of_light

__all__ = ['set_BC_single_particle', 'set_BC_particles', 'set_BC_single_particle_positions', 'set_BC_positions']

def set_BC_single_particle(x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """
    Applies boundary conditions (BCs) to a single particle's position and velocity.

    Args:
        x_n (jnp.ndarray): Particle position as a 1D array [x, y, z].
        v_n (jnp.ndarray): Particle velocity as a 1D array [vx, vy, vz].
        q (float): Particle charge.
        q_m (float): Charge-to-mass ratio of the particle.
        dx (float): Grid spacing.
        grid (jnp.ndarray): Discretized grid positions.
        box_size_x, box_size_y, box_size_z (float): Box dimensions in x, y, and z directions.
        BC_left, BC_right (int): Boundary conditions for left and right boundaries in the x-direction.
            0: Periodic
            1: Reflective
            2: Absorbing

    Returns:
        tuple: Updated position (x_n), velocity (v_n), charge (q), and charge-to-mass ratio (q_m).
    """
    # Apply periodic BCs in y and z directions
    x_n1 = (x_n[1] + box_size_y / 2) % box_size_y - box_size_y / 2
    x_n2 = (x_n[2] + box_size_z / 2) % box_size_z - box_size_z / 2

    # Apply boundary conditions in x-direction
    x_n0 = jnp.where(
        x_n[0] < -box_size_x / 2,
        jnp.where(
            BC_left == 0,  # Periodic
            (x_n[0] + box_size_x / 2) % box_size_x - box_size_x / 2,
            jnp.where(
                BC_left == 1,  # Reflective
                -box_size_x - x_n[0],
                jnp.where(BC_left == 2, grid[0] - 1.5 * dx, x_n[0]),  # Absorbing
            ),
        ),
        jnp.where(
            x_n[0] > box_size_x / 2,
            jnp.where(
                BC_right == 0,  # Periodic
                (x_n[0] + box_size_x / 2) % box_size_x - box_size_x / 2,
                jnp.where(
                    BC_right == 1,  # Reflective
                    box_size_x - x_n[0],
                    jnp.where(BC_right == 2, grid[-1] + 3 * dx, x_n[0]),  # Absorbing
                ),
            ),
            x_n[0],
        ),
    )

    # Update velocities for reflective or absorbing boundaries
    v_n = jnp.where(
        x_n[0] < -box_size_x / 2,
        jnp.where(
            BC_left == 0,  # Periodic
            v_n,
            jnp.where(BC_left == 1, v_n * jnp.array([-1, 1, 1]), jnp.array([0, 0, 0])),  # Reflective or Absorbing
        ),
        jnp.where(
            x_n[0] > box_size_x / 2,
            jnp.where(
                BC_right == 0,  # Periodic
                v_n,
                jnp.where(BC_right == 1, v_n * jnp.array([-1, 1, 1]), jnp.array([0, 0, 0])),  # Reflective or Absorbing
            ),
            v_n,
        ),
    )

    # Nullify charges and charge-to-mass ratio for absorbing BCs
    q   = jnp.where(((x_n[0] < -box_size_x / 2) & (BC_left == 2)) | ((x_n[0] > box_size_x / 2) & (BC_right == 2)), 0, q)
    q_m = jnp.where(((x_n[0] < -box_size_x / 2) & (BC_left == 2)) | ((x_n[0] > box_size_x / 2) & (BC_right == 2)), 0, q_m)

    return jnp.array([x_n0, x_n1, x_n2]), v_n, q, q_m

@jit
def set_BC_particles(xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """
    Applies boundary conditions to all particles in parallel.

    Args:
        xs_n (jnp.ndarray): Positions of all particles, shape (N, 3).
        vs_n (jnp.ndarray): Velocities of all particles, shape (N, 3).
        qs (jnp.ndarray): Charges of all particles, shape (N,).
        ms (jnp.ndarray): Masses of all particles, shape (N,).
        q_ms (jnp.ndarray): Charge-to-mass ratios of all particles, shape (N,).
        Other parameters: Same as set_BCs.

    Returns:
        tuple: Updated positions, velocities, charges, masses, and charge-to-mass ratios for all particles.
    """
    xs_n, vs_n, qs, q_ms = vmap(
        lambda x_n, v_n, q, q_m: set_BC_single_particle(x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right)
    )(xs_n, vs_n, qs, q_ms)
    return xs_n, vs_n, qs, ms, q_ms

def set_BC_single_particle_positions(x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """
    Applies boundary conditions to particle positions only (used for half-step updates).

    Args:
        x_n (jnp.ndarray): Particle position as a 1D array [x, y, z].
        Other parameters: Same as set_BCs.

    Returns:
        jnp.ndarray: Updated particle position [x, y, z].
    """
    x_n1 = (x_n[1] + box_size_y / 2) % box_size_y - box_size_y / 2
    x_n2 = (x_n[2] + box_size_z / 2) % box_size_z - box_size_z / 2

    x_n0 = jnp.where(
        x_n[0] < -box_size_x / 2,
        jnp.where(BC_left == 0, (x_n[0] + box_size_x / 2) % box_size_x - box_size_x / 2, 
                  jnp.where(BC_left == 1, -box_size_x - x_n[0], grid[0] - 1.5 * dx)),  # Absorbing
        jnp.where(
            x_n[0] > box_size_x / 2,
            jnp.where(BC_right == 0, (x_n[0] + box_size_x / 2) % box_size_x - box_size_x / 2,
                      jnp.where(BC_right == 1, box_size_x - x_n[0], grid[-1] + 3 * dx)),  # Absorbing
            x_n[0],
        ),
    )
    return jnp.array([x_n0, x_n1, x_n2])

@jit
def set_BC_positions(xs_n, qs, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """
    Applies boundary conditions to particle positions for all particles during a half-step update.

    Args:
        xs_n (jnp.ndarray): Positions of all particles, shape (N, 3).
        qs (jnp.ndarray): Charges of all particles, shape (N,).
        Other parameters: Same as set_BCs.

    Returns:
        jnp.ndarray: Updated positions of all particles, shape (N, 3).
    """
    xs_n = vmap(lambda x_n: set_BC_single_particle_positions(x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right))(xs_n)
    return xs_n


@jit
def field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field):
    """
    Set the ghost cells for the electric field at the boundaries of the simulation grid. 
    The ghost cells are used to apply boundary conditions and extend the field in the 
    simulation domain based on the selected boundary conditions.

    Args:
        field_BC_left (int): Boundary condition at the left boundary for electric field.
                              0: periodic, 1: reflective, 2: absorbing, 3: custom.
        field_BC_right (int): Boundary condition at the right boundary for electric field.
                              0: periodic, 1: reflective, 2: absorbing, 3: custom.
        E_field (array): Electric field values at each grid point, shape (G, 3).
        B_field (array): Magnetic field values at each grid point, shape (G, 3).
        dx (float): Grid spacing in meters.
        current_t (float): Current simulation time.
        E0 (float): Amplitude of the electric field used in custom boundary conditions.
        k (float): Wave number (related to the frequency of the wave).

    Returns:
        tuple: The electric field ghost cells at the left and right boundaries, each of shape (3,).
    """
    field_ghost_cell_L = jnp.where(field_BC_left == 0, E_field[-1],
                         jnp.where(field_BC_left == 1, E_field[0],
                         jnp.where(field_BC_left == 2, jnp.array([0, -2*speed_of_light*B_field[0, 2] - E_field[0, 1], 2*speed_of_light*B_field[0, 1] - E_field[0, 2]]),
                                   jnp.array([0, 0, 0]))))
    field_ghost_cell_R = jnp.where(field_BC_right == 0, E_field[0],
                         jnp.where(field_BC_right == 1, E_field[-1],
                         jnp.where(field_BC_right == 2, jnp.array([0, 3 * E_field[-1, 1] - 2 * speed_of_light * B_field[-1, 2], 3 * E_field[-1, 2] + 2 * speed_of_light * B_field[-1, 1]]),
                                   jnp.array([0, 0, 0]))))
    return field_ghost_cell_L, field_ghost_cell_R


@jit
def field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field):
    """
    Set the ghost cells for the magnetic field at the boundaries of the simulation grid. 
    The ghost cells are used to apply boundary conditions and extend the magnetic field 
    in the simulation domain based on the selected boundary conditions.

    Args:
        field_BC_left (int): Boundary condition at the left boundary for magnetic field.
                              0: periodic, 1: reflective, 2: absorbing, 3: custom.
        field_BC_right (int): Boundary condition at the right boundary for magnetic field.
                              0: periodic, 1: reflective, 2: absorbing, 3: custom.
        B_field (array): Magnetic field values at each grid point, shape (G, 3).
        E_field (array): Electric field values at each grid point, shape (G, 3).

    Returns:
        tuple: The magnetic field ghost cells at the left and right boundaries, each of shape (3,).
    """
    field_ghost_cell_L = jnp.where(field_BC_left == 0, B_field[-1],
                         jnp.where(field_BC_left == 1, B_field[0],
                         jnp.where(field_BC_left == 2, jnp.array([0, 3 * B_field[0, 1] - (2 / speed_of_light) * E_field[0, 2], 3 * B_field[0, 2] + (2 / speed_of_light) * E_field[0, 1]]),
                                   jnp.array([0, 0, 0]))))
    field_ghost_cell_R = jnp.where(field_BC_right == 0, B_field[0],
                         jnp.where(field_BC_right == 1, B_field[-1],
                         jnp.where(field_BC_right == 2, jnp.array([0, -(2 / speed_of_light) * E_field[-1, 2] - B_field[-1, 1], (2 / speed_of_light) * E_field[-1, 1] - B_field[-1, 2]]),
                                   jnp.array([0, 0, 0]))))
    return field_ghost_cell_L, field_ghost_cell_R

@jit
def field_2_ghost_cells(field_BC_left, field_BC_right, field):
    """
    This function adds ghost cells to the field array, which is used for interpolation when 
    accessing field values at particle positions. Ghost cells are added to the left and 
    right boundaries based on the specified boundary conditions for the particles.

    Ghost cells are needed for simulations to handle boundary effects by using the appropriate 
    field values at the boundaries. This is especially important in simulations where particles 
    can cross boundary regions, and the electric and magnetic fields must be extended beyond 
    the simulation domain.

    Args:
        field_BC_left  (array): Boundary condition values for the left boundary of the particle grid, shape (N,).
        field_BC_right (array): Boundary condition values for the right boundary of the particle grid, shape (N,).
        field (array): The field values on the grid, shape (G, 3), where G is the number of grid points.

    Returns:
        tuple: A tuple containing:
            - field_ghost_cell_L2 (array): Ghost cell field values for the left boundary, shape (3,).
            - field_ghost_cell_L1 (array): Ghost cell field values for the left boundary, shape (3,).
            - field_ghost_cell_R (array): Ghost cell field values for the right boundary, shape (3,).
    """

    field_ghost_cell_L2 = jnp.where(field_BC_left==0,field[-2],
                          jnp.where(field_BC_left==1,field[1],
                          jnp.where(field_BC_left==2,jnp.array([0,0,0]),
                                    jnp.array([0,0,0]))))
    field_ghost_cell_L1 = jnp.where(field_BC_left==0,field[-1],
                          jnp.where(field_BC_left==1,field[0],
                          jnp.where(field_BC_left==2,jnp.array([0,0,0]),
                                    jnp.array([0,0,0]))))
    
    field_ghost_cell_R = jnp.where(field_BC_right==0,field[0],
                         jnp.where(field_BC_right==1,field[-1],
                         jnp.where(field_BC_right==2,jnp.array([0,0,0]),
                                   jnp.array([0,0,0]))))

    return field_ghost_cell_L2, field_ghost_cell_L1, field_ghost_cell_R
