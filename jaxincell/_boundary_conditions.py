from jax import jit, vmap
import jax.numpy as jnp
from ._constants import speed_of_light

__all__ = ['set_BC_single_particle', 'set_BC_particles', 'set_BC_single_particle_positions', 'set_BC_positions']

def _reflect_position(x, length):
    phase = (x + length / 2) % (2 * length)
    return length / 2 - jnp.abs(phase - length)

def set_BC_single_particle(x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z,
                           BC_left, BC_right, mixed_BC_weight=1., COR_left=1., COR_right=1., max_vx=1.):
    """One resolved wall impact; q and m carry the same returned fraction.

    BC3 returns ``mixed_BC_weight``; BC4 returns ``max(1-abs(vx)/max_vx,0)``,
    where max_vx is a prescribed wall speed, independent of other markers.
    Restitution reduces both the normal speed and the remaining drift distance.
    """
    left = (x_n[0] < -box_size_x/2) | ((x_n[0] == -box_size_x/2) & (v_n[0] < 0))
    right = (x_n[0] > box_size_x/2) | ((x_n[0] == box_size_x/2) & (v_n[0] > 0))
    hit = left | right
    code = jnp.where(left, BC_left, BC_right)
    face = jnp.where(left, -box_size_x/2, box_size_x/2)
    restitution = jnp.where(left, COR_left, COR_right)
    fraction = jnp.where(code == 2, 0., jnp.where(code == 3, mixed_BC_weight,
                        jnp.where(code == 4, jnp.clip(1-jnp.abs(v_n[0])/jnp.where(max_vx > 0, max_vx, 1.), 0., 1.), 1.)))
    fraction = jnp.where(hit & (code != 0), fraction, 1.)
    q, m = q*fraction, m*fraction
    lost = hit & (code >= 2) & (fraction <= 0)
    normal = jnp.where(code == 0, (x_n[0]+box_size_x/2) % box_size_x-box_size_x/2,
                       face-restitution*(x_n[0]-face))
    normal = jnp.where(hit | (code == 0), normal, x_n[0])
    normal = jnp.where(lost, jnp.where(left, grid[0]-1.5*dx, grid[-1]+3*dx), normal)
    elastic_walls = (BC_left == 1) & (BC_right == 1) & (COR_left == 1) & (COR_right == 1)
    normal = jnp.where(elastic_walls & hit, _reflect_position(x_n[0], box_size_x), normal)
    normal = jnp.where(~lost & (jnp.abs(normal) > box_size_x/2), jnp.nan, normal)
    position = jnp.array([normal, (x_n[1]+box_size_y/2) % box_size_y-box_size_y/2,
                         (x_n[2]+box_size_z/2) % box_size_z-box_size_z/2])
    velocity = v_n.at[0].set(jnp.where(hit & (code != 0), -restitution*v_n[0], v_n[0]))
    reflections = jnp.floor((jnp.abs(x_n[0])-box_size_x/2)/box_size_x)+1
    velocity = velocity.at[0].set(jnp.where(elastic_walls & hit,
                    v_n[0]*jnp.where(reflections % 2 == 0, 1., -1.), velocity[0]))
    return position, jnp.where(lost, 0., velocity), q, jnp.where(lost, 0., q_m), m


@jit
def set_BC_particles(xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z,
                     BC_left, BC_right, mixed_BC_weight=1., COR_left=1., COR_right=1.,
                     mixed_BC_velocity_scale=speed_of_light):
    """Apply the prescribed wall law to each marker independently."""
    x, v, q, qm, m = vmap(lambda x, v, q, qm, m: set_BC_single_particle(
        x, v, q, qm, m, dx, grid, box_size_x, box_size_y, box_size_z,
        BC_left, BC_right, mixed_BC_weight, COR_left, COR_right, mixed_BC_velocity_scale))(
        xs_n, vs_n, qs, q_ms, ms)
    return x, v, q, m, qm


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

    hit_left_boundary = x_n[0] < -box_size_x / 2
    hit_right_boundary = x_n[0] > box_size_x / 2

    x_n0 = jnp.select(
        [hit_left_boundary, hit_right_boundary],
        [jnp.select(
            [BC_left == 0, BC_left == 1, BC_left == 2, BC_left == 3, BC_left == 4],
            [(x_n[0] + box_size_x / 2) % box_size_x - box_size_x / 2, -box_size_x - x_n[0], grid[0] - 1.5 * dx, -box_size_x - x_n[0], -box_size_x - x_n[0]]
        ), jnp.select(
            [BC_right == 0, BC_right == 1, BC_right == 2, BC_right == 3, BC_right == 4],
            [(x_n[0] + box_size_x / 2) % box_size_x - box_size_x / 2, box_size_x - x_n[0], grid[-1] + 3 * dx, box_size_x - x_n[0], box_size_x - x_n[0]]
        )],
        x_n[0]
    )

    x_n0 = jnp.where((BC_left == 1) & (BC_right == 1), _reflect_position(x_n[0], box_size_x), x_n0)
    return jnp.array([jnp.where((BC_left == 0) & (BC_right == 0),
                                     (x_n[0]+box_size_x/2) % box_size_x-box_size_x/2, x_n0), x_n1, x_n2])

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
