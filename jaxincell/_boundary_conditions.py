from jax import jit, vmap
import jax.numpy as jnp
from ._constants import speed_of_light

__all__ = ['set_BC_single_particle', 'set_BC_particles', 'set_BC_single_particle_positions', 'set_BC_positions']

def _particle_boundary_map(x, dx, grid, box, BC_left, BC_right):
    """Map a free-flight endpoint and return reflection and collection flags."""
    length = box[0]
    left, right = x[0] < -length/2, x[0] > length/2
    hit = left | right
    code = jnp.where(left, BC_left, BC_right)
    periods = jnp.asarray(box)
    wrapped = (x + periods/2) % periods - periods/2
    normal = jnp.where(code == 0, wrapped[0], jnp.where(left, -length-x[0], length-x[0]))
    normal = jnp.where(code == 2, jnp.where(left, grid[0]-1.5*dx, grid[-1]+3*dx), normal)
    elastic = (BC_left == 1) & (BC_right == 1) & hit
    phase = (x[0]+length/2) % (2*length)
    normal = jnp.where(elastic, length/2-jnp.abs(phase-length), normal)
    normal = jnp.where(hit, normal, x[0])
    reflections = jnp.ceil((jnp.abs(x[0])-length/2)/length)
    factor = jnp.where(hit & (code == 1), -1., 1.)
    factor = jnp.where(elastic, jnp.where(reflections % 2 == 0, 1., -1.), factor)
    return jnp.array([normal, wrapped[1], wrapped[2]]), factor, hit & (code == 2)


def set_BC_single_particle(x_n, v_n, q, q_m, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """Map position, normal velocity and live charge for boundary codes 0, 1, 2."""
    position, factor, lost = _particle_boundary_map(
        x_n, dx, grid, (box_size_x, box_size_y, box_size_z), BC_left, BC_right)
    velocity = jnp.array([factor*v_n[0], v_n[1], v_n[2]])
    return position, jnp.where(lost, 0., velocity), jnp.where(lost, 0., q), jnp.where(lost, 0., q_m)


@jit
def set_BC_particles(xs_n, vs_n, qs, ms, q_ms, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """Map particles in parallel; absorption retains the legacy mass array."""
    xs_n, vs_n, qs, q_ms = vmap(lambda x, v, q, qm: set_BC_single_particle(
        x, v, q, qm, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right))(xs_n, vs_n, qs, q_ms)
    return xs_n, vs_n, qs, ms, q_ms


def set_BC_single_particle_positions(x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """Apply the same endpoint map without changing velocity or particle weight."""
    return _particle_boundary_map(x_n, dx, grid, (box_size_x, box_size_y, box_size_z), BC_left, BC_right)[0]


@jit
def set_BC_positions(xs_n, qs, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right):
    """Map positions in parallel, independently of particle charge."""
    return vmap(lambda x: set_BC_single_particle_positions(
        x, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right))(xs_n)


@jit
def field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field):
    """Electric ghosts for scalar boundary codes and field arrays of shape (G, 3).

    Code 0 wraps endpoints, 1 copies the adjacent field in the implemented
    reflective ghost convention, and 2 couples transverse E/B for radiation.
    This copy convention is not a general PEC/PMC boundary model.
    Returns left/right arrays of shape (3,).
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
    """Magnetic ghosts for scalar boundary codes and field arrays of shape (G, 3).

    Code 0 wraps endpoints, 1 copies the adjacent field in the implemented
    reflective ghost convention, and 2 couples transverse B/E for radiation.
    This copy convention is not a general PEC/PMC boundary model.
    Returns left/right arrays of shape (3,).
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
    """Interpolation ghosts for scalar boundary codes and a (G, 3) field.

    Code 0 wraps, 1 uses the implemented mirrored-index convention, and 2
    supplies zeros. Returns two left ghosts and one right ghost, each shape (3,).
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
