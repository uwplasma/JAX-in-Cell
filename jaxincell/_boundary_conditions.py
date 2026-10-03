from jax import jit, vmap
import jax.numpy as jnp
from ._constants import speed_of_light

__all__ = ['set_BC_single_particle', 'set_BC_particles', 'set_BC_single_particle_positions', 'set_BC_positions']

def _particle_boundary_map(x, vx, dx, grid, box, BC_left, BC_right,
                           mixed_BC_weight=1., COR_left=1., COR_right=1., max_vx=1., collected=False):
    """Common position map and impact factors; only two elastic walls permit repeated impacts."""
    length = box[0]
    left = (x[0] < -length/2) | ((x[0] == -length/2) & (vx < 0))
    right = (x[0] > length/2) | ((x[0] == length/2) & (vx > 0))
    hit = left | right
    code = jnp.where(left, BC_left, BC_right)
    face = jnp.where(left, -length/2, length/2)
    restitution = jnp.where(left, COR_left, COR_right)
    fraction = jnp.where(code == 2, 0., jnp.where(code == 3, mixed_BC_weight,
                        jnp.where(code == 4, jnp.clip(1-jnp.abs(vx)/jnp.where(max_vx > 0, max_vx, 1.), 0., 1.), 1.)))
    fraction = jnp.where(hit & (code != 0), fraction, 1.)
    lost = hit & (code >= 2) & (fraction <= 0)
    parked = collected & hit & (code != 0)
    periods = jnp.asarray(box, dtype=x.dtype)
    wrapped = (x + periods/2) % periods - periods/2
    normal = jnp.where(code == 0, wrapped[0], face-restitution*(x[0]-face))
    normal = jnp.where(hit | (code == 0), normal, x[0])
    elastic = (BC_left == 1) & (BC_right == 1) & (COR_left == 1) & (COR_right == 1)
    phase = (x[0] + length/2) % (2*length)
    normal = jnp.where(elastic & hit, length/2-jnp.abs(phase-length), normal)
    normal = jnp.where(lost, jnp.where(left, grid[0]-1.5*dx, grid[-1]+3*dx), normal)
    normal = jnp.where(parked, x[0], normal)
    normal = jnp.where((code != 0) & ~(lost | parked) & (jnp.abs(normal) > length/2), jnp.nan, normal)
    reflections = jnp.floor((jnp.abs(x[0])-length/2)/length)+1
    speed_factor = jnp.where(hit & (code != 0), -restitution, 1.)
    speed_factor = jnp.where(elastic & hit, jnp.where(reflections % 2 == 0, 1., -1.), speed_factor)
    # This scalar choice lets XLA discard all impact logic for paired periodic walls.
    periodic = (BC_left == 0) & (BC_right == 0)
    position = jnp.where(periodic, wrapped, jnp.array([normal, wrapped[1], wrapped[2]]))
    return position, jnp.where(periodic, 1., speed_factor), jnp.where(periodic, 1., fraction), (lost | parked) & jnp.logical_not(periodic)

def set_BC_single_particle(x_n, v_n, q, q_m, m, dx, grid, box_size_x, box_size_y, box_size_z,
                           BC_left, BC_right, mixed_BC_weight=1., COR_left=1., COR_right=1., max_vx=1.):
    """One resolved wall impact; q and m carry the same returned fraction.

    BC3 returns ``mixed_BC_weight``; BC4 returns ``max(1-abs(vx)/max_vx,0)``,
    where max_vx is a prescribed wall speed, independent of other markers.
    Restitution reduces both the normal speed and the remaining drift distance.
    """
    position, speed_factor, fraction, lost = _particle_boundary_map(
        x_n, v_n[0], dx, grid, (box_size_x, box_size_y, box_size_z), BC_left, BC_right,
        mixed_BC_weight, COR_left, COR_right, max_vx, jnp.all((q == 0) & (m == 0)))
    velocity = jnp.array([speed_factor*v_n[0], v_n[1], v_n[2]])
    return position, jnp.where(lost, 0., velocity), q*fraction, jnp.where(lost, 0., q_m), m*fraction


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
    """Map positions with unit restitution/return and no directional wall impact.

    Positions alone cannot determine restitution, returned weight or collection
    history. Use the full-state mapper for those laws and exact outgoing impacts.
    """
    return _particle_boundary_map(x_n, 0., dx, grid, (box_size_x, box_size_y, box_size_z), BC_left, BC_right)[0]

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
    return vmap(lambda x_n: set_BC_single_particle_positions(
        x_n, dx, grid, box_size_x, box_size_y, box_size_z, BC_left, BC_right))(xs_n)


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
