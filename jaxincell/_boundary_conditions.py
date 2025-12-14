from jax import jit, vmap
import jax.numpy as jnp
from ._constants import speed_of_light

__all__ = ['set_BC_particles', 'set_BC_positions']

@jit
def set_BC_particles(xs, vs, qs, ms, q_ms, dx, grid, Lx, Ly, Lz, BC_left, BC_right):
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
    x, y, z = xs[:,0], xs[:,1], xs[:,2]

    # periodic y/z
    y = (y + Ly/2) % Ly - Ly/2
    z = (z + Lz/2) % Lz - Lz/2

    left  = x < -Lx/2
    right = x >  Lx/2

    # periodic wrap for x (only where out of bounds)
    x_per = (x + Lx/2) % Lx - Lx/2

    # reflective
    x_ref_left  = -Lx - x
    x_ref_right =  Lx - x

    # absorbing “park outside”
    x_abs_left  = grid[0] - 1.5*dx
    x_abs_right = grid[-1] + 3.0*dx

    # choose per side
    xL = jnp.where(BC_left == 0, x_per,
         jnp.where(BC_left == 1, x_ref_left,  x_abs_left))
    xR = jnp.where(BC_right == 0, x_per,
         jnp.where(BC_right == 1, x_ref_right, x_abs_right))

    x = jnp.where(left,  xL, x)
    x = jnp.where(right, xR, x)

    # velocity update on reflective / absorbing
    flipx = jnp.array([-1.0, 1.0, 1.0])
    vL = jnp.where(BC_left  == 1, vs * flipx, jnp.zeros_like(vs))
    vR = jnp.where(BC_right == 1, vs * flipx, jnp.zeros_like(vs))
    vs = jnp.where(left[:,None],  jnp.where(BC_left==0, vs, vL), vs)
    vs = jnp.where(right[:,None], jnp.where(BC_right==0, vs, vR), vs)

    # zero charge/qm if absorbing
    absorb = (left & (BC_left==2)) | (right & (BC_right==2))
    qs   = jnp.where(absorb[:,None], 0.0, qs)
    q_ms = jnp.where(absorb[:,None], 0.0, q_ms)

    xs = jnp.stack([x,y,z], axis=1)
    return xs, vs, qs, ms, q_ms


@jit
def set_BC_positions(xs_n, qs, dx, grid, Lx, Ly, Lz, BC_left, BC_right,
                     zero_q_on_absorb=False):
    """
    Vectorized positions-only BCs (half-step). Same logic as set_BC_particles,
    but only updates xs (and optionally qs if absorbing).
    xs_n: (N,3)
    qs:   (N,) or (N,1)  (only used if zero_q_on_absorb=True)
    """
    x, y, z = xs_n[:, 0], xs_n[:, 1], xs_n[:, 2]

    # periodic y/z
    y = (y + Ly / 2) % Ly - Ly / 2
    z = (z + Lz / 2) % Lz - Lz / 2

    left  = x < -Lx / 2
    right = x >  Lx / 2

    # periodic wrap for x
    x_per = (x + Lx / 2) % Lx - Lx / 2

    # reflective maps
    x_ref_left  = -Lx - x
    x_ref_right =  Lx - x

    # absorbing “park outside”
    x_abs_left  = grid[0]  - 1.5 * dx
    x_abs_right = grid[-1] + 3.0 * dx

    # choose x value for each side depending on BC type
    xL = jnp.where(BC_left  == 0, x_per,
         jnp.where(BC_left  == 1, x_ref_left,  x_abs_left))
    xR = jnp.where(BC_right == 0, x_per,
         jnp.where(BC_right == 1, x_ref_right, x_abs_right))

    x = jnp.where(left,  xL, x)
    x = jnp.where(right, xR, x)

    xs_n = jnp.stack([x, y, z], axis=1)

    # Optional: zero charge for absorbing (useful if you “kill” particles at half-step)
    if zero_q_on_absorb:
        absorb = (left & (BC_left == 2)) | (right & (BC_right == 2))
        qs = jnp.where(absorb.reshape(-1, 1) if qs.ndim == 2 else absorb, 0.0, qs)
        return xs_n, qs

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
