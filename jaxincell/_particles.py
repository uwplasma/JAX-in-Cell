from jax import vmap, jit
import jax.numpy as jnp
from ._constants import speed_of_light as c
from ._sources import get_S2_weights_and_indices_periodic_CN, get_S2_weights_and_indices
__all__ = ['fields_to_particles_grid', 'fields_to_particles_periodic_CN','rotation', 'boris_step', 'boris_step_relativistic']


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
    x = x_n[0]
    grid_size = field.shape[0]

    # Fast S2 gather with BC support (no ghost-cell array building per particle)
    idx, w, active = get_S2_weights_and_indices(
        x, dx, grid_start, grid_size, field_BC_left, field_BC_right
    )
    w = jnp.where(active, w, 0.0)

    # field[idx] has shape (3,3) if field is (G,3) and idx is (3,)
    return jnp.tensordot(w, field[idx], axes=(0, 0))

@jit
def fields_to_particles_periodic_CN(x_n, field, dx, grid_start):
    """
    Interpolates field to particle using Periodic BCs.
    Args:
        field: The field array (size N).
        grid_start: Physical position of field[0].
    """
    x = x_n[0]
    grid_size = field.shape[0]
    
    # Get wrapped indices and weights
    indices, weights = get_S2_weights_and_indices_periodic_CN(x, dx, grid_start, grid_size)
    
    # Gather (Dot Product)
    # Since indices are already wrapped (0 to N-1), this is safe.
    E_particle = jnp.dot(weights, field[indices])
    
    return E_particle


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
    # q_ms: (N,1) or (N,)
    qm = q_ms[:, 0] if (q_ms.ndim == 2) else q_ms  # (N,)

    v_minus = vs_n + (qm[:, None] * E_fields_at_x) * (dt / 2)

    t = (qm[:, None] * B_fields_at_x) * (dt / 2)             # (N,3)
    t2 = jnp.sum(t * t, axis=1, keepdims=True)               # (N,1)
    s = 2 * t / (1.0 + t2)                                   # (N,3)

    v_prime = v_minus + jnp.cross(v_minus, t)
    v_plus  = v_minus + jnp.cross(v_prime, s)

    v_new = v_plus + (qm[:, None] * E_fields_at_x) * (dt / 2)
    x_new = xs_nplushalf + dt * v_new
    return x_new, v_new

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
    q_s = q_s[:, 0] if (q_s.ndim == 2) else q_s
    m_s = m_s[:, 0] if (m_s.ndim == 2) else m_s

    def single_particle_step(x, v, q, m, E, B):
        # Compute initial momentum
        v2 = jnp.sum((v / c) ** 2)
        v2 = jnp.minimum(v2, 1.0 - 1e-15)
        gamma_n = 1.0 / jnp.sqrt(1.0 - v2)

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