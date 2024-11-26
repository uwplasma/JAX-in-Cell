from jax import jit
import jax.numpy as jnp
from sources import calculate_charge_density
from boundary_conditions import field_ghost_cells_E, field_ghost_cells_B
from constants import epsilon_0, speed_of_light

@jit
def E_from_Poisson_equation(xs_n, qs, dx, grid, part_BC_left, part_BC_right):
    """
    Solve for the electric field at t=0 (E0) using the charge density distribution 
    and applying Gauss's law in a 1D system.

    Args:
        xs_n (array): Particle positions at a given time (t), shape (N,).
        qs (array): Particle charges, shape (N,).
        dx (float): Grid spacing in meters.
        grid (array): Grid points in space, shape (G,).
        part_BC_left (int): Boundary condition at the left boundary (0: periodic, 1: reflective, 2: absorbing).
        part_BC_right (int): Boundary condition at the right boundary (0: periodic, 1: reflective, 2: absorbing).
    
    Returns:
        array: The electric field at each grid point due to the particles, shape (G,).
    """
    # Calculate charge density from particles
    charge_density = calculate_charge_density(xs_n, qs, dx, grid, part_BC_left, part_BC_right)
    
    # Construct divergence matrix for solving Poisson equation (Gauss's Law)
    divergence_matrix = jnp.diag(jnp.ones(len(grid)))-jnp.diag(jnp.ones(len(grid)-1),k=-1)
    
    # Solve for the electric field using Poisson's equation in the 1D case
    E_field_from_Poisson = (dx / epsilon_0) * jnp.linalg.solve(divergence_matrix, charge_density)
    return E_field_from_Poisson


@jit
def curlE(E_field, B_field, dx, dt, field_BC_left, field_BC_right, current_t, E0, k):
    """
    Compute the curl of the electric field, which is related to the time derivative of 
    the magnetic field in Maxwell's equations (Faraday's law).

    Args:
        E_field (array): Electric field at each grid point, shape (G, 3).
        B_field (array): Magnetic field at each grid point, shape (G, 3).
        dx (float): Grid spacing.
        dt (float): Time step.
        field_BC_left (int): Left boundary condition for fields (0: periodic, 1: reflective, 2: absorbing).
        field_BC_right (int): Right boundary condition for fields.
        current_t (float): Current time step.
        E0 (float): Initial electric field magnitude (optional, used for field scaling).
        k (int): A constant or time step factor.

    Returns:
        array: The curl of the electric field, which is the source of the magnetic field.
    """
    # Set ghost cells for boundary conditions
    ghost_cell_L, ghost_cell_R = field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field, dx, current_t, E0, k)
    E_field = jnp.insert(E_field, 0, ghost_cell_L, axis=0)
    E_field = jnp.append(E_field, jnp.array([ghost_cell_R]), axis=0)

    # Compute the curl using the finite difference approximation for 1D (only d/dx)
    dFz_dx = (E_field[1:-1, 2] - E_field[0:-2, 2]) / dx
    dFy_dx = (E_field[1:-1, 1] - E_field[0:-2, 1]) / dx

    # Return the curl in the 3D vector form (only z and y components are non-zero)
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)), -dFz_dx, dFy_dx]))


@jit
def curlB(B_field, E_field, dx, dt, field_BC_left, field_BC_right, current_t, B0, k):
    """
    Compute the curl of the magnetic field, which is related to the time derivative of 
    the electric field in Maxwell's equations (Ampère's law with Maxwell correction).

    Args:
        B_field (array): Magnetic field at each grid point, shape (G, 3).
        E_field (array): Electric field at each grid point, shape (G, 3).
        dx (float): Grid spacing.
        dt (float): Time step.
        field_BC_left (int): Left boundary condition for fields.
        field_BC_right (int): Right boundary condition for fields.
        current_t (float): Current time step.
        B0 (float): Initial magnetic field magnitude (optional).
        k (int): A constant or time step factor.

    Returns:
        array: The curl of the magnetic field, which is the source of the electric field.
    """
    # Set ghost cells for boundary conditions
    ghost_cell_L, ghost_cell_R = field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field, dx, current_t, B0, k)
    B_field = jnp.insert(B_field, 0, ghost_cell_L, axis=0)
    B_field = jnp.append(B_field, jnp.array([ghost_cell_R]), axis=0)

    # Roll B-field to align with the correct grid positions for curl calculation
    B_field = jnp.roll(B_field, -1, axis=0)

    # Compute the curl using the finite difference approximation for 1D (only d/dx)
    dFz_dx = (B_field[1:-1, 2] - B_field[0:-2, 2]) / dx
    dFy_dx = (B_field[1:-1, 1] - B_field[0:-2, 1]) / dx

    # Return the curl in the 3D vector form (only z and y components are non-zero)
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)), -dFz_dx, dFy_dx]))


@jit
def field_update1(E_fields, B_fields, dx, dt_2, j, field_BC_left, field_BC_right, current_t, E0=0, k=0):
    """
    Update the electric and magnetic fields based on Maxwell's equations using the
    half-step field update algorithm. Update first the electric field and then the magnetic field.

    Args:
        E_fields (array): Electric field at each grid point, shape (G, 3).
        B_fields (array): Magnetic field at each grid point, shape (G, 3).
        dx (float): Grid spacing.
        dt_2 (float): Half time step.
        j (array): Current density at each grid point, shape (G, 3).
        field_BC_left (int): Left boundary condition for fields.
        field_BC_right (int): Right boundary condition for fields.
        current_t (float): Current time step.
        E0 (float): Initial electric field magnitude (optional).
        k (int): A constant or time step factor.

    Returns:
        tuple: Updated electric and magnetic fields, each of shape (G, 3).
    """
    # Update the magnetic field (Faraday's law)
    curl_B = curlB(B_fields, E_fields, dx, dt_2, field_BC_left, field_BC_right, current_t, E0 / speed_of_light, k)
    E_fields += dt_2*((speed_of_light**2)*curl_B-(j/epsilon_0))

    # Update the electric field (Ampère's law with Maxwell's correction)
    curl_E = curlE(E_fields, B_fields, dx, dt_2, field_BC_left, field_BC_right, current_t, E0, k)
    B_fields -= dt_2*curl_E

    return E_fields, B_fields

@jit
def field_update2(E_fields, B_fields, dx, dt_2, j, field_BC_left, field_BC_right, current_t, E0=0, k=0):
    """
    Update the electric and magnetic fields based on Maxwell's equations using the
    half-step field update algorithm. Update first the magnetic field and then the electric field.

    Args:
        E_fields (array): Electric field at each grid point, shape (G, 3).
        B_fields (array): Magnetic field at each grid point, shape (G, 3).
        dx (float): Grid spacing.
        dt_2 (float): Half time step.
        j (array): Current density at each grid point, shape (G, 3).
        field_BC_left (int): Left boundary condition for fields.
        field_BC_right (int): Right boundary condition for fields.
        current_t (float): Current time step.
        E0 (float): Initial electric field magnitude (optional).
        k (int): A constant or time step factor.

    Returns:
        tuple: Updated electric and magnetic fields, each of shape (G, 3).
    """
    # Update the electric field (Ampère's law with Maxwell's correction)
    curl_E = curlE(E_fields, B_fields, dx, dt_2, field_BC_left, field_BC_right, current_t, E0, k)
    B_fields -= dt_2*curl_E

    # Update the magnetic field (Faraday's law)
    curl_B = curlB(B_fields, E_fields, dx, dt_2, field_BC_left, field_BC_right, current_t, E0 / speed_of_light, k)
    E_fields += dt_2*((speed_of_light**2)*curl_B-(j/epsilon_0))

    return E_fields, B_fields
