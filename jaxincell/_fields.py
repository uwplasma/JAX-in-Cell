from jax import jit
import jax.numpy as jnp
from ._sources import calculate_charge_density
from ._boundary_conditions import field_ghost_cells_E, field_ghost_cells_B
from ._constants import epsilon_0, speed_of_light

__all__ = ['E_from_Gauss_1D_FFT', 'E_from_Poisson_1D_FFT', 'E_from_Gauss_1D_Cartesian', 'curlE', 'curlB', 'field_update']

@jit
def E_from_Gauss_1D_FFT(charge_density, dx):
    """
    Solve for the electric field E = -d(phi)/dx using FFT, 
    where phi is derived from the 1D Gauss' law equation.
    Parameters:
    charge_density : 1D numpy array, source term (right-hand side of Poisson equation)
    dx : float, grid spacing in the x-direction
    Returns:
    E : 1D numpy array, electric field
    """
    # Get the number of grid points
    nx = len(charge_density)
    # Create wavenumbers in Fourier space (k_x)
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    # Perform 1D FFT of the source term
    charge_density_k = jnp.fft.fft(charge_density)
    # Avoid division by zero for the k = 0 mode
    kx = kx.at[0].set(1.0)  # Prevent division by zero
    # Solve for the electric field in Fourier space
    E_k = -1j * charge_density_k / kx / epsilon_0  # Electric field in Fourier space
    # Inverse FFT to transform back to spatial domain
    E = jnp.fft.ifft(E_k).real
    return E

@jit
def E_from_Poisson_1D_FFT(charge_density, dx):
    """
    Solve for the electric field E = -d(phi)/dx using FFT, 
    where phi is derived from the 1D Poisson equation.
    Parameters:
    charge_density : 1D numpy array, source term (right-hand side of Poisson equation)
    dx : float, grid spacing in the x-direction
    Returns:
    E : 1D numpy array, electric field
    """
    # Get the number of grid points
    nx = len(charge_density)
    # Create wavenumbers in Fourier space (k_x)
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    # Perform 1D FFT of the source term
    charge_density_k = jnp.fft.fft(charge_density)
    # Avoid division by zero for the k = 0 mode
    kx = kx.at[0].set(1.0)  # Prevent division by zero
    # Solve Poisson equation in Fourier space
    phi_k = -charge_density_k / kx**2 / epsilon_0
    # Set the k = 0 mode of phi_k to 0 to ensure a zero-average solution
    phi_k = phi_k.at[0].set(0.0)
    # Compute electric field from potential in Fourier space
    E_k = 1j * kx * phi_k
    # Inverse FFT to transform back to spatial domain
    E = jnp.fft.ifft(E_k).real
    return E

@jit
def E_from_Gauss_1D_Cartesian(charge_density, dx):
    """
    Solve for the electric field at t=0 (E0) using the charge density distribution 
    and applying Gauss's law in a 1D system.

    Args:
        charge_density : 1D numpy array, source term (right-hand side of Gauss equation)
        dx : float, grid spacing in the x-direction
    
    Returns:
        array: The electric field at each grid point due to the particles, shape (G,).
    """
    # Construct divergence matrix for solving Gauss' Law
    divergence_matrix = jnp.diag(jnp.ones(len(charge_density)))-jnp.diag(jnp.ones(len(charge_density)-1),k=-1)
    divergence_matrix.at[0,-1].set(-1)
    
    # Solve for the electric field using Gauss' law in the 1D case
    E_field_from_Gauss = (dx / epsilon_0) * jnp.linalg.solve(divergence_matrix, charge_density)
    return E_field_from_Gauss


@jit
def curlE(E_field, B_field, dx, dt, field_BC_left, field_BC_right):
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

    Returns:
        array: The curl of the electric field, which is the source of the magnetic field.
    """
    # Set ghost cells for boundary conditions
    ghost_cell_L, ghost_cell_R = field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field)
    E_field = jnp.insert(E_field, 0, ghost_cell_L, axis=0)
    E_field = jnp.append(E_field, jnp.array([ghost_cell_R]), axis=0)
    
    # Compute the curl using the finite difference approximation for 1D (only d/dx)
    dFz_dx = (E_field[1:-1, 2] - E_field[0:-2, 2]) / dx
    dFy_dx = (E_field[1:-1, 1] - E_field[0:-2, 1]) / dx

    # Return the curl in the 3D vector form (only z and y components are non-zero)
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)), -dFz_dx, dFy_dx]))


@jit
def curlB(B_field, E_field, dx, dt, field_BC_left, field_BC_right):
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

    Returns:
        array: The curl of the magnetic field, which is the source of the electric field.
    """
    # Set ghost cells for boundary conditions
    ghost_cell_L, ghost_cell_R = field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field)
    B_field = jnp.insert(B_field, 0, ghost_cell_L, axis=0)
    B_field = jnp.append(B_field, jnp.array([ghost_cell_R]), axis=0)

    #If taking E_i = B_(i+1) - B_i (since B-fields defined on centers), roll by -1 first. 
    B_field = jnp.roll(B_field, -1, axis=0)

    # Compute the curl using the finite difference approximation for 1D (only d/dx)
    dFz_dx = (B_field[1:-1, 2] - B_field[0:-2, 2]) / dx
    dFy_dx = (B_field[1:-1, 1] - B_field[0:-2, 1]) / dx

    # Return the curl in the 3D vector form (only z and y components are non-zero)
    return jnp.transpose(jnp.array([jnp.zeros(len(dFz_dx)), -dFz_dx, dFy_dx]))

@jit
def field_update(E_fields, B_fields, dx, dt, j, field_BC_left, field_BC_right):
    """
    Update the electric and magnetic fields based on Maxwell's equations

    Args:
        E_fields (array): Electric field at each grid point, shape (G, 3).
        B_fields (array): Magnetic field at each grid point, shape (G, 3).
        dx (float): Grid spacing.
        dt (float): Time step.
        j (array): Current density at each grid point, shape (G, 3).
        field_BC_left (int): Left boundary condition for fields.
        field_BC_right (int): Right boundary condition for fields.

    Returns:
        tuple: Updated electric and magnetic fields, each of shape (G, 3).
    """
    # Ampère's law with Maxwell's correction
    curl_E = curlE(E_fields, B_fields, dx, dt, field_BC_left, field_BC_right)

    # Faraday's law
    curl_B = curlB(B_fields, E_fields, dx, dt, field_BC_left, field_BC_right)
    
    # Update the Fields
    B_fields -= dt*curl_E
    E_fields += dt*((speed_of_light**2)*curl_B-(j/epsilon_0))

    return E_fields, B_fields
