from jax import jit, lax
import jax.numpy as jnp
from functools import partial
from ._sources import calculate_charge_density
from ._boundary_conditions import field_ghost_cells_E, field_ghost_cells_B
from ._constants import epsilon_0, speed_of_light

__all__ = [
    'E_from_Gauss_1D_Cartesian',
    'curlE',
    'curlB',
    'field_update',
    'field_update1',
    'field_update2',
    'project_E_to_satisfy_Gauss',
]

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

@jit
def field_update1(E_fields, B_fields, dx, dt, j, field_BC_left, field_BC_right):
    #First, update E (Ampere's)
    curl_B = curlB(B_fields, E_fields, dx, dt, field_BC_left, field_BC_right)
    E_fields += dt*((speed_of_light**2)*curl_B-(j/epsilon_0))
    #Then, update B (Faraday's)
    curl_E = curlE(E_fields, B_fields, dx, dt, field_BC_left, field_BC_right)
    B_fields -= dt*curl_E
    return E_fields,B_fields

@jit
def field_update2(E_fields, B_fields, dx, dt, j, field_BC_left, field_BC_right):
    #First, update B (Faraday's)
    curl_E = curlE(E_fields, B_fields, dx, dt, field_BC_left, field_BC_right)
    B_fields -= dt*curl_E
    #Then, update E (Ampere's)
    curl_B = curlB(B_fields, E_fields, dx, dt, field_BC_left, field_BC_right)
    E_fields += dt*((speed_of_light**2)*curl_B-(j/epsilon_0))
    return E_fields,B_fields


# ---------------------------------------------------------------------------
#  Projection of E_x to enforce Gauss's law after a full EM Maxwell update
# ---------------------------------------------------------------------------

@jit
def _project_E_fft(E_field, charge_density, dx):
    """
    FFT-based Gauss projection in 1D (periodic), but using the SAME
    discrete divergence operator as in diagnostics/_project_E_cartesian:

        div(E)_i = (E_i - E_{i-1}) / dx

    In Fourier space, the symbol of this backward-difference operator is
        D(k) = (1 - exp(-i k dx)) / dx.

    We enforce  D(k) * E_k = rho_k / epsilon_0  for k != 0,
    and keep the k=0 (DC) Ex mode unchanged.
    """

    Ex = E_field[:, 0]            # (G,)
    nx = Ex.shape[0]

    # Wavenumbers
    kx = jnp.fft.fftfreq(nx, d=dx) * 2.0 * jnp.pi

    # Backward-difference symbol in Fourier space
    D = (1.0 - jnp.exp(-1j * kx * dx)) / dx  # shape (G,)

    # FFT of Ex and rho
    Ex_k  = jnp.fft.fft(Ex)
    rho_k = jnp.fft.fft(charge_density)

    # Avoid division by zero at k=0
    D_safe = jnp.where(jnp.abs(D) == 0.0, 1.0 + 0.0j, D)

    # Solve D(k) * Ex_new_k = rho_k / epsilon_0
    Ex_new_k = rho_k / (epsilon_0 * D_safe)

    # Preserve the DC component from the original Ex (does not affect Gauss)
    Ex_new_k = Ex_new_k.at[0].set(Ex_k[0])

    # Back to real space
    Ex_new = jnp.fft.ifft(Ex_new_k).real

    # Optional: preserve mean Ex to avoid artificial DC changes
    Ex_new = Ex_new - jnp.mean(Ex_new) + jnp.mean(Ex)

    return E_field.at[:, 0].set(Ex_new)

@jit
def _project_E_cartesian(E_field, charge_density, dx):
    Ex = E_field[:, 0]
    nx = Ex.shape[0]

    # Build the same D Ex as in E_from_Gauss_1D_Cartesian, but without forming the matrix
    def div_op(E):
        # periodic backward difference
        Em1 = jnp.roll(E, 1)
        return E - Em1

    divE_star = div_op(Ex)
    rhs = (dx / epsilon_0) * charge_density

    # We want D (Ex + deltaE) = rhs => D deltaE = rhs - D Ex
    s = rhs - divE_star

    # Enforce compatibility (sum s = 0) by removing the mean
    s = s - jnp.mean(s)

    # Solve D deltaE = s with cumulative sum:
    # deltaE_0 = 0, deltaE_i = deltaE_{i-1} + s_i
    deltaE = jnp.cumsum(s)

    # Remove the mean of deltaE to avoid introducing a DC shift
    deltaE = deltaE - jnp.mean(deltaE)

    Ex_new = Ex + deltaE
    return E_field.at[:, 0].set(Ex_new)


@partial(jit, static_argnums=(3,))
def project_E_to_satisfy_Gauss(E_field, charge_density, dx, method):
    """
    Full EM + Gauss projection step.

    Parameters
    ----------
    E_field : (G, 3) array
        Electric field after a full Maxwell update (Ampère + Faraday).
    charge_density : (G,) array
        Charge density ρ at the same time level as this E_field.
    dx : float
        Grid spacing.
    method : int
        0 : no projection (NOT used here; handled in caller)
        1 : FFT-based projection (spectral divergence / Laplacian)
        2 : direct-space projection (divergence matrix, Cartesian)
        3 : alias for 1 (FFT) for backward compatibility

    Returns
    -------
    E_field_projected : (G, 3) array
        Same as E_field but with E_x minimally corrected so that
        ∇·E = ρ/ε₀ (in the chosen discrete sense).
    """
    # Map method -> index for lax.switch:
    #   1 -> 0  (FFT)
    #   2 -> 1  (Cartesian)
    #   3 -> 2  (FFT again, as alias)
    idx = jnp.clip(method - 1, 0, 2)

    return lax.switch(
        idx,
        (
            _project_E_fft,        # idx 0 => method 1
            _project_E_cartesian,  # idx 1 => method 2
            _project_E_fft,        # idx 2 => method 3 (alias FFT)
        ),
        E_field,
        charge_density,
        dx,
    )