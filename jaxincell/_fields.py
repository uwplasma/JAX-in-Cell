from jax import jit, lax
import jax.numpy as jnp
from ._boundary_conditions import field_ghost_cells_E, field_ghost_cells_B
from ._constants import epsilon_0, speed_of_light

__all__ = ['curlE', 'curlB', 'field_update', 'field_update1', 'field_update2', 'enforce_gauss_1d']

@jit
def delta_Ex_from_div_error_1D_FFT(Ex, charge_density, dx):
    """
    Periodic divergence-error corrector for backward-difference div:
      g = (Ex - roll(Ex,1))/dx - rho/eps0
      Find δEx s.t. (δEx - roll(δEx,1))/dx = -g, with zero-mean δEx.
    """
    nx = charge_density.shape[0]

    # divergence error
    g = (Ex - jnp.roll(Ex, 1)) / dx - charge_density / epsilon_0

    # project out k=0 (compatibility) to avoid any residual mean component
    g = g - jnp.mean(g)

    g_k = jnp.fft.fft(g)
    kx = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi

    # Backward-difference symbol
    Db = (1.0 - jnp.exp(-1j * kx * dx)) / dx
    Db = Db.at[0].set(1.0 + 0.0j)         # avoid divide-by-zero

    dEx_k = -g_k / Db
    dEx_k = dEx_k.at[0].set(0.0 + 0.0j)   # enforce zero-mean δEx gauge

    dEx = jnp.fft.ifft(dEx_k).real
    dEx = dEx - jnp.mean(dEx)             # robust zero-mean in real space
    return dEx

@jit
def enforce_gauss_1d(E_field, charge_density, dx, field_BC_left, field_BC_right,
                     relax=1.0, neutralize_periodic=True):
    """
    Enforce div(E)=rho/eps0 by correcting Ex with a 1D Gauss-law-consistent field.

    - Periodic: FFT-based (zero-mean) solve (fast, spectrally accurate).
    - Nonperiodic: integrate from left boundary with Ex(left)=0 reference.
      (O(G), robust for reflective/absorbing/custom-like cases.)

    relax in [0,1]: 1 = full correction, <1 = partial (smaller energy kick).
    neutralize_periodic: subtract mean(rho) for periodic compatibility (k=0).

    Only Ex is modified; Ey/Ez are preserved.
    """
    periodic = (field_BC_left == 0) & (field_BC_right == 0)

    def _periodic(_):
        Ex0 = E_field[:, 0]
        rho = charge_density
        rho = lax.cond(neutralize_periodic, lambda r: r - jnp.mean(r), lambda r: r, rho)

        dEx = delta_Ex_from_div_error_1D_FFT(Ex0, rho, dx)
        Ex1 = Ex0 + relax * dEx
        return E_field.at[:, 0].set(Ex1)

    def _nonperiodic(_):
        # target Ex consistent with backward-div and Ex(-1)=0 reference
        rhs = dx * charge_density / epsilon_0
        Ex_tgt = jnp.cumsum(rhs)
        Ex0 = E_field[:, 0]
        Ex1 = Ex0 + relax * (Ex_tgt - Ex0)
        return E_field.at[:, 0].set(Ex1)

    return lax.cond(periodic, _periodic, _nonperiodic, operand=None)


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

    # Backward difference on staggered E: dE/dx at B locations
    # d/dx E[i] ≈ (E[i] - E[i-1]) / dx
    E_im1 = jnp.concatenate([ghost_cell_L[None, :], E_field[:-1]], axis=0)
    
    # Compute the curl using the finite difference approximation for 1D (only d/dx)
    dEz_dx = (E_field[:, 2] - E_im1[:, 2]) / dx
    dEy_dx = (E_field[:, 1] - E_im1[:, 1]) / dx

    # Return the curl in the 3D vector form (only z and y components are non-zero)
    return jnp.transpose(jnp.array([jnp.zeros_like(dEz_dx), -dEz_dx, dEy_dx]))


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

    # Forward difference on staggered B: dB/dx at E locations
    # d/dx B[i] ≈ (B[i+1] - B[i]) / dx, with ghost_R providing B[G]
    B_ip1 = jnp.concatenate([B_field[1:], ghost_cell_R[None, :]], axis=0)

    # Compute the curl using the finite difference approximation for 1D (only d/dx)
    dBz_dx = (B_ip1[:, 2] - B_field[:, 2]) / dx
    dBy_dx = (B_ip1[:, 1] - B_field[:, 1]) / dx

    # Return the curl in the 3D vector form (only z and y components are non-zero)
    return jnp.transpose(jnp.array([jnp.zeros_like(dBz_dx), -dBz_dx, dBy_dx]))

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