from jax import jit, lax, vmap
import jax.numpy as jnp
from ._boundary_conditions import field_ghost_cells_E, field_ghost_cells_B
from ._constants import epsilon_0, speed_of_light
from ._metric_tensor import metric_bundle

__all__ = ['E_from_Gauss_1D_FFT', 'E_from_Poisson_1D_FFT', 'E_from_Gauss_1D_Cartesian', 'curlE',
           'curlB', 'field_update', 'field_update1', 'field_update2']

@jit
def _geom_at_grid(t, x_grid, metric_cfg):
    """
    For a diagonal, zero-shift metric g = diag(g00, g11, g22, g33):
      alpha = sqrt(-g00)/c           (lapse)
      S = (sqrt(g11), sqrt(g22), sqrt(g33))   (orthonormal scales)
    Vectorized over x_grid (shape (G,)).
    Returns:
      alpha: (G,)
      S:     (G,3)

    For diagonal, zero-shift metrics: return alpha(x)=sqrt(-g00)/c and
    the spatial scale factors S_i(x)=sqrt(g_ii). Vectorized over x_grid.
    """
    def one_x(x):
        mb = metric_bundle(t, x, metric_cfg["kind"], speed_of_light, **metric_cfg.get("params", {}))
        g  = mb["g"]
        alpha = jnp.sqrt(-g[0,0]) / speed_of_light
        Sx = jnp.sqrt(g[1,1]); Sy = jnp.sqrt(g[2,2]); Sz = jnp.sqrt(g[3,3])
        return alpha, jnp.array([Sx, Sy, Sz])
    alphas, S = vmap(one_x)(x_grid)       # alphas: (G,), S: (G,3)
    return alphas, S

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
def curlE(E_field, B_field, dx, dt, field_BC_left, field_BC_right, invS1=None):
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


    Local-frame curl for 1D (x-only variation). If invS1 is provided (shape (G,)),
    we use d/dx_phys = invS1 * d/dx; otherwise assume invS1 = 1 (flat x metric).
    """
    if invS1 is None:
        invS1 = 1.0

    # ghost cells for E
    ghost_L, ghost_R = field_ghost_cells_E(field_BC_left, field_BC_right, E_field, B_field)
    E = jnp.insert(E_field, 0, ghost_L, axis=0)
    E = jnp.append(E, jnp.array([ghost_R]), axis=0)

    dFz_dx = (E[1:-1, 2] - E[0:-2, 2]) / dx
    dFy_dx = (E[1:-1, 1] - E[0:-2, 1]) / dx

    dFz_dx = invS1 * dFz_dx
    dFy_dx = invS1 * dFy_dx

    return jnp.stack([jnp.zeros_like(dFz_dx), -dFz_dx, dFy_dx], axis=1)


@jit
def curlB(B_field, E_field, dx, dt, field_BC_left, field_BC_right, invS1=None):
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

    Local-frame curl for 1D (x-only variation). If invS1 is provided, uses physical derivative.
    """
    if invS1 is None:
        invS1 = 1.0

    ghost_L, ghost_R = field_ghost_cells_B(field_BC_left, field_BC_right, B_field, E_field)
    B = jnp.insert(B_field, 0, ghost_L, axis=0)
    B = jnp.append(B, jnp.array([ghost_R]), axis=0)

    # stagger handling
    B = jnp.roll(B, -1, axis=0)

    dFz_dx = (B[1:-1, 2] - B[0:-2, 2]) / dx
    dFy_dx = (B[1:-1, 1] - B[0:-2, 1]) / dx

    dFz_dx = invS1 * dFz_dx
    dFy_dx = invS1 * dFy_dx

    return jnp.stack([jnp.zeros_like(dFz_dx), -dFz_dx, dFy_dx], axis=1)

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
def field_update1(E_fields, B_fields, dx, dt, J, field_BC_left, field_BC_right, grid, t, metric_cfg):
    """
    Local-frame Maxwell update (Ampère then Faraday) for diagonal, zero-shift metrics.
      E_loc = S * E,  B_loc = S * B,  J_loc = S * J
      dt_loc = α dt
      ∂t_loc E_loc =  c^2 curl_phys B_loc - J_loc/ε0
      ∂t_loc B_loc = -      curl_phys E_loc
    with curl_phys ≡ (1/S1) d/dx applied by passing invS1 to curl stencils.
    """
    alphas, S = _geom_at_grid(t, grid, metric_cfg)   # (G,), (G,3)
    S1 = S[:, 0]; invS1 = 1.0 / (S1 + 1e-30)
    dt_loc = (alphas[:, None]) * dt

    # Map to local orthonormal frame
    E_loc = S * E_fields
    B_loc = S * B_fields
    J_loc = S * J

    # Ampère (local)
    curl_B_loc = curlB(B_loc, E_loc, dx, dt, field_BC_left, field_BC_right, invS1=invS1)
    E_loc = E_loc + dt_loc * ( (speed_of_light**2) * curl_B_loc - J_loc/epsilon_0 )

    # Faraday (local)
    curl_E_loc = curlE(E_loc, B_loc, dx, dt, field_BC_left, field_BC_right, invS1=invS1)
    B_loc = B_loc - dt_loc * ( curl_E_loc )

    # Map back to coordinate components
    invS = 1.0 / (S + 1e-30)
    E = invS * E_loc
    B = invS * B_loc
    return E, B

@jit
def field_update2(E_fields, B_fields, dx, dt, J, field_BC_left, field_BC_right, grid, t, metric_cfg):
    """
    Same as field_update1 but Faraday first, then Ampère (leapfrog symmetry).
    """
    alphas, S = _geom_at_grid(t, grid, metric_cfg)
    S1 = S[:, 0]; invS1 = 1.0 / (S1 + 1e-30)
    dt_loc = (alphas[:, None]) * dt

    E_loc = S * E_fields
    B_loc = S * B_fields
    J_loc = S * J

    # Faraday first
    curl_E_loc = curlE(E_loc, B_loc, dx, dt, field_BC_left, field_BC_right, invS1=invS1)
    B_loc = B_loc - dt_loc * ( curl_E_loc )

    # Then Ampère
    curl_B_loc = curlB(B_loc, E_loc, dx, dt, field_BC_left, field_BC_right, invS1=invS1)
    E_loc = E_loc + dt_loc * ( (speed_of_light**2) * curl_B_loc - J_loc/epsilon_0 )

    invS = 1.0 / (S + 1e-30)
    E = invS * E_loc
    B = invS * B_loc
    return E, B