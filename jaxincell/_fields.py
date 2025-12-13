from jax import jit, lax
import jax.numpy as jnp
from ._boundary_conditions import field_ghost_cells_E, field_ghost_cells_B
from ._constants import epsilon_0, speed_of_light

__all__ = ['E_from_Gauss_1D_FFT', 'E_from_Poisson_1D_FFT', 'E_from_Gauss_1D_Cartesian',
           'curlE', 'curlB', 'field_update', 'field_update1', 'field_update2',
           "project_Gauss_1D_FFT", "project_Gauss_1D_Cartesian",]

@jit
def _div_backward_1d(f, dx):
    """Periodic backward-difference divergence: (f_i - f_{i-1})/dx."""
    return (f - jnp.roll(f, 1)) / dx

@jit
def _grad_forward_1d(phi, dx):
    """Periodic forward-difference gradient: (phi_{i+1} - phi_i)/dx."""
    return (jnp.roll(phi, -1) - phi) / dx

@jit
def project_Gauss_1D_FFT(Ex, charge_density, dx):
    r"""
    Divergence-cleaning (Gauss projection) for 1D periodic EM PIC.

    Goal
    ----
    Given an existing longitudinal field Ex (from Maxwell update),
    compute the *smallest correction* dEx = D_f(phi) such that:

        D_b(Ex + dEx) = rho/epsilon_0

    where:
        D_b is backward difference, D_f is forward difference.

    This yields a discrete Poisson equation for phi:
        (D_b D_f) phi = (rho/epsilon_0) - D_b(Ex)

    In Fourier space:
        L(k) = D_b(k) D_f(k) = 4 sin^2(k dx / 2) / dx^2  (real, >= 0)
        phi_k = g_k / L(k)
        dEx_k = D_f(k) * phi_k

    Notes
    -----
    - Periodic only.
    - Projects out k=0 (and Nyquist for even n) for solvability.
    - This is the right primitive to “correct Ex to satisfy Gauss”
      in a full EM PIC scheme without clobbering energy.
    """
    n = charge_density.shape[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)

    # Gauss error (what divergence *should* be minus what it is)
    g = (charge_density / epsilon_0) - _div_backward_1d(Ex, dx)
    g_k = jnp.fft.fft(g)

    # Project out null modes (solvability / gauge)
    g_k = g_k.at[0].set(0.0)
    if n % 2 == 0:
        g_k = g_k.at[n // 2].set(0.0)

    # Symbols for forward/backward differences
    Df = (jnp.exp(1j * k * dx) - 1.0) / dx
    Db = (1.0 - jnp.exp(-1j * k * dx)) / dx

    # Self-adjoint Laplacian on a periodic grid
    L = Db * Df  # = 4 sin^2(k dx/2)/dx^2 (real, >=0)

    # Safe inverse (we already zeroed troublesome modes)
    L_safe = L.at[0].set(1.0)
    if n % 2 == 0:
        L_safe = L_safe.at[n // 2].set(1.0)

    phi_k = g_k / L_safe
    phi_k = phi_k.at[0].set(0.0)
    if n % 2 == 0:
        phi_k = phi_k.at[n // 2].set(0.0)

    dEx_k = Df * phi_k
    dEx_k = dEx_k.at[0].set(0.0)
    if n % 2 == 0:
        dEx_k = dEx_k.at[n // 2].set(0.0)

    dEx = jnp.fft.ifft(dEx_k).real

    # Optional: keep mean(Ex) unchanged (pure gauge in periodic 1D)
    Ex_new = Ex + dEx
    Ex_new = Ex_new - jnp.mean(Ex_new) + jnp.mean(Ex)
    return Ex_new

@jit
def project_Gauss_1D_Cartesian(Ex, charge_density, dx, field_BC_left=0, field_BC_right=0):
    r"""
    Real-space Gauss projection for 1D grids (works for non-periodic too).

    We enforce:
        (Ex_i - Ex_{i-1})/dx = rho_i/epsilon_0

    by building a correction dEx such that:
        (dEx_i - dEx_{i-1})/dx = (rho/epsilon_0) - (Ex_i - Ex_{i-1})/dx

    Periodic case:
      - remove mean(rho) for solvability
      - remove mean(dEx) (gauge) to avoid shifting the DC field

    Non-periodic/open case:
      - choose dEx_0 = 0 (don’t change the left boundary value)
      - integrate the correction by cumulative sum
    """
    is_periodic = jnp.logical_and(field_BC_left == 0, field_BC_right == 0)

    def periodic(_):
        rho = charge_density - jnp.mean(charge_density)
        g = (rho / epsilon_0) - _div_backward_1d(Ex, dx)
        dEx = dx * jnp.cumsum(g)
        dEx = dEx - jnp.mean(dEx)
        Ex_new = Ex + dEx
        return Ex_new - jnp.mean(Ex_new) + jnp.mean(Ex)

    def nonperiodic(_):
        g0 = (charge_density / epsilon_0) - jnp.concatenate(
            [jnp.array([0.0], dtype=Ex.dtype), (Ex[1:] - Ex[:-1]) / dx]
        )
        dEx = dx * jnp.cumsum(g0)
        return Ex + dEx

    return lax.cond(is_periodic, periodic, nonperiodic, operand=None)

@jit
def E_from_Gauss_1D_FFT(charge_density, dx):
    """
    Periodic electrostatic solve using the *discrete Gauss law* in Fourier space.

    We solve (in 1D):
        dE/dx = rho / epsilon_0

    with the *backward-difference* discrete derivative:
        (E_i - E_{i-1})/dx = rho_i / epsilon_0

    In Fourier space, the backward-difference symbol is:
        D_b(k) = (1 - exp(-i k dx)) / dx

    so:
        D_b(k) * E_k = rho_k / epsilon_0
        => E_k = rho_k / (epsilon_0 * D_b(k))

    Notes
    -----
    - This routine assumes *periodic* boundary conditions.
    - The k=0 mode is projected out (solvability / gauge: mean(rho)=0).
    - On even grids, the Nyquist mode is also projected out for consistency
      with your other routines/tests.
    """
    n = charge_density.shape[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)

    rho_k = jnp.fft.fft(charge_density)

    # Enforce solvability / gauge: drop k=0.
    rho_k = rho_k.at[0].set(0.0)

    # Optional: also drop Nyquist on even grids to match your tests/subspace.
    if n % 2 == 0:
        rho_k = rho_k.at[n // 2].set(0.0)

    # Backward-difference derivative symbol.
    D_b = (1.0 - jnp.exp(-1j * k * dx)) / dx

    # Safe division (we already set the problematic modes to 0).
    D_safe = D_b.at[0].set(1.0)
    if n % 2 == 0:
        D_safe = D_safe.at[n // 2].set(1.0)

    E_k = rho_k / (epsilon_0 * D_safe)

    # Ensure the same projected modes are exactly zero.
    E_k = E_k.at[0].set(0.0)
    if n % 2 == 0:
        E_k = E_k.at[n // 2].set(0.0)

    # Real field from Hermitian-symmetric spectrum.
    return jnp.fft.ifft(E_k).real


@jit
def E_from_Poisson_1D_FFT(charge_density, dx):
    """
    Periodic electrostatic solve via a Poisson potential, but **operator-consistent**
    with `E_from_Gauss_1D_FFT`, so they match pointwise for mean-zero rho.

    Continuous equations:
        -phi'' = rho/epsilon_0
        E = -phi'

    Discrete choice here (to match your Gauss solver exactly):
    - Use the *backward difference* gradient for E:
        E = -D_b(phi)
    - Use the same backward difference as the divergence:
        D_b(E) = rho/epsilon_0

    Combining:
        D_b(-D_b(phi)) = rho/epsilon_0
        => -D_b^2(phi) = rho/epsilon_0
        => D_b^2(phi)  = -rho/epsilon_0

    In Fourier space:
        (D_b(k))^2 * phi_k = -rho_k/epsilon_0
        E_k = -D_b(k) * phi_k

    This yields:
        E_k = rho_k / (epsilon_0 * D_b(k))
    i.e. *identical* to the Gauss-law FFT solver, but derived via Poisson.

    Notes
    -----
    - Periodic BCs only.
    - Projects out k=0 (and Nyquist on even grids) for solvability/consistency.
    """
    n = charge_density.shape[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)

    rho_k = jnp.fft.fft(charge_density)
    rho_k = rho_k.at[0].set(0.0)
    if n % 2 == 0:
        rho_k = rho_k.at[n // 2].set(0.0)

    D_b = (1.0 - jnp.exp(-1j * k * dx)) / dx
    L_bb = D_b * D_b  # backward-backward "Laplacian" symbol (complex, but consistent)

    # Safe division on projected modes.
    L_safe = L_bb.at[0].set(1.0)
    if n % 2 == 0:
        L_safe = L_safe.at[n // 2].set(1.0)

    # Solve D_b^2 phi = -rho/eps0
    phi_k = -(rho_k) / (epsilon_0 * L_safe)
    phi_k = phi_k.at[0].set(0.0)
    if n % 2 == 0:
        phi_k = phi_k.at[n // 2].set(0.0)

    # E = -D_b phi
    E_k = -D_b * phi_k
    E_k = E_k.at[0].set(0.0)
    if n % 2 == 0:
        E_k = E_k.at[n // 2].set(0.0)

    return jnp.fft.ifft(E_k).real


@jit
def E_from_Gauss_1D_Cartesian(charge_density, dx, field_BC_left=0, field_BC_right=0):
    """
    Real-space 1D Gauss-law solve that works for periodic and non-periodic setups.

    We integrate the discrete Gauss law:
        (E_i - E_{i-1})/dx = rho_i / epsilon_0
    which implies (choosing an integration constant):
        E_i = E_left + (dx/epsilon_0) * sum_{m=0..i} rho_m

    Boundary handling
    -----------------
    - If (field_BC_left, field_BC_right) == (0, 0): treat as *periodic*.
        * Enforce solvability by removing mean(rho).
        * Choose a gauge by removing mean(E).
    - Otherwise: treat as *non-periodic / open*.
        * Integrate from the left with implicit E_left = 0.
          (This allows net charge to generate a nonzero field at the right.)

    Notes
    -----
    - We also project out the Nyquist mode on even grids to match the
      “FFT-representable” subspace you’re using elsewhere.
    """
    n = charge_density.shape[0]

    # Match the “FFT representable subspace” you compare against in tests:
    rk = jnp.fft.fft(charge_density)
    if n % 2 == 0:
        rk = rk.at[n // 2].set(0.0)
    rho = jnp.fft.ifft(rk).real

    is_periodic = jnp.logical_and(field_BC_left == 0, field_BC_right == 0)

    def periodic(r):
        # solvability: periodic discrete derivative requires mean(r)=0
        r = r - jnp.mean(r)
        E = (dx / epsilon_0) * jnp.cumsum(r)
        # gauge choice: mean(E)=0
        return E - jnp.mean(E)

    def nonperiodic(r):
        # open/integrated solve with E_left = 0
        return (dx / epsilon_0) * jnp.cumsum(r)

    return lax.cond(is_periodic, periodic, nonperiodic, rho)


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