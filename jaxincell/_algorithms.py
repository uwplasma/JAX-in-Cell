import jax.numpy as jnp
from jax import lax,  vmap, jit
from functools import partial

from ._sources import current_density, calculate_charge_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._particles import (
    fields_to_particles_grid,
    boris_step, boris_step_relativistic,
    boris_step_with_force, boris_step_relativistic_with_force
)

from ._constants import speed_of_light, epsilon_0, elementary_charge, mass_electron, mass_proton, gravitational_constant
from ._fields import (field_update, E_from_Gauss_1D_Cartesian, E_from_Gauss_1D_FFT, delta_leapfrog_step,
                      E_from_Poisson_1D_FFT, field_update1, field_update2, grad1d_periodic)

try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib

__all__ = ['Boris_step', 'CN_step']

def split_species_charge_density(positions, qs, dx, grid_E, particle_BC_left, particle_BC_right, N_e):
    # electrons: [0:N_e), ions: [N_e: ]
    rho_e = calculate_charge_density(positions[:N_e], qs[:N_e], dx, grid_E, particle_BC_left, particle_BC_right)
    rho_i = calculate_charge_density(positions[N_e:], qs[N_e:], dx, grid_E, particle_BC_left, particle_BC_right)
    return rho_e, rho_i

#Boris step
def Boris_step(carry, step_index, parameters, dx, dt, grid, box_size,
               particle_BC_left, particle_BC_right,
               field_BC_left, field_BC_right,
               field_solver, use_gravity):

    (E_field, B_field, positions_minus1_2, positions,
     positions_plus1_2, velocities, qs, ms, q_ms, delta, delta_prev) = carry

    # -- advance fields half step (standard PIC)
    J = current_density(positions_minus1_2, positions, positions_plus1_2, velocities,
                        qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    E_field, B_field = field_update1(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)

    # external fields
    total_E = E_field + parameters["external_electric_field"]
    total_B = B_field + parameters["external_magnetic_field"]

    # ---- geometric fields: α = 1 + δ  (keep to all orders via log1p; effectively 2nd order in δ)
    alpha_grid     = 1.0 + delta
    ln_alpha_grid  = jnp.log1p(delta)            # ln(α) ≈ δ - δ^2/2
    grad_ln_alpha  = grad1d_periodic(ln_alpha_grid, dx)          # ∂x ln α on E-grid
    ln_alpha_prev  = jnp.log1p(delta_prev)
    dot_ln_alpha_grid = (ln_alpha_grid - ln_alpha_prev) / dt     # ∂t ln α on E-grid

    # scalar interpolation helper (reuse quadratic vector interpolator)
    def interp_scalar_on_Egrid(x_n, scalar_grid):
        scalar_vec = jnp.stack([scalar_grid,
                                jnp.zeros_like(scalar_grid),
                                jnp.zeros_like(scalar_grid)], axis=1)
        val_vec = fields_to_particles_grid(x_n, scalar_vec, dx, grid + dx/2, grid[0],
                                           field_BC_left, field_BC_right)
        return val_vec[0]  # x-component holds the scalar

    # interpolate δ, ∂x ln α, ∂t ln α at x_{n+1/2}
    interp_delta      = vmap(lambda x: interp_scalar_on_Egrid(x, delta))(positions_plus1_2)
    interp_grad_ln    = vmap(lambda x: interp_scalar_on_Egrid(x, grad_ln_alpha))(positions_plus1_2)   # scalar ∂x ln α
    interp_dot_ln     = vmap(lambda x: interp_scalar_on_Egrid(x, dot_ln_alpha_grid))(positions_plus1_2)

    # Interpolate E,B to particles (E on E-stagger, B on cell-centers)
    def interpolate_fields(x_n):
        E = fields_to_particles_grid(x_n, total_E, dx, grid + dx/2, grid[0], field_BC_left, field_BC_right)
        B = fields_to_particles_grid(x_n, total_B, dx, grid,         grid[0] - dx/2, field_BC_left, field_BC_right)
        return E, B
    E_field_at_x, B_field_at_x = vmap(interpolate_fields)(positions_plus1_2)

    # ---- longitudinal field rescaling (2nd order in δ)
    delta_val = interp_delta
    Ex = E_field_at_x[:, 0]
    scale = 1.0 + 1.5 * delta_val + 0.375 * (delta_val ** 2)
    E_field_at_x = E_field_at_x.at[:, 0].set(Ex * scale)

    # =========================
    #   GR force (geodesic) in a conformal metric  g = α η
    #   Using coordinate-time form:
    #   a_GR = - (c^2/2) (1 - v^2/c^2) ∇ ln α  +  (1/2) (∂t ln α) (-1 + v^2/c^2) v
    #   (Here ∇ ln α has only x-component in our 1D geometry.)
    #   We then map to momentum-force:
    #     F_ext = m [ γ a_⊥ + γ^3 a_∥ ]
    #   so we can use the standard relativistic Boris with an extra force.
    # =========================
    c2  = speed_of_light ** 2
    v2  = jnp.sum(velocities**2, axis=1)                      # (N,)
    gamma_sr = 1.0 / jnp.sqrt(jnp.maximum(1.0 - v2 / c2, 1e-30))

    # spatial piece: only x-component for grad ln α
    a_space_x = -0.5 * (c2) * (1.0 - v2 / c2) * interp_grad_ln     # (N,)
    a_space   = jnp.stack([a_space_x,
                           jnp.zeros_like(a_space_x),
                           jnp.zeros_like(a_space_x)], axis=1)     # (N,3)

    # time-derivative piece: parallel to v
    a_time = 0.5 * interp_dot_ln[:, None] * (-1.0 + v2 / c2)[:, None] * velocities  # (N,3)

    a_GR = a_space + a_time   # (N,3)

    # decompose a_GR into components parallel / perpendicular to v
    v_norm = jnp.sqrt(jnp.maximum(v2, 1e-30))
    v_hat  = velocities / v_norm[:, None]
    a_par_mag = jnp.sum(a_GR * v_hat, axis=1)                      # (N,)
    a_par     = a_par_mag[:, None] * v_hat                         # (N,3)
    a_perp    = a_GR - a_par                                       # (N,3)

    m_s = ms[:, 0]                   # (N,)
    F_ext = (gamma_sr[:, None]   * m_s[:, None]) * a_perp \
          + (gamma_sr[:, None]**3 * m_s[:, None]) * a_par          # (N,3)

    # ---- PRE half-step: apply half of GR force symmetrically
    # For the relativistic pusher with extra force we pass F_ext and it will do ±(qE+F)/2
    # (For the nonrelativistic path, we pass acceleration a_GR.)
    positions_plus3_2, velocities_plus1 = lax.cond(
        parameters["relativistic"],
        lambda _: boris_step_relativistic_with_force(
            dt, positions_plus1_2, velocities, qs[:, 0], ms[:, 0], E_field_at_x, B_field_at_x, F_ext
        ),
        lambda _: boris_step_with_force(
            dt, positions_plus1_2, velocities, q_ms, E_field_at_x, B_field_at_x, a_GR
        ),
        operand=None
    )

    # make positions consistent with post-kick velocity (already returned above)
    # Apply particle BCs
    positions_plus3_2, velocities_plus1, qs, ms, q_ms = set_BC_particles(
        positions_plus3_2, velocities_plus1, qs, ms, q_ms, dx, grid,
        *box_size, particle_BC_left, particle_BC_right
    )
    positions_plus1 = set_BC_positions(
        positions_plus3_2 - (dt / 2) * velocities_plus1,
        qs, dx, grid, *box_size, particle_BC_left, particle_BC_right
    )

    # -- advance fields the other half-step
    J = current_density(positions_plus1_2, positions_plus1, positions_plus3_2, velocities_plus1,
                        qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    E_field, B_field = field_update2(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)

    # optional Gauss solver with weak-field source renormalization ρ_eff = α^{-1/2} ρ ≈ (1 - δ/2 + 3/8 δ^2) ρ
    if field_solver != 0:
        charge_density = calculate_charge_density(positions, qs, dx, grid + dx / 2,
                                                  particle_BC_left, particle_BC_right)
        rho_eff = (1.0 - 0.5 * delta + 0.375 * (delta ** 2)) * charge_density
        switcher = {1: E_from_Gauss_1D_FFT, 2: E_from_Gauss_1D_Cartesian, 3: E_from_Poisson_1D_FFT}
        E_field = E_field.at[:, 0].set(switcher[field_solver](rho_eff, dx))

    # rotate time levels for particles
    positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
    velocities = velocities_plus1
    positions  = positions_plus1

    # ---- δ update (wave eqn) with your source
    if use_gravity:
        Np = parameters["number_pseudoelectrons"]
        rho_e, rho_i = split_species_charge_density(positions, qs, dx, grid + dx/2,
                                                    particle_BC_left, particle_BC_right, Np)
        qe = parameters["electron_charge_over_elementary_charge"] * elementary_charge
        qi = parameters["ion_charge_over_elementary_charge"]      * elementary_charge
        ne = rho_e / qe
        ni = rho_i / qi

        m_e = mass_electron
        m_i = parameters["ion_mass_over_proton_mass"] * mass_proton
        m2n_sum = (m_e**2) * ne + (m_i**2) * ni

        total_Ex = (E_field[:, 0] + parameters["external_electric_field"][:, 0])

        kappa  = parameters["kappa"] * 8 * jnp.pi * gravitational_constant / (speed_of_light**4)
        Lambda = parameters["Lambda"]
        source = -(speed_of_light**2) * Lambda \
                 + kappa * (speed_of_light**2) * epsilon_0 * (total_Ex**2) \
                 + kappa * (speed_of_light**5) * m2n_sum

        delta_next = delta_leapfrog_step(delta, delta_prev, dx, dt, source)
        delta_prev, delta = delta, delta_next

    # pack carry + diagnostics
    carry = (E_field, B_field, positions_minus1_2, positions,
             positions_plus1_2, velocities, qs, ms, q_ms, delta, delta_prev)

    charge_density_out = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right)
    step_data = (positions, velocities, E_field, B_field, J, charge_density_out, delta)

    return carry, step_data



# Implicit Crank-Nicolson step
@partial(jit, static_argnames=('num_substeps', 'particle_BC_left', 'particle_BC_right', 'field_BC_left', 'field_BC_right','use_gravity'))
def CN_step(carry, step_index, parameters, dx, dt, grid, box_size,
                                  particle_BC_left, particle_BC_right,
                                  field_BC_left, field_BC_right, num_substeps, use_gravity):
    (E_field, B_field, positions,
    velocities, qs, ms, q_ms, delta, delta_prev) = carry

    E_new=E_field
    B_new=B_field
    positions_new=positions+ (dt) * velocities
    velocities_new=velocities
    # initialize the array of half-substep positions for Picard iterations
    positions_sub1_2_all_init = jnp.repeat(positions[None, ...], num_substeps, axis=0)

    # Picard iteration of solution for next step
    substep_indices = jnp.arange(num_substeps)
    def picard_step(pic_carry, _):
        _, E_new, pos_fix, _, vel_fix, _, qs_prev, ms_prev, q_ms_prev, pos_stag_arr = pic_carry
        E_avg = 0.5 * (E_field + E_new)
        dtau  = dt / num_substeps


        interp_E = partial(fields_to_particles_grid, dx=dx, grid=grid + dx/2, grid_start=grid[0],
                        field_BC_left=field_BC_left, field_BC_right=field_BC_right)
        # substepping
        def substep_loop(sub_carry, step_idx):
            pos_sub, vel_sub, qs_sub, ms_sub, q_ms_sub, pos_stag_arr = sub_carry
            pos_stag_prev = pos_stag_arr[step_idx]

            # Interpolate Ex and δ (and ∂xδ) at the staggered mid-positions
            E_mid = vmap(interp_E, in_axes=(0, None))(pos_stag_prev, E_avg)  # shape (N,3)

            # δ lives on the E-grid; get δ and ∂xδ there and interpolate like a scalar
            delta_grad = grad1d_periodic(delta, dx)

            def interp_scalar_on_Egrid(x_n, scalar_grid):
                # reuse fields_to_particles_grid by putting scalar in x-slot
                scalar_vec = jnp.stack([scalar_grid, jnp.zeros_like(scalar_grid), jnp.zeros_like(scalar_grid)], axis=1)
                val_vec = fields_to_particles_grid(x_n, scalar_vec, dx, grid + dx/2, grid[0], field_BC_left, field_BC_right)
                return val_vec[0]

            delta_mid   = vmap(lambda x: interp_scalar_on_Egrid(x, delta))(pos_stag_prev)       # (N,)
            deltax_mid  = vmap(lambda x: interp_scalar_on_Egrid(x, delta_grad))(pos_stag_prev)  # (N,)

            delta_val = delta_mid                        # (N,)
            Ex = E_mid[:, 0]
            scale = 1.0 + 1.5 * delta_val + 0.375 * (delta_val ** 2)
            E_mid = E_mid.at[:, 0].set(Ex * scale)

            # delta_dot is fixed for this CN iteration (built outside substep_loop)
            delta_dot_grid = (delta - delta_prev) / dt

            # inside substep_loop(), after you have pos_stag_prev:
            deltat_mid = vmap(lambda x: interp_scalar_on_Egrid(x, delta_dot_grid))(pos_stag_prev)

            # spatial gravity (already there)
            g_space_mid = -0.5 * (speed_of_light**2) * (1.0 + delta_mid) * deltax_mid

            # time-derivative gravity: + dot{delta} * v_x  (use current vel_sub for explicit midpoint)
            g_tx_mid = deltat_mid * vel_sub[:, 0]

            g_mid_x = g_space_mid + g_tx_mid
            g_mid   = jnp.stack([g_mid_x, jnp.zeros_like(g_mid_x), jnp.zeros_like(g_mid_x)], axis=1)

            # velocity update:
            vel_new = vel_sub + (q_ms_sub * E_mid + g_mid) * dtau
            
            vel_mid = 0.5 * (vel_sub + vel_new)
            pos_new = pos_sub + vel_mid * dtau

            # Apply boundary conditions
            pos_new, vel_mid, qs_new, ms_new, q_ms_new = set_BC_particles(
                pos_new, vel_mid, qs_sub, ms_sub, q_ms_sub,
                dx, grid, *box_size, particle_BC_left, particle_BC_right
            )

            pos_stag_new = set_BC_positions(
                pos_new - 0.5*dtau*vel_mid,
                qs_new, dx, grid,
                *box_size, particle_BC_left, particle_BC_right
            )
            # Update half substep positions
            pos_stag_arr = pos_stag_arr.at[step_idx].set(pos_stag_new)

            # half step current density
            J_sub = current_density(
                pos_sub, pos_stag_new, pos_new, vel_mid, qs_new, dx, dtau, grid,
                grid[0] - dx/2, particle_BC_left, particle_BC_right
            )

            return (pos_new, vel_new, qs_new, ms_new, q_ms_new, pos_stag_arr), J_sub * dtau

        # initial substep carry 
        sub_init = (
            pos_fix, vel_fix,
            qs_prev, ms_prev, q_ms_prev,
            pos_stag_arr
        )

        # run through all substeps
        (pos_final, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr), J_accs = lax.scan(
            substep_loop,
            sub_init,
            substep_indices
        )

        # Sum over substep to get next step current with eletric field
        J_iter = jnp.sum(J_accs, axis=0) / dt
        mean_J = jnp.mean(J_iter, axis=0)
        E_next = E_field - (dt / epsilon_0) * (J_iter - mean_J)

        return (
            (E_new, E_next, pos_fix, pos_final, vel_fix, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr),
            J_iter
        )


    # Picard iteration
    picard_init = (E_new, positions, positions_new, velocities,velocities_new, qs, ms, q_ms, positions_sub1_2_all_init)
    tol = parameters["tolerance_Picard_iterations_implicit_CN"]
    max_iter = parameters["max_number_of_Picard_iterations_implicit_CN"]

    positions_sub1_2_all_init = jnp.tile(positions[None, ...], (num_substeps, 1, 1))
    E_old = E_new
    delta_E0 = jnp.array(jnp.inf)
    iter_idx0 = jnp.array(0)

    picard_init = (E_old, E_new, positions, positions_new, velocities, velocities_new, qs, ms, q_ms, positions_sub1_2_all_init)
    state0 = (picard_init, jnp.zeros_like(E_new), delta_E0, iter_idx0)
    
    def cond_fn(state):
        _, _, delta_E, i = state
        return jnp.logical_and(delta_E > tol, i < max_iter)

    def body_fn(state):
        carry, _, _, i = state
        
        E_old = carry[0]

        new_carry, J_iter = picard_step(carry, None)
        E_next = new_carry[1]

        delta_E = jnp.abs(jnp.max(E_next - E_old)) / (jnp.max(jnp.abs(E_next)))
        return (new_carry, J_iter, delta_E, i + 1)

    final_carry, J, _, _ = lax.while_loop(cond_fn, body_fn, state0)
    (E_old, E_new, _, positions_new, _, velocities_new, _, _, _, _) = final_carry
    
    # ---------- δ update (leapfrog) ----------
    if use_gravity:
        # species densities on E-grid
        Np = parameters["number_pseudoelectrons"]
        
        # electrons have negative q, ions positive q in your setup
        mask_e = (qs[:, 0] < 0).astype(qs.dtype).reshape(-1, 1)
        mask_i = (qs[:, 0] > 0).astype(qs.dtype).reshape(-1, 1)

        # deposit with masked charges (positions shape unchanged)
        rho_e = calculate_charge_density(positions_new, qs * mask_e, dx, grid + dx/2,
                                        particle_BC_left, particle_BC_right)
        rho_i = calculate_charge_density(positions_new, qs * mask_i, dx, grid + dx/2,
                                        particle_BC_left, particle_BC_right)

        # number densities
        qe = parameters["electron_charge_over_elementary_charge"] * elementary_charge
        qi = parameters["ion_charge_over_elementary_charge"]      * elementary_charge
        ne = rho_e / qe
        ni = rho_i / qi

        m_e = mass_electron
        m_i = parameters["ion_mass_over_proton_mass"] * mass_proton
        m2n_sum = (m_e**2) * ne + (m_i**2) * ni

        total_Ex = (E_new[:, 0] + parameters["external_electric_field"][:, 0])

        kappa = parameters["kappa"] * 8 * jnp.pi * gravitational_constant / speed_of_light**4
        Lambda = parameters["Lambda"]
        source = -(speed_of_light**2) * Lambda \
                + kappa * (speed_of_light**2) * epsilon_0 * (total_Ex**2) \
                + kappa * (speed_of_light**5) * m2n_sum

        delta_next = delta_leapfrog_step(delta, delta_prev, dx, dt, source)
        delta_prev, delta = delta, delta_next

    # Update carrys for next step
    E_field = E_new
    B_field = B_new
    positions_plus1= positions_new
    velocities_plus1 = velocities_new
    
    charge_density = calculate_charge_density(positions_new, qs, dx, grid, particle_BC_left, particle_BC_right)

    carry = (E_field, B_field, positions_plus1, velocities_plus1, qs, ms, q_ms, delta, delta_prev)
    
    # Collect data
    step_data = (positions_plus1, velocities_plus1, E_field, B_field, J, charge_density, delta)
    
    return carry, step_data
