import jax.numpy as jnp
from jax import lax,  vmap, jit
from functools import partial
from ._sources import (current_density_periodic_CN, calculate_charge_density, current_density,
                       Jx_from_continuity_periodic_fft, deposit_S2_scalar)
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._particles import fields_to_particles_periodic_CN,fields_to_particles_grid, boris_step, boris_step_relativistic
from ._constants import speed_of_light, epsilon_0
from ._fields import (curlB, curlE, field_update1, field_update2, enforce_gauss_1d)


try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib

__all__ = ['Boris_step', 'CN_step']

#Boris step
def Boris_step(carry, step_index, parameters, dx, dt, grid, box_size,
                      particle_BC_left, particle_BC_right,
                      field_BC_left, field_BC_right):

    (E_field, B_field, positions_minus1_2, positions,
    positions_plus1_2, velocities, qs, ms, q_ms) = carry

    fpasses  = parameters["filter_passes"]
    falpha   = parameters["filter_alpha"]
    fstrides = parameters["filter_strides"]  # digital filter for ρ and J (Birdsall & Langdon style)
    
    E_grid_start = grid[0] + dx/2    # E on edges (right-edge convention)
    B_grid_start = grid[0]           # B on centers
    
    J = current_density(positions, positions, positions_plus1_2, velocities,
                qs, dx, dt/2, grid, E_grid_start, particle_BC_left, particle_BC_right,
                filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
                field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    E_field, B_field = field_update1(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)
    
    # Add external fields
    total_E = E_field + parameters["external_electric_field"]
    total_B = B_field + parameters["external_magnetic_field"]

    def interpolate_fields(x_n):
        E = fields_to_particles_grid(x_n, total_E, dx, grid + dx/2, E_grid_start, field_BC_left, field_BC_right)
        B = fields_to_particles_grid(x_n, total_B, dx, grid,        B_grid_start, field_BC_left, field_BC_right)
        return E, B

    E_field_at_x, B_field_at_x = vmap(interpolate_fields)(positions_plus1_2)

    # Particle update: Boris pusher
    positions_plus3_2, velocities_plus1 = lax.cond(
        parameters["relativistic"],
        lambda _: boris_step_relativistic(dt, positions_plus1_2, velocities, qs, ms, E_field_at_x, B_field_at_x),
        lambda _: boris_step(dt, positions_plus1_2, velocities, q_ms, E_field_at_x, B_field_at_x),
        operand=None
    )

    # Apply boundary conditions
    positions_plus3_2, velocities_plus1, qs, ms, q_ms = set_BC_particles(
        positions_plus3_2, velocities_plus1, qs, ms, q_ms, dx, grid,
        *box_size, particle_BC_left, particle_BC_right)
    
    positions_plus1 = set_BC_positions(positions_plus3_2 - (dt / 2) * velocities_plus1,
                                    qs, dx, grid, *box_size, particle_BC_left, particle_BC_right)

    J = current_density(positions_plus1_2, positions_plus1, positions_plus1, velocities_plus1,
                qs, dx, dt/2, grid, E_grid_start, particle_BC_left, particle_BC_right,
                filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
                field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    E_field, B_field = field_update2(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)
    
    charge_density_out = calculate_charge_density(
        positions_plus1, qs, dx, grid, particle_BC_left, particle_BC_right,
        filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
        field_BC_left=field_BC_left, field_BC_right=field_BC_right
    )

    E_field = enforce_gauss_1d(
        E_field, charge_density_out, dx, field_BC_left, field_BC_right,
        relax=parameters.get("gauss_relax", 1.0),
        neutralize_periodic=parameters.get("gauss_neutralize_periodic", True)
    )

    # Update positions and velocities
    positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
    velocities = velocities_plus1
    positions = positions_plus1

    # Prepare state for the next step
    carry = (E_field, B_field, positions_minus1_2, positions,
            positions_plus1_2, velocities, qs, ms, q_ms)


    step_data = (positions, velocities, E_field, B_field, J, charge_density_out)
    
    return carry, step_data



# Implicit Crank-Nicolson step
@partial(jit, static_argnames=('num_substeps', 'particle_BC_left', 'particle_BC_right',
                               'field_BC_left', 'field_BC_right'))
def CN_step(carry, step_index, parameters, dx, dt, grid, box_size, particle_BC_left, particle_BC_right,
            field_BC_left, field_BC_right, num_substeps):
    (E_field, B_field, positions,
    velocities, qs, ms, q_ms) = carry
    
    # Implicit method does not have digital filtering for ρ and J
    # Filtering parameters are ignored here

    # Grid Definitions (Staggered)
    # E is stored on the same grid indices as rho in this codebase
    E_grid_start = grid[0] + dx/2
    # B is staggered by -dx/2 relative to E
    B_grid_start = grid[0]
    
    grid_size = len(grid)
    
    # Initial Guess (Predictor)
    E_new = E_field
    B_new = B_field
    positions_new = positions + dt * velocities
    velocities_new = velocities
    
    # Init substep array
    substep_indices = jnp.arange(num_substeps)
    
    c_sq = speed_of_light**2
    dtau = dt / num_substeps

    def picard_step(pic_carry, _):
        _, E_guess, B_guess, pos_fix, _, vel_fix, _, qs_prev, ms_prev, q_ms_prev = pic_carry
        
        # ---------------------------------------------------------------------
        # 1. Faraday's Law: B^{n+1} = B^n - dt * Curl(E^{n+1/2})
        # ---------------------------------------------------------------------
        E_avg_for_Faraday = 0.5 * (E_field + E_guess)
        
        # Use Periodic Backward Difference for E -> B
        curl_E = curlE(E_avg_for_Faraday, B_field, dx, dt, field_BC_left, field_BC_right)
        B_next = B_field - dt * curl_E
        
        # ---------------------------------------------------------------------
        # 2. Particle Push
        # ---------------------------------------------------------------------
        B_avg_for_Push = 0.5 * (B_field + B_next)
        interp_E_fn = partial(fields_to_particles_periodic_CN, dx=dx, grid_start=E_grid_start)
        interp_B_fn = partial(fields_to_particles_periodic_CN, dx=dx, grid_start=B_grid_start)
        
        def substep_loop(sub_carry, step_idx):
            pos_sub, vel_sub, pos_stag, qs_sub, ms_sub, q_ms_sub = sub_carry

            # Gather
            E_mid = vmap(interp_E_fn, in_axes=(0, None))(pos_stag, E_avg_for_Faraday)
            B_mid = vmap(interp_B_fn, in_axes=(0, None))(pos_stag, B_avg_for_Push)
            
            # Boris Velocity Update (Rotation + Push)
            def push_nonrel(_):
                _, vel_new = boris_step(dtau, pos_stag, vel_sub, q_ms_sub, E_mid, B_mid)
                return vel_new

            def push_rel(_):
                _, vel_new = boris_step_relativistic(dtau, pos_stag, vel_sub, qs_sub, ms_sub, E_mid, B_mid)
                return vel_new

            vel_new = lax.cond(parameters["relativistic"], push_rel, push_nonrel, operand=None)

            vel_mid = 0.5 * (vel_sub + vel_new)
            pos_new = pos_sub + vel_mid * dtau

            # BCs
            pos_new, vel_mid, qs_new, ms_new, q_ms_new = set_BC_particles(
                pos_new, vel_mid, qs_sub, ms_sub, q_ms_sub,
                dx, grid, *box_size, particle_BC_left, particle_BC_right
            )

            pos_stag_new = set_BC_positions(
                pos_new - 0.5 * dtau * vel_mid,
                qs_new, dx, grid, *box_size, particle_BC_left, particle_BC_right
            )

            # Current Deposition
            J_sub = current_density_periodic_CN(pos_stag, vel_mid, qs_new, dx, E_grid_start, grid_size)

            return (pos_new, vel_new, pos_stag_new, qs_new, ms_new, q_ms_new), J_sub * dtau

        pos_stag0 = set_BC_positions(
            positions + 0.5 * dtau * velocities,
            qs, dx, grid, *box_size, particle_BC_left, particle_BC_right
        )
        sub_init = (positions, velocities, pos_stag0, qs, ms, q_ms)
        (pos_final, vel_final, _, qs_final, ms_final, q_ms_final), J_accs = lax.scan(
            substep_loop, sub_init, substep_indices
        )
        J_iter = jnp.sum(J_accs, axis=0) / dt
        mean_J = jnp.mean(J_iter, axis=0)
        
        # ---------------------------------------------------------------------
        # 3. Ampere's Law: E^{n+1} = E^n + dt * c^2 * Curl(B^{n+1/2}) - ...
        # ---------------------------------------------------------------------
        # Use Periodic Forward Difference for B -> E
        curl_B = curlB(B_avg_for_Push, E_field, dx, dt, field_BC_left, field_BC_right)
        E_next = E_field + dt * (c_sq * curl_B - (1/epsilon_0) * (J_iter - mean_J))

        return (
            (E_guess, E_next, B_next, pos_fix, pos_final, vel_fix, vel_final, qs_final, ms_final, q_ms_final),
            J_iter
        )

    # --- Picard Loop ---
    picard_init = (E_new, E_new, B_new, positions, positions_new, velocities, velocities_new, qs, ms, q_ms)
    
    tol = parameters["tolerance_Picard_iterations_implicit_CN"]
    max_iter = parameters["max_number_of_Picard_iterations_implicit_CN"]

    # Initial state
    state0 = (picard_init, jnp.zeros_like(E_new), jnp.array(jnp.inf), jnp.array(0))
    
    def cond_fn(state):
        _, _, delta_E, i = state
        return jnp.logical_and(delta_E > tol, i < max_iter)

    def body_fn(state):
        carry, _, _, i = state
        E_guess = carry[1]

        new_carry, J_iter = picard_step(carry, None)
        E_calculated = new_carry[1]

        delta_E = jnp.max(jnp.abs(E_calculated - E_guess)) / (jnp.max(jnp.abs(E_calculated)) + 1e-12)
        
        return (new_carry, J_iter, delta_E, i + 1)

    final_carry, J, _, _ = lax.while_loop(cond_fn, body_fn, state0)
    
    (E_prev, E_final, B_final, _, positions_new, _, velocities_new,
     qs_final, ms_final, q_ms_final) = final_carry

    charge_density_out = calculate_charge_density(
        positions_new, qs_final, dx, grid,
        particle_BC_left, particle_BC_right,
        filter_passes=0, filter_alpha=0.5, filter_strides=(1, 2, 4),
        field_BC_left=field_BC_left, field_BC_right=field_BC_right
    )

    # Store Data
    E_field = E_final
    B_field = B_final
    positions_plus1 = positions_new
    velocities_plus1 = velocities_new

    carry = (E_field, B_field, positions_plus1, velocities_plus1, qs_final, ms_final, q_ms_final)
    step_data = (positions_plus1, velocities_plus1, E_field, B_field, J, charge_density_out)
    return carry, step_data