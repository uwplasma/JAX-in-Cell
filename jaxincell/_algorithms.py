import jax.numpy as jnp
from jax import lax,  vmap, jit
from functools import partial

from ._sources import current_density, calculate_charge_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._particles import fields_to_particles_grid, boris_step, boris_step_relativistic, v_from_p, p_from_v
from ._constants import speed_of_light, epsilon_0, elementary_charge, mass_electron, mass_proton
from ._fields import (field_update, E_from_Gauss_1D_Cartesian, E_from_Gauss_1D_FFT,
                      E_from_Poisson_1D_FFT, field_update1, field_update2)
from ._filters import filter_vector_field

__all__ = ['Boris_step', 'CN_step']

#Boris step
@partial(jit, static_argnames=('particle_BC_left', 'particle_BC_right', 'field_BC_left', 'field_BC_right', 'field_solver'))
def Boris_step(carry, step_index, parameters, dx, dt, grid, box_size,
                      particle_BC_left, particle_BC_right,
                      field_BC_left, field_BC_right,
                      field_solver):
    
    fpasses  = parameters["filter_passes"]
    falpha   = parameters["filter_alpha"]
    fstrides = parameters["filter_strides"]

    (E_field, B_field, positions_minus1_2, positions,
    positions_plus1_2, momenta, qs, ms, q_ms) = carry

    velocities = v_from_p(momenta, ms)  # (N,3) used by current deposition and BCs
    
    J = current_density(positions_minus1_2, positions, positions_plus1_2, velocities,
                qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)
    J = filter_vector_field(J, passes=fpasses, alpha=falpha, strides=fstrides)
    E_field, B_field = field_update1(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)
    
    # Add external fields
    total_E = E_field + parameters["external_electric_field"]
    total_B = B_field + parameters["external_magnetic_field"]

    interp_E = partial(fields_to_particles_grid,
                       dx=dx, grid=grid + dx/2, grid_start=grid[0],
                       field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    interp_B = partial(fields_to_particles_grid,
                       dx=dx, grid=grid, grid_start=grid[0] - dx/2,
                       field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    def interpolate_fields(x_n):
        return interp_E(x_n, total_E), interp_B(x_n, total_B)

    E_field_at_x, B_field_at_x = vmap(interpolate_fields)(positions_plus1_2)

    # Particle update: Boris pusher
    positions_plus3_2, momenta_plus1 = lax.cond(
        parameters["relativistic"],
        lambda _: boris_step_relativistic(dt, positions_plus1_2, momenta, qs, ms, E_field_at_x, B_field_at_x),
        lambda _: boris_step(dt, positions_plus1_2, momenta, q_ms, ms, E_field_at_x, B_field_at_x),
        operand=None
    )

    # Apply boundary conditions
    positions_plus3_2, momenta_plus1, qs, ms, q_ms = set_BC_particles(
        positions_plus3_2, momenta_plus1, qs, ms, q_ms, dx, grid,
        *box_size, particle_BC_left, particle_BC_right)
    
    velocities_plus1 = v_from_p(momenta_plus1, ms)

    positions_plus1 = set_BC_positions(positions_plus3_2 - (dt / 2) * velocities_plus1,
                                    qs, dx, grid, *box_size, particle_BC_left, particle_BC_right)

    J = current_density(positions_plus1_2, positions_plus1, positions_plus3_2, velocities_plus1,
                qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right)  
    J = filter_vector_field(J, passes=fpasses, alpha=falpha, strides=fstrides)
    E_field, B_field = field_update2(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)
    
    if field_solver != 0:
        charge_density = calculate_charge_density(positions, qs, dx, grid + dx / 2, particle_BC_left, particle_BC_right, filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides)
        switcher = {
            1: E_from_Gauss_1D_FFT,
            2: E_from_Gauss_1D_Cartesian,
            3: E_from_Poisson_1D_FFT,
        }
        E_field = E_field.at[:,0].set(switcher[field_solver](charge_density, dx))

    # Update positions and velocities
    positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
    momenta    = momenta_plus1
    velocities = velocities_plus1
    positions  = positions_plus1

    # Prepare state for the next step
    charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right,
                                              filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides)
    carry = (E_field, B_field, positions_minus1_2, positions,
            positions_plus1_2, momenta, qs, ms, q_ms)

    # Collect data for storage
    step_data = (positions, velocities, momenta, E_field, B_field, J, charge_density)
    
    return carry, step_data

# Implicit Crank-Nicolson step
@partial(jit, static_argnames=('num_substeps', 'particle_BC_left', 'particle_BC_right', 'field_BC_left', 'field_BC_right'))
def CN_step(carry, step_index, parameters, dx, dt, grid, box_size,
                                  particle_BC_left, particle_BC_right,
                                  field_BC_left, field_BC_right, num_substeps):
    
    fpasses  = parameters["filter_passes"]
    falpha   = parameters["filter_alpha"]
    fstrides = parameters["filter_strides"]

    (E_field, B_field, positions, velocities, qs, ms, q_ms) = carry

    # -------------------- hoisted constants / closures --------------------
    dtau = dt / num_substeps
    grid_for_E   = grid + dx/2        # E stagger
    start_for_E  = grid[0]            # left start for E
    grid_for_J   = grid               # for current deposition
    start_for_J  = grid[0] - dx/2     # J stagger

    # Prebuild interpolation closure once
    interp_E = partial(
        fields_to_particles_grid,
        dx=dx, grid=grid_for_E, grid_start=start_for_E,
        field_BC_left=field_BC_left, field_BC_right=field_BC_right,
    )
    # ---------------------------------------------------------------------

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

        # substepping
        def substep_loop(sub_carry, step_idx):
            pos_sub, vel_sub, qs_sub, ms_sub, q_ms_sub, pos_stag_arr = sub_carry
            pos_stag_prev = pos_stag_arr[step_idx]

            E_mid = vmap(interp_E, in_axes=(0, None))(pos_stag_prev, E_avg)

            vel_new = vel_sub + (q_ms_sub * E_mid) * dtau
            vel_mid = 0.5 * (vel_sub + vel_new)
            pos_new = pos_sub + vel_mid * dtau

            # Apply boundary conditions
            pos_new, vel_mid, qs_new, ms_new, q_ms_new = set_BC_particles(
                pos_new, vel_mid, qs_sub, ms_sub, q_ms_sub,
                dx, grid, *box_size, particle_BC_left, particle_BC_right
            )

            pos_stag_new = set_BC_positions(
                pos_new - 0.5*dtau*vel_mid,
                qs_new, dx, grid, *box_size, particle_BC_left, particle_BC_right
            )
            # Update half substep positions
            pos_stag_arr = pos_stag_arr.at[step_idx].set(pos_stag_new)

            # half step current density
            J_sub = current_density(pos_sub, pos_stag_new, pos_new, vel_mid, qs_new, dx, dtau,
                grid_for_J, start_for_J, particle_BC_left, particle_BC_right)
            J_sub = filter_vector_field(J_sub, passes=fpasses, alpha=falpha, strides=fstrides)

            return (pos_new, vel_new, qs_new, ms_new, q_ms_new, pos_stag_arr), J_sub * dtau

        # initial substep carry 
        sub_init = (pos_fix, vel_fix, qs_prev, ms_prev, q_ms_prev, pos_stag_arr)

        # run through all substeps
        (pos_final, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr), J_accs = lax.scan(
            substep_loop, sub_init, substep_indices
        )

        # Sum over substep to get next step current with eletric field
        J_iter = jnp.sum(J_accs, axis=0) / dt
        mean_J = jnp.mean(J_iter, axis=0)
        E_next = E_field - (dt / epsilon_0) * (J_iter - mean_J)

        return (E_new, E_next, pos_fix, pos_final, vel_fix, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr), J_iter

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

    # Update carrys for next step
    E_field = E_new
    B_field = B_new
    positions_plus1= positions_new
    velocities_plus1 = velocities_new
    momenta_plus1 = ms * velocities_plus1

    charge_density = calculate_charge_density(positions_new, qs, dx, grid, particle_BC_left, particle_BC_right, filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides)

    carry = (E_field, B_field, positions_plus1, velocities_plus1, qs, ms, q_ms)
    
    # Collect data
    step_data = (positions_plus1, velocities_plus1, momenta_plus1, E_field, B_field, J, charge_density)
    
    return carry, step_data
