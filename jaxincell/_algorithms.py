import jax.numpy as jnp
from jax import lax,  vmap, jit, random
from functools import partial
from ._sources import current_density_periodic_CN, calculate_charge_density, current_density
from ._boundary_conditions import set_BC_positions, set_BC_particles
from ._particles import fields_to_particles_periodic_CN,fields_to_particles_grid, boris_step, boris_step_relativistic
from ._constants import speed_of_light, epsilon_0
from ._fields import (E_from_Gauss_1D_Cartesian, E_from_Gauss_1D_FFT, curlB,curlE,
                      E_from_Poisson_1D_FFT, field_update1, field_update2)

try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib

__all__ = ['Boris_step', 'CN_step']

@jit
def collision_kernel(v1, v2, q1, q2, m1, m2, n_local, dt, coulomb_log, key):
    """
    Takizuka-Abe Binary Collision Kernel.
    """

    # --- 2. Calculate Scattering Physics ---
    # Relative velocity
    u_vec = v1 - v2
    u_mag = jnp.linalg.norm(u_vec) + 1e-10  # Prevent div/0
    
    # Reduced mass: m_red = (m1*m2)/(m1+m2)
    m_red = (m1 * m2) / (m1 + m2)
    
    variance_numerator =  ((q1 * q2)**2) * n_local * coulomb_log * dt
    variance_denominator = 8.0 * jnp.pi * (epsilon_0**2) * (m_red**2) * (u_mag**3)
    
    variance_delta = variance_numerator / variance_denominator
    
    # Cap variance to prevent numerical explosions at extremely low relative velocities
    variance_delta = jnp.minimum(variance_delta, 10.0)
    
    sigma_delta = jnp.sqrt(variance_delta)
    # Generate Random Angles
    key, k1, k2 = random.split(key, 3)
    delta = random.normal(k1) * sigma_delta
    phi   = random.uniform(k2, minval=0, maxval=2 * jnp.pi)

    # --- 3. Vector Rotation (Geometry) ---
    # Determine perpendicular vector n1
    arbitrary = jnp.where(jnp.abs(u_vec[2]) < 0.8 * u_mag, 
                          jnp.array([0., 0., 1.]), 
                          jnp.array([1., 0., 0.]))
    
    n1 = jnp.cross(u_vec, arbitrary)
    n1 = n1 / (jnp.linalg.norm(n1) + 1e-10)
    n2 = jnp.cross(u_vec, n1) / u_mag

    # Rotate u_vec
    delta_sq = delta**2
    sin_theta = (2.0 * delta) / (1.0 + delta_sq)
    cos_theta = (1.0 - delta_sq) / (1.0 + delta_sq)
    
    u_new = (u_vec * cos_theta + 
             u_mag * sin_theta * (n1 * jnp.cos(phi) + n2 * jnp.sin(phi)))
    
    # Change in velocity
    delta_u = u_new - u_vec

    # --- 4. Update Velocities (Conservation of Momentum) ---
    # v1 changes by (m2/M_total) * delta_u
    # v2 changes by (m1/M_total) * delta_u
    
    v1_new = v1 + (m2 / (m1 + m2)) * delta_u
    v2_new = v2 - (m1 / (m1 + m2)) * delta_u
    
    return v1_new, v2_new



@partial(jit, static_argnames=['num_cells'])
def apply_binary_collisions(positions, velocities, qs, ms, dt, dx, grid_start, num_cells, parameters, rng_key):
    """
    Classical Takizuka-Abe Pairing:
    1. Intra-species (Like-Like)
    2. Inter-species (Binary matching index J, truncating excess)
    """
    N = len(positions)
    coulomb_log = parameters["coulomb_logarithm"]
    weight = parameters["weight"]
    qs = jnp.squeeze(qs)
    ms = jnp.squeeze(ms)

    # 1. Assign particles to cells
    cell_indices = jnp.floor((positions[:, 0] - grid_start) / dx).astype(int)
    cell_indices = jnp.clip(cell_indices, 0, num_cells - 1)

    # Species flag: 1 for Ion, 0 for Electron
    is_ion = jnp.where(qs > 0, 1, 0)

    # --- DENSITY CALCULATION ---
    # We must explicitly define `length` for bincount to be JIT-compatible.
    # We route opposite species to an out-of-bounds index (num_cells) and slice it off.
    e_counts = jnp.bincount(jnp.where(is_ion == 0, cell_indices, num_cells), length=num_cells + 1)[:-1]
    i_counts = jnp.bincount(jnp.where(is_ion == 1, cell_indices, num_cells), length=num_cells + 1)[:-1]

    n_e_local = (e_counts[cell_indices] * weight) / dx
    n_i_local = (i_counts[cell_indices] * weight) / dx
    n_min_local = jnp.minimum(n_e_local, n_i_local) # Use lower one for inter-species

    # Split PRNG keys
    k1, k2, k3 = random.split(rng_key, 3)

    # ==========================================
    # PASS 1: INTRA-SPECIES (Like-Like)
    # ==========================================
    # Shuffle first to randomize which pairs meet within the same species
    rand_perm = random.permutation(k1, N)
    c_shuf = cell_indices[rand_perm]
    ion_shuf = is_ion[rand_perm]

    # Sort by Cell, then Species. 
    # This groups all e's and i's together inside each cell.
    sort_keys_intra = c_shuf * 10 + ion_shuf
    intra_perm = jnp.argsort(sort_keys_intra)
    final_intra_perm = rand_perm[intra_perm]

    # Gather data for Intra-pass
    v_intra = velocities[final_intra_perm]
    q_intra = qs[final_intra_perm] / weight
    m_intra = ms[final_intra_perm] / weight
    c_intra = cell_indices[final_intra_perm]
    ion_flag_intra = is_ion[final_intra_perm]
    
    # Use respective density (e or i) for like-like collisions
    n_intra = jnp.where(ion_flag_intra == 1, n_i_local[final_intra_perm], n_e_local[final_intra_perm])

    idx_1 = jnp.arange(0, N - 1, 2)
    idx_2 = jnp.arange(1, N, 2)

    # Mask: Must be same cell AND same species
    mask_intra = (c_intra[idx_1] == c_intra[idx_2]) & (ion_flag_intra[idx_1] == ion_flag_intra[idx_2])

    keys_intra = random.split(k2, len(idx_1))
    v1_out_intra, v2_out_intra = vmap(collision_kernel, in_axes=(0,0,0,0,0,0,0,None,None,0))(
        v_intra[idx_1], v_intra[idx_2], q_intra[idx_1], q_intra[idx_2], 
        m_intra[idx_1], m_intra[idx_2], n_intra[idx_1], dt, coulomb_log, keys_intra
    )

    v1_final_intra = jnp.where(mask_intra[:, None], v1_out_intra, v_intra[idx_1])
    v2_final_intra = jnp.where(mask_intra[:, None], v2_out_intra, v_intra[idx_2])

    v_after_intra = jnp.zeros_like(velocities)
    v_after_intra = v_after_intra.at[idx_1].set(v1_final_intra)
    v_after_intra = v_after_intra.at[idx_2].set(v2_final_intra)
    if N % 2 != 0: v_after_intra = v_after_intra.at[-1].set(v_intra[-1])

    # ==========================================
    # PASS 2: INTER-SPECIES (Binary J-matching)
    # ==========================================
    # We need to find the local index "J" for each particle. 
    # Since the array is currently sorted by (Cell, Species), we look for boundaries.
    changes = jnp.concatenate([
        jnp.array([True]), 
        (c_intra[1:] != c_intra[:-1]) | (ion_flag_intra[1:] != ion_flag_intra[:-1])
    ])
    # JAX trick to create local arange counters: [0, 1, 2, 0, 1, 0, ...]
    group_start_indices = lax.cummax(jnp.where(changes, jnp.arange(N), 0))
    J_ranks_intra = jnp.arange(N) - group_start_indices

    # Map arrays back to original particle indices temporarily to re-sort
    inv_intra_perm = jnp.argsort(final_intra_perm)
    v_base_inter = v_after_intra[inv_intra_perm]
    J_ranks = J_ranks_intra[inv_intra_perm]

    # SORT 2: Sort by (Cell, J_rank, Species)
    # This guarantees the J-th electron and J-th ion in a cell are adjacent!
    sort_keys_inter = cell_indices * (N * 10) + J_ranks * 10 + is_ion
    inter_perm = jnp.argsort(sort_keys_inter)

    v_inter = v_base_inter[inter_perm]
    q_inter = qs[inter_perm] / weight
    m_inter = ms[inter_perm] / weight
    c_inter = cell_indices[inter_perm]
    ion_flag_inter = is_ion[inter_perm]
    J_inter = J_ranks[inter_perm]
    n_min_inter = n_min_local[inter_perm]

    # Mask: Must be same cell, SAME J, but DIFFERENT species!
    # If a particle is truncated, it won't have a J match, and this mask becomes False.
    mask_inter = (c_inter[idx_1] == c_inter[idx_2]) & \
                 (J_inter[idx_1] == J_inter[idx_2]) & \
                 (ion_flag_inter[idx_1] != ion_flag_inter[idx_2])

    keys_inter = random.split(k3, len(idx_1))
    v1_out_inter, v2_out_inter = vmap(collision_kernel, in_axes=(0,0,0,0,0,0,0,None,None,0))(
        v_inter[idx_1], v_inter[idx_2], q_inter[idx_1], q_inter[idx_2], 
        m_inter[idx_1], m_inter[idx_2], n_min_inter[idx_1], dt, coulomb_log, keys_inter
    )

    v1_final_inter = jnp.where(mask_inter[:, None], v1_out_inter, v_inter[idx_1])
    v2_final_inter = jnp.where(mask_inter[:, None], v2_out_inter, v_inter[idx_2])

    v_final_sorted = jnp.zeros_like(velocities)
    v_final_sorted = v_final_sorted.at[idx_1].set(v1_final_inter)
    v_final_sorted = v_final_sorted.at[idx_2].set(v2_final_inter)
    if N % 2 != 0: v_final_sorted = v_final_sorted.at[-1].set(v_inter[-1])

    # Final unsort
    inv_inter_perm = jnp.argsort(inter_perm)
    return v_final_sorted[inv_inter_perm]

#Boris step
def Boris_step(carry, step_index, parameters, dx, dt, grid, box_size,
                      particle_BC_left, particle_BC_right,
                      field_BC_left, field_BC_right,
                      field_solver):

    (E_field, B_field, positions_minus1_2, positions,
        positions_plus1_2, velocities, qs, ms, q_ms, rng_key) = carry

    fpasses  = parameters["filter_passes"]
    falpha   = parameters["filter_alpha"]
    fstrides = parameters["filter_strides"]  # digital filter for ρ and J (Birdsall & Langdon style)

    # --- 1. Nanbu/Takizuka-Abe Collision Step ---
    # We apply collisions to the current velocities 'velocities' (v_n)
    # before they are used to push the position or updated to v_{n+1}

    rng_key, col_key = random.split(rng_key)
    num_cells = len(grid)
    velocities = apply_binary_collisions(
        positions, velocities, qs, ms, dt, dx, grid[0], num_cells,
        parameters, col_key
    )

    J = current_density(positions_minus1_2, positions, positions_plus1_2, velocities,
                qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right,
                filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
                field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    E_field, B_field = field_update1(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)
    
    # Add external fields
    total_E = E_field + parameters["external_electric_field"]
    total_B = B_field + parameters["external_magnetic_field"]

    # Interpolate fields to particle positions
    def interpolate_fields(x_n):
        E = fields_to_particles_grid(x_n, total_E, dx, grid + dx/2, grid[0], field_BC_left, field_BC_right)
        B = fields_to_particles_grid(x_n, total_B, dx, grid, grid[0] - dx/2, field_BC_left, field_BC_right)
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

    J = current_density(positions_plus1_2, positions_plus1, positions_plus3_2, velocities_plus1,
                qs, dx, dt, grid, grid[0] - dx / 2, particle_BC_left, particle_BC_right,
                filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
                field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    E_field, B_field = field_update2(E_field, B_field, dx, dt/2, J, field_BC_left, field_BC_right)
    
    if field_solver != 0:
        charge_density = calculate_charge_density(positions, qs, dx, grid + dx / 2, particle_BC_left, particle_BC_right,
                                                  filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
                                                  field_BC_left=field_BC_left, field_BC_right=field_BC_right)
        switcher = {
            1: E_from_Gauss_1D_FFT,
            2: E_from_Gauss_1D_Cartesian,
            3: E_from_Poisson_1D_FFT,
        }
        E_field = E_field.at[:,0].set(switcher[field_solver](charge_density, dx))

    # Update positions and velocities
    positions_minus1_2, positions_plus1_2 = positions_plus1_2, positions_plus3_2
    velocities = velocities_plus1
    positions = positions_plus1

    # Prepare state for the next step
    new_carry = (E_field, B_field, positions_minus1_2, positions,
                positions_plus1_2, velocities, qs, ms, q_ms, rng_key)

    # Collect data for storage
    charge_density = calculate_charge_density(positions, qs, dx, grid, particle_BC_left, particle_BC_right,
                                              filter_passes=fpasses, filter_alpha=falpha, filter_strides=fstrides,
                                              field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    step_data = (positions, velocities, E_field, B_field, J, charge_density)
    
    return new_carry, step_data



# Implicit Crank-Nicolson step
@partial(jit, static_argnames=('num_substeps', 'particle_BC_left', 'particle_BC_right', 'field_BC_left', 'field_BC_right'))
def CN_step(carry, step_index, parameters, dx, dt, grid, box_size,
                                  particle_BC_left, particle_BC_right,
                                  field_BC_left, field_BC_right, num_substeps):
    (E_field, B_field, positions,
    velocities, qs, ms, q_ms) = carry
    
    # Implicit method does not have digital filtering for ρ and J
    # Filtering parameters are ignored here

    # Grid Definitions (Staggered)
    E_grid_start = grid[0] + dx/2  # E at i + 1/2
    B_grid_start = grid[0] - dx/2  # B at i (shifted by -1/2 relative to E-grid logic)
    grid_size = len(grid)
    
    # Initial Guess (Predictor)
    E_new = E_field
    B_new = B_field
    positions_new = positions + dt * velocities
    velocities_new = velocities
    
    # Init substep array
    positions_sub1_2_all_init = jnp.repeat(positions[None, ...], num_substeps, axis=0)
    substep_indices = jnp.arange(num_substeps)
    
    c_sq = speed_of_light**2
    dtau = dt / num_substeps

    def picard_step(pic_carry, _):
        _, E_guess, B_guess, pos_fix, _, vel_fix, _, qs_prev, ms_prev, q_ms_prev, pos_stag_arr = pic_carry
        
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
        # interp_E_fn = partial(fields_to_particles_grid, dx=dx, grid=grid, grid_start=E_grid_start, field_BC_left=field_BC_left, field_BC_right=field_BC_right)
        interp_B_fn = partial(fields_to_particles_periodic_CN, dx=dx, grid_start=B_grid_start)
        
        def substep_loop(sub_carry, step_idx):
            pos_sub, vel_sub, qs_sub, ms_sub, q_ms_sub, pos_stag_arr = sub_carry
            pos_stag_prev = pos_stag_arr[step_idx]

            # Gather
            E_mid = vmap(interp_E_fn, in_axes=(0, None))(pos_stag_prev, E_avg_for_Faraday)
            B_mid = vmap(interp_B_fn, in_axes=(0, None))(pos_stag_prev, B_avg_for_Push)
            
            # Boris Velocity Update (Rotation + Push)
            _,vel_new = boris_step(dtau, pos_stag_prev, vel_sub, q_ms, E_mid, B_mid)
            
            vel_mid = 0.5 * (vel_sub + vel_new)
            pos_new = pos_sub + vel_mid * dtau

            # BCs
            pos_new, vel_mid, qs_new, ms_new, q_ms_new = set_BC_particles(
                pos_new, vel_mid, qs_sub, ms_sub, q_ms_sub,
                dx, grid, *box_size, particle_BC_left, particle_BC_right
            )
            pos_stag_new = set_BC_positions(
                pos_new - 0.5*dtau*vel_mid,
                qs_new, dx, grid, *box_size, particle_BC_left, particle_BC_right
            )
            pos_stag_arr = pos_stag_arr.at[step_idx].set(pos_stag_new)

            # Current Deposition
            J_sub = current_density_periodic_CN(
                            pos_stag_prev, vel_mid, qs, dx, 
                            E_grid_start, grid_size
                        )

            return (pos_new, vel_new, qs_new, ms_new, q_ms_new, pos_stag_arr), J_sub * dtau

        sub_init = (pos_fix, vel_fix, qs_prev, ms_prev, q_ms_prev, pos_stag_arr)
        (pos_final, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr_new), J_accs = lax.scan(
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
            (E_guess, E_next, B_next, pos_fix, pos_final, vel_fix, vel_final, qs_final, ms_final, q_ms_final, pos_stag_arr_new),
            J_iter
        )

    # --- Picard Loop ---
    picard_init = (E_new, E_new, B_new, positions, positions_new, velocities, velocities_new, qs, ms, q_ms, positions_sub1_2_all_init)
    
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

        delta_E = jnp.abs(jnp.max(E_calculated - E_guess)) / (jnp.max(jnp.abs(E_calculated)) + 1e-12)
        
        return (new_carry, J_iter, delta_E, i + 1)

    final_carry, J, _, _ = lax.while_loop(cond_fn, body_fn, state0)
    
    (E_prev, E_final, B_final, _, positions_new, _, velocities_new, _, _, _, _) = final_carry

    # Store Data
    E_field = E_final
    B_field = B_final
    positions_plus1 = positions_new
    velocities_plus1 = velocities_new
    
    charge_density = calculate_charge_density(positions_new, qs, dx, grid, particle_BC_left, particle_BC_right,
                                              filter_passes=0, filter_alpha=0.5, filter_strides=(1, 2, 4),
                                              field_BC_left=field_BC_left, field_BC_right=field_BC_right)
    carry = (E_field, B_field, positions_plus1, velocities_plus1, qs, ms, q_ms)
    step_data = (positions_plus1, velocities_plus1, E_field, B_field, J, charge_density)
    
    return carry, step_data