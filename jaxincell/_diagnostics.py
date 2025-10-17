from jax import lax, vmap
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from ._constants import epsilon_0, mu_0, speed_of_light as c
from ._metric_tensor import metric_bundle

__all__ = ['diagnostics']



# ---------- Geometry helpers (GR) ----------

def _geom_at_grid_time(t, x_grid, metric_cfg):
    """
    Return alpha(x) [G,], S(x)=[G,3], sqrt_gamma(x) [G,] at time t on the grid.
    """
    def one_x(x):
        mb = metric_bundle(t, x, metric_cfg["kind"], c, **metric_cfg.get("params", {}))
        g  = mb["g"]
        alpha = jnp.sqrt(-g[0,0]) / c
        S1 = jnp.sqrt(g[1,1]); S2 = jnp.sqrt(g[2,2]); S3 = jnp.sqrt(g[3,3])
        sqrt_gamma = S1 * S2 * S3
        return alpha, jnp.array([S1, S2, S3]), sqrt_gamma
    alphas, S, sqrtg = vmap(one_x)(x_grid)      # alphas: (G,), S: (G,3), sqrtg: (G,)
    return alphas, S, sqrtg

def _geom_at_positions_time(t, x_pos, metric_cfg):
    """
    Metric at particle positions x_pos[...,0] (N,1 or (N,3) but we use x).
    Returns alpha_p [N,], S_p [N,3].
    """
    def one_x(x):
        mb = metric_bundle(t, x, metric_cfg["kind"], c, **metric_cfg.get("params", {}))
        g  = mb["g"]
        alpha = jnp.sqrt(-g[0,0]) / c
        return alpha, jnp.array([jnp.sqrt(g[1,1]), jnp.sqrt(g[2,2]), jnp.sqrt(g[3,3])])
    alphas, S = vmap(one_x)(x_pos)
    return alphas, S

# ---------- Local-frame energy helpers (GR) ----------

def _em_energy_split_gr_at_time(E_gx, B_gx, grid, t, metric_cfg, dx):
    """
    One time slice: split EM energy into electric and magnetic parts in the local orthonormal frame,
    integrate with proper volume √γ dx.
    Inputs: E_gx, B_gx: (G,3) in coordinate components on the E/B grids (same G here).
    """
    _, S, sqrtg = _geom_at_grid_time(t, grid, metric_cfg)    # (G,), (G,3), (G,)
    E_loc = S * E_gx
    B_loc = S * B_gx
    E2 = jnp.sum(E_loc**2, axis=1)                   # (G,)
    B2 = jnp.sum(B_loc**2, axis=1)                   # (G,)
    uE = 0.5 * epsilon_0 * E2                        # (G,)
    uB = 0.5 * (B2 / mu_0)                           # (G,)
    UE = jnp.sum(uE * sqrtg) * dx
    UB = jnp.sum(uB * sqrtg) * dx
    return UE, UB

def _em_energy_split_gr(E_tgx, B_tgx, grid, dt, plasma_frequency, metric_cfg, dx):
    """
    Vectorized over time: returns UE(t), UB(t) each shape (T,).
    """
    T = E_tgx.shape[0]
    t_arr = jnp.arange(T) * dt * plasma_frequency
    UE, UB = vmap(lambda tt, Eg, Bg: _em_energy_split_gr_at_time(Eg, Bg, grid, tt, metric_cfg, dx))(
        t_arr, E_tgx, B_tgx
    )
    return UE, UB

def _kinetic_energy_species_gr(v_tnj, m_n1, pos_tnj, dt, plasma_frequency, metric_cfg):
    """
    Local-frame kinetic energy for one species.
      v_tnj: (T,N,3) coordinate velocities,
      m_n1:  (N,1) masses,
      pos_tnj: (T,N,3) particle positions if available, else None (homogeneous-in-x fallback).
    Returns: K(t) shape (T,)
    """
    T, N, _ = v_tnj.shape
    m_n = m_n1[:, 0]                                 # (N,)
    t_arr = jnp.arange(T) * dt * plasma_frequency

    if pos_tnj is not None:
        # Sample metric at particle positions (x = pos[...,0])
        def one_time(tt, v_nj, x_nj):
            alpha_p, S_p = _geom_at_positions_time(tt, x_nj[:, 0], metric_cfg)  # (N,), (N,3)
            v_loc = (S_p / alpha_p[:, None]) * v_nj
            beta2 = jnp.sum(v_loc**2, axis=1) / (c*c)
            gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta2, 1e-30))
            return jnp.sum((gamma - 1.0) * m_n * (c*c))
        K = vmap(one_time)(t_arr, v_tnj, pos_tnj)    # (T,)
    else:
        # Homogeneous-in-x metric: evaluate at grid center x≈0 (exact for FLRW/Bianchi I)
        def one_time(tt, v_nj):
            alpha, S, _ = _geom_at_grid_time(tt, jnp.array([0.0]), metric_cfg)  # shapes (1,), (1,3), (1,)
            alpha = alpha[0]; S = S[0]
            v_loc = (S / alpha) * v_nj
            beta2 = jnp.sum(v_loc**2, axis=1) / (c*c)
            gamma = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta2, 1e-30))
            return jnp.sum((gamma - 1.0) * m_n * (c*c))
        K = vmap(one_time)(t_arr, v_tnj)            # (T,)

    return K

def diagnostics(output):
    E_field_over_time = output['electric_field']
    grid              = output['grid']
    dt                = output['dt']
    total_steps       = output['total_steps']
    mass_electrons    = output["mass_electrons"][0]
    mass_ions         = output["mass_ions"][0]
    dx                = output['dx']

    # array_to_do_fft_on = charge_density_over_time[:,len(grid)//2]
    array_to_do_fft_on = E_field_over_time[:,len(grid)//2,0]
    array_to_do_fft_on = (array_to_do_fft_on-jnp.mean(array_to_do_fft_on))/jnp.max(array_to_do_fft_on)
    plasma_frequency = output['plasma_frequency']

    fft_values = lax.slice(fft(array_to_do_fft_on), (0,), (total_steps//2,))
    freqs = fftfreq(total_steps, d=dt)[:total_steps//2]*2*jnp.pi # d=dt specifies the time step
    magnitude = jnp.abs(fft_values)
    peak_index = jnp.argmax(magnitude)
    dominant_frequency = jnp.abs(freqs[peak_index])

    def integrate(y, dx): return 0.5 * (jnp.asarray(dx) * (y[..., 1:] + y[..., :-1])).sum(-1)
    # def integrate(y, dx): return jnp.sum(y, axis=-1) * dx
    
    abs_E_squared              = jnp.sum(output['electric_field']**2, axis=-1)
    abs_externalE_squared      = jnp.sum(output['external_electric_field']**2, axis=-1)
    integral_E_squared         = integrate(abs_E_squared, dx=output['dx'])
    integral_externalE_squared = integrate(abs_externalE_squared, dx=output['dx'])
    
    abs_B_squared              = jnp.sum(output['magnetic_field']**2, axis=-1)
    abs_externalB_squared      = jnp.sum(output['external_magnetic_field']**2, axis=-1)
    integral_B_squared         = integrate(abs_B_squared, dx=output['dx'])
    integral_externalB_squared = integrate(abs_externalB_squared, dx=output['dx'])
    
    # Velocities
    ve = output['velocity_electrons']        # (T, Ne, 3)
    vi = output['velocity_ions']             # (T, Ni, 3)

    # Per-particle masses (Ne,1) / (Ni,1) -> broadcast over time
    me = output['mass_electrons']            # (Ne,1)
    mi = output['mass_ions']                 # (Ni,1)
    me_b = me[None, :, 0]                    # (T, Ne) via broadcast
    mi_b = mi[None, :, 0]                    # (T, Ni)

    # Lorentz gamma (clamped for numerical safety)
    beta2_e = jnp.sum(ve**2, axis=-1) / (c**2)             # (T, Ne)
    beta2_i = jnp.sum(vi**2, axis=-1) / (c**2)             # (T, Ni)
    gamma_e = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta2_e, 1e-30))
    gamma_i = 1.0 / jnp.sqrt(jnp.maximum(1.0 - beta2_i, 1e-30))

    # Kinetic energy per species: sum over particles, keep time axis
    v_electrons_squared = jnp.sum(jnp.sum(output['velocity_electrons']**2, axis=-1), axis=-1)
    v_ions_squared      = jnp.sum(jnp.sum(output['velocity_ions']**2     , axis=-1), axis=-1)

    # Only do relativistic if Boris (algorithm 0) and relativistic=True
    algo_is_zero = jnp.equal(jnp.asarray(output['time_evolution_algorithm']), 0)
    relativistic_flag = jnp.asarray(output['relativistic'])
    algo_rel_check = jnp.where(algo_is_zero,
        jnp.where(jnp.logical_not(relativistic_flag), False, True), False)
    kinetic_energy_electrons = lax.cond(algo_rel_check,
        lambda _: jnp.sum((gamma_e - 1.0) * me_b * (c**2), axis=1),  # (T,)
        lambda _: (1/2) * mass_electrons * v_electrons_squared,      # (T,)
        operand=None)
    kinetic_energy_ions = lax.cond(algo_rel_check,
        lambda _: jnp.sum((gamma_i - 1.0) * mi_b * (c**2), axis=1),  # (T,)
        lambda _: (1/2) * mass_ions * v_ions_squared,  # (T,)
        operand=None)

    # ---------- Gauss' law deviation (1D) ----------
    # Use Ex component; periodic central difference
    Ex_tg = output['electric_field'][:,:,0]                  # (T, G)
    rho_tg = output['charge_density']                        # (T, G)
    # periodic roll for central diff
    dEx_dx = (jnp.roll(Ex_tg, -1, axis=1) - jnp.roll(Ex_tg, 1, axis=1)) / (2.0*dx)
    rhs = rho_tg / epsilon_0
    num = jnp.linalg.norm(dEx_dx - rhs, axis=1)
    den = jnp.maximum(jnp.linalg.norm(rhs, axis=1), 1e-300)
    gauss_rel_error = num / den

    # ---------- Momentum relative error ----------
    # convert to (T, Ne, 1) broadcast for multiply
    me_b3 = me[None, :, :]                      # (1,Ne,1) -> broadcast to (T,Ne,1)
    mi_b3 = mi[None, :, :]                      # (1,Ni,1)

    momentum_electrons = lax.cond(algo_rel_check,
        lambda _: jnp.sum((gamma_e[..., None] * me_b3) * ve, axis=1),  # (T, 3)
        lambda _: jnp.sum(me_b3 * ve, axis=1),                # (T, 3)
        operand=None)
    momentum_ions = lax.cond(algo_rel_check,
        lambda _: jnp.sum((gamma_i[..., None] * mi_b3) * vi, axis=1),  # (T, 3)
        lambda _: jnp.sum(mi_b3 * vi, axis=1),                # (T, 3)
        operand=None)
    total_momentum = momentum_electrons + momentum_ions                                         # (T, 3)

    P0   = total_momentum[0]
    numP = jnp.linalg.norm(total_momentum - P0, axis=1)
    denP = jnp.maximum(jnp.linalg.norm(P0), 1e-300)
    momentum_rel_error = numP / denP

    output.update({
        'electric_field_energy_density': (epsilon_0/2) * abs_E_squared,
        'electric_field_energy':         (epsilon_0/2) * integral_E_squared,
        'magnetic_field_energy_density': 1/(2*mu_0)    * abs_B_squared,
        'magnetic_field_energy':         1/(2*mu_0)    * integral_B_squared,
        'dominant_frequency': dominant_frequency,
        'plasma_frequency':   plasma_frequency,
        'kinetic_energy_electrons': kinetic_energy_electrons,
        'kinetic_energy_ions':      kinetic_energy_ions,
        'kinetic_energy':           kinetic_energy_electrons + kinetic_energy_ions,
        'momentum_electrons': momentum_electrons,
        'momentum_ions': momentum_ions,
        'total_momentum': total_momentum,
        'external_electric_field_energy_density': (epsilon_0/2) * abs_externalE_squared,
        'external_electric_field_energy':         (epsilon_0/2) * integral_externalE_squared,
        'external_magnetic_field_energy_density': 1/(2*mu_0)    * abs_externalB_squared,
        'external_magnetic_field_energy':         1/(2*mu_0)    * integral_externalB_squared,
        'gauss_rel_error': gauss_rel_error,
        'momentum_rel_error': momentum_rel_error
    })
    
    total_energy = (output["electric_field_energy"] + output["external_electric_field_energy"] +
                    output["magnetic_field_energy"] + output["external_magnetic_field_energy"] +
                    output["kinetic_energy"])
    
    output.update({'total_energy': total_energy})

    # ---------- GR-correct energies (when metric != flat) ----------
    metric_cfg = output.get("metric", {"kind": 0, "params": {}})
    kind = jnp.asarray(metric_cfg.get("kind", 0))
    use_gr = kind != 0

    # Flat energies already computed above:
    UE_flat = output['electric_field_energy']          # (T,)
    UB_flat = output['magnetic_field_energy']          # (T,)
    Ke_flat = output['kinetic_energy_electrons']       # (T,)
    Ki_flat = output['kinetic_energy_ions']            # (T,)

    # GR energies (compute; cheap compared to the run)
    UE_gr, UB_gr = _em_energy_split_gr(
        output['electric_field'], output['magnetic_field'],
        output['grid'], dt, plasma_frequency, metric_cfg, dx
    )
    pos_e = output.get('position_electrons', None)
    pos_i = output.get('position_ions', None)
    Ke_gr = _kinetic_energy_species_gr(output['velocity_electrons'], output['mass_electrons'], pos_e, dt, plasma_frequency, metric_cfg)
    Ki_gr = _kinetic_energy_species_gr(output['velocity_ions'],      output['mass_ions'],      pos_i, dt, plasma_frequency, metric_cfg)

    # Select per metric
    UE_sel = jnp.where(use_gr, UE_gr, UE_flat)
    UB_sel = jnp.where(use_gr, UB_gr, UB_flat)
    Ke_sel = jnp.where(use_gr, Ke_gr, Ke_flat)
    Ki_sel = jnp.where(use_gr, Ki_gr, Ki_flat)

    # Totals; keep your externals added to total (they remain flat definitions)
    total_energy_sel = UE_sel + UB_sel + Ke_sel + Ki_sel \
                        + output["external_electric_field_energy"] \
                        + output["external_magnetic_field_energy"]

    # Overwrite the same keys (always) so plotting stays simple;
    # also store the GR split explicitly (useful on flat too; equals the flat values).
    output.update({
        'electric_field_energy': UE_sel,
        'magnetic_field_energy': UB_sel,
        'kinetic_energy_electrons': Ke_sel,
        'kinetic_energy_ions':      Ki_sel,
        'kinetic_energy':           Ke_sel + Ki_sel,
        'total_energy':             total_energy_sel,
        'electric_field_energy_GR': UE_gr,
        'magnetic_field_energy_GR': UB_gr,
    })