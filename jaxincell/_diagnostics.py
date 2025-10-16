from jax import lax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from ._constants import epsilon_0, mu_0, speed_of_light as c

__all__ = ['diagnostics']

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
    KEe_t = (1/2) * mass_electrons * v_electrons_squared#jnp.sum((gamma_e - 1.0) * me_b * (c**2), axis=1)   # (T,)
    KEi_t = (1/2) * mass_ions      * v_ions_squared#jnp.sum((gamma_i - 1.0) * mi_b * (c**2), axis=1)   # (T,)

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

    pe_t = jnp.sum((gamma_e[..., None] * me_b3) * ve, axis=1)   # (T, 3)
    pi_t = jnp.sum((gamma_i[..., None] * mi_b3) * vi, axis=1)   # (T, 3)
    P_tot = pe_t + pi_t                                         # (T, 3)

    P0   = P_tot[0]
    numP = jnp.linalg.norm(P_tot - P0, axis=1)
    denP = jnp.maximum(jnp.linalg.norm(P0), 1e-300)
    momentum_rel_error = numP / denP

    output.update({
        'electric_field_energy_density': (epsilon_0/2) * abs_E_squared,
        'electric_field_energy':         (epsilon_0/2) * integral_E_squared,
        'magnetic_field_energy_density': 1/(2*mu_0)    * abs_B_squared,
        'magnetic_field_energy':         1/(2*mu_0)    * integral_B_squared,
        'dominant_frequency': dominant_frequency,
        'plasma_frequency':   plasma_frequency,
        'kinetic_energy_electrons': KEe_t,
        'kinetic_energy_ions':      KEi_t,
        'kinetic_energy':           KEe_t + KEi_t,
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