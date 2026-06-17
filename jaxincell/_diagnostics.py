from jax import lax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from ._constants import epsilon_0, mu_0

__all__ = ['diagnostics']

def diagnostics(output):
    # --- Keep legacy split first (unchanged) ---
    isel = (output["charges"] >= 0)[:, 0]  # cannot use masks in jitted functions
    esel = (output["charges"] <  0)[:, 0]
    segregated = {
        "position_electrons": output["positions"] [:, esel, :],
        "velocity_electrons": output["velocities"][:, esel, :],
        "mass_electrons":     output["masses"]    [   esel],
        "charge_electrons":   output["charges"]   [   esel],
        "position_ions":      output["positions"] [:, isel, :],
        "velocity_ions":      output["velocities"][:, isel, :],
        "mass_ions":          output["masses"]    [   isel],
        "charge_ions":        output["charges"]   [   isel],
    }
    output.update(**segregated)

    # --- NEW: multi-species view, fully additive/back-compat ---
    # Group by (q, m) exact pairs
    import numpy as np
    q = np.asarray(output["charges"]).reshape(-1)
    m = np.asarray(output["masses"]).reshape(-1)
    qm = np.stack([q, m], axis=1)
    unique_pairs, labels = np.unique(qm, axis=0, return_inverse=True)

    species_list = []
    for si, (qv, mv) in enumerate(unique_pairs):
        mask = (labels == si)
        pos_s = output["positions"][:, mask, :]
        vel_s = output["velocities"][:, mask, :]

        # Names chosen to keep the first negative = "electrons", first positive = "ions"
        if qv < 0 and not any(sp.get("name") == "electrons" for sp in species_list):
            name = "electrons"
        elif qv > 0 and not any(sp.get("name") == "ions" for sp in species_list):
            name = "ions"
        else:
            name = f"species_{si}"

        species_list.append({
            "name": name,
            "charge": float(qv),
            "mass": float(mv),
            "positions": pos_s,
            "velocities": vel_s,
        })

    output["species"] = species_list

    # --- Preserve legacy memory behavior exactly ---
    del output["positions"]
    del output["velocities"]
    del output["masses"]
    del output["charges"]

    # CHANGED: Use reshape(-1) to flatten shape from (N, 1) to (N,)
    # This allows broadcasting against velocity array of shape (Time, N)
    mass_electrons_array = output["mass_electrons"].reshape(-1)
    mass_ions_array      = output["mass_ions"].reshape(-1)

    # --- All your existing energy/FFT/diagnostics code continues below unchanged ---
    E_field_over_time = output['electric_field']
    grid              = output['grid']

    # Make sure these are plain Python scalars for indexing / fftfreq
    total_steps_val = output['total_steps']
    dt_val          = output['dt']

    # Coerce JAX scalars or numpy scalars to Python ints/floats
    total_steps = int(total_steps_val)
    dt          = float(dt_val)

    # Now safe to use in len/indexing/XLA primitives on all Python versions
    # (including 3.8)
    # ------------------------------------------------------------------
    # FFT-based dominant frequency at the middle grid point
    array_to_do_fft_on = E_field_over_time[:, len(grid)//2, 0]
    array_to_do_fft_on = (array_to_do_fft_on - jnp.mean(array_to_do_fft_on)) / jnp.max(array_to_do_fft_on)
    plasma_frequency = output['plasma_frequency']

    half = total_steps // 2
    fft_full = fft(array_to_do_fft_on)
    fft_values = lax.slice(fft_full, (0,), (half,))
    freqs = fftfreq(total_steps, d=dt)[:half] * 2 * jnp.pi
    magnitude = jnp.abs(fft_values)
    peak_index = jnp.argmax(magnitude)
    dominant_frequency = jnp.abs(freqs[peak_index])

    # def integrate(y, dx):
    #     return 0.5 * (jnp.asarray(dx) * (y[..., 1:] + y[..., :-1])).sum(-1)
    
    def integrate(y, dx):
        return jnp.sum(y, axis=-1) * dx

    abs_E_squared              = jnp.sum(output['electric_field']**2, axis=-1)
    abs_externalE_squared      = jnp.sum(output['external_electric_field']**2, axis=-1)
    integral_E_squared         = integrate(abs_E_squared, dx=output['dx'])
    integral_externalE_squared = integrate(abs_externalE_squared, dx=output['dx'])

    abs_B_squared              = jnp.sum(output['magnetic_field']**2, axis=-1)
    abs_externalB_squared      = jnp.sum(output['external_magnetic_field']**2, axis=-1)
    integral_B_squared         = integrate(abs_B_squared, dx=output['dx'])
    integral_externalB_squared = integrate(abs_externalB_squared, dx=output['dx'])

    # CHANGED: Calculate v^2 per particle (sum over spatial dims x,y,z only)
    # Shape becomes (Time, N_particles)
    v_sq_electrons_per_particle = jnp.sum(output['velocity_electrons']**2, axis=-1)
    v_sq_ions_per_particle      = jnp.sum(output['velocity_ions']**2,      axis=-1)

    # CHANGED: Calculate KE = sum(0.5 * m_i * v_i^2)
    # Mass (N,) broadcasts against V_sq (Time, N) -> Result (Time, N)
    # Sum over axis=-1 (particles) -> Result (Time,)
    total_ke_electrons = 0.5 * jnp.sum(mass_electrons_array * v_sq_electrons_per_particle, axis=-1)
    total_ke_ions      = 0.5 * jnp.sum(mass_ions_array      * v_sq_ions_per_particle,      axis=-1)
    
    # Debug print (optional, can be removed)
    # print(mass_electrons_array) 

    output.update({ 
        'electric_field_energy_density': (epsilon_0/2) * abs_E_squared,
        'electric_field_energy':         (epsilon_0/2) * integral_E_squared,
        'magnetic_field_energy_density': 1/(2*mu_0)    * abs_B_squared,
        'magnetic_field_energy':         1/(2*mu_0)    * integral_B_squared,
        'dominant_frequency': dominant_frequency,
        'plasma_frequency':   plasma_frequency,
        
        # Updated kinetic energy keys to use the corrected totals
        'kinetic_energy':           total_ke_electrons + total_ke_ions,
        'kinetic_energy_electrons': total_ke_electrons,
        'kinetic_energy_ions':      total_ke_ions,
        
        'external_electric_field_energy_density': (epsilon_0/2) * abs_externalE_squared,
        'external_electric_field_energy':         (epsilon_0/2) * integral_externalE_squared,
        'external_magnetic_field_energy_density': 1/(2*mu_0)    * abs_externalB_squared,
        'external_magnetic_field_energy':         1/(2*mu_0)    * integral_externalB_squared
    })

    total_energy = (output["electric_field_energy"] + output["external_electric_field_energy"] +
                    output["magnetic_field_energy"] + output["external_magnetic_field_energy"] +
                    output["kinetic_energy"])

    output.update({'total_energy': total_energy})