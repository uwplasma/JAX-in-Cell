import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from ._constants import epsilon_0, mu_0, boltzmann_constant, speed_of_light

__all__ = ['diagnostics']

def diagnostics(output):
    """Add diagnostics in place, preserving the raw arrays for repeated calls."""
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
    mass, velocity = output["masses"].reshape(-1), output["velocities"]
    v2 = jnp.sum(velocity ** 2, axis=-1)
    if output.get("solver_parameters", {}).get("relativistic", output.get("relativistic", False)):
        root = jnp.sqrt(1 - v2 / speed_of_light ** 2)
        gamma, kinetic_p = 1 / root, mass * v2 / (root * (1 + root))
    else:
        gamma, kinetic_p = jnp.ones_like(v2), 0.5 * mass * v2
    momentum_p = (mass * gamma)[..., None] * velocity

    # Simulation populations have identities independent of their charge, mass or weight.
    # Keep the old charge/mass grouping for dictionaries without population metadata.
    import numpy as np
    q = np.asarray(output["charges"]).reshape(-1)
    m = np.asarray(output["masses"]).reshape(-1)
    if "species_integer_index" in output:
        labels = np.asarray(output["species_integer_index"]).reshape(-1)
    else:
        _, labels = np.unique(np.stack([q, m], axis=1), axis=0, return_inverse=True)
    names = [f"{kind}.{sp.get('user_label', label)}"
             for kind in ("electrons", "ions")
             for label, sp in output.get("species_parameters", {}).get(kind, {}).items()]
    weights = jnp.asarray(output.get("weights", jnp.ones_like(output["masses"]))).reshape(-1)

    species_list = []
    for si in np.unique(labels):
        mask = (labels == si)
        pos_s = output["positions"][:, mask, :]
        vel_s = output["velocities"][:, mask, :]
        w_s, m_s = weights[mask], jnp.asarray(m[mask])
        norm = jnp.sum(w_s)
        norm = jnp.where(norm > 0, norm, 1.0)
        qv = float(output["charge_integer_lookup"][si]) if "charge_integer_lookup" in output else float(jnp.sum(q[mask]) / norm)
        mv = float(output["mass_integer_lookup"][si]) if "mass_integer_lookup" in output else float(jnp.sum(m_s) / norm)
        mean = jnp.sum(w_s[None, :, None] * vel_s, axis=1) / norm
        temperature = jnp.sum(m_s[None, :, None] * (vel_s - mean[:, None, :]) ** 2, axis=1) / (boltzmann_constant * norm)

        # Legacy names for dictionaries without configured population labels.
        if "species_integer_index" in output and si < len(names):
            name = names[si]
        elif qv < 0 and not any(sp.get("name") == "electrons" for sp in species_list):
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
            "weights": w_s,
            "temperature_components": temperature,
            "temperature": jnp.mean(temperature, axis=-1),
            "kinetic_energy": jnp.sum(kinetic_p[:, mask], axis=-1),
        })

    output["species"] = species_list

    E_field_over_time = output['electric_field']
    grid              = output['grid']
    dt_val          = output['dt']
    total_steps = E_field_over_time.shape[0]
    dt = float(output["time_array"][1] - output["time_array"][0]) if "time_array" in output and total_steps > 1 else float(dt_val)

    # FFT-based dominant frequency at the middle grid point
    array_to_do_fft_on = E_field_over_time[:, len(grid)//2, 0]
    array_to_do_fft_on = array_to_do_fft_on - jnp.mean(array_to_do_fft_on)
    plasma_frequency = output['plasma_frequency']

    half = total_steps // 2 + 1
    fft_values = fft(array_to_do_fft_on)[:half]
    freqs = fftfreq(total_steps, d=dt)[:half] * 2 * jnp.pi
    magnitude = jnp.abs(fft_values)
    peak_index = jnp.argmax(magnitude)
    dominant_frequency = jnp.abs(freqs[peak_index])

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

    total_ke_electrons = jnp.sum(kinetic_p[:, esel], axis=-1)
    total_ke_ions = jnp.sum(kinetic_p[:, isel], axis=-1)

    def temperature(velocities, masses, weights):
        total = jnp.sum(weights)
        total = jnp.where(total > 0, total, 1.)
        mean = jnp.sum(weights[None, :, None] * velocities, axis=1, keepdims=True) / total
        return jnp.sum(masses[None, :, None] * (velocities - mean) ** 2, axis=1) / (boltzmann_constant * total)

    output.update({ 
        'electric_field_energy_density': (epsilon_0/2) * abs_E_squared,
        'electric_field_energy':         (epsilon_0/2) * integral_E_squared,
        'magnetic_field_energy_density': 1/(2*mu_0)    * abs_B_squared,
        'magnetic_field_energy':         1/(2*mu_0)    * integral_B_squared,
        'dominant_frequency': dominant_frequency,
        'plasma_frequency':   plasma_frequency,
        
        'kinetic_energy':           total_ke_electrons + total_ke_ions,
        'kinetic_energy_electrons': total_ke_electrons,
        'kinetic_energy_ions':      total_ke_ions,
        'temperature_electrons': temperature(velocity[:, esel], mass[esel], weights[esel]),
        'temperature_ions': temperature(velocity[:, isel], mass[isel], weights[isel]),
        
        'external_electric_field_energy_density': (epsilon_0/2) * abs_externalE_squared,
        'external_electric_field_energy':         (epsilon_0/2) * integral_externalE_squared,
        'external_magnetic_field_energy_density': 1/(2*mu_0)    * abs_externalB_squared,
        'external_magnetic_field_energy':         1/(2*mu_0)    * integral_externalB_squared
    })

    total_energy = (output["electric_field_energy"] + output["external_electric_field_energy"] +
                    output["magnetic_field_energy"] + output["external_magnetic_field_energy"] +
                    output["kinetic_energy"])

    output.update({'total_energy': total_energy})

    total_momentum = jnp.sum(momentum_p, axis=-2)
    momentum_scale = jnp.sum(jnp.linalg.norm(momentum_p[0], axis=-1))
    output.update({
        'total_momentum': total_momentum,
        'momentum_error_rel': jnp.linalg.norm(total_momentum - total_momentum[0], axis=-1) / (momentum_scale + 1e-300),
    })

    # Gauss's law residual dE_x/dx - rho/epsilon_0 (backward difference, the stencil the
    # field solve uses), as max over x relative to max |rho/epsilon_0|; measures charge conservation
    if 'charge_density' in output:
        Ex  = E_field_over_time[..., 0]
        rhs = output['charge_density'] / epsilon_0
        periodic = int(output.get('field_BC_left', 0)) == 0 and int(output.get('field_BC_right', 0)) == 0
        Ex_left = jnp.roll(Ex, 1, axis=-1) if periodic else jnp.pad(Ex[:, :-1], ((0, 0), (1, 0)))
        gauss_residual = (Ex - Ex_left) / output['dx'] - rhs
        if periodic:  # a periodic E can only match rho up to its mean
            gauss_residual = gauss_residual - jnp.mean(gauss_residual, axis=-1, keepdims=True)
        gauss_error_Linf = jnp.max(jnp.abs(gauss_residual), axis=-1)
        output.update({
            'gauss_error_Linf': gauss_error_Linf,
            'gauss_error_Linf_rel': gauss_error_Linf / (jnp.max(jnp.abs(rhs), axis=-1) + 1e-300),
        })
