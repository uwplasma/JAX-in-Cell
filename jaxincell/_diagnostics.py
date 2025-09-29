from jax import lax
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from ._constants import epsilon_0, mu_0

__all__ = ['diagnostics']

def diagnostics(output):

    # weight arrays are redundant w.r.t. charge/mass arrays,
    # but they are convenient for histogram plotting
    output["weights"] = jnp.ones_like(output["charges"])
    for ii, species in enumerate(output["species"]):
        ssel = (output['species_ids'] == ii+2)  # hardcoded offset here is not great!!
        output["weights"] = output["weights"].at[ssel].set( species["weight_ratio"] )

    isel = (output["charges"] >= 0)[:,0]  # cannot use masks in jitted functions
    esel = (output["charges"] <  0)[:,0]
    segregated = {
        "position_electrons": output["positions"] [:, esel, :],
        "velocity_electrons": output["velocities"][:, esel, :],
        "mass_electrons":     output["masses"]    [   esel],
        "charge_electrons":   output["charges"]   [   esel],
        "weight_electrons":   output["weights"]   [   esel],
        "position_ions":      output["positions"] [:, isel, :],
        "velocity_ions":      output["velocities"][:, isel, :],
        "mass_ions":          output["masses"]    [   isel],
        "charge_ions":        output["charges"]   [   isel],
        "weight_ions":        output["weights"]   [   isel],
    }
    output.update(**segregated)
    del output["positions"]
    del output["velocities"]
    del output["masses"]
    del output["charges"]

    E_field_over_time = output['electric_field']
    grid              = output['grid']
    dt                = output['dt']
    total_steps       = output['total_steps']

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

    KE_electrons = (1/2) * jnp.expand_dims(output['mass_electrons'][...,0], 0) * jnp.sum(output['velocity_electrons']**2, axis=-1)
    KE_ions      = (1/2) * jnp.expand_dims(output['mass_ions']     [...,0], 0) * jnp.sum(output['velocity_ions']**2,      axis=-1)
    KE_electrons = jnp.sum(KE_electrons, axis=-1)
    KE_ions      = jnp.sum(KE_ions,      axis=-1)

    output.update({
        'electric_field_energy_density': (epsilon_0/2) * abs_E_squared,
        'electric_field_energy':         (epsilon_0/2) * integral_E_squared,
        'magnetic_field_energy_density': 1/(2*mu_0)    * abs_B_squared,
        'magnetic_field_energy':         1/(2*mu_0)    * integral_B_squared,
        'dominant_frequency': dominant_frequency,
        'plasma_frequency':   plasma_frequency,
        'kinetic_energy':           KE_electrons + KE_ions,
        'kinetic_energy_electrons': KE_electrons,
        'kinetic_energy_ions':      KE_ions,
        'external_electric_field_energy_density': (epsilon_0/2) * abs_externalE_squared,
        'external_electric_field_energy':         (epsilon_0/2) * integral_externalE_squared,
        'external_magnetic_field_energy_density': 1/(2*mu_0)    * abs_externalB_squared,
        'external_magnetic_field_energy':         1/(2*mu_0)    * integral_externalB_squared
    })

    total_energy = (output["electric_field_energy"] + output["external_electric_field_energy"] +
                    output["magnetic_field_energy"] + output["external_magnetic_field_energy"] +
                    output["kinetic_energy"])

    output.update({'total_energy': total_energy})
