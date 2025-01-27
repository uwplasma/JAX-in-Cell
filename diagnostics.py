from jax import vmap
from constants import epsilon_0
import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq
from constants import epsilon_0, charge_electron, mass_electron, mass_proton, speed_of_light, mu_0
import matplotlib.pyplot as plt

def diagnostics(output, print_to_terminal=True):
    E_field_over_time = output['electric_field']
    grid = output['grid']
    dt = output['dt']
    total_steps = output['total_steps']
    
    # array_to_do_fft_on = charge_density_over_time[:,len(grid)//2]
    array_to_do_fft_on = E_field_over_time[:,len(grid)//2,0]
    array_to_do_fft_on = (array_to_do_fft_on-jnp.mean(array_to_do_fft_on))/jnp.max(array_to_do_fft_on)
    plasma_frequency = output['plasma_frequency']

    fft_values = fft(array_to_do_fft_on)[:total_steps//2]
    freqs = fftfreq(total_steps, d=dt)[:total_steps//2]*2*jnp.pi # d=dt specifies the time step
    magnitude = jnp.abs(fft_values)
    peak_index = jnp.argmax(magnitude)
    dominant_frequency = jnp.abs(freqs[peak_index])

    if print_to_terminal:
        print(f"Dominant FFT frequency (f): {dominant_frequency} Hz")
        print(f"Plasma frequency (w_p):     {plasma_frequency} Hz")
        # print(f"Error: {jnp.abs(dominant_frequency - plasma_frequency) / plasma_frequency * 100:.2f}%")
    
    abs_E_squared              = jnp.sum(output['electric_field']**2, axis=-1)
    abs_externalE_squared      = jnp.sum(output['external_electric_field']**2, axis=-1)
    integral_E_squared         = jnp.trapezoid(abs_E_squared, dx=output['dx'], axis=-1)
    integral_externalE_squared = jnp.trapezoid(abs_externalE_squared, dx=output['dx'], axis=-1)
    
    abs_B_squared              = jnp.sum(output['magnetic_field']**2, axis=-1)
    abs_externalB_squared      = jnp.sum(output['external_magnetic_field']**2, axis=-1)
    integral_B_squared         = jnp.trapezoid(abs_B_squared, dx=output['dx'], axis=-1)
    integral_externalB_squared = jnp.trapezoid(abs_externalB_squared, dx=output['dx'], axis=-1)
    
    v_electrons_squared = jnp.sum(jnp.sum(output['velocity_electrons']**2, axis=-1), axis=-1)
    v_ions_squared      = jnp.sum(jnp.sum(output['velocity_ions']**2     , axis=-1), axis=-1)
    
    output.update({
        'electric_field_energy_density': (epsilon_0/2) * abs_E_squared,
        'electric_field_energy':         (epsilon_0/2) * integral_E_squared,
        'magnetic_field_energy_density': 1/(2*mu_0)    * abs_B_squared,
        'magnetic_field_energy':         1/(2*mu_0)    * integral_B_squared,
        'dominant_frequency': dominant_frequency,
        'plasma_frequency':   plasma_frequency,
        'kinetic_energy':     (1/2) * mass_electron * output['weight'] * v_electrons_squared + (1/2) * mass_proton * output['weight'] * v_ions_squared,
        'kinetic_energy_electrons': (1/2) * mass_electron * output['weight'] * v_electrons_squared,
        'kinetic_energy_ions':      (1/2) * mass_proton   * output['weight'] * v_ions_squared,
        'external_electric_field_energy_density': (epsilon_0/2) * abs_externalE_squared,
        'external_electric_field_energy':         (epsilon_0/2) * integral_externalE_squared,
        'external_magnetic_field_energy_density': 1/(2*mu_0)    * abs_externalB_squared,
        'external_magnetic_field_energy':         1/(2*mu_0)    * integral_externalB_squared
    })
    
    total_energy = (output["electric_field_energy"] + output["external_electric_field_energy"] +
                    output["magnetic_field_energy"] + output["external_magnetic_field_energy"] +
                    output["kinetic_energy"])
    
    output.update({'total_energy': total_energy})