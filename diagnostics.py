import jax.numpy as jnp
from jax.numpy.fft import fft, fftfreq

def system_parameters():
    plasma_frequency = jnp.sqrt(no_pseudoelectrons * weight * charge_electron**2 / (mass_electron * epsilon_0 * length))

def diagnostics():
    # array_to_do_fft_on = charge_density_over_time[:,len(grid)//2]
    array_to_do_fft_on = E_field_over_time[:,len(grid)//2,0]
    array_to_do_fft_on = (array_to_do_fft_on-jnp.mean(array_to_do_fft_on))/jnp.max(array_to_do_fft_on)
    plasma_frequency = system_parameters()

    fft_values = fft(array_to_do_fft_on)[:total_steps//2]
    freqs = fftfreq(total_steps, d=dt)[:total_steps//2]*2*jnp.pi # d=dt specifies the time step
    magnitude = jnp.abs(fft_values)
    peak_index = jnp.argmax(magnitude)
    dominant_frequency = jnp.abs(freqs[peak_index])

    print(f"Dominant FFT frequency (f): {dominant_frequency} Hz")
    print(f"Plasma frequency (w_p):     {plasma_frequency} Hz")
    print(f"Error: {jnp.abs(dominant_frequency - plasma_frequency) / plasma_frequency * 100:.2f}%")