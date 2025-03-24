# Example script to run the simulation and plot the results
import time
from jax import block_until_ready
from jaxincell import simulation, plot, load_parameters

# Read parameters from TOML file
input_parameters, solver_parameters = load_parameters('example_input.toml')

n_simulations = 1 # Check that first simulation takes longer due to JIT compilation

# Run the simulation
for i in range(n_simulations):
    if i>0: input_parameters["print_info"] = False
    start = time.time()
    output = block_until_ready(simulation(input_parameters, **solver_parameters))
    print(f"Run #{i+1}: Wall clock time: {time.time()-start}s")

# Plot the results
# plot(output)

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from matplotlib.animation import FuncAnimation

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

len_grid = len(output["grid"])
box_size_x = output["length"]
max_velocity_electrons = 1.2 * jnp.max(output["velocity_electrons"])
n_electrons = output["number_pseudoelectrons"]
total_steps = output["total_steps"]
time = output["time_array"] * output["plasma_frequency"]

time_index = 600

electron_phase_histograms = vmap(
    lambda pos, vel: jnp.histogram2d(
        pos, vel, bins=[len_grid, len_grid],
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_electrons, max_velocity_electrons]]
    )[0]
)(output["position_electrons"][:, :, 0], output["velocity_electrons"][:, :, 0])
electron_phase_histograms /= n_electrons

# Perform FFT along both the position and velocity directions and take its absolute value
electron_phase_histograms_FFT = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(electron_phase_histograms, axes=(1, 2)), axes=(1, 2)))
kx = jnp.fft.fftshift(2 * jnp.pi * jnp.fft.fftfreq(len_grid, d=box_size_x / len_grid))
kv = jnp.fft.fftshift(2 * jnp.pi * jnp.fft.fftfreq(len_grid, d=(2 * max_velocity_electrons) / len_grid))

# Plot the results in real space
electron_phase_plot = axes[0].imshow(
    jnp.zeros((len_grid, len_grid)), aspect="auto", origin="lower", cmap="twilight", interpolation="sinc",
    extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_electrons, max_velocity_electrons],
    vmin=jnp.min(electron_phase_histograms), vmax=jnp.max(electron_phase_histograms)
)
axes[0].set(xlabel="Electron Position (m)", ylabel="Electron Velocity (m/s)")
electron_phase_text = axes[0].text(
    0.5, 0.9, "", transform=axes[0].transAxes,
    ha="center", va="top", fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

# Plot the results in Fourier space
electron_phase_plot_fft = axes[1].imshow(
    jnp.zeros((len_grid, len_grid)), aspect="auto", origin="lower", cmap="twilight", interpolation="sinc",
    extent=[kx.min(), kx.max(), kv.min(), kv.max()],
    vmin=jnp.min(electron_phase_histograms_FFT), vmax=jnp.max(electron_phase_histograms_FFT)
)
axes[1].set(xlabel=r"Electron Position Fourier ($m^{-1}$)", ylabel=r"Electron Velocity Fourier ($s/m$)")

# frames_to_show = jnp.arange(0, total_steps, 0.006 * total_steps).astype(int)
frames_to_show = total_steps
def update(frame):
    electron_phase_plot.set_array(electron_phase_histograms[frame].T)  # Convert JAX to NumPy
    electron_phase_plot_fft.set_array(electron_phase_histograms_FFT[frame].T)  # Convert JAX to NumPy
    electron_phase_text.set_text(f"Time: {time[frame]:.1f} * ωₚ")
    return (electron_phase_plot, electron_phase_plot_fft, electron_phase_text,)

ani = FuncAnimation(fig, update, frames=frames_to_show, blit=True, interval=3, repeat_delay=1000)

plt.show()
