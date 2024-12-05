from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from constants import speed_of_light, mass_proton, mass_electron

def plot_results(output):
    # Precompute constants
    v_th = output["vth_electrons_over_c"] * speed_of_light
    grid = output["grid"]
    box_size_x = output["length"]
    total_steps = output["total_steps"]
    sqrtmemi = jnp.sqrt(mass_electron / mass_proton)
    
    max_velocity_electrons = max(1.2 * jnp.max(output["velocity_electrons"]), 
                                 5 * jnp.abs(v_th) + jnp.abs(output["electron_drift_speed"]))
    max_velocity_ions = max(1.0 * jnp.max(output["velocity_ions"]),
                            sqrtmemi * 0.3 * jnp.abs(v_th) * jnp.sqrt(output["ion_temperature_over_electron_temperature"]) + jnp.abs(output["ion_drift_speed"]))
    
    bins_position = len(grid)
    bins_velocity = len(grid)*2
    
    time = output["time_array"] * output["plasma_frequency"]

    # Precompute histograms
    ve_over_vi = max_velocity_electrons / max_velocity_ions
    bins_position = jnp.linspace(-box_size_x / 2, box_size_x / 2, len(grid) + 1)
    bins_velocity_electrons = jnp.linspace(-max_velocity_electrons, max_velocity_electrons, len(grid) + 1)
    bins_velocity_ions = jnp.linspace(-max_velocity_ions*ve_over_vi, max_velocity_ions*ve_over_vi, len(grid) + 1)
    
    compute_histogram = lambda data, bins: vmap(lambda x: jnp.histogram(x, bins=bins)[0])(data)
    position_hist_electrons = compute_histogram(output["position_electrons"][:, :, 0], bins_position)
    position_hist_ions = compute_histogram(output["position_ions"][:, :, 0], bins_position)
    velocity_hist_electrons = compute_histogram(output["velocity_electrons"][:, :, 0], bins_velocity_electrons)
    velocity_hist_ions = compute_histogram(output["velocity_ions"][:, :, 0]*ve_over_vi, bins_velocity_ions)

    max_y_positions = max(jnp.max(position_hist_electrons), jnp.max(position_hist_ions))
    max_y_velocities = max(jnp.max(velocity_hist_electrons), jnp.max(velocity_hist_ions))

    # Setup plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    plt.subplots_adjust(hspace=0.7, wspace=0.4)

    # Electric field plot
    im1 = axes[0, 0].imshow(
        output["electric_field"][:, :, 0], aspect="auto", cmap="RdBu", origin="lower",
        extent=[grid[0], grid[-1], 0, time[-1]]
    )
    axes[0, 0].set(title="Electric Field", xlabel="Position (m)", ylabel=r"Time ($\omega_{pe}^{-1}$)")
    fig.colorbar(im1, ax=axes[0, 0], label="Electric field (V/m)")

    # Charge density plot
    im2 = axes[0, 1].imshow(
        output["charge_density"], aspect="auto", cmap="RdBu", origin="lower",
        extent=[grid[0], grid[-1], 0, time[-1]]
    )
    axes[0, 1].set(title="Charge Density", xlabel="Position (m)", ylabel=r"Time ($\omega_{pe}^{-1}$)")
    fig.colorbar(im2, ax=axes[0, 1], label="Charge density (C/m³)")

    # Mean charge density and energy error
    total_energy = (output["electric_field_energy"] +
                    output["magnetic_field_energy"] +
                    output["kinetic_energy"])
    axes[0, 2].plot(time[3:], jnp.abs(jnp.mean(output["charge_density"], axis=-1))[3:], 
                    label="Mean charge density (C/m³)")
    axes[0, 2].plot(time[3:], jnp.abs(total_energy[3:] - total_energy[3]) / total_energy[3], 
                    label="Relative energy error")
    axes[0, 2].set(title="Mean Charge Density and Energy Error", xlabel=r"Time ($\omega_{pe}^{-1}$)", 
                   ylabel="Charge density/Energy error", yscale="log")
    axes[0, 2].legend()

    # Energy plots
    axes[0, 3].plot(time, output["electric_field_energy"]/output["electric_field_energy"][0], label="Electric field energy")
    axes[0, 3].plot(time, output["kinetic_energy"]/output["kinetic_energy"][0], label="Kinetic energy")
    axes[0, 3].plot(time, output["kinetic_energy_electrons"]/output["kinetic_energy_electrons"][0], label="Kinetic energy electrons")
    axes[0, 3].plot(time, output["kinetic_energy_ions"]/output["kinetic_energy_ions"][0], label="Kinetic energy ions")
    axes[0, 3].plot(time, total_energy/total_energy[0], label="Total energy")
    axes[0, 3].set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel="Energy (J)", yscale="log", ylim=[1e-2, None])
    axes[0, 3].legend()

    # Animated phase space
    position_electron_line, = axes[1, 0].plot([], [], lw=2, color="red", label="Electron position")
    position_ion_line, = axes[1, 0].plot([], [], lw=2, color="blue", label="Ion position")
    axes[1, 0].set_xlim(-box_size_x / 2, box_size_x / 2)
    axes[1, 0].set_ylim(0, max_y_positions)
    axes[1, 0].set(xlabel="Position (m)", ylabel="Number of particles")
    axes[1, 0].legend()

    velocity_electron_line, = axes[1, 1].plot([], [], lw=2, color="red", label="Electron velocity")
    velocity_ion_line, = axes[1, 1].plot([], [], lw=2, color="blue", label=f"Ion velocity x {ve_over_vi:.2f}")
    axes[1, 1].set_xlim(-max_velocity_electrons, max_velocity_electrons)
    axes[1, 1].set_ylim(0, max_y_velocities)
    axes[1, 1].set(xlabel="Velocity (m/s)", ylabel="Number of particles")
    axes[1, 1].legend()
    
    # Precompute phase space histograms
    electron_phase_histograms = vmap(
        lambda pos, vel: jnp.histogram2d(
            pos, vel, bins=[len(grid), bins_velocity],
            range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_electrons, max_velocity_electrons]]
        )[0]
    )(output["position_electrons"][:, :, 0], output["velocity_electrons"][:, :, 0])

    ion_phase_histograms = vmap(
        lambda pos, vel: jnp.histogram2d(
            pos, vel, bins=[len(grid), bins_velocity],
            range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_ions, max_velocity_ions]]
        )[0]
    )(output["position_ions"][:, :, 0], output["velocity_ions"][:, :, 0])
    
    # Initialize phase space plots
    electron_phase_plot = axes[1, 2].imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight", interpolation="sinc",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_electrons, max_velocity_electrons],
        vmin=jnp.min(electron_phase_histograms),vmax=jnp.max(electron_phase_histograms)
    )
    ion_phase_plot = axes[1, 3].imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight", interpolation="sinc",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_ions, max_velocity_ions],
        vmin=jnp.min(ion_phase_histograms),vmax=jnp.max(ion_phase_histograms)
    )
    axes[1, 2].set(xlabel="Electron Position (m)", ylabel="Electron Velocity (m/s)")
    axes[1, 3].set(xlabel="Ion Position (m)", ylabel="Ion Velocity (m/s)")
    
    def update(frame):
        # Update position histograms
        position_electron_line.set_data(bins_position[:-1], position_hist_electrons[frame])
        position_ion_line.set_data(bins_position[:-1], position_hist_ions[frame])

        # Update velocity histograms
        velocity_electron_line.set_data(bins_velocity_electrons[:-1], velocity_hist_electrons[frame])
        velocity_ion_line.set_data(bins_velocity_ions[:-1], velocity_hist_ions[frame])

        # Update phase space plots using precomputed histograms
        electron_phase_plot.set_data(electron_phase_histograms[frame].T)
        ion_phase_plot.set_data(ion_phase_histograms[frame].T)

        return (position_electron_line, position_ion_line, 
                velocity_electron_line, velocity_ion_line,
                electron_phase_plot, ion_phase_plot)

    ani = FuncAnimation(fig, update, frames=total_steps, blit=True, interval=10*300/total_steps, repeat_delay=10000)

    plt.tight_layout()
    plt.show()
