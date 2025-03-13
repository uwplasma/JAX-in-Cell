from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ._fields import E_from_Gauss_1D_FFT, E_from_Poisson_1D_FFT, E_from_Gauss_1D_Cartesian
from ._constants import speed_of_light

__all__ = ['plot']

def plot(output):
    # Precompute constants
    v_th = output["vth_electrons_over_c"] * speed_of_light
    grid = output["grid"]
    box_size_x = output["length"]
    total_steps = output["total_steps"]
    sqrtmemi = jnp.sqrt(output["mass_electrons"][0] / output["mass_ions"][0])
    
    max_velocity_electrons = max(1.2 * jnp.max(output["velocity_electrons"]), 
                                 5 * jnp.abs(v_th) + jnp.abs(output["electron_drift_speed"]))
    max_velocity_ions = max(1.0 * jnp.max(output["velocity_ions"]),
                            sqrtmemi * 0.3 * jnp.abs(v_th) * jnp.sqrt(output["ion_temperature_over_electron_temperature"]) + jnp.abs(output["ion_drift_speed"]))
    
    bins_position = len(grid)
    bins_velocity = len(grid)
    
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
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

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
    
    # Different Electric Field Calculations
    ## Precompute E fields with vmap
    E_fields_Gauss_1D_Cartesian = vmap(lambda charge_density: E_from_Gauss_1D_Cartesian(charge_density, output["dx"]))(output["charge_density"])
    E_fields_Gauss_1D_FFT = vmap(lambda charge_density: E_from_Gauss_1D_FFT(charge_density, output["dx"]))(output["charge_density"])    
    E_fields_Poisson_1D_FFT = vmap(lambda charge_density: E_from_Poisson_1D_FFT(charge_density, output["dx"]))(output["charge_density"])

    E_field_line_Gauss_1D_Cartesian, = axes[0, 2].plot([], [], lw=2, color="red",   linestyle='-',  label="E field from Gauss 1D Cartesian")
    E_field_line_Gauss_1D_FFT,       = axes[0, 2].plot([], [], lw=2, color="blue",  linestyle='--', label="E field from Gauss 1D FFT")
    E_field_line_Poisson_1D_FFT,     = axes[0, 2].plot([], [], lw=2, color="green", linestyle='-.', label="E field from Poisson 1D FFT")
    axes[0, 2].set_xlim(-box_size_x / 2, box_size_x / 2)
    # set y lim to min and max of all E fields
    axes[0, 2].set_ylim(jnp.min(jnp.array([E_fields_Gauss_1D_Cartesian, E_fields_Gauss_1D_FFT, E_fields_Poisson_1D_FFT])),
                        jnp.max(jnp.array([E_fields_Gauss_1D_Cartesian, E_fields_Gauss_1D_FFT, E_fields_Poisson_1D_FFT])))
    axes[0, 2].set(xlabel="Position (m)", ylabel="Electric Field (V/m)")
    axes[0, 2].legend(loc='upper right')

    # Mean charge density and energy error
    axes[2, 0].plot(time, jnp.abs(jnp.mean(output["charge_density"], axis=-1))*1e12, 
                    label="Mean charge density (C/m³) x 1e12")
    axes[2, 0].plot(time, jnp.abs(output["total_energy"] - output["total_energy"][0]) / output["total_energy"][0], 
                    label="Relative energy error")
    axes[2, 0].set(title="Mean Charge Density and Energy Error", xlabel=r"Time ($\omega_{pe}^{-1}$)", 
                   ylabel="Charge density/Energy error", yscale="log")
    axes[2, 0].legend()

    # Energy plots
    axes[1, 0].plot(time, output["electric_field_energy"], label="Electric field energy")
    axes[1, 0].plot(time, output["kinetic_energy"], label="Kinetic energy")
    axes[1, 0].plot(time, output["kinetic_energy_electrons"], label="Kinetic energy electrons")
    axes[1, 0].plot(time, output["kinetic_energy_ions"], label="Kinetic energy ions")
    axes[1, 0].plot(time, output["total_energy"], label="Total energy")
    axes[1, 0].set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)", ylabel="Energy (J)", yscale="log", ylim=[1e-5, None])
    axes[1, 0].legend()

    # Animated phase space
    position_electron_line, = axes[1, 1].plot([], [], lw=2, color="red", label="Electron position")
    position_ion_line,      = axes[1, 1].plot([], [], lw=2, color="blue", label="Ion position")
    axes[1, 1].set_xlim(-box_size_x / 2, box_size_x / 2)
    axes[1, 1].set_ylim(0, max_y_positions)
    axes[1, 1].set(xlabel="Position (m)", ylabel="Number of particles")
    axes[1, 1].legend()

    velocity_electron_line, = axes[1, 2].plot([], [], lw=2, color="red", label="Electron velocity")
    velocity_ion_line,      = axes[1, 2].plot([], [], lw=2, color="blue", label=f"Ion velocity x {ve_over_vi:.2f}")
    axes[1, 2].set_xlim(-max_velocity_electrons, max_velocity_electrons)
    axes[1, 2].set_ylim(0, max_y_velocities)
    axes[1, 2].set(xlabel="Velocity (m/s)", ylabel="Number of particles")
    axes[1, 2].legend()
    
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
    electron_phase_plot = axes[2, 1].imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight", interpolation="sinc",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_electrons, max_velocity_electrons],
        vmin=jnp.min(electron_phase_histograms),vmax=jnp.max(electron_phase_histograms)
    )
    axes[2, 1].set(xlabel="Electron Position (m)", ylabel="Electron Velocity (m/s)")
    electron_phase_text = axes[2, 1].text(
        0.5, 0.9, "", transform=axes[2, 1].transAxes,
        ha="center", va="top", fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    ion_phase_plot = axes[2, 2].imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight", interpolation="sinc",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_ions, max_velocity_ions],
        vmin=jnp.min(ion_phase_histograms),vmax=jnp.max(ion_phase_histograms)
    )
    axes[2, 2].set(xlabel="Ion Position (m)", ylabel="Ion Velocity (m/s)")
    
    
    def update(frame):
        # Update position histograms
        position_electron_line.set_data(bins_position[:-1], position_hist_electrons[frame])
        position_ion_line.set_data(bins_position[:-1], position_hist_ions[frame])

        # Update velocity histograms
        velocity_electron_line.set_data(bins_velocity_electrons[:-1], velocity_hist_electrons[frame])
        velocity_ion_line.set_data(bins_velocity_ions[:-1], velocity_hist_ions[frame])

        # Update phase space plots using precomputed histograms
        electron_phase_plot.set_array(electron_phase_histograms[frame].T)
        ion_phase_plot.set_array(ion_phase_histograms[frame].T)
        
        electron_phase_text.set_text(f"Time: {time[frame]:.1f} * ωₚ")
        
        # Update electric field lines
        E_field_line_Gauss_1D_Cartesian.set_data(grid, E_fields_Gauss_1D_Cartesian[frame] )
        E_field_line_Gauss_1D_FFT.set_data(grid, E_fields_Gauss_1D_FFT[frame])
        E_field_line_Poisson_1D_FFT.set_data(grid, E_fields_Poisson_1D_FFT[frame])

        return (position_electron_line, position_ion_line, 
                velocity_electron_line, velocity_ion_line,
                electron_phase_plot, ion_phase_plot, electron_phase_text,
                E_field_line_Gauss_1D_Cartesian, E_field_line_Gauss_1D_FFT, E_field_line_Poisson_1D_FFT)

    ani = FuncAnimation(fig, update, frames=total_steps, blit=True, interval=1, repeat_delay=1000)

    plt.tight_layout()
    plt.show()
