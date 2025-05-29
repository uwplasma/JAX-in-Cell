import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jax import vmap
from ._constants import speed_of_light
import numpy as np

__all__ = ['plot']

def plot(output, direction="x", threshold=1e-12):
    def is_nonzero(field):
        return jnp.max(jnp.abs(field)) > threshold

    v_th = output[f"vth_electrons_over_c_{direction}"] * speed_of_light
    grid = output["grid"]
    time = output["time_array"] * output["plasma_frequency"]
    total_steps = output["total_steps"]
    box_size_x = output["length"]

    direction_index = {"x": 0, "y": 1, "z": 2}[direction]

    # Determine which vector fields have nonzero components
    def add_field_components(field, unit, label_prefix):
        components = []
        for i, axis in enumerate("xyz"):
            data = output[field][:, :, i]
            if is_nonzero(data):
                components.append({
                    "data": data,
                    "title": f"{label_prefix} along {axis}",
                    "xlabel": f"{axis} Position (m)",
                    "ylabel": r"Time ($\omega_{pe}^{-1}$)",
                    "cbar": f"{label_prefix} ({unit})"
                })
        return components

    plots_to_make = []
    plots_to_make += add_field_components("electric_field", "V/m", "Electric Field")
    plots_to_make += add_field_components("magnetic_field", "T", "Magnetic Field")
    plots_to_make += add_field_components("current_density", "A/m²", "Current Density")

    # Charge density (always plotted)
    plots_to_make.append({
        "data": output["charge_density"],
        "title": "Charge Density",
        "xlabel": f"{direction} Position (m)",
        "ylabel": r"Time ($\omega_{pe}^{-1}$)",
        "cbar": "Charge density (C/m³)"
    })

    # Compute phase space histograms
    sqrtmemi = jnp.sqrt(output["mass_electrons"][0] / output["mass_ions"][0])
    max_velocity_electrons = max(1.2 * jnp.max(output["velocity_electrons"]),
                                 5 * jnp.abs(v_th) + jnp.abs(output[f"electron_drift_speed_{direction}"]))
    max_velocity_ions = max(1.0 * jnp.max(output["velocity_ions"]),
                            sqrtmemi * 0.3 * jnp.abs(v_th) * jnp.sqrt(output[f"ion_temperature_over_electron_temperature_{direction}"]) +
                            jnp.abs(output[f"ion_drift_speed_{direction}"]))
    ve_over_vi = max_velocity_electrons / max_velocity_ions
    bins_velocity = max(min(len(grid), 111), 71)

    electron_phase_histograms = vmap(lambda pos, vel: jnp.histogram2d(
        pos, vel, bins=[len(grid), bins_velocity],
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_electrons, max_velocity_electrons]])[0]
    )(output["position_electrons"][:, :, direction_index], output["velocity_electrons"][:, :, direction_index])

    ion_phase_histograms = vmap(lambda pos, vel: jnp.histogram2d(
        pos, vel, bins=[len(grid), bins_velocity],
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_ions, max_velocity_ions]])[0]
    )(output["position_ions"][:, :, direction_index], output["velocity_ions"][:, :, direction_index])

    # Grid layout
    ncols = 3
    n_field_plots = len(plots_to_make)
    n_total_plots = n_field_plots + 2  # add 2 for phase space
    nrows = (n_total_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.5 * nrows), squeeze=False)

    def plot_field(ax, field_data, title, xlabel, ylabel, cbar_label):
        im = ax.imshow(field_data, aspect="auto", cmap="RdBu", origin="lower",
                       extent=[grid[0], grid[-1], time[0], time[-1]])
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im, ax=ax, label=cbar_label)
        return im

    # Plot all field plots
    for i, plot_info in enumerate(plots_to_make):
        row, col = divmod(i, ncols)
        plot_field(axes[row, col], plot_info["data"],
                   plot_info["title"], plot_info["xlabel"],
                   plot_info["ylabel"], plot_info["cbar"])

    # Phase space plots
    phase_index = n_field_plots
    electron_row, electron_col = divmod(phase_index, ncols)
    ion_row, ion_col = divmod(phase_index + 1, ncols)

    electron_ax = axes[electron_row, electron_col]
    ion_ax = axes[ion_row, ion_col]

    electron_plot = electron_ax.imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_electrons, max_velocity_electrons],
        vmin=jnp.min(electron_phase_histograms), vmax=jnp.max(electron_phase_histograms))
    electron_ax.set(xlabel=f"Electron Position {direction} (m)",
                    ylabel=f"Electron Velocity {direction} (m/s)",
                    title="Electron Phase Space")

    ion_plot = ion_ax.imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_ions, max_velocity_ions],
        vmin=jnp.min(ion_phase_histograms), vmax=jnp.max(ion_phase_histograms))
    ion_ax.set(xlabel=f"Ion Position {direction} (m)",
               ylabel=f"Ion Velocity {direction} (m/s)",
               title="Ion Phase Space")

    # Time label
    animated_time_text = electron_ax.text(0.5, 0.9, "", transform=electron_ax.transAxes,
                                          ha="center", va="top", fontsize=12,
                                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Get flat list of axes
    axes_flat = np.ravel(axes)

    # Track index of next available subplot
    used_axes = len(plots_to_make)+2
    total_axes = len(axes_flat)
    
    print(f"Total axes: {total_axes}, Used axes: {used_axes}")

    if total_axes >= used_axes + 1:
        energy_ax = axes_flat[used_axes]
        energy_ax.plot(time, output["total_energy"],          label="Total energy")
        energy_ax.plot(time, output["kinetic_energy_electrons"], label="Kinetic energy electrons")
        energy_ax.plot(time, output["kinetic_energy_ions"],   label="Kinetic energy ions")
        energy_ax.plot(time, output["electric_field_energy"], label="Electric field energy")
        if jnp.max(output["magnetic_field_energy"]) > 1e-10:
            energy_ax.plot(time, output["magnetic_field_energy"], label="Magnetic field energy")
        energy_ax.plot(time[2:], jnp.abs(jnp.mean(output["charge_density"][2:], axis=-1))*1e12, label=r"Mean $\rho \times 10^{12}$")
        energy_ax.plot(time[2:], jnp.abs(output["total_energy"][2:] - output["total_energy"][2]) / output["total_energy"][2], label="Relative energy error")
        energy_ax.set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)",
                    ylabel="Energy (J)", yscale="log", ylim=[1e-7, None])
        energy_ax.legend(fontsize=7)
        

    def update(frame):
        electron_plot.set_array(electron_phase_histograms[frame].T)
        ion_plot.set_array(ion_phase_histograms[frame].T)
        animated_time_text.set_text(f"Time: {time[frame]:.2f} ωₚ")
        return electron_plot, ion_plot, animated_time_text

    ani = FuncAnimation(fig, update, frames=total_steps, blit=True, interval=1, repeat_delay=1000)

    plt.tight_layout()
    plt.show()
