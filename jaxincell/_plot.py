import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jax.debug import print as jprint
from ._constants import speed_of_light
from matplotlib.animation import FuncAnimation

__all__ = ['plot']

def plot(output, direction="x", threshold=1e-12):
    def is_nonzero(field):
        return jnp.max(jnp.abs(field)) > threshold

    grid = output["grid"]
    time = output["time_array"] * output["plasma_frequency"]
    total_steps = output["total_steps"]
    box_size_x = output["length"]
    number_pseudoelectrons = output["number_pseudoelectrons"]

    if len(direction) == 1 and direction in "xyz":
        direction1 = direction
        direction2 = None
        direction_index1 = {"x": 0, "y": 1, "z": 2}[direction]
        second_direction = False
        direction_index2 = None
        vth_e_1 = output["vth_electrons_over_c_" + direction1] * speed_of_light
        vth_i_1 = jnp.sqrt(jnp.abs(output["ion_temperature_over_electron_temperature_" + direction1])) * output["vth_electrons_over_c_" + direction1] * speed_of_light
    elif len(direction) == 2 and all(c in "xyz" for c in direction):
        direction1, direction2 = direction
        direction_index1 = {"x": 0, "y": 1, "z": 2}[direction[0]]
        direction_index2 = {"x": 0, "y": 1, "z": 2}[direction[1]]
        second_direction = True
        vth_e_1 = output["vth_electrons_over_c_" + direction1] * speed_of_light
        vth_i_1 = jnp.sqrt(jnp.abs(output["ion_temperature_over_electron_temperature_" + direction1])) * output["vth_electrons_over_c_" + direction1] * speed_of_light
        vth_e_2 = output["vth_electrons_over_c_" + direction2] * speed_of_light
        vth_i_2 = jnp.sqrt(jnp.abs(output["ion_temperature_over_electron_temperature_" + direction2])) * output["vth_electrons_over_c_" + direction2] * speed_of_light
    else:
        raise ValueError("direction must be one or two of 'x', 'y', or 'z'")
    # direction_index = {"x": 0, "y": 1, "z": 2}[direction]

    # Determine which vector fields have nonzero components
    def add_field_components(field, unit, label_prefix):
        components = []
        for i, axis in enumerate("xyz"):
            data = output[field][:, :, i]
            if is_nonzero(data):
                components.append({
                    "data": data,
                    "title": f"{label_prefix} in the {axis} direction",
                    "xlabel": f"x Position (m)",
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
        "xlabel": f"x Position (m)",
        "ylabel": r"Time ($\omega_{pe}^{-1}$)",
        "cbar": "Charge density (C/m³)"
    })

    # Compute phase space histograms
    sqrtmemi = jnp.sqrt(output["mass_electrons"][0] / output["mass_ions"][0])
    max_velocity_electrons_1 = max(1.0 * jnp.max(output["velocity_electrons"][:, :, direction_index1]),
                                 2.5 * jnp.abs(vth_e_1)
                                 + jnp.abs(output[f"electron_drift_speed_{direction1}"]))
    max_velocity_ions_1      = max(1.0 * jnp.max(output["velocity_ions"][:, :, direction_index1]),
                                 sqrtmemi * 0.3 * jnp.abs(vth_i_1) * jnp.sqrt(output[f"ion_temperature_over_electron_temperature_{direction1}"])
                                 + jnp.abs(output[f"ion_drift_speed_{direction1}"]))
    max_velocity_ions_1 = float(jnp.asarray(max_velocity_ions_1))
    bins_velocity = max(min(len(grid), 111), 71)

    electron_phase_histograms = vmap(lambda pos, vel: jnp.histogram2d(
        pos, vel, bins=[len(grid), bins_velocity],
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_electrons_1, max_velocity_electrons_1]])[0]
    )(output["position_electrons"][:, :, direction_index1], output["velocity_electrons"][:, :, direction_index1])

    ion_phase_histograms = vmap(lambda pos, vel: jnp.histogram2d(
        pos, vel, bins=[len(grid), bins_velocity],
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_ions_1, max_velocity_ions_1]])[0]
    )(output["position_ions"][:, :, direction_index1], output["velocity_ions"][:, :, direction_index1])

    # Grid layout
    ncols = 3
    n_field_plots = len(plots_to_make)
    n_total_plots = n_field_plots + 2  # add 2 for phase space
    if second_direction:
        n_total_plots += 3 # add 2 for phase space in second direction and 1 for xy locations
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
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_electrons_1, max_velocity_electrons_1],
        vmin=jnp.min(electron_phase_histograms), vmax=jnp.max(electron_phase_histograms))
    electron_ax.set(xlabel=f"Electron Position {direction1} (m)",
                    ylabel=f"Electron Velocity {direction1} (m/s)",
                    title=f"Electron Phase Space {direction1}")
    fig.colorbar(electron_plot, ax=electron_ax, label="Electron count")

    ion_plot = ion_ax.imshow(
        jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight",
        extent=[-box_size_x / 2, box_size_x / 2, -max_velocity_ions_1, max_velocity_ions_1],
        vmin=jnp.min(ion_phase_histograms), vmax=jnp.max(ion_phase_histograms))
    ion_ax.set(xlabel=f"Ion Position {direction1} (m)",
               ylabel=f"Ion Velocity {direction1} (m/s)",
               title=f"Ion Phase Space {direction1}")
    fig.colorbar(ion_plot, ax=ion_ax, label="Ion count")

    # Time label
    animated_time_text = electron_ax.text(0.5, 0.9, "", transform=electron_ax.transAxes,
                                          ha="center", va="top", fontsize=12,
                                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Get flat list of axes
    axes_flat = np.ravel(axes)

    # Track index of next available subplot
    used_axes = len(plots_to_make)+2
    total_axes = len(axes_flat)

    if total_axes >= used_axes + 1:
        energy_ax = axes_flat[used_axes]
        energy_ax.plot(time, output["total_energy"],          label="Total energy")
        energy_ax.plot(time, output["kinetic_energy_electrons"], label="Kinetic energy electrons")
        energy_ax.plot(time, output["kinetic_energy_ions"],   label="Kinetic energy ions")
        energy_ax.plot(time, output["electric_field_energy"], label="Electric field energy")
        if jnp.max(output["magnetic_field_energy"]) > 1e-10:
            energy_ax.plot(time, output["magnetic_field_energy"], label="Magnetic field energy")
        energy_ax.plot(time[2:], jnp.abs(jnp.mean(output["charge_density"][2:], axis=-1))*1e15, label=r"Mean $\rho \times 10^{15}$")
        energy_ax.plot(time[1:], jnp.abs(output["total_energy"][1:] - output["total_energy"][0]) / output["total_energy"][0], label="Relative energy error")
        energy_ax.set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)",
                    ylabel="Energy (J)", yscale="log", ylim=[1e-7, None])
        energy_ax.legend(fontsize=7)
    
    if second_direction:
        electron_ax2 = axes_flat[used_axes + 1]
        ion_ax2 = axes_flat[used_axes + 2]
        positions_ax = axes_flat[used_axes + 3]
        
        # Phase space in second direction
        max_velocity_electrons_2 = max(1.0 * jnp.max(output["velocity_electrons"][:, :, direction_index2]),
                                    2.5 * jnp.abs(vth_e_2) + jnp.abs(output[f"electron_drift_speed_{direction2}"]))
        max_velocity_ions_2 = max(1.0 * jnp.max(output["velocity_ions"][:, :, direction_index2]),
                                sqrtmemi * 0.3 * jnp.abs(vth_i_2) * jnp.sqrt(output[f"ion_temperature_over_electron_temperature_{direction2}"]) +
                                jnp.abs(output[f"ion_drift_speed_{direction2}"]))
        
        max_velocity_electrons_12 = max(max_velocity_electrons_1, max_velocity_electrons_2)
        max_velocity_ions_12      = max(max_velocity_ions_1, max_velocity_ions_2)
        electron_phase_histograms2 = vmap(lambda pos, vel: jnp.histogram2d(
            pos, vel, bins=[len(grid), bins_velocity],
            range=[[-max_velocity_electrons_12, max_velocity_electrons_12], [-max_velocity_electrons_12, max_velocity_electrons_12]])[0]
        )(output["velocity_electrons"][:, :, direction_index1], output["velocity_electrons"][:, :, direction_index2])

        ion_phase_histograms2 = vmap(lambda pos, vel: jnp.histogram2d(
            pos, vel, bins=[len(grid), bins_velocity],
            range=[[-max_velocity_ions_12, max_velocity_ions_12], [-max_velocity_ions_12, max_velocity_ions_12]])[0]
        )(output["velocity_ions"][:, :, direction_index1], output["velocity_ions"][:, :, direction_index2])

        electron_plot2 = electron_ax2.imshow(
            jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight",
            extent=[-max_velocity_electrons_12, max_velocity_electrons_12, -max_velocity_electrons_12, max_velocity_electrons_12],
            vmin=jnp.min(electron_phase_histograms2), vmax=jnp.max(electron_phase_histograms2))
        electron_ax2.set(xlabel=f"Electron Velocity {direction1} (m/s)",
                         ylabel=f"Electron Velocity {direction2} (m/s)",
                         title=f"Electron Phase Space v{direction1} vs v{direction2}")

        ion_plot2 = ion_ax2.imshow(
            jnp.zeros((len(grid), bins_velocity)), aspect="auto", origin="lower", cmap="twilight",
            extent=[-max_velocity_ions_12, max_velocity_ions_12, -max_velocity_ions_12, max_velocity_ions_12],
            vmin=jnp.min(ion_phase_histograms2), vmax=jnp.max(ion_phase_histograms2))
        ion_ax2.set(xlabel=f"Ion Velocity {direction1} (m/s)",
                    ylabel=f"Ion Velocity {direction2} (m/s)",
                    title=f"Ion Phase Space v{direction1} vs v{direction2}")
        
        B_field_densities = output["magnetic_field_energy_density"]
        B0 = np.asarray(B_field_densities[0])
        global_max = np.asarray(B_field_densities).max()
        if B0.ndim != 2: B0 = np.tile(B0, reps=(10, 1))
        scat_r = positions_ax.scatter([0], [0], marker='<', color='red', label='Electrons')
        scat_b = positions_ax.scatter([0], [0], marker='>', color='blue', label='Ions')
        im = positions_ax.imshow(B0.T, extent=[-box_size_x/2, box_size_x/2, -box_size_x/2, box_size_x/2],
                    origin='lower', interpolation='bilinear', vmin=0, vmax=global_max, aspect='auto')
        cb = fig.colorbar(im, ax=positions_ax, label='B-field density')
        positions_ax.set_xlim([-box_size_x/2, box_size_x/2])
        positions_ax.set_ylim([-box_size_x/2, box_size_x/2])
        positions_ax.set_xlabel(direction2)
        positions_ax.set_ylabel(direction1)
        positions_ax.legend(loc='upper right', fontsize=7)
        rng = np.random.default_rng(42)
        n_particles_to_plot = min(100, number_pseudoelectrons)
        subset_ions      = rng.choice(number_pseudoelectrons, size=n_particles_to_plot, replace=False)
        subset_electrons = rng.choice(number_pseudoelectrons, size=n_particles_to_plot, replace=False)
        scat_r.set_animated(True)
        scat_b.set_animated(True)
        im.set_animated(True)
        
        # Time label
        animated_time_text2 = positions_ax.text(0.5, 0.9, "", transform=positions_ax.transAxes,
                                            ha="center", va="top", fontsize=12,
                                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add marker explanation text
        positions_ax.text(0.02, 0.98, "Electrons: '<'", color="red",
                  transform=positions_ax.transAxes, ha="left", va="top",
                  fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        positions_ax.text(0.02, 0.92, "Ions: '>'", color="blue",
                  transform=positions_ax.transAxes, ha="left", va="top",
                  fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
    def update(frame):
        electron_plot.set_array(electron_phase_histograms[frame].T)
        ion_plot.set_array(ion_phase_histograms[frame].T)
        animated_time_text.set_text(f"Time: {time[frame]:.2f} ωₚ")
        if second_direction:
            electron_plot2.set_array(electron_phase_histograms2[frame].T)
            ion_plot2.set_array(ion_phase_histograms2[frame].T)
            
            x_electrons = np.asarray(output["position_electrons"][frame, subset_electrons, direction_index2])
            z_electrons = np.asarray(output["position_electrons"][frame, subset_electrons, direction_index1])
            x_ions = np.asarray(output["position_ions"][frame, subset_ions, direction_index2])
            z_ions = np.asarray(output["position_ions"][frame, subset_ions, direction_index1])
            scat_r.set_offsets(np.stack([x_electrons, z_electrons], axis=-1))
            scat_b.set_offsets(np.stack([x_ions, z_ions], axis=-1))
            B = np.asarray(B_field_densities[frame])
            if B.ndim != 2: B = np.tile(B, reps=(10, 1))
            im.set_array(B.T)
            animated_time_text2.set_text(f"Time: {time[frame]:.2f} ωₚ")

            return (electron_plot, ion_plot, electron_plot2, ion_plot2, animated_time_text, scat_r, scat_b, im, animated_time_text2)
        else:
            return (electron_plot, ion_plot, animated_time_text)

    ani = FuncAnimation(fig, update, frames=total_steps, blit=True, interval=1, repeat_delay=1000)

    plt.tight_layout()
    plt.show()
