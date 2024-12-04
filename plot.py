from jax import vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
from constants import speed_of_light, mass_proton, mass_electron
from matplotlib.animation import FuncAnimation

def plot_results(output):
    v_th = output["vth_electrons_over_c"] * speed_of_light
    grid = output["grid"]
    box_size_x = output["length"]
    total_steps = output["total_steps"]
    sqrtmime = jnp.sqrt(mass_proton/mass_electron)
    max_velocity_electrons = max(1.2*jnp.max(output['velocity_electrons']),                                       5   * jnp.abs(v_th) + jnp.abs(output["electron_drift_speed"]))
    max_velocity_ions      = max(1.0*jnp.max(output['velocity_ions']     ), jnp.sqrt(mass_electron/mass_proton) * 0.3 * jnp.abs(v_th) + jnp.abs(output["ion_drift_speed"]))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    plt.subplots_adjust(hspace=0.7, wspace=0.4)

    # Electric field plot
    im1 = axes[0, 0].imshow(
        output["electric_field"][:, :, 0], aspect='auto', cmap='RdBu', origin='lower',
        extent=[output["grid"][0], output["grid"][-1], 0, total_steps * output["dt"]]
    )
    axes[0, 0].set(title='Electric Field', xlabel='Position (m)', ylabel='Time (s)')
    fig.colorbar(im1, ax=axes[0, 0], label='Electric field (V/m)')

    # Charge density plot
    im2 = axes[0, 1].imshow(
        output["charge_density"], aspect='auto', cmap='RdBu', origin='lower',
        extent=[output["grid"][0], output["grid"][-1], 0, total_steps * output["dt"]]
    )
    axes[0, 1].set(title='Charge Density', xlabel='Position (m)', ylabel='Time (s)')
    fig.colorbar(im2, ax=axes[0, 1], label='Charge density (C/m³)')
    
    # Mean charge density and energy error
    total_energy = output["electric_field_energy"] + output["magnetic_field_energy"] + output["kinetic_energy"]
    axes[0, 2].plot(output["time_array"][3:], jnp.abs(jnp.mean(output["charge_density"], axis=-1))[3:], label='Mean charge density (C/m³)')
    axes[0, 2].plot(output["time_array"][3:], jnp.abs(total_energy[3:]-total_energy[3])/total_energy[3], label='Relative energy error')
    axes[0, 2].set(title='Mean Charge Density and Energy Error', xlabel='Time (s)', ylabel='Charge density/Energy error', yscale='log')
    axes[0, 2].legend()

    # Electric field energy
    axes[0, 3].plot(output["time_array"], output["electric_field_energy"], label='Electric field energy')
    axes[0, 3].plot(output["time_array"], output["kinetic_energy"], label='Kinetic energy')
    axes[0, 3].plot(output["time_array"], output["kinetic_energy_electrons"], label='Kinetic energy electrons')
    axes[0, 3].plot(output["time_array"], output["kinetic_energy_ions"], label='Kinetic energy ions')
    axes[0, 3].plot(output["time_array"], total_energy, label='Total energy')
    axes[0, 3].set(title='Energy', xlabel='Time (s)', ylabel='Energy (J)')
    axes[0, 3].legend()
    
    # Find the maximum histogram value across all frames for both electrons and ions
    bins = jnp.linspace(-box_size_x / 2, box_size_x / 2, len(grid) + 1)
    electron_histograms = vmap(lambda x: jnp.histogram(x, bins=bins)[0])(output['position_electrons'][:, :, 0])
    ion_histograms      = vmap(lambda x: jnp.histogram(x, bins=bins)[0])(output['position_ions'][:, :, 0])
    max_y_all_frames_positions = jnp.max(jnp.concatenate([electron_histograms, ion_histograms]))
    
    bins_electrons = jnp.linspace(-max_velocity_electrons, max_velocity_electrons, len(grid) + 1)
    bins_ions      = jnp.linspace(-max_velocity_ions     , max_velocity_ions     , len(grid) + 1)
    electron_histograms = vmap(lambda x: jnp.histogram(x, bins=bins_electrons)[0])(output['velocity_electrons'][:, :, 0])
    ion_histograms      = vmap(lambda x: jnp.histogram(x, bins=bins_ions     )[0])(output['velocity_ions'][:, :, 0])
    max_y_all_frames_velocities = jnp.max(jnp.concatenate([electron_histograms, ion_histograms]))

    # Animated phase space for electrons
    def update(frame):
        # Distribution of electrons and ions
        axes[1, 0].clear()
        axes[1, 0].hist(output['position_electrons'][frame, :, 0], jnp.linspace(-box_size_x / 2, box_size_x / 2, len(grid) + 1), color='red' , label='electrons', alpha=0.5)
        axes[1, 0].hist(output['position_ions'][frame, :, 0]     , jnp.linspace(-box_size_x / 2, box_size_x / 2, len(grid) + 1), color='blue', label='ions'     , alpha=0.5)
        axes[1, 0].set(title=f'Positions at timestep {frame}/{total_steps}', xlabel='Position (m)', ylabel='Number of particles')
        axes[1, 0].set_xlim(-box_size_x / 2, box_size_x / 2)
        axes[1, 0].set_ylim(0, max_y_all_frames_positions)
        axes[1, 0].legend(loc='lower right')
        
        axes[1, 1].clear()
        me_over_mi = max_velocity_electrons / max_velocity_ions
        axes[1, 1].hist(           output['velocity_electrons'][frame, :, 0], jnp.linspace(-max_velocity_electrons      , max_velocity_electrons      , len(grid) + 1), color='red' , label='electrons'      , alpha=0.5)
        axes[1, 1].hist(me_over_mi*output['velocity_ions'     ][frame, :, 0], jnp.linspace(-max_velocity_ions*me_over_mi, max_velocity_ions*me_over_mi, len(grid) + 1), color='blue', label=f'ions*{me_over_mi:.1e}', alpha=0.5)
        axes[1, 1].set(title=f'Velocity Distribution at timestep {frame}/{total_steps}', xlabel='Velocity (m/s)', ylabel='Number of particles')
        axes[1, 1].set_ylim(0, max_y_all_frames_velocities)
        axes[1, 1].set_xlim(-max_velocity_electrons, max_velocity_electrons)
        axes[1, 1].legend(loc='upper right')
        
        axes[1, 2].clear()
        h = axes[1, 2].hist2d(
            output['position_electrons'][frame, :, 0],
            output['velocity_electrons'][frame, :, 0],
            bins=jnp.array([len(grid), 50]),
            range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_electrons, max_velocity_electrons]],
            cmap='plasma'
        )
        # axes[1, 2].set(title=f'Phase Space of Electrons at timestep {frame}/{total_steps}')
        axes[1, 2].set(xlabel='Electron Position (m)', ylabel='Electron Velocity (m/s)')

        axes[1, 3].clear()
        h = axes[1, 3].hist2d(
            output['position_ions'][frame, :, 0],
            output['velocity_ions'][frame, :, 0],
            bins=jnp.array([len(grid), 50]),
            range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_ions, max_velocity_ions]],
            cmap='viridis'
        )
        # axes[1, 3].set(title=f'Phase Space of Ions at timestep {frame}/{total_steps}')
        axes[1, 3].set(xlabel='Ion Position (m)', ylabel='Ion Velocity (m/s)')

    ani = FuncAnimation(fig, update, frames=output["total_steps"], blit=False, interval=1, repeat_delay=1000)

    # Add static colorbars for context
    h_electrons = axes[1, 2].hist2d(
        output['position_electrons'][0, :, 0],
        output['velocity_electrons'][0, :, 0],
        bins=jnp.array([len(grid), 50]),
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_electrons, max_velocity_electrons]],
        cmap='plasma'
    )
    fig.colorbar(h_electrons[3], ax=axes[1, 2], label='Number of electrons')

    h_ions = axes[1, 3].hist2d(
        output['position_ions'][0, :, 0],
        output['velocity_ions'][0, :, 0],
        bins=jnp.array([len(grid), 50]),
        range=[[-box_size_x / 2, box_size_x / 2], [-max_velocity_ions, max_velocity_ions]],
        cmap='viridis'
    )
    fig.colorbar(h_ions[3], ax=axes[1, 3], label='Number of ions')

    plt.tight_layout()
    plt.show()