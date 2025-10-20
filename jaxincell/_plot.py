import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ._constants import speed_of_light
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
import time as time_package
import platform

__all__ = ['plot']

def _frame_indices(total_steps: int, steps_to_plot: int | None) -> np.ndarray:
    """Pick evenly spaced, unique frame indices from 0..total_steps-1."""
    if steps_to_plot is None or steps_to_plot >= total_steps:
        return np.arange(int(total_steps), dtype=int)
    # evenly spaced, include endpoints
    idx = np.linspace(0, int(total_steps) - 1, int(steps_to_plot), dtype=int)
    # unique & sorted (linspace can repeat when total_steps is small)
    return np.unique(idx)

def _ensure_interactive_backend(save_path):
    """
    If the current backend is non-interactive (Agg) and the user didn't request saving,
    switch to an interactive GUI backend (MacOSX on mac, else TkAgg).
    If saving, prefer Agg for speed and stability.
    """
    import matplotlib.pyplot as plt

    try:
        be = plt.get_backend().lower()
        if save_path is None:
            # we want to show a window
            if be.endswith("agg"):
                target = "MacOSX" if platform.system() == "Darwin" else "TkAgg"
                plt.switch_backend(target)   # interactive
        else:
            # we are saving: switch to Agg for speed/consistency if not already
            if not be.endswith("agg"):
                plt.switch_backend("Agg")
    except Exception:
        # Best effort; if switch fails (e.g., no GUI available), just continue.
        pass

def plot(
    output,
    direction="x",
    threshold=1e-12,
    save_path=None,     # '...mp4', '...gif', '...png', etc.
    dpi=130,            # 
    fps=30,             # 
    interval_ms=None,   # if None, computed from fps
    crf=23,             # H.264 quality for MP4
    steps_to_plot=None,
    bins_v=None,                 # velocity bins (int)
    bins_x=None,                # position bins (int)
    points_per_species=None,     # subsample particle columns for histograms
    seed=7,                      # for reproducible subsampling
):
    def is_nonzero(field):
        return jnp.max(jnp.abs(field)) > threshold
    
    t0 = time_package.time()
    print(f"[plot] Start: building summary (direction='{direction}', frames={int(output['total_steps'])}) -> {save_path or 'interactive window'}")

    rng = np.random.default_rng(seed)

    def _maybe_subsample_columns(A, nmax):
        # A: [F, N, ...]  ->  subsample along axis=1 (particles)
        if (nmax is None) or (A.shape[1] <= nmax):
            return A
        idx = rng.choice(A.shape[1], size=int(nmax), replace=False)
        return A[:, idx]

    _ensure_interactive_backend(save_path)

    grid = output["grid"]
    time = output["time_array"] * output["plasma_frequency"]
    total_steps = output["total_steps"]
    if interval_ms is None:
        interval_ms = int(1000 / max(1, fps))
        
    # choose frames to render
    _frames = _frame_indices(total_steps, steps_to_plot)
    # convenience “views”
    time_full = output["time_array"] * output["plasma_frequency"]
    time = np.asarray(time_full)[_frames]

    box_size_x = output["length"]
    number_pseudoelectrons = output["number_pseudoelectrons"]
    
    # Downselect time dimension ONCE for all arrays used in imshow/phase hist:
    E = output["electric_field"][_frames]      # [F, Nx, 3]
    B = output["magnetic_field"][_frames]      # [F, Nx, 3]
    J = output["current_density"][_frames]     # [F, Nx, 3]
    rho = output["charge_density"][_frames]    # [F, Nx]
    vel_e = output["velocity_electrons"][_frames]  # [F, Ne, 3]
    vel_i = output["velocity_ions"][_frames]       # [F, Ni, 3]
    pos_e = output["position_electrons"][_frames]  # [F, Ne, 3]
    pos_i = output["position_ions"][_frames]       # [F, Ni, 3]

    if points_per_species is not None:
        pos_e = _maybe_subsample_columns(pos_e, points_per_species)
        vel_e = _maybe_subsample_columns(vel_e, points_per_species)
        pos_i = _maybe_subsample_columns(pos_i, points_per_species)
        vel_i = _maybe_subsample_columns(vel_i, points_per_species)

    # Energies: also slice to frames to avoid plotting huge series
    EE = output.get("electric_field_energy", None)
    BE = output.get("magnetic_field_energy", None)
    TE = output.get("total_energy", None)
    KEe = output.get("kinetic_energy_electrons", None)
    KEi = output.get("kinetic_energy_ions", None)
    def _slice_if(a): return None if a is None else np.asarray(a)[_frames]
    EE, BE, TE, KEe, KEi = map(_slice_if, (EE, BE, TE, KEe, KEi))

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
            data = {"electric_field": E, "magnetic_field": B, "current_density": J}[field][:, :, i]
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
        "data": rho,
        "title": "Charge Density",
        "xlabel": f"x Position (m)",
        "ylabel": r"Time ($\omega_{pe}^{-1}$)",
        "cbar": "Charge density (C/m³)"
    })

    # Compute phase space histograms
    sqrtmemi = jnp.sqrt(output["mass_electrons"][0] / output["mass_ions"][0])[0]
    max_velocity_electrons_1 = min(max(1.2 * jnp.max(jnp.abs(output["velocity_electrons"][:, :, direction_index1])),
                                 2.5 * jnp.abs(vth_e_1)
                                 + jnp.abs(output[f"electron_drift_speed_{direction1}"])), 1.1*speed_of_light)
    max_velocity_electrons_1 = float(jnp.asarray(max_velocity_electrons_1))
    max_velocity_ions_1      = min(max(1.2 * jnp.max(jnp.abs(output["velocity_ions"][:, :, direction_index1])),
                                 sqrtmemi * 0.3 * jnp.abs(vth_i_1) * jnp.sqrt(output[f"ion_temperature_over_electron_temperature_{direction1}"])
                                 + jnp.abs(output[f"ion_drift_speed_{direction1}"])), 1.1*speed_of_light)
    max_velocity_ions_1 = float(jnp.asarray(max_velocity_ions_1))

    # velocity bins
    if bins_v is None:
        bins_velocity = max(min(len(grid), 111), 71)
    else:
        bins_velocity = int(max(8, bins_v))

    # independent x-bin count; FAR smaller than len(grid)
    if bins_x is None:
        bins_x = min(len(grid), 160)    # good speed/quality tradeoff
    else:
        bins_x = int(max(16, bins_x))

    def _hist2d_stack(X, V, vmax_abs):
        H = np.empty((len(_frames), bins_x, bins_velocity), dtype=np.float32)
        x_edges = np.linspace(-box_size_x/2, box_size_x/2, bins_x+1)
        v_edges = np.linspace(-vmax_abs,     vmax_abs,     bins_velocity+1)
        for k in range(len(_frames)):
            # h: [bins_x, bins_v]
            h, _, _ = np.histogram2d(np.asarray(X[k, :]), np.asarray(V[k, :]),
                                    bins=[x_edges, v_edges])
            H[k] = h
        return H

    electron_phase_histograms = _hist2d_stack(
        pos_e[:, :, direction_index1], vel_e[:, :, direction_index1], max_velocity_electrons_1
    )
    ion_phase_histograms = _hist2d_stack(
        pos_i[:, :, direction_index1], vel_i[:, :, direction_index1], max_velocity_ions_1
    )

    # Grid layout
    ncols = 3
    n_field_plots = len(plots_to_make)
    n_total_plots = n_field_plots + 2  # +2 = (electron phase, ion phase)
    # --- CHANGE 1: detect whether we'll draw an Energy panel and count it ---
    want_energy = any(x is not None for x in (EE, BE, TE, KEe, KEi))
    metric_cfg = output.get("metric", {"kind": 0, "params": {}})
    kind = jnp.asarray(metric_cfg.get("kind", 0))
    use_gr = (kind != 0)
    if use_gr:
        PB = output.get("poynting_balance_residual", None)
        GR = output.get("gr_energy_balance_residual", None)
        want_energy = want_energy or (PB is not None) or (GR is not None)
    if want_energy:
        n_total_plots += 1
    if second_direction:
        n_total_plots += 3  # +2 for v-v phase; +1 for xy locations
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
        np.zeros((bins_x, bins_velocity), dtype=np.float32),  # <--- was len(grid)
        aspect="auto", origin="lower", cmap="twilight",
        extent=[-box_size_x/2, box_size_x/2, -max_velocity_electrons_1, max_velocity_electrons_1],
        vmin=float(np.min(electron_phase_histograms)),
        vmax=float(np.max(electron_phase_histograms)),
    )
    electron_ax.set(xlabel=f"Electron Position {direction1} (m)",
                    ylabel=f"Electron Velocity {direction1} (m/s)",
                    title=f"Electron Phase Space {direction1}")
    fig.colorbar(electron_plot, ax=electron_ax, label="Electron count")

    ion_plot = ion_ax.imshow(
        np.zeros((bins_x, bins_velocity), dtype=np.float32),  # <--- was len(grid)
        aspect="auto", origin="lower", cmap="twilight",
        extent=[-box_size_x/2, box_size_x/2, -max_velocity_ions_1, max_velocity_ions_1],
        vmin=float(np.min(ion_phase_histograms)),
        vmax=float(np.max(ion_phase_histograms)),
    )
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

    # only place energy panel if we counted it, then advance used_axes ---
    if want_energy and (total_axes >= used_axes + 1):
        energy_ax = axes_flat[used_axes]
        
        # Build local-frame fields and Tij, and ∂t γ_ij via finite differences on time.
        metric_cfg = output.get("metric", {"kind": 0, "params": {}})
        kind = jnp.asarray(metric_cfg.get("kind", 0))
        use_gr = (kind != 0)

        if EE is not None:  energy_ax.plot(time, EE,  label="Electric field energy")
        if BE is not None and (np.max(np.asarray(BE)) > 1e-10):
            energy_ax.plot(time, BE, label="Magnetic field energy")

        if use_gr:
            # slice GR diagnostics to selected frames if present
            PB = output.get("poynting_balance_residual", None)
            GR = output.get("gr_energy_balance_residual", None)

            def _match_time_to_series(series):
                """Return a time array (in ω_pe^-1 units) that matches the length of `series`."""
                t_full = np.asarray(output["time_array"]) * float(output["plasma_frequency"])
                n = len(series)
                T = len(t_full)
                # common patterns from finite differences / centered stencils:
                if n == T:
                    return t_full
                if n == T - 1:
                    return t_full[1:]          # e.g., forward/backward diff
                if n == T - 2:
                    return t_full[1:-1]        # e.g., centered diff
                # Fallback: linearly map onto the full time span
                return np.linspace(t_full[0], t_full[-1], n)

            if PB is not None:
                PB = np.asarray(PB)
                tPB = _match_time_to_series(PB)
                energy_ax.plot(tPB, PB, label="Poynting balance residual")

            if GR is not None:
                GR = np.asarray(GR)
                tGR = _match_time_to_series(GR)
                energy_ax.plot(tGR, GR, label="GR energy balance residual")

        else:
            if TE is not None:  energy_ax.plot(time, TE,  label="Total energy")
            if KEe is not None: energy_ax.plot(time, KEe, label="Kinetic energy electrons")
            if KEi is not None: energy_ax.plot(time, KEi, label="Kinetic energy ions")
            if TE is not None:
                rel = np.abs(TE - TE[0]) / (np.abs(TE[0]) + 1e-300)
                energy_ax.plot(time, rel, label="Relative energy error")

        energy_ax.set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)",
                    ylabel="Energy (J)", yscale="log", ylim=[1e-5, None])
        energy_ax.legend(fontsize=7, loc="lower right")
        used_axes += 1  # we consumed one axis slot
    
    # guard + start second-direction block at current used_axes ---
    if second_direction:
        if total_axes < used_axes + 3:
            second_direction = False  # not enough slots; skip gracefully
        else:
            electron_ax2 = axes_flat[used_axes + 0]
            ion_ax2      = axes_flat[used_axes + 1]
            positions_ax = axes_flat[used_axes + 2]
        
        # Phase space in second direction
        max_velocity_electrons_2 = max(
            1.0 * jnp.max(vel_e[:, :, direction_index2]),
            2.5 * jnp.abs(vth_e_2) + jnp.abs(output[f"electron_drift_speed_{direction2}"])
        )
        max_velocity_ions_2 = max(
            1.0 * jnp.max(vel_i[:, :, direction_index2]),
            sqrtmemi * 0.3 * jnp.abs(vth_i_2) * jnp.sqrt(output[f"ion_temperature_over_electron_temperature_{direction2}"])
            + jnp.abs(output[f"ion_drift_speed_{direction2}"])
        )
        max_velocity_electrons_12 = float(np.max([max_velocity_electrons_1, max_velocity_electrons_2]))
        max_velocity_ions_12      = float(np.max([max_velocity_ions_1,      max_velocity_ions_2]))

        def _hist2d_vv(Va, Vb, vabs):
            H = np.empty((len(_frames), bins_velocity, bins_velocity), dtype=np.float32)
            v_edges = np.linspace(-vabs, vabs, bins_velocity+1)
            for k in range(len(_frames)):
                h, _, _ = np.histogram2d(np.asarray(Va[k, :]), np.asarray(Vb[k, :]),
                                        bins=[v_edges, v_edges])
                H[k] = h.T
            return H

        electron_phase_histograms2 = _hist2d_vv(
            vel_e[:, :, direction_index1], vel_e[:, :, direction_index2], max_velocity_electrons_12
        )
        ion_phase_histograms2 = _hist2d_vv(
            vel_i[:, :, direction_index1], vel_i[:, :, direction_index2], max_velocity_ions_12
        )

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
        
        B_field_densities = np.asarray(output["magnetic_field_energy_density"])[_frames]
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
        n_particles_to_plot = min(100, pos_e.shape[1])
        subset_ions      = rng.choice(pos_i.shape[1], size=n_particles_to_plot, replace=False)
        subset_electrons = rng.choice(pos_e.shape[1], size=n_particles_to_plot, replace=False)
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
        electron_plot.set_array(electron_phase_histograms[frame])
        ion_plot.set_array(ion_phase_histograms[frame])
        animated_time_text.set_text(f"Time: {time[frame]:.2f} ωₚ")
        if second_direction:
            electron_plot2.set_array(electron_phase_histograms2[frame].T)
            ion_plot2.set_array(ion_phase_histograms2[frame].T)

            x_electrons = np.asarray(pos_e[frame, subset_electrons, direction_index2])
            z_electrons = np.asarray(pos_e[frame, subset_electrons, direction_index1])
            x_ions      = np.asarray(pos_i[frame, subset_ions,      direction_index2])
            z_ions      = np.asarray(pos_i[frame, subset_ions,      direction_index1])

            scat_r.set_offsets(np.stack([x_electrons, z_electrons], axis=-1))
            scat_b.set_offsets(np.stack([x_ions, z_ions], axis=-1))

            B = np.asarray(B_field_densities[frame])
            if B.ndim != 2: B = np.tile(B, reps=(10, 1))
            im.set_array(B.T)
            animated_time_text2.set_text(f"Time: {time[frame]:.2f} ωₚ")

            return (electron_plot, ion_plot, electron_plot2, ion_plot2, animated_time_text, scat_r, scat_b, im, animated_time_text2)
        else:
            return (electron_plot, ion_plot, animated_time_text)

    ani = FuncAnimation(
    fig, update, frames=len(_frames), blit=True, interval=interval_ms,
    repeat_delay=1000, cache_frame_data=False
    )

    plt.tight_layout()
    # --- SAVE/SHOW ---
    if save_path is not None:
        # If saving a static image (e.g. PNG), we want the LAST timestep on the figure:
        def _update_to_last_frame():
            # drive the artists to the final state
            update(len(_frames) - 1)

        _save_animation_or_image(
            ani, fig, save_path,
            fps=fps, dpi=dpi, crf=crf,
            last_frame_updater=_update_to_last_frame
        )

        print(f"[plot] Done: wrote {save_path} in {time_package.time()-t0:.2f}s")
    else:
        print(f"[plot] Done: displayed in {time_package.time()-t0:.2f}s")
        plt.show()

        

def _save_animation_or_image(ani, fig, save_path, fps=30, dpi=130, crf=23, last_frame_updater=True):
    ext = str(save_path).lower().rsplit(".", 1)[-1]

    if ext in ("mp4", "m4v", "mov"):
        try:
            # Try macOS hardware encoder first (HUGE speedup)
            hw_writer = FFMpegWriter(
                fps=fps,
                codec="h264_videotoolbox",  # Apple VideoToolbox (hardware)
                extra_args=[
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt", "yuv420p",
                    "-b:v", "6M",                 # target bitrate (tweak as needed)
                    "-maxrate", "10M",
                    "-bufsize", "20M",
                    "-profile:v", "main",
                    "-movflags", "+faststart",
                    "-r", str(int(fps)),          # constant frame rate
                    "-threads", "0",
                    "-vsync","cfr"
                ],
            )
            ani.save(save_path, writer=hw_writer, dpi=dpi)
            plt.close(fig)
            return
        except Exception as e:
            print(f"[save] Hardware encoder failed ({e!r}); falling back to libx264…")

        try:
            # CPU fallback: libx264 ultrafast
            sw_writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=[
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-vsync", "cfr",
                    "-pix_fmt", "yuv420p",
                    "-preset", "ultrafast",
                    "-tune", "fastdecode",
                    "-crf", str(int(crf)),
                    "-movflags", "+faststart",
                    "-r", str(int(fps)),
                    "-threads", "0",
                    "-g", str(int(fps) * 2),      # larger GOP for speed
                    "-bf", "0",                   # fewer B-frames => faster
                ],
            )
            ani.save(save_path, writer=sw_writer, dpi=dpi)
            plt.close(fig)
            return
        except Exception as e2:
            print(f"FFmpeg (libx264) failed ({e2!r}). Falling back to GIF…")
            # falls through to GIF branch

    elif ext == "gif":
        import os, subprocess, tempfile
        # 1) write a fast MP4 to a temp file
        with tempfile.TemporaryDirectory() as td:
            tmp_mp4 = os.path.join(td, "anim.mp4")
            try:
                # Prefer hardware; fall back to software automatically
                try:
                    hw_writer = FFMpegWriter(
                        fps=fps,
                        codec="h264_videotoolbox",
                        extra_args=[
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-pix_fmt", "yuv420p",
                            "-b:v", "6M",
                            "-maxrate", "10M",
                            "-bufsize", "20M",
                            "-r", str(int(fps)),
                            "-movflags", "+faststart",
                            "-threads", "0",
                        ],
                    )
                    ani.save(tmp_mp4, writer=hw_writer, dpi=dpi)
                except Exception:
                    sw_writer = FFMpegWriter(
                        fps=fps,
                        codec="libx264",
                        extra_args=[
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-pix_fmt", "yuv420p",
                            "-preset", "ultrafast",
                            "-crf", "28",
                            "-r", str(int(fps)),
                            "-movflags", "+faststart",
                            "-threads", "0",
                            "-g", str(int(fps) * 2),
                            "-bf", "0",
                        ],
                    )
                    ani.save(tmp_mp4, writer=sw_writer, dpi=dpi)
                plt.close(fig)
            except Exception as e:
                print(f"[save] Could not render intermediate MP4 ({e!r}). Falling back to PillowWriter...")
                try:
                    writer = PillowWriter(fps=fps)
                    ani.save(save_path, writer=writer, dpi=dpi)
                    plt.close(fig)
                    return
                except Exception as e2:
                    print(f"Pillow writer also failed ({e2!r}); showing interactively instead.")
                    plt.show()
                    return

            # 2) MP4 -> optimized GIF (palette) using ffmpeg (very fast)
            # palette improves quality and speed; scale keeps even dims not required for GIF, but ok.
            palette = os.path.join(td, "pal.png")
            # Generate palette
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp4, "-vf",
                 f"fps={int(fps)},scale=iw:ih:flags=fast_bilinear,palettegen",
                 palette],
                check=True
            )
            # Apply palette
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp4, "-i", palette, "-lavfi",
                 f"fps={int(fps)},scale=iw:ih:flags=fast_bilinear [x]; [x][1:v] paletteuse=new=1",
                 save_path],
                check=True
            )
            return

    else:
        # Static image (png/pdf/svg). If caller wants the LAST frame, update first.
        if last_frame_updater is not None:
            last_frame_updater()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
