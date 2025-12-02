import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from matplotlib.animation import FuncAnimation

__all__ = ['plot']

def plot(output, direction="x", threshold=1e-12):
    def is_nonzero(field):
        return jnp.max(jnp.abs(field)) > threshold

    grid = output["grid"]
    time = output["time_array"] * output["plasma_frequency"]
    total_steps = output["total_steps"]
    box_size_x = output["length"]

    # NEW: use actual domain bounds (no centering at zero)
    xmin, xmax = float(grid[0]), float(grid[-1])

    # -------- Direction handling (preserve legacy behavior) ----------
    if len(direction) == 1 and direction in "xyz":
        direction1 = direction
        direction_index1 = {"x": 0, "y": 1, "z": 2}[direction1]
    elif len(direction) == 2 and all(c in "xyz" for c in direction):
        direction1, _direction2 = direction
        direction_index1 = {"x": 0, "y": 1, "z": 2}[direction1]
    else:
        raise ValueError("direction must be one or two of 'x', 'y', 'z'")

    # -------- Which field panels to render (unchanged logic) --------
    def add_field_components(field, unit, label_prefix):
        components = []
        for i, axis in enumerate("xyz"):
            data = output[field][:, :, i]
            if is_nonzero(data):
                components.append({
                    "data": data,
                    "title": f"{label_prefix} in the {axis} direction",
                    "xlabel": "x Position (m)",
                    "ylabel": r"Time ($\omega_{pe}^{-1}$)",
                    "cbar": f"{label_prefix} ({unit})"
                })
        return components

    plots_to_make = []
    plots_to_make += add_field_components("electric_field", "V/m", "Electric Field")
    plots_to_make += add_field_components("magnetic_field", "T", "Magnetic Field")
    plots_to_make += add_field_components("current_density", "A/m²", "Current Density")
    plots_to_make.append({
        "data": output["charge_density"],
        "title": "Charge Density",
        "xlabel": "x Position (m)",
        "ylabel": r"Time ($\omega_{pe}^{-1}$)",
        "cbar": "Charge density (C/m³)"
    })

    # -------- Species list (backward-compatible) --------
    species = output.get("species", None)
    if not species:
        species = [
            {"name": "electrons",
             "positions": output["position_electrons"],
             "velocities": output["velocity_electrons"],
             "charge": -1.0},
            {"name": "ions",
             "positions": output["position_ions"],
             "velocities": output["velocity_ions"],
             "charge": +1.0},
        ]

    def _pos1(sp):
        P = sp["positions"]  # (T, N, 3)
        return P[:, :, direction_index1]

    def _vel1(sp):
        V = sp["velocities"]  # (T, N, 3)
        return V[:, :, direction_index1]

    def _first_idx(pred):
        for i, sp in enumerate(species):
            q = sp.get("charge", None)
            if q is None:
                nm = sp.get("name", "").lower()
                q = -1.0 if "electron" in nm else (+1.0 if "ion" in nm else 0.0)
            if pred(q):
                return i
        return None

    e_idx = _first_idx(lambda q: q < 0)
    i_idx = _first_idx(lambda q: q > 0)

    bins_velocity = max(min(len(grid), 111), 71)

    def _histogram_time_series(pos_tn, vel_tn):
        vmax = float(np.max(np.abs(np.asarray(vel_tn)))) if vel_tn.size > 0 else 1.0
        vmax = vmax if vmax > 0 else 1.0
        # CHANGED: positions now in [xmin, xmax] instead of [-L/2, L/2]
        hist = vmap(lambda p, v: jnp.histogram2d(
            p, v,
            bins=[len(grid), bins_velocity],
            range=[[xmin, xmax], [-vmax, vmax]]
        )[0])(pos_tn, vel_tn)
        return hist, vmax

    phase_specs = []

    def _pack(sp, title_prefix):
        pos_tn = _pos1(sp)
        vel_tn = _vel1(sp)
        hist, vmax = _histogram_time_series(pos_tn, vel_tn)
        return {"name": title_prefix, "hist": hist, "vmax": vmax}

    if e_idx is not None:
        phase_specs.append(_pack(species[e_idx], "Electron"))
    if i_idx is not None:
        phase_specs.append(_pack(species[i_idx], "Ion"))

    neg_extra = 0
    pos_extra = 0
    neu_extra = 0

    def _charge_of(sp):
        q = sp.get("charge", None)
        if q is None:
            nm = sp.get("name", "").lower()
            if "electron" in nm:
                return -1.0
            if "ion" in nm:
                return +1.0
            return 0.0
        return float(q)

    for k, sp in enumerate(species):
        if k in (e_idx, i_idx):
            continue
        q = _charge_of(sp)
        if q < 0:
            neg_extra += 1
            display_name = f"electron_{neg_extra}"
        elif q > 0:
            pos_extra += 1
            display_name = f"ion_{pos_extra}"
        else:
            neu_extra += 1
            display_name = f"neutral_{neu_extra}"
        pos_tn = _pos1(sp)
        vel_tn = _vel1(sp)
        hist, vmax = _histogram_time_series(pos_tn, vel_tn)
        phase_specs.append({"name": display_name, "hist": hist, "vmax": vmax})

    # -------- Layout: field panels + per-species phase-space ----------
    ncols = 3
    n_field = len(plots_to_make)
    n_phase = len(phase_specs)
    have_energy = "total_energy" in output
    n_total_core = n_field + n_phase
    n_total = n_total_core + (1 if have_energy else 0)

    nrows = (n_total + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.6 * nrows), squeeze=False)

    def plot_field(ax, field_data, title, xlabel, ylabel, cbar_label):
        finite = np.isfinite(field_data)
        vbnd = np.max(np.abs(field_data[finite])) if np.any(finite) else 1.0
        im = ax.imshow(field_data, aspect="auto", cmap="RdBu", origin="lower",
                       extent=[grid[0], grid[-1], time[0], time[-1]],
                       vmin=-vbnd, vmax=vbnd)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(im, ax=ax, label=cbar_label)
        return im

    # Field panels
    for i, info in enumerate(plots_to_make):
        r, c = divmod(i, ncols)
        plot_field(axes[r, c], info["data"], info["title"], info["xlabel"], info["ylabel"], info["cbar"])

    # Phase-space panels
    axes_flat = np.ravel(axes)
    start_idx = n_field
    phase_images = []
    time_labels = []

    for k, spec in enumerate(phase_specs):
        ax = axes_flat[start_idx + k]
        vmax = spec["vmax"]
        # CHANGED: extent x-range now [xmin, xmax] (no centering at zero)
        im = ax.imshow(
            jnp.zeros((len(grid), max(min(len(grid), 111), 71))),
            aspect="auto", origin="lower", cmap="twilight",
            extent=[xmin, xmax, -vmax, vmax],
            vmin=jnp.min(spec["hist"]), vmax=jnp.max(spec["hist"])
        )

        if k == 0 and spec["name"] == "Electron":
            ax.set(xlabel=f"Electron Position {direction1} (m)",
                   ylabel=f"Electron Velocity {direction1} (m/s)",
                   title=f"Electron Phase Space {direction1}")
            cbar_label = "Electron count"
        elif k == 1 and spec["name"] == "Ion":
            ax.set(xlabel=f"Ion Position {direction1} (m)",
                   ylabel=f"Ion Velocity {direction1} (m/s)",
                   title=f"Ion Phase Space {direction1}")
            cbar_label = "Ion count"
        else:
            nm = spec["name"]
            ax.set(xlabel=f"{nm} Position {direction1} (m)",
                   ylabel=f"{nm} Velocity {direction1} (m/s)",
                   title=f"{nm} Phase Space ({direction1})")
            cbar_label = f"{nm} count"

        fig.colorbar(im, ax=ax, label=cbar_label)
        phase_images.append(im)

        tt = ax.text(0.5, 0.9, "", transform=ax.transAxes,
                     ha="center", va="top", fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        time_labels.append(tt)

    # Energy panel
    if have_energy:
        eax = axes_flat[n_total - 1]
        eax.plot(time, output["total_energy"], label="Total energy")
        if "kinetic_energy_electrons" in output:
            eax.plot(time, output["kinetic_energy_electrons"], label="Kinetic energy electrons")
        if "kinetic_energy_ions" in output:
            eax.plot(time, output["kinetic_energy_ions"], label="Kinetic energy ions")
        if "electric_field_energy" in output:
            eax.plot(time, output["electric_field_energy"], label="Electric field energy")
        if "magnetic_field_energy" in output and jnp.max(output["magnetic_field_energy"]) > 1e-10:
            eax.plot(time, output["magnetic_field_energy"], label="Magnetic field energy")
        if "charge_density" in output:
            eax.plot(time[2:], jnp.abs(jnp.mean(output["charge_density"][2:], axis=-1))*1e15,
                     label=r"Mean $\rho \times 10^{15}$")
        if "total_energy" in output:
            eax.plot(time[1:], jnp.abs(output["total_energy"][1:] - output["total_energy"][0]) / output["total_energy"][0],
                     label="Relative energy error")
        eax.set(title="Energy", xlabel=r"Time ($\omega_{pe}^{-1}$)",
                ylabel="Energy (J)", yscale="log", ylim=[1e-7, None])
        eax.legend(fontsize=7)

    # Animation
    def update(frame):
        for im, spec, tt in zip(phase_images, phase_specs, time_labels):
            im.set_array(spec["hist"][frame].T)
            tt.set_text(f"Time: {time[frame]:.2f} ωₚ")
        return tuple(phase_images + time_labels)

    ani = FuncAnimation(fig, update, frames=total_steps, blit=True, interval=1, repeat_delay=1000)
    plt.tight_layout()
    plt.show()
