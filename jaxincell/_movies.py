# --- ADDITIONS TO _movies.py ---

__all__ = ["particle_box_movie", "wave_spectrum_movie", "phase_space_movie"]

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.collections import LineCollection

# --------------------------- helpers reused ---------------------------

def _to_np(x):
    return np.asarray(x)

def _robust_bounds(a, lo=2, hi=98, eps=1e-12):
    vmin = float(np.percentile(a, lo))
    vmax = float(np.percentile(a, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        pad = eps if vmin == 0 else abs(vmin) * 1e-6
        return vmin - pad, vmax + pad
    return vmin, vmax

# ===============================================================
# 1) Wave + Spectrum + Energy (investor-friendly “physics dashboard”)
# ===============================================================
def wave_spectrum_movie(
    output,
    direction="x",
    fps=60,
    interval_ms=1000//60,
    save_path=None,
    spectrogram_cmap="magma",
    field_cmap="coolwarm",
):
    """
    **Wave growth dashboard**:
      - Top-left: progressive E(x,t) ribbon (audiences instantly see waves move/grow)
      - Top-right: live |Ê(k)| spectrum bars (which modes dominate?)
      - Bottom: energy time-series (total, kinetic e/i, electric & magnetic)

    Parameters
    ----------
    output : dict
        Needs keys:
          grid [Nx], time_array [T], plasma_frequency (scalar), total_steps (int),
          electric_field [T,Nx,3], kinetic_energy_electrons [T], kinetic_energy_ions [T],
          electric_field_energy [T], magnetic_field_energy [T], total_energy [T]
    direction : {'x','y','z'}
        Which E-component to visualize / FFT.
    """
    assert direction in "xyz"
    di = {"x":0, "y":1, "z":2}[direction]

    # Unpack
    grid  = _to_np(output["grid"])
    T     = int(_to_np(output["total_steps"]))
    time  = _to_np(output["time_array"]) * float(_to_np(output["plasma_frequency"]))
    Eall  = _to_np(output["electric_field"][:, :, di])  # [T, Nx]
    Nx    = grid.size

    # Energies (robust to missing magnetic field energy)
    KEe = _to_np(output.get("kinetic_energy_electrons", np.zeros(T)))
    KEi = _to_np(output.get("kinetic_energy_ions",     np.zeros(T)))
    EE  = _to_np(output.get("electric_field_energy",   np.zeros(T)))
    BE  = _to_np(output.get("magnetic_field_energy",   np.zeros(T)))
    TE  = _to_np(output.get("total_energy",            KEe+KEi+EE+BE))

    # Robust color scaling for E ribbon
    e_vmin, e_vmax = _robust_bounds(Eall, 2, 98)

    # Precompute spectrum |Ê(k)| with rfft for speed
    Ek = np.abs(np.fft.rfft(Eall, axis=1))  # [T, Nk]
    Ek_vis = np.log1p(Ek)                   # log-like for dynamic range
    k = np.fft.rfftfreq(Nx, d=(grid[1]-grid[0]))
    s_vmin, s_vmax = _robust_bounds(Ek_vis, 2, 98)

    # Figure layout
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.2], width_ratios=[2.0, 1.2], hspace=0.25, wspace=0.25)
    ax_ribbon = fig.add_subplot(gs[0, 0])
    ax_spec   = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, :])

    # --- Top-left: progressive E(x,t) ribbon
    # Start blank and "paint" rows as time flows (gives motion even if waves are standing)
    buf = np.zeros_like(Eall)
    im_ribbon = ax_ribbon.imshow(
        buf, origin="lower", aspect="auto", cmap=field_cmap,
        extent=[grid[0], grid[-1], time[0], time[-1]],
        vmin=e_vmin, vmax=e_vmax, animated=True,
    )
    ax_ribbon.set_title(f"Electric field $E_{direction}(x,t)$")
    ax_ribbon.set_xlabel(f"{direction}-position (m)")
    ax_ribbon.set_ylabel(r"Time ($\omega_{pe}^{-1}$)")

    # --- Top-right: spectrum bars
    bars = ax_spec.bar(k, Ek_vis[0], width=(k[1]-k[0]) if len(k) > 1 else 1.0, align="center")
    ax_spec.set_xlim(k[0], k[-1] if len(k)>1 else k[0]+1)
    ax_spec.set_ylim(s_vmin, s_vmax)
    ax_spec.set_xlabel("Wavenumber k (1/m)")
    ax_spec.set_ylabel(r"$\log(1+|\hat{E}(k)|)$")
    ax_spec.set_title("Mode content")

    # --- Bottom: energy timeline
    (ltotal,) = ax_energy.plot(time, np.zeros_like(time), lw=2.0, color="black", label="Total")
    (lke_e,)  = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#0066cc", label="Kinetic (e⁻)")
    (lke_i,)  = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#00aa66", label="Kinetic (ions)")
    (le,)     = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#aa0000", label="E-field")
    (lb,)     = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#aa6600", label="B-field")
    ax_energy.set_title("Energies")
    ax_energy.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    ax_energy.set_ylabel("Energy (J)")
    # Set sensible y-lims using the final ranges
    y_all = np.stack([TE, KEe, KEi, EE, BE], axis=0)
    ymin, ymax = _robust_bounds(y_all, 2, 98)
    pad = 0.08 * (ymax - ymin + 1e-12)
    ax_energy.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_energy.grid(alpha=0.25)
    ax_energy.legend(fontsize=8, loc="upper right")

    time_text = ax_spec.text(0.02, 0.92, "", transform=ax_spec.transAxes,
                             ha="left", va="top",
                             bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                             animated=True)

    artists = [im_ribbon, time_text] + list(bars) + [ltotal, lke_e, lke_i, le, lb]
    artists = tuple(artists)

    def _update(ti):
        # paint ribbon up to ti
        buf[ti] = Eall[ti]
        im_ribbon.set_array(buf)

        # spectrum bars at ti
        h = Ek_vis[ti]
        for b, val in zip(bars, h):
            b.set_height(val)

        # energies up to ti  (keep x & y lengths equal)
        ltotal.set_data(time[:ti+1], TE[:ti+1])
        lke_e.set_data(time[:ti+1], KEe[:ti+1])
        lke_i.set_data(time[:ti+1], KEi[:ti+1])
        le.set_data(time[:ti+1], EE[:ti+1])
        lb.set_data(time[:ti+1], BE[:ti+1])

        time_text.set_text(f"t = {time[ti]:.2f}  $\\omega_{{pe}}^{{-1}}$")
        return artists

    ani = FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=True, repeat=False)

    # Save or show
    if save_path is not None:
        ext = save_path.split(".")[-1].lower()
        if ext in ("mp4", "m4v", "mov"):
            Writer = writers.get("ffmpeg", None)
            if Writer is None:
                print("ffmpeg not available; showing instead.")
                plt.show()
            else:
                ani.save(save_path, writer=Writer(fps=fps), dpi=120)
        elif ext == "gif":
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            print(f"Unknown extension '.{ext}', not saving. Showing instead.")
            plt.show()
    else:
        plt.show()

    return ani


# ===============================================================
# 2) Phase-space movie (x–v) for electrons/ions with live points
# ===============================================================
def phase_space_movie(
    output,
    direction="x",
    species="both",           # 'electrons', 'ions', or 'both'
    bins_x=160,
    bins_v=120,
    fps=60,
    interval_ms=1000//60,
    save_path=None,
    points_per_species=200,   # overlay live points
    seed=7,
    cmap="twilight",
):
    """
    **Phase-space showcase**:
      - Heatmaps of f(x,v) for electrons and ions (choose either or both)
      - Overlay a subset of live particles (crisp visual for trapped/passing)

    Parameters
    ----------
    output : dict
        Needs keys:
          grid [Nx], length (scalar), time_array [T], plasma_frequency (scalar), total_steps (int),
          position_electrons [T,Ne,3], velocity_electrons [T,Ne,3],
          position_ions [T,Ni,3], velocity_ions [T,Ni,3]
    species : str
        'electrons' | 'ions' | 'both'
    """
    assert direction in "xyz"
    di = {"x":0, "y":1, "z":2}[direction]

    rng = np.random.default_rng(seed)

    # Unpack
    grid = _to_np(output["grid"])
    Lx   = float(_to_np(output["length"]))
    T    = int(_to_np(output["total_steps"]))
    time = _to_np(output["time_array"]) * float(_to_np(output["plasma_frequency"]))

    Xe = _to_np(output["position_electrons"][:, :, di])
    Ve = _to_np(output["velocity_electrons"][:, :, di])
    Xi = _to_np(output["position_ions"][:, :, di])
    Vi = _to_np(output["velocity_ions"][:, :, di])

    Ne = Xe.shape[1]
    Ni = Xi.shape[1]

    # subsets for points overlay
    idx_e = rng.choice(Ne, size=min(points_per_species, Ne), replace=False) if Ne>0 else np.array([])
    idx_i = rng.choice(Ni, size=min(points_per_species, Ni), replace=False) if Ni>0 else np.array([])

    # phase-space ranges (robust)
    v_e_vmin, v_e_vmax = _robust_bounds(Ve, 2, 98) if Ne>0 else (-1.0, 1.0)
    v_i_vmin, v_i_vmax = _robust_bounds(Vi, 2, 98) if Ni>0 else (-1.0, 1.0)

    # histogram bins
    x_edges = np.linspace(-Lx/2, Lx/2, bins_x+1)
    v_edges_e = np.linspace(v_e_vmin, v_e_vmax, bins_v+1)
    v_edges_i = np.linspace(v_i_vmin, v_i_vmax, bins_v+1)

    # Precompute histograms frame-by-frame (fast enough; avoids work in the loop)
    def _hist2d_time(X, V, vx_edges):
        H = np.empty((T, bins_v, bins_x), dtype=float)  # store as [T, bins_v, bins_x]
        for t in range(T):
            h, _, _ = np.histogram2d(X[t], V[t], bins=[x_edges, vx_edges])  # h: [bins_x, bins_v]
            H[t] = h.T  # now fits: [bins_v, bins_x]
        return H

    show_e = species in ("electrons", "both") and Ne > 0
    show_i = species in ("ions", "both") and Ni > 0

    He = _hist2d_time(Xe, Ve, v_edges_e) if show_e else None  # [T, bins_v, bins_x]
    Hi = _hist2d_time(Xi, Vi, v_edges_i) if show_i else None

    # figure layout
    if show_e and show_i:
        fig, (ax_e, ax_i) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=False, sharey=False)
    elif show_e:
        fig, ax_e = plt.subplots(1, 1, figsize=(6.6, 4.6))
        ax_i = None
    elif show_i:
        fig, ax_i = plt.subplots(1, 1, figsize=(6.6, 4.6))
        ax_e = None
    else:
        raise ValueError("No particles to show for requested species.")

    # electrons panel
    if show_e:
        im_e = ax_e.imshow(
            He[0], origin="lower", aspect="auto", cmap=cmap,
            extent=[-Lx/2, Lx/2, v_edges_e[0], v_edges_e[-1]],
            animated=True
        )
        ax_e.set_title(f"Electron phase space  $x$–$v_{direction}$")
        ax_e.set_xlabel("x (m)")
        ax_e.set_ylabel(f"v{direction} (m/s)")
        # overlay points
        scat_e = ax_e.scatter(Xe[0, idx_e], Ve[0, idx_e], s=6, c="#ff3333", alpha=0.9, edgecolor="none", animated=True)
    else:
        im_e = scat_e = None

    # ions panel
    if show_i:
        im_i = ax_i.imshow(
            Hi[0], origin="lower", aspect="auto", cmap=cmap,
            extent=[-Lx/2, Lx/2, v_edges_i[0], v_edges_i[-1]],
            animated=True
        )
        ax_i.set_title(f"Ion phase space  $x$–$v_{direction}$")
        ax_i.set_xlabel("x (m)")
        ax_i.set_ylabel(f"v{direction} (m/s)")
        scat_i = ax_i.scatter(Xi[0, idx_i], Vi[0, idx_i], s=6, c="#3388ff", alpha=0.9, edgecolor="none", animated=True)
    else:
        im_i = scat_i = None

    # time text
    if show_e:
        time_text = ax_e.text(0.02, 0.95, "", transform=ax_e.transAxes,
                              ha="left", va="top",
                              bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                              animated=True)
    else:
        time_text = ax_i.text(0.02, 0.95, "", transform=ax_i.transAxes,
                              ha="left", va="top",
                              bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                              animated=True)

    artists = [time_text]
    if im_e is not None: artists.append(im_e)
    if scat_e is not None: artists.append(scat_e)
    if im_i is not None: artists.append(im_i)
    if scat_i is not None: artists.append(scat_i)
    artists = tuple(artists)

    def _update(ti):
        if im_e is not None:
            im_e.set_array(He[ti])
            scat_e.set_offsets(np.column_stack([Xe[ti, idx_e], Ve[ti, idx_e]]))
        if im_i is not None:
            im_i.set_array(Hi[ti])
            scat_i.set_offsets(np.column_stack([Xi[ti, idx_i], Vi[ti, idx_i]]))
        time_text.set_text(f"t = {time[ti]:.2f}  $\\omega_{{pe}}^{{-1}}$")
        return artists

    ani = FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=True, repeat=False)

    # Save or show
    if save_path is not None:
        ext = save_path.split(".")[-1].lower()
        if ext in ("mp4", "m4v", "mov"):
            Writer = writers.get("ffmpeg", None)
            if Writer is None:
                print("ffmpeg not available; showing instead.")
                plt.show()
            else:
                ani.save(save_path, writer=Writer(fps=fps), dpi=120)
        elif ext == "gif":
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            print(f"Unknown extension '.{ext}', not saving. Showing instead.")
            plt.show()
    else:
        plt.show()

    return ani

def _unwrap_periodic(x, L):
    x = _to_np(x)
    y = (x + L/2.0) % L
    dy = np.diff(y, axis=0)
    jumps = np.where(dy >  L/2, -L, 0.0) + np.where(dy < -L/2,  L, 0.0)
    corr = np.cumsum(jumps, axis=0)
    corr = np.vstack([np.zeros((1, x.shape[1])), corr])
    y_unwrapped = y + corr
    return y_unwrapped - L/2.0

def _build_trail_segments(xu, y, t_idx, trail_len, fade=False):
    start = max(0, t_idx - trail_len + 1)
    window = xu[start:t_idx+1]                 # [W, N]
    if window.shape[0] < 2:
        return np.empty((0, 2, 2)), None
    W, N = window.shape
    x0 = window[:-1, :]
    x1 = window[1:,  :]
    y0 = np.broadcast_to(y, (W-1, N))
    y1 = y0
    segs = np.stack([np.stack([x0, y0], axis=-1),
                     np.stack([x1, y1], axis=-1)], axis=2)  # [(W-1), N, 2, 2]
    segs = segs.reshape(-1, 2, 2)
    if not fade:
        return segs, None
    alphas_t = np.linspace(0.15, 1.0, W-1)
    alphas = np.repeat(alphas_t, N)
    return segs, alphas

def particle_box_movie(
    output,
    direction="x",
    trail_len=40,
    n_electrons=150,
    n_ions=150,
    jitter=0.08,
    fps=60,
    interval_ms=1000//60,
    save_path=None,
    seed=42,
    show_field=True,          # NEW: overlay field + line subplot
    field_alpha=0.35,         # transparency of field ribbon
    field_cmap="coolwarm",    # pretty divergent colormap
):
    """
    Cinematic particle box with optional live electric field overlay & line plot.

    - Top panel: particles + fading trails + semi-transparent E(x) ribbon (wow!)
    - Bottom panel: instantaneous E(x) curve

    Parameters
    ----------
    output : dict
        Needs keys: grid, length, time_array, plasma_frequency, total_steps,
        position_electrons [T,Ne,3], position_ions [T,Ni,3], electric_field [T,Nx,3].
    direction : {'x','y','z'}
        Which component/axis to visualize (and which E-component to show).
    trail_len : int
        Trail length in frames.
    n_electrons, n_ions : int
        Max particles to display (subsampled for speed).
    jitter : float
        Vertical jitter for particle rows to look “cloudy”.
    fps, interval_ms : int
        Playback/save speed settings.
    save_path : str or None
        If provided: saves to MP4 (ffmpeg) or GIF (pillow).
    seed : int
        RNG seed for subsampling & jitter.
    show_field : bool
        If True, overlay E-field and show E(x) subplot.
    field_alpha : float
        Alpha for the field ribbon overlay.
    field_cmap : str
        Matplotlib colormap for the field ribbon.
    """
    assert direction in "xyz"
    di = {"x":0, "y":1, "z":2}[direction]

    rng = np.random.default_rng(seed)

    # --- data ---
    grid    = _to_np(output["grid"])
    Lx      = float(_to_np(output["length"]))
    time    = _to_np(output["time_array"]) * float(_to_np(output["plasma_frequency"]))
    T       = int(_to_np(output["total_steps"]))
    pos_e   = _to_np(output["position_electrons"][:, :, di])  # [T, Ne]
    pos_i   = _to_np(output["position_ions"][:, :, di])       # [T, Ni]

    # Optional field (component along `direction`)
    if show_field:
        E_all  = _to_np(output["electric_field"][:, :, di])   # [T, Nx]
        # Robust global vmin/vmax for nice colorscales:
        # Use percentiles to avoid a single spike ruining the palette
        vmin = float(np.percentile(E_all, 2))
        vmax = float(np.percentile(E_all, 98))
        if vmin == vmax:  # fallback
            vmax = vmin + (1e-12 if vmin == 0 else abs(vmin)*1e-6)

    Ne = pos_e.shape[1]
    Ni = pos_i.shape[1]

    # Subsample
    n_e = min(n_electrons, Ne)
    n_i = min(n_ions, Ni)
    idx_e = rng.choice(Ne, size=n_e, replace=False) if Ne > n_e else np.arange(Ne)
    idx_i = rng.choice(Ni, size=n_i, replace=False) if Ni > n_i else np.arange(Ni)

    Xe = pos_e[:, idx_e]   # [T, n_e]
    Xi = pos_i[:, idx_i]   # [T, n_i]

    # Unwrap for trails
    Xe_u = _unwrap_periodic(Xe, Lx)
    Xi_u = _unwrap_periodic(Xi, Lx)

    # Vertical rows (fixed) for “aquarium” look
    y_e_center = +0.6
    y_i_center = -0.6
    ye = y_e_center + jitter * rng.standard_normal(n_e)
    yi = y_i_center + jitter * rng.standard_normal(n_i)

    # --- figure layout ---
    if show_field:
        fig, (ax, ax_field) = plt.subplots(
            2, 1, figsize=(10, 6.2),
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.28}
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 3.6))

    # TOP: particle aquarium
    ax.set_xlim(-Lx/2, Lx/2)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(f"{direction}-position (m)")
    ax.set_yticks([])
    ax.set_title(f"Charged particles in a periodic box — {direction}-axis")
    ax.axvline(-Lx/2, color="k", lw=1, alpha=0.5)
    ax.axvline(+Lx/2, color="k", lw=1, alpha=0.5)

    # Optional E-field ribbon overlay
    if show_field:
        # Create a slim 2D tile from the 1D E(x) so we can color the strip
        Nx = grid.size
        ribbon_rows = 120  # vertical resolution of the ribbon
        E0 = E_all[0]  # [Nx]
        E_img = np.tile(E0[None, :], (ribbon_rows, 1))  # [H, Nx]

        # Place as full background (y from -1 to 1); transparency via alpha
        im_field = ax.imshow(
            E_img, origin="lower", aspect="auto", cmap=field_cmap,
            extent=[grid[0], grid[-1], -1.0, 1.0], alpha=field_alpha,
            vmin=vmin, vmax=vmax, animated=True,
        )
    else:
        im_field = None

    # Points (animated)
    scat_e = ax.scatter(Xe[0], ye, s=10, c="red",  marker="o", alpha=0.95, zorder=3, animated=True)
    scat_i = ax.scatter(Xi[0], yi, s=14, c="blue", marker="o", alpha=0.95, zorder=3, animated=True)

    # Trails
    lc_e = LineCollection([], colors=[(1, 0, 0, 0.6)], linewidths=1.2, zorder=2, animated=True)
    lc_i = LineCollection([], colors=[(0, 0, 1, 0.6)], linewidths=1.2, zorder=2, animated=True)
    ax.add_collection(lc_e)
    ax.add_collection(lc_i)

    # Time label
    time_text = ax.text(0.015, 0.95, "", transform=ax.transAxes,
                        ha="left", va="top",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                        animated=True)

    # BOTTOM: E(x) line plot
    if show_field:
        ax_field.set_xlim(grid[0], grid[-1])
        # sensible y-lims based on robust global vmin/vmax
        pad = 0.05 * (vmax - vmin)
        ax_field.set_ylim(vmin - pad, vmax + pad)
        ax_field.set_xlabel(f"{direction}-position (m)")
        ax_field.set_ylabel(f"E{direction} (V/m)")
        (line_field,) = ax_field.plot(grid, E_all[0], lw=1.5, color="black", animated=True)
        ax_field.grid(alpha=0.25)

    # Pre-create blit artists
    artists = [scat_e, scat_i, lc_e, lc_i, time_text]
    if im_field is not None:
        artists.append(im_field)
    if show_field:
        artists.append(line_field)
    artists = tuple(artists)

    def _update(frame):
        # Update points
        scat_e.set_offsets(np.column_stack([Xe[frame], ye]))
        scat_i.set_offsets(np.column_stack([Xi[frame], yi]))

        # Update trails
        segs_e, alpha_e = _build_trail_segments(Xe_u, ye, frame, trail_len, fade=True)
        segs_i, alpha_i = _build_trail_segments(Xi_u, yi, frame, trail_len, fade=True)
        lc_e.set_segments(segs_e)
        lc_i.set_segments(segs_i)
        if alpha_e is not None:
            cols = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (len(alpha_e), 1))
            cols[:, 3] = alpha_e
            lc_e.set_colors(cols)
        if alpha_i is not None:
            cols = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), (len(alpha_i), 1))
            cols[:, 3] = alpha_i
            lc_i.set_colors(cols)

        # Field overlay + line
        if show_field:
            E = E_all[frame]  # [Nx]
            E_img = np.tile(E[None, :], (im_field.get_array().shape[0], 1))
            im_field.set_array(E_img)
            line_field.set_ydata(E)

        time_text.set_text(f"t = {time[frame]:.2f}  $\\omega_{{pe}}^{{-1}}$")
        return artists

    ani = FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=True, repeat=False)

    # Save or show
    if save_path is not None:
        ext = save_path.split(".")[-1].lower()
        if ext in ("mp4", "m4v", "mov"):
            Writer = writers.get("ffmpeg", None)
            if Writer is None:
                print("ffmpeg writer not available; showing interactively instead.")
                plt.show()
            else:
                ani.save(save_path, writer=Writer(fps=fps), dpi=120)
        elif ext == "gif":
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            print(f"Unknown extension '.{ext}', not saving. Showing interactively instead.")
            plt.show()
    else:
        plt.show()

    return ani