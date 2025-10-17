# --- ADDITIONS TO _movies.py ---

__all__ = ["particle_box_movie", "wave_spectrum_movie", "phase_space_movie"]

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
import shutil  # to detect if ffmpeg is available
import time

# --------------------------- helpers reused ---------------------------

def _save_animation(ani, fig, save_path, fps=30, dpi=110, crf=23):
    """
    Save animation to MP4 (libx264, yuv420p) with even-dimension fix.
    Falls back to GIF if ffmpeg is missing or fails.
    """
    import os
    from matplotlib.animation import FFMpegWriter, PillowWriter

    ext = os.path.splitext(save_path)[1].lower()
    if ext in (".mp4", ".m4v", ".mov"):
        try:
            writer = FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-crf", str(int(crf)),
                    "-movflags", "+faststart",
                    # ensure even width/height (required by yuv420p)
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-threads", "0",           # let ffmpeg use all cores,
                    "-preset", "veryfast",     # <<— add this (or "faster"/"superfast"/"ultrafast")
                ],
            )
            ani.save(save_path, writer=writer, dpi=dpi)
            plt.close(fig)
            return
        except Exception as e:
            # Fall back to GIF with same basename
            gif_path = os.path.splitext(save_path)[0] + ".gif"
            print(f"FFmpeg failed ({e!r}); falling back to GIF: {gif_path}")
            try:
                writer = PillowWriter(fps=fps)
                ani.save(gif_path, writer=writer)
                plt.close(fig)
                return
            except Exception as e2:
                print(f"Pillow writer also failed ({e2!r}); showing interactively instead.")
                plt.show()
                return
    elif ext == ".gif":
        try:
            writer = PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
            plt.close(fig)
            return
        except Exception as e:
            print(f"Pillow writer failed ({e!r}); showing interactively instead.")
            plt.show()
            return
    else:
        print(f"Unknown extension '{ext}', not saving. Showing interactively instead.")
        plt.show()
        return

def _to_np(x):
    return np.asarray(x)

def _robust_bounds(a, lo=0.1, hi=99.9, eps=1e-12):
    vmin = float(np.percentile(a, lo))
    vmax = float(np.percentile(a, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        pad = eps if vmin == 0 else abs(vmin) * 1e-2
        return vmin - pad, vmax + pad
    return vmin, vmax

# ===============================================================
# 1) Wave + Spectrum + Energy (investor-friendly “physics dashboard”)
# ===============================================================
def wave_spectrum_movie(
    output,
    direction="x",
    fps=30,
    interval_ms=None,             # if None, computed from fps
    save_path=None,
    spectrogram_cmap="magma",
    field_cmap="coolwarm",
    show_B=True,                  # NEW: include B-field
    crf=23,                       # NEW: H.264 quality (lower = better quality, bigger file)
    dpi=110,                      # NEW: lower dpi for smaller files
    b_direction="auto",
):
    """
    Wave growth dashboard:
      - E(x,t) ribbon + |Ê(k)| spectrum
      - (optional) B(x,t) ribbon + |B̂(k)| spectrum
      - Total / kinetic(e,i) / E / B energies

    Parameters
    ----------
    output : dict with standard fields
    direction : {'x','y','z'}  component to visualize
    show_B : bool              include B evolution panels
    crf : int                  H.264 quality (18~28 reasonable)
    dpi : int                  output dpi for saved movie frames
    """
    assert direction in "xyz"
    di = {"x":0, "y":1, "z":2}[direction]

    t0 = time.time()
    print(f"[wave_spectrum_movie] Start: preparing frames at {fps} fps -> {save_path or 'interactive window'}")

    # Unpack
    grid  = _to_np(output["grid"])
    T     = int(_to_np(output["total_steps"]))
    time  = _to_np(output["time_array"]) * float(_to_np(output["plasma_frequency"]))
    Nx    = grid.size
    if interval_ms is None:
        interval_ms = int(1000/max(1, fps))

    Eall  = _to_np(output["electric_field"][:, :, di])  # [T, Nx]
    if show_B:
        if b_direction == "auto":
            # choose the B component with the largest RMS across time & x
            Ball_all = _to_np(output["magnetic_field"])  # [T, Nx, 3]
            # prevent NaNs
            Ball_all = np.nan_to_num(Ball_all, nan=0.0)
            rms = np.sqrt(np.mean(Ball_all**2, axis=(0,1)))  # [3]
            bi = int(np.argmax(rms))
            b_dir_label = "xyz"[bi]
        else:
            assert b_direction in "xyz"
            bi = {"x":0, "y":1, "z":2}[b_direction]
            b_dir_label = b_direction

        Ball = Ball_all[:, :, bi]
    else:
        Ball = None
        bi = None
        b_dir_label = None

    # Energies
    KEe = _to_np(output.get("kinetic_energy_electrons", np.zeros(T)))
    KEi = _to_np(output.get("kinetic_energy_ions",     np.zeros(T)))
    EE  = _to_np(output.get("electric_field_energy",   np.zeros(T)))
    BE  = _to_np(output.get("magnetic_field_energy",   np.zeros(T)))
    TE  = _to_np(output.get("total_energy",            KEe+KEi+EE+BE))

    # Robust scaling for ribbons
    e_vmin, e_vmax = _robust_bounds(Eall, 0.1, 99.9)
    if show_B:
        b_vmin, b_vmax = _robust_bounds(Ball, 0.1, 99.9)

    # Spectra (precompute)
    Ek = np.abs(np.fft.rfft(Eall, axis=1))   # [T, Nk]
    Ek_vis = np.log1p(Ek)
    if show_B:
        Bk = np.abs(np.fft.rfft(Ball, axis=1))
        Bk_vis = np.log1p(Bk)
    k = np.fft.rfftfreq(Nx, d=(grid[1]-grid[0]))
    sE_vmin, sE_vmax = _robust_bounds(Ek_vis, 0.1, 99.9)
    if show_B:
        sB_vmin, sB_vmax = _robust_bounds(Bk_vis, 0.1, 99.9)

    # Figure layout
    nrows = 3 if show_B else 2
    fig = plt.figure(figsize=(11.8, 6.8 if show_B else 5.2), dpi=dpi)
    if show_B:
        gs = fig.add_gridspec(nrows, 2, height_ratios=[1.6, 1.6, 1.2], width_ratios=[2.0, 1.2], hspace=0.28, wspace=0.28)
        ax_ribbonE = fig.add_subplot(gs[0, 0])
        ax_specE   = fig.add_subplot(gs[0, 1])
        ax_ribbonB = fig.add_subplot(gs[1, 0])
        ax_specB   = fig.add_subplot(gs[1, 1])
        ax_energy  = fig.add_subplot(gs[2, :])
    else:
        gs = fig.add_gridspec(nrows, 2, height_ratios=[2.0, 1.2], width_ratios=[2.0, 1.2], hspace=0.28, wspace=0.28)
        ax_ribbonE = fig.add_subplot(gs[0, 0])
        ax_specE   = fig.add_subplot(gs[0, 1])
        ax_energy  = fig.add_subplot(gs[1, :])
        ax_ribbonB = ax_specB = None

    # --- E ribbon
    bufE = np.zeros_like(Eall)
    im_ribbonE = ax_ribbonE.imshow(
        bufE, origin="lower", aspect="auto", cmap=field_cmap,
        extent=[grid[0], grid[-1], time[0], time[-1]],
        vmin=e_vmin, vmax=e_vmax, animated=True,
    )
    ax_ribbonE.set_title(f"Electric field $E_{direction}(x,t)$")
    ax_ribbonE.set_xlabel(f"{direction}-position (m)")
    ax_ribbonE.set_ylabel(r"Time ($\omega_{pe}^{-1}$)")

    # --- E spectrum
    barsE = ax_specE.bar(k, Ek_vis[0], width=(k[1]-k[0]) if len(k) > 1 else 1.0, align="center")
    ax_specE.set_xlim(k[0], k[-1] if len(k)>1 else k[0]+1)
    ax_specE.set_ylim(sE_vmin, sE_vmax)
    ax_specE.set_xlabel("Wavenumber k (1/m)")
    ax_specE.set_ylabel(r"$\log(1+|\hat{E}(k)|)$")
    ax_specE.set_title("E-mode content")

    # --- B ribbon/spectrum (optional)
    if show_B:
        bufB = np.zeros_like(Ball)
        im_ribbonB = ax_ribbonB.imshow(
            bufB, origin="lower", aspect="auto", cmap=field_cmap,
            extent=[grid[0], grid[-1], time[0], time[-1]],
            vmin=b_vmin, vmax=b_vmax, animated=True,
        )
        ax_ribbonB.set_title(f"Magnetic field $B_{b_dir_label}(x,t)$")
        ax_ribbonB.set_xlabel(f"{direction}-position (m)")
        ax_ribbonB.set_ylabel(r"Time ($\omega_{pe}^{-1}$)")

        barsB = ax_specB.bar(k, Bk_vis[0], width=(k[1]-k[0]) if len(k) > 1 else 1.0, align="center")
        ax_specB.set_xlim(k[0], k[-1] if len(k)>1 else k[0]+1)
        ax_specB.set_ylim(sB_vmin, sB_vmax)
        ax_specB.set_xlabel("Wavenumber k (1/m)")
        ax_specB.set_ylabel(r"$\log(1+|\hat{B}(k)|)$")
        ax_specB.set_title(f"B-mode content ($B_{b_dir_label}$)")

    # --- Energy panel
    (ltotal,) = ax_energy.plot(time, np.zeros_like(time), lw=2.0, color="black", label="Total")
    (lke_e,)  = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#0066cc", label="Kinetic (e⁻)")
    (lke_i,)  = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#00aa66", label="Kinetic (ions)")
    (le,)     = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#aa0000", label="E-field")
    (lb,)     = ax_energy.plot(time, np.zeros_like(time), lw=1.6, color="#aa6600", label="B-field")
    ax_energy.set_title("Energies")
    ax_energy.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
    ax_energy.set_ylabel("Energy (J)")
    y_all = np.stack([TE, KEe, KEi, EE, BE], axis=0)
    ymin, ymax = _robust_bounds(y_all, 0.1, 99.9)
    pad = 0.08 * (ymax - ymin + 1e-12)
    ax_energy.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax_energy.grid(alpha=0.25)
    ax_energy.legend(fontsize=8, loc="upper right")

    time_textE = ax_specE.text(0.02, 0.92, "", transform=ax_specE.transAxes,
                               ha="left", va="top",
                               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                               animated=True)
    time_textB = None
    if show_B:
        time_textB = ax_specB.text(0.02, 0.92, "", transform=ax_specB.transAxes,
                                   ha="left", va="top",
                                   bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                                   animated=True)

    artists = [im_ribbonE, time_textE] + list(barsE) + [ltotal, lke_e, lke_i, le, lb]
    if show_B:
        artists = [*artists, im_ribbonB, time_textB, *list(barsB)]
    artists = tuple(artists)

    def _update(ti):
        # paint ribbons
        bufE[ti] = Eall[ti]; im_ribbonE.set_array(bufE)
        hE = Ek_vis[ti]
        for b, val in zip(barsE, hE): b.set_height(val)

        if show_B:
            bufB[ti] = Ball[ti]; im_ribbonB.set_array(bufB)
            hB = Bk_vis[ti]
            for b, val in zip(barsB, hB): b.set_height(val)

        # energies
        ltotal.set_data(time[:ti+1], TE[:ti+1])
        lke_e.set_data(time[:ti+1], KEe[:ti+1])
        lke_i.set_data(time[:ti+1], KEi[:ti+1])
        le.set_data(time[:ti+1], EE[:ti+1])
        lb.set_data(time[:ti+1], BE[:ti+1])

        time_textE.set_text(f"t = {time[ti]:.2f}  $\\omega_{{pe}}^{{-1}}$")
        if time_textB is not None:
            time_textB.set_text(f"t = {time[ti]:.2f}  $\\omega_{{pe}}^{{-1}}$")
        return artists

    ani = FuncAnimation(fig, _update, frames=T, interval=interval_ms, blit=True, repeat=False)
    if save_path is not None:
        _save_animation(ani, fig, save_path, fps=fps, dpi=dpi, crf=crf)
        print(f"[wave_spectrum_movie] Done: wrote {save_path} in {time.time()-t0:.2f}s")
    else:
        print(f"[wave_spectrum_movie] Done: displayed in {time.time()-t0:.2f}s")
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

    t0 = time.time()
    print(f"[phase_space_movie] Start: preparing frames at {fps} fps -> {save_path or 'interactive window'}")

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
    v_e_vmin, v_e_vmax = _robust_bounds(Ve, 0.1, 99.9) if Ne>0 else (-1.0, 1.0)
    v_i_vmin, v_i_vmax = _robust_bounds(Vi, 0.1, 99.9) if Ni>0 else (-1.0, 1.0)

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
    if save_path is not None:
        _save_animation(ani, fig, save_path, fps=fps, dpi=110, crf=23)
        print(f"[phase_space_movie] Done: wrote {save_path} in {time.time()-t0:.2f}s")
    else:
        print(f"[phase_space_movie] Done: displayed in {time.time()-t0:.2f}s")
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

    t0 = time.time()
    print(f"[particle_box_movie] Start: preparing frames at {fps} fps -> {save_path or 'interactive window'}")

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
        vmin = float(np.percentile(E_all, 0.1))
        vmax = float(np.percentile(E_all, 99.9))
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
    if save_path is not None:
        _save_animation(ani, fig, save_path, fps=fps, dpi=110, crf=23)
        print(f"[particle_box_movie] Done: wrote {save_path} in {time.time()-t0:.2f}s")
    else:
        print(f"[particle_box_movie] Done: displayed in {time.time()-t0:.2f}s")
        plt.show()

    return ani