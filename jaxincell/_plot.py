# jaxincell/_plot.py
import os
import shutil
import subprocess
from functools import lru_cache

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

import jax.numpy as jnp

__all__ = ["plot"]

# ======================================================================================
# Helpers (kept small + classroom-friendly)
# ======================================================================================

_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
max_bins_phase_space = 251  # max number of velocity bins in phase space plots
min_bins_phase_space = 111  # min number of velocity bins in phase space plots

def _parse_direction(direction: str) -> List[str]:
    """
    direction:
      - "x"  -> ["x"]
      - "xz" -> ["x","z"]
      - "xy" -> ["x","y"]
    """
    if not isinstance(direction, str):
        raise TypeError("direction must be a string like 'x' or 'xz'.")

    direction = direction.strip().lower()
    if len(direction) not in (1, 2) or any(c not in "xyz" for c in direction):
        raise ValueError("direction must be one or two of 'x', 'y', or 'z' (e.g. 'x', 'xz').")

    # Keep order, disallow duplicates like "xx"
    dirs = list(direction)
    if len(dirs) == 2 and dirs[0] == dirs[1]:
        raise ValueError("direction with two letters must be distinct (e.g. 'xz', not 'xx').")
    return dirs


def _robust_abs_max(a, q: float = 99.0, eps: float = 1e-30) -> float:
    """Robust symmetric scale: percentile(|a|)."""
    an = np.asarray(a)
    return float(max(np.percentile(np.abs(an), q), eps))


def _robust_vmax_from_samples(v_tn: np.ndarray, q: float = 99.5, pad: float = 1.25, eps: float = 1e-30) -> float:
    """
    Robust symmetric velocity span based on percentile(|v|) across the provided samples.
    This is what fixes the "ions look frozen because the v-axis is too wide" issue.
    """
    val = np.percentile(np.abs(v_tn), q)
    return float(max(pad * val, eps))


def _make_overlay_axes(fig: plt.Figure, ax_base: plt.Axes) -> plt.Axes:
    """
    Transparent axes placed exactly on top of ax_base. Used for instantaneous lines
    so they don't "climb" on the time axis.
    """
    bb = ax_base.get_position()
    ax_ov = fig.add_axes([bb.x0, bb.y0, bb.width, bb.height], frameon=False)
    ax_ov.patch.set_alpha(0.0)
    ax_ov.set_zorder(ax_base.get_zorder() + 10)

    # Keep it visually clean: no ticks/labels (avoids clashes with the colorbar).
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    return ax_ov


def _is_nonzero(field: jnp.ndarray, threshold: float) -> bool:
    return bool(jnp.max(jnp.abs(field)) > threshold)


def _combine_by_charge_sign(output: dict, want_negative: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return (positions, velocities) for:
      want_negative=True  -> all q < 0 (electrons + any extra negative species)
      want_negative=False -> all q > 0 (ions + any extra positive species)

    We try in order:
      1) legacy split from diagnostics(): position_electrons/ions, velocity_electrons/ions
      2) output["species"] list from diagnostics() (multi-species view)
    """
    if want_negative and ("position_electrons" in output) and ("velocity_electrons" in output):
        return output["position_electrons"], output["velocity_electrons"]
    if (not want_negative) and ("position_ions" in output) and ("velocity_ions" in output):
        return output["position_ions"], output["velocity_ions"]

    if "species" not in output:
        raise RuntimeError(
            "Could not find electron/ion velocities. "
            "Call diagnostics(output) before plot(output)."
        )

    pos_list, vel_list = [], []
    for sp in output["species"]:
        q = float(sp["charge"])
        if want_negative and q < 0:
            pos_list.append(sp["positions"])
            vel_list.append(sp["velocities"])
        if (not want_negative) and q > 0:
            pos_list.append(sp["positions"])
            vel_list.append(sp["velocities"])

    if not vel_list:
        raise RuntimeError("No particles found for the requested charge sign.")
    return jnp.concatenate(pos_list, axis=1), jnp.concatenate(vel_list, axis=1)

@lru_cache(maxsize=1)
def _ffmpeg_encoders_text() -> str:
    """
    Return the output of `ffmpeg -encoders` as a string, or "" if ffmpeg missing.
    Cached so we only run the subprocess once.
    """
    if shutil.which("ffmpeg") is None:
        return ""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out
    except Exception:
        return ""


def _ffmpeg_has_encoder(name: str) -> bool:
    txt = _ffmpeg_encoders_text()
    if not txt:
        return False
    # encoder lines typically contain: " V....D h264_videotoolbox ..."
    return f" {name} " in txt or f"\t{name} " in txt


def _auto_codec_order_for_mp4() -> List[str]:
    """
    Prefer codecs that are broadly playable by QuickTime / Windows players.
    (H.264 first; HEVC only if explicitly requested or as a later fallback.)
    """
    candidates = [
        # macOS hardware (fast + compatible)
        "h264_videotoolbox",
        # NVIDIA
        "h264_nvenc",
        # Intel QuickSync
        "h264_qsv",
        # AMD AMF (Windows)
        "h264_amf",
        # software fallback
        "libx264",

        # HEVC options (smaller, but more compatibility pitfalls)
        "hevc_videotoolbox",
        "hevc_nvenc",
        "hevc_qsv",
        "hevc_amf",
        "libx265",
    ]
    return [c for c in candidates if _ffmpeg_has_encoder(c)]


def _make_ffmpeg_writer_auto(
    out_path: str,
    fps: int,
    crf: Optional[int],
    preset: Optional[str],
    pix_fmt: str = "yuv420p",
    codec_override: Optional[str] = None,
):
    """
    Auto-select codec + args optimized for small files *and* player compatibility.

    Key compatibility rules:
      - Force even dimensions (yuv420p/h264 common requirement).
      - If HEVC in MP4, tag as hvc1 for QuickTime.
    """
    from matplotlib.animation import FFMpegWriter

    ext = os.path.splitext(out_path)[1].lower()
    if ext not in (".mp4", ".m4v", ".mov", ".webm"):
        ext = ".mp4"

    # Always force even pixel dims (Matplotlib dpi/figsize can produce odd sizes)
    vf_even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    if ext == ".webm":
        codec = "libvpx-vp9" if _ffmpeg_has_encoder("libvpx-vp9") else "libx264"
        extra = ["-vf", vf_even, "-pix_fmt", pix_fmt]
        if codec == "libvpx-vp9":
            q = crf if crf is not None else 38
            extra = ["-b:v", "0", "-crf", str(q), "-row-mt", "1", "-speed", "4", "-vf", vf_even, "-pix_fmt", pix_fmt]
        return FFMpegWriter(fps=fps, codec=codec, bitrate=-1, extra_args=extra)

    # MP4 family: pick codec
    if codec_override is not None:
        if not _ffmpeg_has_encoder(codec_override):
            raise RuntimeError(f"Requested codec '{codec_override}' not available in your ffmpeg.")
        codec = codec_override
    else:
        codecs = _auto_codec_order_for_mp4()
        codec = codecs[0] if codecs else "libx264"

    # Defaults
    if preset is None:
        preset = "veryfast" if codec in ("libx264", "libx265") else None

    if crf is None:
        crf = 32 if ("hevc" in codec or codec == "libx265") else 30

    # MP4 tags for Apple players
    tag_args: List[str] = []
    if "hevc" in codec or codec == "libx265":
        tag_args = ["-tag:v", "hvc1"]  # QuickTime compatibility for HEVC in MP4 
    elif "h264" in codec or codec == "libx264":
        tag_args = ["-tag:v", "avc1"]

    extra_args: List[str] = ["-vf", vf_even, "-pix_fmt", pix_fmt, "-movflags", "+faststart"] + tag_args

    if codec in ("libx264", "libx265"):
        if preset is not None:
            extra_args = ["-preset", preset] + extra_args
        extra_args = ["-crf", str(crf)] + extra_args
    else:
        # Hardware encoders: CRF support varies; keep it conservative.
        # If you want *smallest* with HEVC, prefer libx265 (slower but predictable).
        pass

    return FFMpegWriter(fps=fps, codec=codec, bitrate=-1, extra_args=extra_args)


def _pdf_over_frames_numpy(v_frames_n: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    PDF histogram for many frames, fast-ish numpy implementation.

    v_frames_n: shape (F, N)
    edges: shape (B+1,) covering the full plotting range
    returns: pdf shape (F, B), with integral ~ 1 for each frame.
    """
    v = np.asarray(v_frames_n, dtype=np.float64)
    F, N = v.shape
    B = len(edges) - 1

    # bin index in [0, B-1]
    idx = np.searchsorted(edges, v, side="right") - 1
    idx = np.clip(idx, 0, B - 1)

    counts = np.zeros((F, B), dtype=np.float32)
    f_idx = np.repeat(np.arange(F, dtype=np.int32), N)
    np.add.at(counts, (f_idx, idx.reshape(-1)), 1.0)

    widths = np.diff(edges).astype(np.float32)
    pdf = counts / (N * widths[None, :])
    return pdf


@dataclass
class _PrecomputedPhaseSpace:
    # counts per frame, shape (F, Xbins, Vbins)
    e_counts: np.ndarray
    i_counts: np.ndarray
    v_edges_e: np.ndarray
    v_edges_i: np.ndarray
    v_range_e: Tuple[float, float]
    v_range_i: Tuple[float, float]
    norm_e: LogNorm
    norm_i: LogNorm


@dataclass
class _PrecomputedPDF:
    # pdf per frame, shape (F, B)
    e_pdf: np.ndarray
    i_pdf: np.ndarray
    v_centers: np.ndarray
    scale_e0: float
    scale_i0: float


# ======================================================================================
# Main plot()
# ======================================================================================

def plot(
    output,
    direction: str = "x",
    threshold: float = 1e-12,
    save_mp4: str | None = None,
    fps: int = 30,
    dpi: int = 150,
    show: bool = True,
    animation_interval: int = 1,
    save_stride: int = 1,          # downsample frames for saving only (1 = keep all)
    save_dpi: Optional[int] = None,# if None, uses dpi; recommend 60 for small files
    save_crf: Optional[int] = None,# if None, uses codec-dependent default (small)
    save_preset: Optional[str] = None,  # for libx264/libx265 hardware encoders
    save_codec: Optional[str] = None,  # e.g. "libx264", "h264_videotoolbox", "libx265", "hevc_videotoolbox"
):
    """
    Production-ready plotting/animation for JAX-in-Cell outputs.

    What you get:
      1) Heatmaps (x vs time): E, B (nonzero components), charge density.
         - IMPORTANT: heatmap color limits are fixed over the whole run using a robust
           percentile, so growth/decay in time is visible (no per-frame re-normalization).

      2) Instantaneous E(x,t) overlay:
         - For each plotted electric-field component, we draw a line on top of the heatmap.
         - The line is normalized by a single GLOBAL robust scale over the whole run,
           so amplitude growth is visible.
         - Overlay axes have no ticks (prevents clashes with colorbars).

      3) Distribution functions f(v,t) (LAB FRAME; NO drift centering):
         - Shown as clean line plots in their own subplot(s) (no current-density heatmap).
         - Solid: current frame
         - Dashed: initial (t=0), labeled "(initial)"

      4) Phase space (x vs v) for electrons and ions for each requested component:
         - Uses a robust velocity range per species per component, so ion dynamics remains visible.
         - Uses LogNorm on counts (with +1 internally) so low-density structure is visible.

    Multi-species:
      - Uses diagnostics() legacy split if present (velocity_electrons/velocity_ions).
      - Otherwise combines output["species"] by charge sign (q<0 as electrons, q>0 as ions).
    """
    # ----------------------------
    # Parse directions and basic arrays
    # ----------------------------
    dirs = _parse_direction(direction)
    dir_indices = [_AXIS_TO_INDEX[d] for d in dirs]

    grid = np.asarray(output["grid"])
    time = np.asarray(output["time_array"]) * float(np.asarray(output["plasma_frequency"]))
    total_steps = int(output["total_steps"])
    box_size_x = float(output["length"])

    # frames we will actually render
    nframes = int(len(time))

    # SHOW: always every simulation frame
    frames_show = np.arange(nframes, dtype=np.int32)

    # SAVE: can downsample independently for smaller/faster files
    save_stride = max(1, int(save_stride))
    frames_save = np.arange(0, nframes, save_stride, dtype=np.int32)


    # multi-species safe combined arrays
    pos_e, vel_e = _combine_by_charge_sign(output, want_negative=True)
    pos_i, vel_i = _combine_by_charge_sign(output, want_negative=False)

    # to numpy for plotting/hist
    x_e = np.asarray(pos_e[:, :, 0])  # spatial axis is always x in this codebase
    x_i = np.asarray(pos_i[:, :, 0])

    # ----------------------------
    # Decide which heatmaps to show (CURRENT DENSITY REMOVED by request)
    # ----------------------------
    heatmaps = []

    def add_vector_field(field: str, unit: str, label_prefix: str):
        if field not in output:
            return
        for comp_i, axis in enumerate("xyz"):
            data = output[field][:, :, comp_i]
            if _is_nonzero(data, threshold):
                heatmaps.append(
                    dict(
                        field=field,
                        component=axis,
                        data=np.asarray(data),
                        title=f"{label_prefix} in the {axis} direction",
                        xlabel="x Position (m)",
                        ylabel=r"Time ($\omega_{pe}^{-1}$)",
                        cbar=f"{label_prefix} ({unit})",
                    )
                )

    add_vector_field("electric_field", "V/m", "Electric Field")
    add_vector_field("magnetic_field", "T", "Magnetic Field")

    # charge density always
    if "charge_density" in output:
        heatmaps.append(
            dict(
                field="charge_density",
                component=None,
                data=np.asarray(output["charge_density"]),
                title="Charge Density",
                xlabel="x Position (m)",
                ylabel=r"Time ($\omega_{pe}^{-1}$)",
                cbar=r"Charge density (C/m$^3$)",
            )
        )

    # ----------------------------
    # Precompute PDFs + phase space for each requested velocity component
    # ----------------------------
    bins_v = int(max(min_bins_phase_space, min(max_bins_phase_space, len(grid))))
    bins_x = int(len(grid))

    pre_pdf: Dict[str, _PrecomputedPDF] = {}
    pre_ps: Dict[str, _PrecomputedPhaseSpace] = {}

    for d, di in zip(dirs, dir_indices):
        ve = np.asarray(vel_e[:, :, di])
        vi = np.asarray(vel_i[:, :, di])

        # Robust velocity spans (THIS fixes ion "no dynamics" view)
        vmax_e = max(_robust_vmax_from_samples(ve, q=99.5, pad=1.25), 1e-30)
        vmax_i = max(_robust_vmax_from_samples(vi, q=99.5, pad=1.25), 1e-30)

        v_edges_e = np.linspace(-vmax_e, vmax_e, bins_v + 1)
        v_edges_i = np.linspace(-vmax_i, vmax_i, bins_v + 1)

        # Use a single common v_centers for plotting the two species:
        # choose the wider range and interpolate the narrower if needed (keep simple: common = max span)
        vmax_common = max(vmax_e, vmax_i)
        v_edges = np.linspace(-vmax_common, vmax_common, bins_v + 1)
        v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

        # PDFs over rendered frames only
        e_pdf = _pdf_over_frames_numpy(ve, v_edges)
        i_pdf = _pdf_over_frames_numpy(vi, v_edges)

        # Scale each species by its INITIAL max (axis fixed; bump/drift shows naturally)
        scale_e0 = float(max(np.max(e_pdf[0]), 1e-30))
        scale_i0 = float(max(np.max(i_pdf[0]), 1e-30))

        pre_pdf[d] = _PrecomputedPDF(
            e_pdf=e_pdf,
            i_pdf=i_pdf,
            v_centers=v_centers,
            scale_e0=scale_e0,
            scale_i0=scale_i0,
        )

        # Phase space histograms over rendered frames only
        x_range = (-box_size_x / 2, box_size_x / 2)
        v_range_e = (-vmax_e, vmax_e)
        v_range_i = (-vmax_i, vmax_i)

        e_counts = np.empty((nframes, bins_x, bins_v), dtype=np.float32)
        i_counts = np.empty((nframes, bins_x, bins_v), dtype=np.float32)

        for t in range(nframes):
            e_counts[t] = np.histogram2d(
                x_e[t], ve[t],
                bins=[bins_x, bins_v],
                range=[x_range, v_range_e],
            )[0].astype(np.float32)

            i_counts[t] = np.histogram2d(
                x_i[t], vi[t],
                bins=[bins_x, bins_v],
                range=[x_range, v_range_i],
            )[0].astype(np.float32)


        # LogNorm for visibility at low counts (add 1 in the images)
        e_vmax = float(max(np.percentile(e_counts + 1.0, 99.5), 2.0))
        i_vmax = float(max(np.percentile(i_counts + 1.0, 99.5), 2.0))
        norm_e = LogNorm(vmin=1.0, vmax=e_vmax)
        norm_i = LogNorm(vmin=1.0, vmax=i_vmax)

        pre_ps[d] = _PrecomputedPhaseSpace(
            e_counts=e_counts,
            i_counts=i_counts,
            v_edges_e=v_edges_e,
            v_edges_i=v_edges_i,
            v_range_e=v_range_e,
            v_range_i=v_range_i,
            norm_e=norm_e,
            norm_i=norm_i,
        )

    # ----------------------------
    # Layout (heatmaps + f(v) panels + phase space panels + energy)
    # ----------------------------
    ncols = 3
    n_heat = len(heatmaps)
    n_fv = len(dirs)              # one f(v) panel per requested velocity component
    n_ps = 2 * len(dirs)          # electron + ion phase space per component
    n_energy = 1                  # if there is space
    n_total = n_heat + n_fv + n_ps + n_energy
    nrows = int(np.ceil(n_total / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.8 * nrows), squeeze=False)

    # ---- draw heatmaps with fixed clim (robust over the whole run) ----
    E_heat_axes: Dict[Tuple[str, str], plt.Axes] = {}  # (field, component) -> ax
    heatmap_images = []
    idx = 0
    for hm in heatmaps:
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        data = hm["data"]

        vlim = _robust_abs_max(data, q=99.0)
        im = ax.imshow(
            data,
            aspect="auto",
            cmap="RdBu",
            origin="lower",
            extent=[grid[0], grid[-1], time[0], time[-1]],
            vmin=-vlim,
            vmax=vlim,
        )
        ax.set_title(hm["title"])
        ax.set_xlabel(hm["xlabel"])
        ax.set_ylabel(hm["ylabel"])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(hm["cbar"])

        heatmap_images.append(im)

        if hm["field"] == "electric_field" and hm["component"] is not None:
            E_heat_axes[(hm["field"], hm["component"])] = ax

        idx += 1

    # ---- f(v) panels (clean line plots; replaces current density subplot) ----
    fv_axes: Dict[str, plt.Axes] = {}
    fv_lines: Dict[str, Dict[str, plt.Line2D]] = {}  # d -> {"e":..., "i":..., "e0":..., "i0":...}
    fv_time_text: Dict[str, plt.Text] = {}

    for d in dirs:
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        fv_axes[d] = ax

        pdf = pre_pdf[d]
        v = pdf.v_centers

        # initial dashed
        (l_e0,) = ax.plot(v, pdf.e_pdf[0] / pdf.scale_e0, linestyle="--", linewidth=1.8, label="e− (initial)")
        (l_i0,) = ax.plot(v, pdf.i_pdf[0] / pdf.scale_i0, linestyle="--", linewidth=1.8, label="i+ (initial)")

        # current solid (initialized at frame 0)
        (l_e,) = ax.plot(v, pdf.e_pdf[0] / pdf.scale_e0, linewidth=2.4, label="e−")
        (l_i,) = ax.plot(v, pdf.i_pdf[0] / pdf.scale_i0, linewidth=2.4, label="i+")

        ax.set_title(rf"Distribution functions $f(v_{d})$")
        ax.set_xlabel(rf"$v_{d}$ (m/s)")
        ax.set_ylabel(r"$f(v)/\max(f_0)$")
        ax.set_ylim(0.0, 1.10)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

        txt = ax.text(
            0.5, 0.92, "", transform=ax.transAxes,
            ha="center", va="top", fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        for artist in (l_e0, l_i0, l_e, l_i, txt):
            artist.set_animated(True)

        fv_lines[d] = {"e": l_e, "i": l_i, "e0": l_e0, "i0": l_i0}
        fv_time_text[d] = txt

        idx += 1

    # ---- phase space panels (x vs v_d) for each d ----
    ps_images: Dict[str, Dict[str, plt.Axes]] = {}
    ps_ims: Dict[str, Dict[str, any]] = {}  # d -> {"e": im, "i": im}
    ps_time_text: Dict[str, plt.Text] = {}

    for d in dirs:
        # electrons
        r, c = divmod(idx, ncols)
        ax_e = axes[r, c]
        ps_images.setdefault(d, {})["e_ax"] = ax_e

        ps = pre_ps[d]
        im_e = ax_e.imshow(
            (ps.e_counts[0] + 1.0).T,
            aspect="auto",
            origin="lower",
            cmap="twilight",
            extent=[-box_size_x / 2, box_size_x / 2, ps.v_range_e[0], ps.v_range_e[1]],
            norm=ps.norm_e,
        )
        ax_e.set_title(rf"Electron phase space $(x, v_{d})$")
        ax_e.set_xlabel("x (m)")
        ax_e.set_ylabel(rf"$v_{d}$ (m/s)")
        cb = fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)
        cb.set_label("counts (log)")

        idx += 1

        # ions
        r, c = divmod(idx, ncols)
        ax_i = axes[r, c]
        ps_images.setdefault(d, {})["i_ax"] = ax_i

        im_i = ax_i.imshow(
            (ps.i_counts[0] + 1.0).T,
            aspect="auto",
            origin="lower",
            cmap="twilight",
            extent=[-box_size_x / 2, box_size_x / 2, ps.v_range_i[0], ps.v_range_i[1]],
            norm=ps.norm_i,
        )
        ax_i.set_title(rf"Ion phase space $(x, v_{d})$")
        ax_i.set_xlabel("x (m)")
        ax_i.set_ylabel(rf"$v_{d}$ (m/s)")
        cb = fig.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04)
        cb.set_label("counts (log)")

        idx += 1

        # one shared time label (put it on electron panel of the first direction)
        if d == dirs[0]:
            txt = ax_e.text(
                0.5, 0.92, "", transform=ax_e.transAxes,
                ha="center", va="top", fontsize=11,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
            txt.set_animated(True)
            ps_time_text[d] = txt

        im_e.set_animated(True)
        im_i.set_animated(True)
        ps_ims[d] = {"e": im_e, "i": im_i}

    # ---- energy panel (static) ----
    # If we ran out of axes, just skip.
    axes_flat = np.ravel(axes)
    if idx < len(axes_flat) and all(k in output for k in ("total_energy", "electric_field_energy", "kinetic_energy_electrons", "kinetic_energy_ions")):
        ax_en = axes_flat[idx]
        ax_en.plot(time, np.asarray(output["total_energy"]), label="Total energy")
        ax_en.plot(time, np.asarray(output["kinetic_energy_electrons"]), label="Kinetic energy electrons")
        ax_en.plot(time, np.asarray(output["kinetic_energy_ions"]), label="Kinetic energy ions")
        ax_en.plot(time, np.asarray(output["electric_field_energy"]), label="Electric field energy")
        if "magnetic_field_energy" in output and np.max(np.asarray(output["magnetic_field_energy"])) > 1e-12:
            ax_en.plot(time, np.asarray(output["magnetic_field_energy"]), label="Magnetic field energy")

        # relative energy error
        te = np.asarray(output["total_energy"])
        denom = float(max(abs(te[0]), 1e-30))
        ax_en.plot(time[1:], np.abs(te[1:] - te[0]) / denom, label="Relative energy error")

        ax_en.set_title("Energy")
        ax_en.set_xlabel(r"Time ($\omega_{pe}^{-1}$)")
        ax_en.set_ylabel("Energy (J)")
        ax_en.set_yscale("log")
        ax_en.legend(fontsize=8, frameon=False)
        idx += 1

    # Hide any unused axes
    for j in range(idx, len(axes_flat)):
        axes_flat[j].axis("off")

    # Tight layout BEFORE overlays (prevents the "not centered" + overlay misalignment issues)
    fig.tight_layout()
    fig.canvas.draw()

    # ----------------------------
    # E(x,t) overlays on E heatmaps (thicker + high-contrast)
    # ----------------------------
    E_overlays: List[Tuple[plt.Line2D, np.ndarray, plt.Text]] = []
    # Precompute normalized E lines only for rendered frames (fast update)
    for (field, comp), ax in E_heat_axes.items():
        comp_i = _AXIS_TO_INDEX[comp]
        E_all = np.asarray(output["electric_field"][:, :, comp_i])  # (T, X)
        E_scale = _robust_abs_max(E_all, q=99.0)  # FIXED over whole run => growth visible
        E_lines = np.clip(E_all / E_scale, -1.0, 1.0)      # (T, X)

        ax_ov = _make_overlay_axes(fig, ax)
        (line,) = ax_ov.plot(grid, E_lines[0], color="black", linewidth=2.6, alpha=0.95)
        ax_ov.set_xlim(grid[0], grid[-1])
        ax_ov.set_ylim(-1.05, 1.05)

        txt = ax_ov.text(
            0.5, 0.92, "", transform=ax_ov.transAxes,
            ha="center", va="top", fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        line.set_animated(True)
        txt.set_animated(True)
        E_overlays.append((line, E_lines, txt))

    # ----------------------------
    # Animation update
    # ----------------------------
    def _render_at_time_index(t: int):
        artists = []

        # phase space updates
        for d in dirs:
            ps = pre_ps[d]
            ps_ims[d]["e"].set_array((ps.e_counts[t] + 1.0).T)
            ps_ims[d]["i"].set_array((ps.i_counts[t] + 1.0).T)
            artists += [ps_ims[d]["e"], ps_ims[d]["i"]]

        # time label (only once)
        if dirs[0] in ps_time_text:
            ps_time_text[dirs[0]].set_text(f"Time: {time[t]:.2f} ωₚ")
            artists.append(ps_time_text[dirs[0]])

        # distribution function updates (solid curves only)
        for d in dirs:
            pdf = pre_pdf[d]
            fv_lines[d]["e"].set_ydata(pdf.e_pdf[t] / pdf.scale_e0)
            fv_lines[d]["i"].set_ydata(pdf.i_pdf[t] / pdf.scale_i0)
            fv_time_text[d].set_text(f"t = {time[t]:.2f} ωₚ")
            artists += [fv_lines[d]["e"], fv_lines[d]["i"], fv_time_text[d]]
            artists += [fv_lines[d]["e0"], fv_lines[d]["i0"]]

        # E overlays
        for line, E_lines, txt in E_overlays:
            line.set_ydata(E_lines[t])
            txt.set_text(f"t = {time[t]:.2f} ωₚ")
            artists += [line, txt]

        return artists


    def update_show(frame_i: int):
        t = int(frames_show[frame_i])
        return _render_at_time_index(t)

    # SHOW animation (always every frame)
    ani_show = FuncAnimation(
        fig, update_show,
        frames=len(frames_show),
        blit=True,
        interval=animation_interval,
        repeat_delay=800,
    )

    # SAVE (optional): make a separate animation with its own frame list
    if save_mp4 is not None:
        try:
            save_dpi_eff = dpi if save_dpi is None else int(save_dpi)

            def update_save(frame_i: int):
                t = int(frames_save[frame_i])
                return _render_at_time_index(t)

            ani_save = FuncAnimation(
                fig, update_save,
                frames=len(frames_save),
                blit=True,
                interval=animation_interval,  # doesn't matter much for saving
                repeat_delay=800,
            )

            writer = _make_ffmpeg_writer_auto(
                out_path=save_mp4,
                fps=fps,
                crf=save_crf,
                preset=save_preset,
                codec_override=save_codec,
            )
            ani_save.save(save_mp4, writer=writer, dpi=save_dpi_eff)
            print(f"Saved animation to: {save_mp4}")
        except Exception as e:
            warnings.warn(
                "Failed to save video. This usually means ffmpeg is not installed or not on PATH.\n"
                f"Error was: {e}\n"
                "Install ffmpeg (e.g. `conda install -c conda-forge ffmpeg`) and try again.",
                RuntimeWarning,
            )


    if show:
        plt.show()
    else:
        plt.close(fig)
