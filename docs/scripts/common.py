"""Shared plotting style and helpers for the documentation figures.

Every figure script in this directory imports this module. Figures are written
to ``docs/_static/figures`` and numerical results that the documentation quotes
are recorded in ``docs/_static/figures/measurements.json`` so that the text and
the plots always come from the same run.
"""
import json
import os
import pathlib
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE.parent / "_static" / "figures"
MEASUREMENTS = FIGURE_DIR / "measurements.json"
EXAMPLES_DIR = HERE.parent.parent / "examples"

# Okabe-Ito palette, colourblind safe. The assignment is fixed across all
# figures: electrons are blue, ions are orange, the explicit scheme is blue,
# the implicit scheme is vermillion, and theory is a black dashed line.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#7F7F7F",
}
C_ELECTRONS = COLORS["blue"]
C_IONS = COLORS["orange"]
C_EXPLICIT = COLORS["blue"]
C_IMPLICIT = COLORS["vermillion"]
C_THEORY = COLORS["black"]
C_FIT = COLORS["green"]
CMAP_SIGNED = "RdBu_r"
CMAP_DENSITY = "viridis"

# One panel of a figure, in inches. Every figure is built from panels of this
# size, so that type and line widths look the same in all of them when the pages
# scale the images to the text width.
PANEL = (9.0, 7.0)
SINGLE = PANEL
WIDE = (2 * PANEL[0], PANEL[1])

# The group's figure style: a heavy frame, ticks turned inward on all four sides
# with visible minor ticks, no grid, and type large enough to read when a figure
# is scaled down to the width of a page or a slide.
STYLE = {
    "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 24, "legend.fontsize": 18,
    "xtick.labelsize": 24, "ytick.labelsize": 24, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True, "axes.linewidth": 3.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 3.0, "ytick.major.width": 3.0, "xtick.major.size": 7.0,
    "ytick.major.size": 7.0, "xtick.minor.width": 2.0, "ytick.minor.width": 2.0,
    "xtick.minor.size": 5.0, "ytick.minor.size": 5.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 3.0, "lines.markersize": 10.0, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 110, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "figure.facecolor": "white", "mathtext.fontset": "dejavusans",
}
plt.rcParams.update(STYLE)


def figure(ncols=1, nrows=1, aspect=None, **kwargs):
    """``ncols`` by ``nrows`` panels of :data:`PANEL` inches each.

    ``aspect`` gives a flatter (or taller) panel, as a fraction of the panel width.
    Remaining keyword arguments go to ``plt.subplots``.
    """
    height = PANEL[0] * aspect if aspect else PANEL[1]
    return plt.subplots(nrows, ncols, figsize=(ncols * PANEL[0], nrows * height), **kwargs)


def compress_png(path, colors=256):
    """Shrink a figure for the repository: quantise to a palette, then optipng.

    Matplotlib output uses far fewer than 256 distinct colours outside the
    colour maps, so an adaptive palette is visually indistinguishable here while
    roughly halving the file. Both steps are optional: if Pillow or optipng is
    missing the original file is left in place.
    """
    path = pathlib.Path(path)
    before = path.stat().st_size
    try:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        image.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE).save(
            path, optimize=True)
    except Exception as error:                       # Pillow missing or unusual image
        print(f"    (palette step skipped: {error})")
    if shutil.which("optipng"):
        subprocess.run(["optipng", "-quiet", "-o5", "-strip", "all", str(path)], check=False)
    after = path.stat().st_size
    return before, after


def savefig(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    before, after = compress_png(path)
    print(f"wrote {path.relative_to(HERE.parent.parent)} "
          f"({before/1024:.0f} kB -> {after/1024:.0f} kB)")
    return path


def record(**values):
    """Merge scalar results into measurements.json (used by the docs text)."""
    import fcntl
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIGURE_DIR / ".measurements.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)     # scripts may run in parallel
        data = json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
        for key, value in values.items():
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            data[key] = value
        MEASUREMENTS.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    for key, value in values.items():
        print(f"  {key} = {value}")


def panel_label(ax, text, x=-0.16, y=1.02):
    """Bold panel letter, for example "(a)", above the top-left corner of ``ax``."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=22, fontweight="bold",
            va="bottom", ha="left")


def plain_log_ticks(axis, lo, hi):
    """Label a logarithmic ``axis`` (``ax.xaxis`` or ``ax.yaxis``) at 1, 2, 5 times
    powers of ten between ``lo`` and ``hi``, written as plain numbers."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ticks = [m * 10.0**e for e in range(-3, 6) for m in (1, 2, 5) if lo <= m * 10.0**e <= hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def fit_growth_rate(time, energy, t_start, t_end):
    """Least-squares slope of ln(energy) on [t_start, t_end].

    Returns the growth (or damping) rate of the field amplitude, that is half
    the slope of the energy, together with the intercept for drawing the fit.
    """
    time = np.asarray(time)
    energy = np.asarray(energy)
    mask = (time >= t_start) & (time <= t_end) & (energy > 0)
    slope, intercept = np.polyfit(time[mask], np.log(energy[mask]), 1)
    return 0.5 * slope, intercept, slope


def quiet_parameters(parameters):
    parameters.setdefault("solver_parameters", {})["print_info"] = False
    return parameters


def silence_progress_bars():
    """jax_tqdm writes a progress bar per run; keep figure logs readable."""
    os.environ.setdefault("TQDM_DISABLE", "1")


def species_for_linear_theory(output):
    """Build the drifting-Maxwellian populations used by the dispersion solvers.

    Reads the cleaned species parameters and the pseudo-particle weights from a
    simulation output so that the theory uses exactly the densities, drifts and
    thermal speeds the run was initialised with. A population created with
    ``velocity_plus_minus_x`` is split into two half-density beams.
    """
    from jaxincell import speed_of_light
    from dispersion import plasma_frequency

    length = float(np.asarray(output["length"]))
    weights = np.asarray(output["weights"]).reshape(-1)
    index = np.asarray(output["species_integer_index"]).reshape(-1)
    charge_lookup = np.asarray(output["charge_integer_lookup"]).reshape(-1)
    mass_lookup = np.asarray(output["mass_integer_lookup"]).reshape(-1)

    populations = []
    integer_index = 0
    for species_type in ("electrons", "ions"):
        for label, sp in output["species_parameters"][species_type].items():
            mask = index == integer_index
            density = float(weights[mask].sum() / length)
            charge = float(charge_lookup[integer_index])
            mass = float(mass_lookup[integer_index])
            wp = plasma_frequency(density, charge, mass)
            u = float(np.asarray(sp["drift_speed_x"]))
            vth = {axis: float(np.asarray(sp[f"vth_over_c_{axis}"])) * speed_of_light for axis in "xyz"}
            base = {"name": f"{species_type}.{sp.get('user_label', label)}", "mass": mass,
                    "charge": charge, "density": density, "vthx": vth["x"],
                    "vthy": vth["y"], "vthz": vth["z"]}
            if bool(sp["velocity_plus_minus_x"]):
                for sign in (+1, -1):
                    populations.append({**base, "wp": wp / np.sqrt(2.0), "u": sign * u,
                                        "vth": vth["x"], "density": density / 2})
            else:
                populations.append({**base, "wp": wp, "u": u, "vth": vth["x"]})
            integer_index += 1
    return populations


def linear_window(time, energy, lower=1e-4, upper=1e-1):
    """Time window of exponential growth: between the last time the energy was
    below ``lower`` * peak and the last time it was below ``upper`` * peak,
    both before the first peak."""
    energy = np.asarray(energy)
    time = np.asarray(time)
    i_peak = int(np.argmax(energy))
    peak = energy[i_peak]
    before = np.arange(i_peak)
    lo = before[energy[:i_peak] < lower * peak]
    hi = before[energy[:i_peak] < upper * peak]
    i_lo = int(lo[-1]) if lo.size else 0
    i_hi = int(hi[-1]) if hi.size else max(i_peak - 1, 1)
    return time[i_lo], time[i_hi]


def phase_space_scatter(ax, x, v, box_length, v_max, color=C_ELECTRONS, size=2.0):
    """Scatter plot of (x / L, v) for one species and one time (small N)."""
    ax.scatter(x / box_length, v, s=size, color=color, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-v_max, v_max)
    ax.grid(False)


def phase_space_hist(ax, x, v, box_length, v_max, weights=None, bins=(70, 90), cmap=CMAP_DENSITY):
    """Weighted 2D histogram of (x / L, v) for one species and one time.

    The colour scale is linear and saturates at the 99.5th percentile of the
    occupied bins so that a few dense bins do not hide the rest."""
    counts, xedges, vedges = np.histogram2d(
        x / box_length, v, bins=bins, range=[[-0.5, 0.5], [-v_max, v_max]], weights=weights)
    counts = counts.T
    vmax = np.percentile(counts[counts > 0], 99.5) if np.any(counts > 0) else 1.0
    im = ax.pcolormesh(xedges, vedges, counts, cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-v_max, v_max)
    ax.grid(False)
    return im


def robust_growth_fit(time, energy, min_r2=0.95, min_duration=15.0, min_efolds=1.5,
                      min_points=12):
    """Fit an exponential to a mode energy, choosing the window by a stated rule.

    Scans windows inside the growth phase (everything before the peak) and keeps
    the **longest** one whose straight-line fit to ``ln(energy)`` reaches
    ``min_r2``. Requiring length rather than the steepest or best-correlated
    window avoids two failure modes: a short window sitting on a noise excursion,
    which can return an arbitrarily large rate, and the seed transient at the
    start of a run, during which the initial perturbation has not yet settled
    onto the growing eigenmode.

    Returns ``None`` when no window qualifies. That is the honest outcome for a
    mode that never grew cleanly, and such modes are left out of the comparison
    with theory rather than being fitted anyway.

    Args:
        time (array): Times, shape ``(S,)``, in units of the inverse plasma frequency.
        energy (array): Mode energy (amplitude squared), shape ``(S,)``.
        min_r2 (float): Required coefficient of determination of the fit.
        min_duration (float): Required window length, in the units of ``time``.
        min_efolds (float): Required growth of the amplitude across the window.
        min_points (int): Required number of samples in the window.

    Returns:
        dict or None: ``{"gamma", "intercept", "slope", "r2", "t0", "t1", "efolds",
        "duration"}`` with ``gamma`` the amplitude growth rate (half the slope of
        the energy), or ``None``.
    """
    time = np.asarray(time)
    energy = np.asarray(energy)
    if (energy > 0).sum() < min_points:
        return None
    i_peak = int(np.argmax(energy))
    if i_peak < min_points:
        return None

    edges = np.unique(np.linspace(0, i_peak, 60).astype(int))
    best = None
    for index, a in enumerate(edges[:-1]):
        for b in edges[index + 1:]:
            t_seg, y = time[a:b + 1], energy[a:b + 1]
            duration = t_seg[-1] - t_seg[0]
            if len(t_seg) < min_points or duration < min_duration or np.any(y <= 0):
                continue
            log_y = np.log(y)
            slope, intercept = np.polyfit(t_seg, log_y, 1)
            if slope <= 0 or 0.5 * slope * duration < min_efolds:
                continue
            residual = log_y - (intercept + slope * t_seg)
            spread = log_y - log_y.mean()
            r2 = 1.0 - residual.dot(residual) / max(spread.dot(spread), 1e-300)
            if r2 < min_r2:
                continue
            if best is None or duration > best["duration"]:
                best = {"gamma": 0.5 * slope, "intercept": float(intercept),
                        "slope": float(slope), "r2": float(r2), "t0": float(t_seg[0]),
                        "t1": float(t_seg[-1]), "efolds": float(0.5 * slope * duration),
                        "duration": float(duration)}
    return best


def mode_window(t, mode_energy, noise_factor=30.0, top=0.1):
    """Exponential-growth window of one Fourier mode: from the last time the mode
    energy was below ``noise_factor`` times its initial level to the last time it was
    below ``top`` times its largest value, both before the first saturation.

    The first saturation is the first time the mode reaches half of its largest
    value. Near the edge of the unstable band the mode can dip after saturating
    and reach its largest value only later; without this cut the window would
    stretch over the saturated phase."""
    i_peak = int(np.argmax(mode_energy >= 0.5 * mode_energy.max()))
    noise = mode_energy[:20].mean()
    lo = np.where(mode_energy[:i_peak] < noise_factor * noise)[0]
    hi = np.where(mode_energy[:i_peak] < top * mode_energy.max())[0]
    t0 = t[lo[-1]] if lo.size else t[0]
    t1 = t[hi[-1]] if hi.size else t[max(i_peak - 1, 1)]
    if t1 <= t0:
        t0, t1 = t[max(i_peak // 4, 1)], t[max(i_peak - 1, 2)]
    return t0, t1


def analyse_two_stream(output):
    """Growth rate of the first Fourier mode of E_x in a two-stream run, and the
    kinetic root for the same populations.

    The rate is half the slope of ln|E_1|^2 over the window of :func:`mode_window`.
    Returns ``(t, gamma_fit, gamma_theory, (t0, t1, intercept, slope), e_folds)``
    with ``t`` in units of the inverse plasma frequency and ``e_folds`` the growth
    of the mode amplitude from its initial level to its peak."""
    from dispersion import electrostatic_epsilon, most_unstable_root
    wpe = float(output["plasma_frequency"])
    t = np.asarray(output["time_array"]) * wpe
    L = float(output["length"])
    Ex = np.asarray(output["electric_field"][:, :, 0])
    mode_energy = (np.abs(np.fft.rfft(Ex, axis=1))[:, 1] / Ex.shape[1]) ** 2
    t0, t1 = mode_window(t, mode_energy)
    gamma_fit, intercept, slope = fit_growth_rate(t, mode_energy, t0, t1)
    populations = species_for_linear_theory(output)
    root = most_unstable_root(lambda w: electrostatic_epsilon(w, 2 * np.pi / L, populations),
                              (-0.5, 0.5), (0.01, 1.0), n_real=21, n_imag=20, scale=wpe)
    gamma_th = root.imag / wpe if root is not None else np.nan
    e_folds = 0.5 * np.log(mode_energy.max() / mode_energy[:20].mean())
    return t, gamma_fit, gamma_th, (t0, t1, intercept, slope), e_folds


def cpu_name():
    """Human-readable processor name (macOS sysctl, Linux /proc/cpuinfo, else platform)."""
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()
