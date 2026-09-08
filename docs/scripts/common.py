"""Shared style and helpers for the documentation figures.

Every figure script here imports this module. Figures go to
``docs/_static/figures``; the numbers the prose quotes are recorded in
``docs/_static/figures/measurements.json`` and pulled into the text as MyST
substitutions, so the text and the committed figures always come from one run.
"""
import json
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

# Okabe-Ito palette, colourblind safe. The assignment is fixed across figures:
# electrons blue, ions orange, explicit blue, implicit vermillion, theory black.
COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
          "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
          "grey": "#7F7F7F"}
C_ELECTRONS = C_EXPLICIT = COLORS["blue"]
C_IONS = COLORS["orange"]
C_IMPLICIT = COLORS["vermillion"]
C_THEORY = COLORS["black"]
C_FIT = COLORS["green"]
CMAP_SIGNED, CMAP_DENSITY = "RdBu_r", "viridis"
SINGLE, WIDE, TALL = (6.0, 3.6), (7.4, 3.4), (6.0, 6.0)

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "lines.linewidth": 1.5, "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def savefig(fig, name, colors=256):
    """Write a figure and shrink it for the repository.

    Matplotlib uses far fewer than 256 distinct colours outside the colour maps,
    so quantising to an adaptive palette is visually indistinguishable and about
    halves the file; optipng then squeezes the rest. Both steps are skipped
    quietly when Pillow or optipng is missing.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    before = path.stat().st_size
    try:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        image.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE).save(path, optimize=True)
    except Exception as error:                       # Pillow missing or an unusual image
        print(f"    (palette step skipped: {error})")
    if shutil.which("optipng"):
        subprocess.run(["optipng", "-quiet", "-o5", "-strip", "all", str(path)], check=False)
    print(f"wrote {path.relative_to(HERE.parent.parent)} "
          f"({before / 1024:.0f} kB -> {path.stat().st_size / 1024:.0f} kB)")
    return path


def record(**values):
    """Merge scalar results into measurements.json, which the docs text quotes."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    for key, value in values.items():
        data[key] = value.item() if isinstance(value, (np.floating, np.integer)) else value
        print(f"  {key} = {data[key]}")
    MEASUREMENTS.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")


def panel_label(ax, text, x=-0.14, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            va="bottom", ha="left")


def mode_amplitude(out, mode):
    """Complex Fourier amplitude of one mode of E_x at every stored step."""
    return np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, mode] / out.E.shape[1]


def maxima(amplitude, above=0.0):
    """Indices of the local maxima of an envelope that stand above ``above``."""
    i = np.arange(1, amplitude.size - 1)
    return i[(amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] > amplitude[2:]) & (amplitude[1:-1] > above)]


def rate_and_frequency(t, amplitude, above=0.0):
    """Growth rate and angular frequency from the maxima of |E_k(t)|: the slope of
    ln|E_k| through them and pi over their mean spacing, since successive maxima
    of a modulus are half a period apart."""
    i = maxima(amplitude, above)
    return np.polyfit(t[i], np.log(amplitude[i]), 1)[0], np.pi / np.mean(np.diff(t[i])), i


def robust_growth_fit(time, energy, min_r2=0.95, min_duration=15.0, min_efolds=1.5, min_points=12):
    """Fit an exponential to a mode energy, choosing the window by a stated rule.

    Scans windows inside the growth phase (everything before the peak) and keeps
    the **longest** one whose straight-line fit to ``ln(energy)`` reaches
    ``min_r2``. Requiring length rather than the steepest or best-correlated
    window avoids two failure modes: a short window sitting on a noise excursion,
    which can return an arbitrarily large rate, and the seed transient at the
    start, during which the perturbation has not yet settled onto the growing
    eigenmode.

    Returns ``None`` when no window qualifies, which is the honest outcome for a
    mode that never grew cleanly; such modes are left out of the comparison with
    theory instead of being fitted anyway.

    Returns:
        dict or None: ``gamma`` (the amplitude rate, half the slope of the
        energy), ``intercept``, ``slope``, ``r2``, ``t0``, ``t1``, ``efolds``,
        ``duration``.
    """
    time, energy = np.asarray(time), np.asarray(energy)
    if (energy > 0).sum() < min_points:
        return None
    i_peak = int(np.argmax(energy))
    if i_peak < min_points:
        return None
    best = None
    edges = np.unique(np.linspace(0, i_peak, 60).astype(int))
    for index, a in enumerate(edges[:-1]):
        for b in edges[index + 1:]:
            t_seg, y = time[a:b + 1], energy[a:b + 1]
            duration = t_seg[-1] - t_seg[0]
            if len(t_seg) < min_points or duration < min_duration or np.any(y <= 0):
                continue
            slope, intercept = np.polyfit(t_seg, np.log(y), 1)
            if slope <= 0 or 0.5 * slope * duration < min_efolds:
                continue
            residual = np.log(y) - (intercept + slope * t_seg)
            spread = np.log(y) - np.log(y).mean()
            r2 = 1.0 - residual.dot(residual) / max(spread.dot(spread), 1e-300)
            if r2 >= min_r2 and (best is None or duration > best["duration"]):
                best = {"gamma": 0.5 * slope, "intercept": float(intercept), "slope": float(slope),
                        "r2": float(r2), "t0": float(t_seg[0]), "t1": float(t_seg[-1]),
                        "efolds": float(0.5 * slope * duration), "duration": float(duration)}
    return best


def phase_space_hist(ax, x, v, box_length, v_max, weights=None, bins=(70, 90), cmap=CMAP_DENSITY):
    """Weighted histogram of (x/L, v) for one species at one time. The colour
    scale saturates at the 99.5th percentile of the occupied bins so that a few
    dense bins do not hide the rest."""
    counts, xedges, vedges = np.histogram2d(x / box_length, v, bins=bins,
                                            range=[[-0.5, 0.5], [-v_max, v_max]], weights=weights)
    counts = counts.T
    vmax = np.percentile(counts[counts > 0], 99.5) if np.any(counts > 0) else 1.0
    image = ax.pcolormesh(xedges, vedges, counts, cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
    ax.set(xlim=(-0.5, 0.5), ylim=(-v_max, v_max))
    ax.grid(False)
    return image


def maxwellian_populations(simulation):
    """The drifting-Maxwellian populations of a Simulation, in the form the
    dispersion solvers want. A species built with ``plus_minus`` is two beams of
    half the density drifting in opposite directions."""
    from dispersion import plasma_frequency
    populations = []
    for s in simulation.species:
        common = {"name": s.name, "vthx": s.vth[0] or 1.0, "vth": s.vth[0] or 1.0,
                  "A": (s.vth[2] / s.vth[0]) ** 2 if s.vth[0] else 1.0}
        if s.plus_minus:
            wp = plasma_frequency(s.density / 2, s.charge_si, s.mass)
            populations += [{**common, "wp": wp, "u": sign * s.drift[0], "density": s.density / 2}
                            for sign in (+1, -1)]
        else:
            populations.append({**common, "wp": plasma_frequency(s.density, s.charge_si, s.mass),
                                "u": s.drift[0], "density": s.density})
    return populations
