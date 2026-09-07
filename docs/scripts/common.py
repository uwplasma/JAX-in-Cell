"""Shared plotting style and helpers for the documentation figures.

Every figure script in this directory imports this module. Figures are written
to ``docs/_static/figures`` and numerical results that the documentation quotes
are recorded in ``docs/_static/figures/measurements.json`` so that the text and
the plots always come from the same run.
"""
import json
import os
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

SINGLE = (6.0, 3.6)
WIDE = (7.4, 3.4)
TALL = (6.0, 6.0)

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.facecolor": "white",
    "mathtext.fontset": "dejavusans",
})


def savefig(fig, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(HERE.parent.parent)}")
    return path


def record(**values):
    """Merge scalar results into measurements.json (used by the docs text)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if MEASUREMENTS.exists():
        data = json.loads(MEASUREMENTS.read_text())
    for key, value in values.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        data[key] = value
    MEASUREMENTS.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")
    for key, value in values.items():
        print(f"  {key} = {value}")


def panel_label(ax, text, x=-0.14, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            va="bottom", ha="left")


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
