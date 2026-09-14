"""Shared style and helpers for the documentation figures.

Every figure script here imports this module. Figures go to
``docs/_static/figures``; the numbers the prose quotes are recorded in
``docs/_static/figures/measurements.json`` and pulled into the text as MyST
substitutions, so the text and the committed figures always come from one run.
Each script's entry under ``_provenance`` in that file says which commit and which
library versions produced its numbers.
"""
import inspect
import json
import platform
import shutil
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE.parent / "_static" / "figures"
MEASUREMENTS = FIGURE_DIR / "measurements.json"

# Okabe-Ito palette, colourblind safe. The assignment is fixed across figures:
# electrons blue, ions orange, explicit blue, implicit vermillion, theory black.
COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
          "purple": "#CC79A7", "sky": "#56B4E9", "black": "#000000", "grey": "#7F7F7F"}
C_ELECTRONS = C_EXPLICIT = COLORS["blue"]
C_IONS, C_IMPLICIT, C_THEORY, C_FIT = COLORS["orange"], COLORS["vermillion"], COLORS["black"], COLORS["green"]
SINGLE, WIDE = (6.0, 3.6), (7.4, 3.4)

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


def _git_revision():
    """The checked-out commit, marked ``-dirty`` when tracked files differ from it."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                             check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if changed else sha


def provenance():
    """What produced a set of numbers: the commit, the library versions, the
    precision and the device."""
    import jax
    import jaxincell
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "not installed"
    return {"git": _git_revision(), "jaxincell": jaxincell.__version__, "jax": jax.__version__,
            "numpy": np.__version__, "scipy": scipy_version,
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
            "backend": jax.default_backend(), "platform": f"{platform.system()} {platform.machine()}"}


def record(**values):
    """Merge scalar results into measurements.json, which the docs text quotes, and
    note under ``_provenance`` which script, commit and versions produced them.

    The documentation quotes double-precision results, so a run in single precision
    (``JAX_ENABLE_X64=0``) is refused rather than recorded over them.
    """
    info = provenance()
    if not info["jax_enable_x64"]:
        raise RuntimeError("refusing to record single-precision measurements: the documentation quotes "
                           "double-precision results. Run without JAX_ENABLE_X64=0.")
    script = Path(inspect.currentframe().f_back.f_code.co_filename).name
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(MEASUREMENTS.read_text()) if MEASUREMENTS.exists() else {}
    for key, value in values.items():
        data[key] = value.item() if isinstance(value, (np.floating, np.integer)) else value
        print(f"  {key} = {data[key]}")
    data.setdefault("_provenance", {})[script] = info
    MEASUREMENTS.write_text(json.dumps(dict(sorted(data.items())), indent=2, sort_keys=True) + "\n")


def panel_label(ax, text, x=-0.14, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold",
            va="bottom", ha="left")


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


def phase_space_hist(ax, x, v, box_length, v_max, weights=None, bins=(70, 90), cmap="viridis"):
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
