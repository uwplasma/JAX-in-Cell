"""The grazing-incidence sheath against GYRAZE: the rehearsal before and after the entrance-plane
fix, and the matched run when it exists.

The runs take hours on a GPU, so this script is not in make_all.py: it reads the folders that
examples/3_advanced/grazing_sheath.py wrote (run.json and profiles.npz) and the GYRAZE output
directory each was compared against::

    python docs/scripts/fig_grazing.py --before=RUN --after=RUN --reference=GYRAZE_DIR \
        [--matched=RUN --matched-reference=GYRAZE_DIR]

``RUN`` is the folder holding ``grazing_sheath/`` (or ``grazing_sheath_matched/``). It writes
grazing_rehearsal.png and, with ``--matched``, grazing_matched.png, and records the
seven-quantity tables in measurements.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.constants import e as e_charge, epsilon_0

from common import COLORS, C_THEORY, figure, panel_label, record, savefig

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
SHORT = {"wall potential [T_e/e]": "wall\npotential", "Debye-sheath drop [T_e/e]": "Debye\ndrop",
         "presheath potential, max |diff| [T_e/e]": r"presheath $\phi$",
         "ion density, max |diff| [n_0]": r"$n_i$", "electron density, max |diff| [n_0]": r"$n_e$",
         "mean ion impact energy [T_e]": "impact\nenergy", "ion flux to the wall [n_0 sqrt(T_e/m_e)]": "ion\nflux"}
KEYS = dict(zip(SHORT, ("wall", "drop", "presheath", "n_i", "n_e", "energy", "flux")))


def load(folder):
    folder = Path(folder).expanduser()
    sub = next(p for p in (folder / "grazing_sheath_matched", folder / "grazing_sheath", folder)
               if (p / "run.json").exists())
    run = json.loads((sub / "run.json").read_text())
    return run, dict(np.load(sub / "profiles.npz"))


def reference_profiles(reference, settings, x):
    """GYRAZE's matched composite (phi, n_i, n_e) at x/rho_s from the entrance plane, joined as in
    the example: the potentials of the two layers add, the densities multiply."""
    reference = Path(reference).expanduser()
    presheath, sheath = np.loadtxt(reference / "phi_n_MP.txt"), np.loadtxt(reference / "phi_n_DS.txt")
    rho_s = settings["rho_s_over_debye"]
    length = settings["length_over_debye"] / rho_s
    rho_B, rho_e = 1 / np.sqrt(1 + settings["temperature_ratio"]), settings["gyro_over_debye"] / rho_s
    d = length - x

    def layer(table, column, unit, beyond):
        end = np.flatnonzero(table[:, column])[-1] + 1
        return np.interp(d, table[:end, 0] * unit, table[:end, column], right=beyond)

    return (layer(presheath, 1, rho_B, np.nan) + layer(sheath, 1, rho_e, 0.0),
            layer(presheath, 2, rho_B, np.nan) * layer(sheath, 2, rho_e, 1.0),
            layer(presheath, 3, rho_B, np.nan) * layer(sheath, 3, rho_e, 1.0))


def coordinates(run, data):
    s = run["settings"]
    debye = np.sqrt(epsilon_0 * s["electron_temperature"] / (s["density"] * e_charge))
    x = data["centres"] / debye / s["rho_s_over_debye"]
    phi = np.interp(data["centres"], data["faces"] + s["length_over_debye"] * debye / 2, data["phi"])
    return x, phi


def draw(runs, reference, name, title):
    """Potential and ion density against the reference, and each compared quantity as
    |difference|/tolerance (a bar under 1 passes)."""
    fig, axes = figure(3)
    x_ref = np.linspace(0, 1, 4000)
    for label, (run, data), color in runs:
        x, phi = coordinates(run, data)
        axes[0].plot(x, phi, color=color, label=label)
        axes[1].plot(x, data["n_i"], color=color, label=label)
        x_ref = np.linspace(0, x.max(), 4000)
    ref = reference_profiles(reference, runs[-1][1][0]["settings"], x_ref)
    axes[0].plot(x_ref, ref[0], "--", color=C_THEORY, label="GYRAZE")
    axes[1].plot(x_ref, ref[1], "--", color=C_THEORY, label="GYRAZE")
    axes[0].set(xlabel=r"$x/\rho_s$ from the entrance plane", ylabel=r"$e\phi/T_e$", title="potential")
    axes[1].set(xlabel=r"$x/\rho_s$ from the entrance plane", ylabel=r"$n_i/n_0$", title="ion density",
                ylim=(0, 1.5))
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].legend(frameon=False, loc="lower left")
    width = 0.8 / len(runs)
    for i, (label, (run, _), color) in enumerate(runs):
        q = run["results"]["comparison"]["quantities"]
        ratio = [q[k]["difference"] / q[k]["tolerance"] for k in SHORT]
        axes[2].bar(np.arange(len(SHORT)) + (i - (len(runs) - 1) / 2) * width, ratio, width, color=color,
                    label=label)
    axes[2].axhline(1.0, color=C_THEORY, ls="--", lw=1.5)
    axes[2].set_xticks(np.arange(len(SHORT)), list(SHORT.values()), fontsize=13)
    axes[2].set(ylabel=r"$|$difference$|$ / tolerance", title="seven quantities; under 1 passes")
    axes[2].legend(frameon=False, loc="upper left")
    for ax, letter in zip(axes, "abc"):
        panel_label(ax, letter)
    fig.tight_layout(w_pad=2.5)
    savefig(fig, name)


def table(prefix, run):
    q = run["results"]["comparison"]["quantities"]
    values = {}
    for k, short in KEYS.items():
        values[f"{prefix}_{short}_diff"] = float(f"{q[k]['difference']:.3g}")
        values[f"{prefix}_{short}_tol"] = float(f"{q[k]['tolerance']:.3g}")
        values[f"{prefix}_{short}_pass"] = q[k]["passed"]
    return values


before, after = load(ARGS["before"]), load(ARGS["after"])
draw([("cold start", before, COLORS["grey"]), ("reservoir start", after, COLORS["blue"])], ARGS["reference"],
     "grazing_rehearsal", r"rehearsal: $m_i/m_e=400$, $\alpha=5^\circ$, three ion transits")
values = {**table("graze_before", before[0]), **table("graze_after", after[0])}
if "matched" in ARGS:
    matched = load(ARGS["matched"])
    draw([("matched run", matched, COLORS["vermillion"])], ARGS["matched-reference"], "grazing_matched",
         r"matched: $m_i/m_e=900$, $\alpha=4^\circ$")
    values.update(table("graze_matched", matched[0]))
record(**values)
