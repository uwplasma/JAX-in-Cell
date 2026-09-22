"""JAX-in-Cell: a differentiable 1D3V particle-in-cell code."""
import sys

from ._config import (Collisions, Domain, Impacts, Solver, Source, Species, boltzmann_constant,
                      elementary_charge, epsilon_0, mass_electron, mass_proton, mu_0, speed_of_light)
from ._diagnostics import (bohm_edge, charge_balance, diagnostics, dominant_frequency, energies,
                           gauss_residual, moment_profiles, potential, temperatures)
from ._archive import load_state, provenance, save_state
from ._simulation import Output, Simulation, load_toml, quiet_start

try:
    from .version import __version__
except ImportError:  # a source tree that was never installed: version.py is written at build time
    __version__ = "unknown"

__all__ = ["Simulation", "Output", "Domain", "Species", "Solver", "Source", "Collisions", "Impacts",
           "load_toml", "save_state", "load_state", "provenance",
           "quiet_start",
           "diagnostics", "energies", "gauss_residual", "charge_balance", "moment_profiles",
           "potential", "temperatures",
           "bohm_edge",
           "dominant_frequency", "plot", "figure", "style",
           "epsilon_0", "mu_0", "speed_of_light", "elementary_charge", "mass_electron",
           "mass_proton", "boltzmann_constant"]


def __getattr__(name):
    """``plot``, ``figure`` and ``style`` import matplotlib on first use, which would otherwise be a
    third of the import time."""
    if name in ("plot", "figure", "style"):
        from . import _plot
        return getattr(_plot, name)
    raise AttributeError(f"module 'jaxincell' has no attribute {name!r}")


USAGE = ("usage: jaxincell input.toml [--steps N] [--seed S] [--save DIR] [--movie FILE] "
         "[--plot | --no-plot]")


def main(argv=None):
    """The ``jaxincell input.toml`` command: run the file, print the energy balance, show the animation.

    Everything ``[run]`` may hold is passed to :meth:`~jaxincell.Simulation.run`, so a file can
    ask for the moments, the particle history or a quieter meter. A command-line run shows
    progress unless the file says otherwise: it is the one place where somebody is watching.

    Five flags, each overriding the file rather than replacing it: ``--steps`` and ``--seed``
    for the two settings a run is usually repeated with, ``--save DIR`` to write the run's
    arrays, its settings and the versions that produced them, ``--movie FILE`` to write the
    animation instead of only showing it, and ``--plot``/``--no-plot`` to override ``[run]
    plot`` -- the last is what a headless machine needs.

    The energy drift is printed when the run kept a particle history: ``store_particles =
    false`` keeps the fields and drops the velocities, so the total energy is not there to
    report and the Gauss residual is.

    The input files in ``inputs/`` are a set to start from.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(USAGE)
        return 1
    import argparse

    parser = argparse.ArgumentParser(prog="jaxincell", usage=USAGE[7:], allow_abbrev=False)
    parser.add_argument("input")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save", metavar="DIR")
    parser.add_argument("--movie", metavar="FILE")
    parser.add_argument("--plot", dest="plot", action="store_true", default=None)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args(argv)

    sim, run = load_toml(args.input)
    settings = {key: value for key, value in run.items() if key != "plot"}
    settings.setdefault("steps", 500)
    settings.setdefault("verbose", True)
    for name in ("steps", "seed"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    out = sim.run(**settings)
    d = diagnostics(out)
    summary = {"steps": int(out.t.shape[0]), "final_time": float(out.t[-1]),
               "gauss_residual": float(d["gauss_residual"][-1])}
    line = (f"steps {summary['steps']}  final time {summary['final_time']:.3e} s  "
            f"gauss residual {summary['gauss_residual']:.2e}")
    if "total" in d:          # the energy balance needs the velocities, so a run that kept no
        total = d["total"]    # particle history has the field energies and not the total
        summary["energy_drift"] = float(abs(total[-1] / total[0] - 1))
        line += f"  energy drift {summary['energy_drift']:.2e}"
    print(line)
    if args.save is not None:
        _write(args.save, args.input, settings, summary, out)
    if args.movie is not None or (args.plot if args.plot is not None else run.get("plot", True)):
        from ._plot import plot
        plot(out, save=args.movie, show=args.plot if args.plot is not None else run.get("plot", True))
    return 0


def _write(folder, path, settings, summary, out):
    """``--save DIR``: the run's arrays, what it was asked for, and what produced it.

    ``fields.npz`` holds the coordinates and the stored fields, and the particle history when
    the run kept one; ``run.json`` holds the settings the run used and the versions, precision,
    device and commit that produced it; and the input file itself is copied in beside them, so
    the folder says what was run as well as what came out. A figure is not written here: an
    animation is ``--movie``.
    """
    import json
    from pathlib import Path

    import numpy as np

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(out, name)) for name in
              ("t", "grid", "faces", "E", "B", "rho", "steps") if getattr(out, name) is not None}
    for name in ("x", "v", "weight", "moments"):
        if getattr(out, name) is not None:
            arrays[name] = np.asarray(getattr(out, name))
    np.savez(folder / "fields.npz", **arrays)
    (folder / Path(path).name).write_text(Path(path).read_text())    # the input, beside its output
    (folder / "run.json").write_text(json.dumps(provenance(
        input=str(path), settings=dict(settings), results=summary), indent=1))
    print(f"wrote {folder}/fields.npz, run.json and {Path(path).name}")
