"""Regenerate every documentation figure.

Run from the repository root with the package installed::

    python docs/scripts/make_all.py

The scripts need scipy and matplotlib in addition to the package dependencies;
Pillow and optipng, if present, shrink the figures. Simulations run
on whatever JAX backend is available; the measurements they record (growth
rates, energy errors, timings) are written to docs/_static/figures/measurements.json,
together with the commit and library versions that produced them, and quoted by the
documentation through MyST substitutions. The documentation quotes double-precision
results, so a single-precision run (JAX_ENABLE_X64=0) is refused.
"""
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "fig_schematics.py",
    "fig_two_stream.py",
    "fig_landau_damping.py",
    "fig_bump_on_tail.py",
    "fig_weibel.py",
    "fig_conservation.py",
    "fig_boundaries.py",
    "fig_wall_reflection.py",
    "fig_sheath.py",
    "fig_sheath_source.py",
    "fig_collisions.py",
    "fig_autodiff.py",
    "fig_scaling.py",
]
# sheath_convergence.py is deliberately not in that list: it is twelve full sheath runs and takes
# about half an hour, where everything above takes minutes. Name it on the command line to run it.


if __name__ == "__main__":
    if os.environ.get("JAX_ENABLE_X64", "1").strip().lower() in ("0", "false", "f", "no", "n", "off"):
        sys.exit("make_all.py records the numbers the documentation quotes, which are double-precision "
                 "results; unset JAX_ENABLE_X64 or set it to 1.")
    selected = sys.argv[1:] or SCRIPTS
    # the repository root first, so that the figures come from the checked-out code
    # rather than from whatever copy of jaxincell happens to be installed
    sys.path.insert(0, str(HERE.parent.parent))
    sys.path.insert(0, str(HERE))
    for script in selected:
        start = time.perf_counter()
        print(f"--- {script}")
        runpy.run_path(str(HERE / script), run_name="__main__")
        print(f"    done in {time.perf_counter() - start:.1f} s")
