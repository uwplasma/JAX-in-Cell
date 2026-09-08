"""Regenerate every documentation figure.

Run from the repository root with the package installed::

    python docs/scripts/make_all.py

The scripts need scipy and matplotlib in addition to the package dependencies;
Pillow and optipng, if present, shrink the figures. Simulations run
on whatever JAX backend is available; the measurements they record (growth
rates, energy errors, timings) are written to docs/_static/figures/measurements.json
and quoted by the documentation through MyST substitutions.
"""
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
    "fig_collisions.py",
    "fig_autodiff.py",
    "fig_scaling.py",
]

if __name__ == "__main__":
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
