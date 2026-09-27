"""One two-stream problem run seven ways: examples/2_intermediate/compare_models.py, run here
and read back, so that the page's figure and numbers are the example's own run."""
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from common import record, savefig

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "2_intermediate" / "compare_models.py"
with tempfile.TemporaryDirectory() as folder:
    here, argv = Path.cwd(), sys.argv
    try:
        os.chdir(folder)
        sys.argv = [str(EXAMPLE)]
        runpy.run_path(str(EXAMPLE), run_name="__main__")
        run = json.loads((Path(folder) / "compare_models" / "run.json").read_text())
    finally:
        os.chdir(here)
        sys.argv = argv
savefig(plt.gcf(), "compare_models")

runs = run["results"]["runs"]
values = dict(compare_particles=run["settings"]["particles"], compare_steps=run["settings"]["steps"],
              compare_kinetic=round(run["results"]["kinetic"], 4))
for name, r in runs.items():
    values[f"compare_{name}_rate"] = round(r["rate"], 4)
    values[f"compare_{name}_reference"] = round(r["reference"], 4)
    values[f"compare_{name}_deviation"] = f"{r['deviation_percent']:+.1f}"
    values[f"compare_{name}_energy"] = f"{r['energy_error']:.0e}"
    values[f"compare_{name}_seconds"] = round(r["seconds"], 1)
record(**values)
