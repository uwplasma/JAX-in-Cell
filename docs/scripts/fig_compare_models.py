"""One two-stream problem run seven ways: examples/2_intermediate/compare_models.py, run here
and read back, so that the page's figure and numbers are the example's own run."""
from common import record, run_example, savefig

run, fig = run_example("2_intermediate/compare_models")
savefig(fig, "compare_models")

runs = run["results"]["runs"]
values = dict(compare_particles=run["settings"]["particles"], compare_steps=run["settings"]["steps"],
              compare_kinetic=round(run["results"]["kinetic"], 4))
for name, r in runs.items():
    values[f"compare_{name}_rate"] = round(r["rate"], 4)
    values[f"compare_{name}_reference"] = round(r["reference"], 4)
    values[f"compare_{name}_deviation"] = f"{r['deviation_percent']:+.1f}"
    values[f"compare_{name}_energy"] = f"{r['energy_error']:.0e}"
    values[f"compare_{name}_seconds"] = round(r["seconds"], 1)
pair = run["results"]["filter_pair"]
values.update(compare_filter_ratio=round(pair["ratio"], 4), compare_filter_sqrt_g=round(pair["sqrt_G"], 4),
              compare_filter_deviation=f"{100 * (pair['ratio'] / pair['sqrt_G'] - 1):+.2f}",
              compare_filter_unfiltered=round(pair["frequency_unfiltered"], 4),
              compare_filter_filtered=round(pair["frequency_filtered"], 4))
record(**values)
