"""External fields on an (x, y, z) grid: examples/2_intermediate/external_fields_3d.py, run here and
read back, so that the page's figure and numbers are the example's own run (about 40 s)."""
from common import record, run_example, savefig

run, fig = run_example("2_intermediate/external_fields_3d")
savefig(fig, "external_fields_3d")
r = run["results"]
values = {}
for ratio, d in r["grad_B"].items():
    values[f"gradb_{ratio}_deviation"] = f"{100 * d['deviation']:+.2f} %"
for angle, m in r["mirror"].items():
    values[f"mirror_{angle}_turning"] = round(m["turning"], 4)
    values[f"mirror_{angle}_mu_spread"] = f"{m['mu_spread']:.0e}"
record(**values)
