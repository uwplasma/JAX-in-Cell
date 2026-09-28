"""What the time integrators conserve: examples/3_advanced/invariants.py, run here and read back,
so that the page's figure and numbers are the example's own run (about a minute)."""
from common import record, run_example, savefig

run, fig = run_example("3_advanced/invariants")
savefig(fig, "invariants")
s, r = run["settings"], run["results"]
values = {}
for w, a, b in zip(s["omega_dt"], r["collisionless"], r["collisional"]):
    tag = f"{w:g}".replace(".", "")
    values[f"centring_{tag}_collisionless"] = f"{a:.2e}"
    values[f"centring_{tag}_collisional"] = f"{b:.2e}"
for n, e in zip(s["picard"], r["picard_energy_error"]):
    values[f"implicit_es_picard_{n}"] = f"{e:.1e}"
values.update(implicit_es_energy=f"{r['implicit_energy_error']:.1e}", implicit_es_gauss=f"{r['implicit_gauss']:.1e}",
              explicit_es_energy=f"{r['explicit_energy_error']:.1e}", explicit_es_gauss=f"{r['explicit_gauss']:.1e}")
record(**values)
