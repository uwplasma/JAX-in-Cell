"""Physical and numerical inputs and the three loadings: examples/1_basic/parameters_and_sampling.py,
run here and read back for its page."""
from common import record, run_example, savefig

run, fig = run_example("1_basic/parameters_and_sampling")
savefig(fig, "parameters_and_sampling")
r = run["results"]
values = dict(sampling_predicted_density=f"{r['predicted_density_spread']:.1e}",
              sampling_predicted_temperature=f"{r['predicted_temperature_error']:.1e}",
              sampling_debye_m=f"{r['debye_length']:.3e}", sampling_omega_pe=f"{r['omega_pe']:.3e}",
              sampling_v_th=f"{r['v_th']:.3e}")
for name, s in r["samplings"].items():
    values[f"sampling_{name}_density"] = f"{s['density_spread']:.1e}"
    values[f"sampling_{name}_temperature"] = f"{s['temperature_error']:+.1e}"
    values[f"sampling_{name}_energy"] = f"{s['electric_energy_end']:.1e}"
record(**values)
