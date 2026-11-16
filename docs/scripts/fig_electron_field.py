"""The electron-field instability and its controls: examples/3_advanced/electron_field.py, run here
and read back, so that the page's figure and numbers are the example's own run (hours on a GPU)."""
from common import record, run_example, savefig

run, fig = run_example("3_advanced/electron_field")
savefig(fig, "electron_field")

s, r = run["settings"], run["results"]
values = dict(efi_debye_lengths=s["debye_lengths"], efi_per_cell=s["per_cell"], efi_realisations=s["realisations"],
              efi_steps=s["steps"], efi_dt=s["dt_wpe"], efi_t_end=s["t_end"],
              efi_fit=f"{s['fit'][0]:.0f}-{s['fit'][1]:.0f}",
              efi_kappa=round(r["kappa_lambda_D"], 3), efi_k_star=round(r["k_star"], 4),
              efi_theory=f"{2 * r['gamma_theory']:.2e}", efi_paper=f"{r['paper_fit']:.1e}",
              efi_frame=f"{r['frame_difference_window']:.1e}", efi_frame_long=f"{r['frame_difference_long']:.1e}",
              efi_ledger=f"{r['ledger_difference']:.0e}")
for name, rate in r["rates"].items():
    key = name.replace(", no field", "_still")
    values[f"efi_{key}_rate"] = f"{rate['energy_rate']:.2e}"
    values[f"efi_{key}_gain"] = f"{rate['gain']:.3g}"
for name, ratio in r["peak_over_eq12"].items():
    values[f"efi_{name}_peak"] = round(ratio, 2)
record(**values)
