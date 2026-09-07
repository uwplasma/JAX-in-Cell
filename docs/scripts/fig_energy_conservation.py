"""Energy conservation of the explicit Boris scheme and the implicit
Crank-Nicolson scheme on the two-stream problem of examples/input.toml."""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready

from common import (C_EXPLICIT, C_IMPLICIT, EXAMPLES_DIR, WIDE, panel_label, quiet_parameters,
                    record, savefig, silence_progress_bars)
from jaxincell import Simulation, diagnostics, load_parameters

silence_progress_bars()
parameters = quiet_parameters(load_parameters(EXAMPLES_DIR / "input.toml"))
runs = {}
for name, algorithm in (("explicit Boris", 0), ("implicit Crank-Nicolson", 1)):
    parameters["solver_parameters"]["time_evolution_algorithm"] = algorithm
    sim = Simulation(parameters)
    output = block_until_ready(sim.run())
    diagnostics(output)
    runs[name] = output

fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.35})
for (name, output), color in zip(runs.items(), (C_EXPLICIT, C_IMPLICIT)):
    wpe = float(output["plasma_frequency"])
    t = np.asarray(output["time_array"]) * wpe
    total = np.asarray(output["total_energy"])
    error = np.abs((total - total[0]) / total[0])
    axes[0].semilogy(t, np.asarray(output["electric_field_energy"]), color=color, label=name)
    axes[1].semilogy(t[1:], np.maximum(error[1:], 1e-17), color=color, label=name)
    key = "explicit" if "explicit" in name else "implicit"
    record(**{f"energy_error_max_{key}": float(error.max()),
              f"energy_error_final_{key}": float(error[-1])})
record(energy_c_dt_over_dx=parameters["domain_parameters"]["timestep_over_spatialstep_times_c"],
       energy_omega_pe_dt=float(runs["explicit Boris"]["dt"] * runs["explicit Boris"]["plasma_frequency"]))
axes[0].set_xlabel(r"$t\,\omega_{pe}$")
axes[0].set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
axes[0].legend(loc="lower right")
panel_label(axes[0], "(a)")
axes[1].set_xlabel(r"$t\,\omega_{pe}$")
axes[1].set_ylabel(r"$|\,\mathcal{E}(t) - \mathcal{E}(0)\,| / \mathcal{E}(0)$")
axes[1].legend(loc="center right")
panel_label(axes[1], "(b)")
savefig(fig, "energy_conservation")
