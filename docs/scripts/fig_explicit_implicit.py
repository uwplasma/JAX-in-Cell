"""The explicit (Boris leapfrog) and implicit (Crank-Nicolson) schemes on the same
two problems: linear Landau damping with a quiet start (the set-up of
``landau.py``) and the two-stream instability of examples/input.toml with four
times its particle number. Top row, the electrostatic energy with the linear
theory; bottom row, the relative error in the total energy."""
import time

import numpy as np
from jax import block_until_ready

from common import (C_EXPLICIT, C_FIT, C_IMPLICIT, C_THEORY, EXAMPLES_DIR, analyse_two_stream,
                    figure, fit_growth_rate, panel_label, quiet_parameters, record, savefig,
                    silence_progress_bars)
from landau import gamma_theory as landau_gamma, maxima_above_floor, mode_one
from landau import quiet_parameters as landau_parameters
from jaxincell import Simulation, diagnostics, load_parameters

silence_progress_bars()
SCHEMES = (("explicit", 0, "explicit (Boris)", C_EXPLICIT, "-"),
           ("implicit", 1, "implicit (Crank-Nicolson)", C_IMPLICIT, "--"))
N_LANDAU, A_K, STEPS_LANDAU = 300000, 0.01, 600


def run(parameters, algorithm):
    parameters["solver_parameters"]["time_evolution_algorithm"] = algorithm
    start = time.perf_counter()
    output = block_until_ready(Simulation(parameters).run())
    seconds = time.perf_counter() - start
    diagnostics(output)
    wpe = float(output["plasma_frequency"])
    total = np.asarray(output["total_energy"])
    return {"output": output, "seconds": seconds, "t": np.asarray(output["time_array"]) * wpe,
            "energy": np.asarray(output["electric_field_energy"]),
            "error": np.abs(total - total[0]) / total[0]}


landau, two_stream = {}, {}
for key, algorithm, *_ in SCHEMES:
    parameters = landau_parameters(N_LANDAU, A_K, STEPS_LANDAU)
    parameters["solver_parameters"].update({"max_number_of_Picard_iterations_implicit_CN": 20,
                                            "number_of_particle_substeps_implicit_CN": 1,
                                            "tolerance_Picard_iterations_implicit_CN": 1e-10})
    landau[key] = run(parameters, algorithm)
    r = landau[key]
    peaks, _ = maxima_above_floor(np.abs(mode_one(r["output"])))
    r["gamma"] = fit_growth_rate(r["t"][peaks], r["energy"][peaks], r["t"][peaks[0]], r["t"][peaks[-1]])[0]
    r["peaks"] = peaks
    del r["output"]

    parameters = quiet_parameters(load_parameters(EXAMPLES_DIR / "input.toml"))
    n_example = parameters["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"]
    for species_type in ("electrons", "ions"):
        parameters["species_parameters"][species_type][f"{species_type}0"]["number_pseudoparticles"] = 4 * n_example
    two_stream[key] = run(parameters, algorithm)
    r = two_stream[key]
    _, _, r["theory"], (t0, t1, _, _), _ = analyse_two_stream(r["output"])
    r["gamma"], r["intercept"], r["slope"] = fit_growth_rate(r["t"], r["energy"], t0, t1)
    r["window"] = (t0, t1)
    r["dt_wpe"] = float(r["output"]["dt"] * r["output"]["plasma_frequency"])
    del r["output"]

ts_theory = two_stream["explicit"]["theory"]
record(**{f"energy_error_max_{key}": float(two_stream[key]["error"].max()) for key in two_stream},
       **{f"energy_error_final_{key}": float(two_stream[key]["error"][-1]) for key in two_stream},
       **{f"landau_energy_error_max_{key}": float(landau[key]["error"].max()) for key in landau},
       **{f"landau_gamma_measured_{key}": float(landau[key]["gamma"]) for key in landau},
       **{f"two_stream_gamma_energy_{key}": float(two_stream[key]["gamma"]) for key in two_stream},
       energy_particles=4 * n_example, energy_c_dt_over_dx=parameters["domain_parameters"]["timestep_over_spatialstep_times_c"],
       energy_omega_pe_dt=two_stream["explicit"]["dt_wpe"], explicit_implicit_two_stream_gamma_theory=ts_theory,
       explicit_implicit_landau_particles=N_LANDAU,
       **{f"explicit_implicit_seconds_{case}_{key}": runs[key]["seconds"]
          for case, runs in (("landau", landau), ("two_stream", two_stream)) for key in runs})

fig, axes = figure(2, 2, sharex="col", gridspec_kw={"wspace": 0.3, "hspace": 0.12})
# (a) Landau damping
ax = axes[0, 0]
for key, _, label, colour, style in SCHEMES:
    ax.semilogy(landau[key]["t"], landau[key]["energy"], color=colour, ls=style, label=label)
r = landau["explicit"]
tt = np.linspace(r["t"][r["peaks"][0]], r["t"][r["peaks"][-1]], 50)
ax.semilogy(tt, r["energy"][r["peaks"][0]] * np.exp(2 * landau_gamma * (tt - tt[0])), ls=":", color=C_THEORY,
            label=rf"$e^{{2\gamma t}}$, kinetic $\gamma = {landau_gamma:.3f}\,\omega_{{pe}}$")
ax.set_xlim(0, 40)
ax.set_ylim(top=300 * r["energy"].max())
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.legend(loc="upper right")
panel_label(ax, "(a)")
# (b) Two-stream instability
ax = axes[0, 1]
for key, _, label, colour, style in SCHEMES:
    ax.semilogy(two_stream[key]["t"], two_stream[key]["energy"], color=colour, ls=style, label=label)
r = two_stream["explicit"]
t0, t1 = r["window"]
tt = np.linspace(t0, t1, 50)
ax.semilogy(tt, np.exp(r["intercept"] + r["slope"] * tt), color=C_FIT, lw=7, alpha=0.5,
            label=rf"fit: $\gamma = {r['gamma']:.3f}\,\omega_{{pe}}$")
ax.semilogy(tt, np.exp(r["intercept"] + r["slope"] * t0) * np.exp(2 * ts_theory * (tt - t0)), ls=":",
            color=C_THEORY, label=rf"$e^{{2\gamma t}}$, kinetic $\gamma = {ts_theory:.3f}\,\omega_{{pe}}$")
ax.set_ylim(top=1e3 * r["energy"].max())
ax.legend(loc="lower right")
panel_label(ax, "(b)")
# (c), (d) Relative error in the total energy
for ax, runs, letter in ((axes[1, 0], landau, "(c)"), (axes[1, 1], two_stream, "(d)")):
    for key, _, label, colour, style in SCHEMES:
        ax.semilogy(runs[key]["t"][1:], np.maximum(runs[key]["error"][1:], 1e-17), color=colour, ls=style,
                    label=label)
    ax.set_xlabel(r"$t\,\omega_{pe}$")
    ax.set_ylim(1e-17, 1e4)
    ax.legend(loc="upper right")
    panel_label(ax, letter)
axes[1, 0].set_xlim(0, 40)
axes[1, 0].set_ylabel(r"$|\,\mathcal{E}(t) - \mathcal{E}(0)\,| / \mathcal{E}(0)$")
savefig(fig, "explicit_implicit")
