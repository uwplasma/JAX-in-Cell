"""Wall-clock time of the explicit scheme against the number of pseudo-particles,
grid points and time steps, measured after compilation on the machine that
generated the documentation figures. Run on an otherwise idle machine (the load
average at the start is recorded with the timings);
``--replot`` redraws the figure from measurements.json without running."""
import json
import os
import sys
import time
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from common import (C_ELECTRONS, plain_log_ticks, C_THEORY, EXAMPLES_DIR, MEASUREMENTS, cpu_name, figure, panel_label,
                    quiet_parameters, record, savefig, silence_progress_bars)

SCANS = ("pseudo-particles per species", "grid points", "time steps")


def key_for(name):
    return "scaling_" + name.replace(" ", "_").replace("-", "_")


def draw(results):
    """results maps a scan name to {"values": [...], "seconds": [...]}."""
    fig, axes = figure(3, 1, gridspec_kw={"wspace": 0.3})
    for ax, name, label, symbol in zip(axes, SCANS, "abc", ("N_p", "N_x", "N_t")):
        values = np.array(results[name]["values"], dtype=float)
        times = np.array(results[name]["seconds"], dtype=float)
        ax.loglog(values, times, "o-", color=C_ELECTRONS, label="run (compiled)")
        slope, intercept = np.polyfit(np.log(values), np.log(times), 1)
        ax.loglog(values, np.exp(intercept) * values**slope, ls="--", color=C_THEORY,
                  label=rf"fit $\propto {symbol}^{{{slope:.2f}}}$")
        ax.set_xlabel(name)
        ax.set_xticks(values[::2])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        plain_log_ticks(ax.yaxis, 0.5 * times.min(), 2 * times.max())
        ax.legend(loc="upper left")
        panel_label(ax, f"({label})")
    axes[0].set_ylabel("wall-clock time (s)")
    savefig(fig, "scaling")


if "--replot" in sys.argv:
    data = json.loads(MEASUREMENTS.read_text())
    draw({name: data[key_for(name)] for name in SCANS})
    sys.exit(0)

import jax
from jax import block_until_ready
from jaxincell import Simulation, load_parameters

silence_progress_bars()
base = quiet_parameters(load_parameters(EXAMPLES_DIR / "input.toml"))
base["domain_parameters"]["total_steps"] = 500


def timed_run(parameters):
    sim = Simulation(parameters)
    start = time.perf_counter()
    block_until_ready(sim.run())
    compile_and_run = time.perf_counter() - start
    start = time.perf_counter()
    block_until_ready(sim.run())
    return time.perf_counter() - start, compile_and_run


def with_particles(n):
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    p["species_parameters"] = {"electrons": {"electrons0": dict(base["species_parameters"]["electrons"]["electrons0"])},
                               "ions": {"ions0": dict(base["species_parameters"]["ions"]["ions0"])}}
    p["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"] = int(n)
    p["species_parameters"]["ions"]["ions0"]["number_pseudoparticles"] = int(n)
    return p


def with_domain(key, value):
    p = with_particles(base["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"])
    p["domain_parameters"] = dict(base["domain_parameters"])
    p["domain_parameters"][key] = int(value)
    return p


scans = {
    "pseudo-particles per species": ([with_particles(n) for n in (1000, 2000, 4000, 8000, 16000, 32000)],
                                     [1000, 2000, 4000, 8000, 16000, 32000]),
    "grid points": ([with_domain("number_grid_points", g) for g in (32, 64, 128, 256, 512)],
                    [32, 64, 128, 256, 512]),
    "time steps": ([with_domain("total_steps", s) for s in (250, 500, 1000, 2000, 4000)],
                   [250, 500, 1000, 2000, 4000]),
}
load_average = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
results = {}
for name, (parameter_list, values) in scans.items():
    times, compiles = [], []
    for p in parameter_list:
        run_time, compile_time = timed_run(p)
        times.append(run_time)
        compiles.append(compile_time - run_time)
        print(f"{name}: {values[len(times) - 1]} -> {run_time:.2f} s (compile {compile_time - run_time:.2f} s)")
    results[name] = {"values": values, "seconds": times, "compile_seconds": compiles}

record(scaling_device=str(jax.devices()[0].platform), scaling_cpu=cpu_name(),
       scaling_load_average=load_average, scaling_cpu_count=os.cpu_count(),
       scaling_jax_version=jax.__version__,
       scaling_seconds_per_step_per_particle=float(results["pseudo-particles per species"]["seconds"][-1]
                                                   / (500 * 2 * 32000)),
       **{key_for(name): value for name, value in results.items()})
draw(results)
