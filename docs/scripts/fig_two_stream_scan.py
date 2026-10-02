"""Two-stream drift scan on CPU and GPU: the wall-clock time of the scan against
the number of pseudo-electrons, and the growth rate against the drift speed for
several particle numbers, compared with the kinetic dispersion relation.

For each particle number, examples/input.toml (the same number of ions as
electrons) is compiled once and run for ten drift speeds passed as runtime
inputs. The time plotted is the sum of the ten runs after compilation; the
compilation is timed separately and recorded. The growth rate is that of the
first Fourier mode of E_x (``analyse_two_stream`` in common.py), and a run is
used only if the mode grows through more than two e-foldings over a window
longer than five inverse plasma frequencies.

    python fig_two_stream_scan.py             # measure on this machine, then draw
    python fig_two_stream_scan.py --measure   # measure and record only (for example on a GPU)
    python fig_two_stream_scan.py --replot    # draw from measurements.json

Measurements are recorded under ``drift_scan_<platform>`` with the device name,
the JAX version and the load average at the start, so a CPU-only machine draws
the GPU curve from the numbers recorded on the GPU.
"""
import json
import os
import sys
import time

import numpy as np

from common import (C_THEORY, COLORS, EXAMPLES_DIR, MEASUREMENTS, analyse_two_stream, cpu_name,
                    figure, panel_label, plain_log_ticks, quiet_parameters, record, savefig, silence_progress_bars,
                    species_for_linear_theory)

PARTICLES = [500, 1000, 2000, 4000, 8000, 16000, 32000]
DRIFTS_OVER_C = [round(v, 3) for v in np.linspace(0.11, 0.20, 10)]
SHOWN = [2000, 8000, 32000]          # particle numbers drawn in panel (b)
PLATFORM_STYLE = {"cpu": ("o-", COLORS["blue"]), "gpu": ("s-", COLORS["vermillion"])}
SHOWN_COLOURS = (COLORS["sky"], COLORS["green"], COLORS["purple"])


def measure():
    import jax
    from jax import block_until_ready
    from jaxincell import Simulation, load_parameters, speed_of_light

    silence_progress_bars()
    device = jax.devices()[0]
    platform = device.platform
    name = cpu_name() if platform == "cpu" else device.device_kind
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    base = quiet_parameters(load_parameters(EXAMPLES_DIR / "input.toml"))

    def inputs(v_over_c):
        return {"electrons": {"electrons0": {"drift_speed_x": v_over_c * speed_of_light}}}

    seconds, compile_seconds, gammas, theory = [], [], [], None
    for n in PARTICLES:
        for species_type in ("electrons", "ions"):
            base["species_parameters"][species_type][f"{species_type}0"]["number_pseudoparticles"] = n
        sim = Simulation(base)
        start = time.perf_counter()
        block_until_ready(sim.run(inputs(DRIFTS_OVER_C[0])))
        first = time.perf_counter() - start
        run_times, rates = [], []
        for v_over_c in DRIFTS_OVER_C:
            start = time.perf_counter()
            output = block_until_ready(sim.run(inputs(v_over_c)))
            run_times.append(time.perf_counter() - start)
            _, gamma_fit, gamma_th, window, e_folds = analyse_two_stream(output)
            usable = e_folds > 2.0 and window[1] - window[0] > 5.0
            rates.append(float(gamma_fit) if usable else float("nan"))
            if theory is None and v_over_c == DRIFTS_OVER_C[0]:
                theory = species_for_linear_theory(output), float(output["plasma_frequency"]), \
                    float(output["length"])
            del output
        seconds.append(float(sum(run_times)))
        compile_seconds.append(float(first - np.mean(run_times)))
        gammas.append(rates)
        print(f"{platform} N = {n}: scan {seconds[-1]:.2f} s, compile {compile_seconds[-1]:.2f} s, "
              f"rates {np.round(rates, 3).tolist()}", flush=True)

    populations, wpe, length = theory
    record(**{f"drift_scan_{platform}": {
        "device": name, "jax_version": jax.__version__, "load_average": load,
        "cpu_count": os.cpu_count(), "particles": PARTICLES, "drifts_over_c": DRIFTS_OVER_C,
        "seconds": seconds, "compile_seconds": compile_seconds, "gamma_measured": gammas}},
        **{f"drift_scan_{platform}_device": name, f"drift_scan_{platform}_jax_version": jax.__version__,
           f"drift_scan_{platform}_seconds_largest": seconds[-1],
           f"drift_scan_{platform}_load_average": load, f"drift_scan_{platform}_cpu_count": os.cpu_count()})
    return populations, wpe, length


def theory_curve(populations, wpe, length, drifts):
    """Most unstable root of the first box mode for each drift speed (in units of c)."""
    from dispersion import electrostatic_epsilon, most_unstable_root
    from jaxincell import speed_of_light

    rates = []
    for v_over_c in drifts:
        pops = [dict(p) for p in populations]
        for p in pops:
            if p["name"].startswith("electrons"):
                p["u"] = np.sign(p["u"]) * v_over_c * speed_of_light
        root = most_unstable_root(lambda w: electrostatic_epsilon(w, 2 * np.pi / length, pops),
                                  (-0.5, 0.5), (0.005, 1.0), n_real=11, n_imag=12, scale=wpe)
        rates.append(root.imag / wpe if root is not None else 0.0)
    return np.maximum(rates, 0.0)


def draw(data):
    fig, (ax1, ax2) = figure(2, 1, gridspec_kw={"wspace": 0.3})
    for platform, (style, colour) in PLATFORM_STYLE.items():
        scan = data.get(f"drift_scan_{platform}")
        if scan is None:
            continue
        ax1.loglog(scan["particles"], scan["seconds"], style, color=colour,
                   label=f"{platform.upper()}: {scan['device']}")
    everything = [t for p in PLATFORM_STYLE if f"drift_scan_{p}" in data for t in data[f"drift_scan_{p}"]["seconds"]]
    plain_log_ticks(ax1.yaxis, 0.8 * min(everything), 1.25 * max(everything))
    ax1.set_xlabel("pseudo-electrons $N$")
    ax1.set_ylabel(f"time of {len(DRIFTS_OVER_C)} runs (s)")
    ax1.legend(loc="upper left")
    panel_label(ax1, "(a)")

    cpu = data["drift_scan_cpu"]
    drifts = np.array(cpu["drifts_over_c"])
    fine = np.array(data["drift_scan_theory_drifts_over_c"])
    ax2.plot(fine, data["drift_scan_theory_gamma"], "-", color=C_THEORY, label="kinetic theory")
    for n, colour in zip(SHOWN, SHOWN_COLOURS):
        rates = np.array(cpu["gamma_measured"][cpu["particles"].index(n)], dtype=float)
        ax2.plot(drifts, rates, "o--", color=colour, label=f"$N = {n}$")
    ax2.set_xlabel(r"drift speed $v_d / c$")
    ax2.set_ylabel(r"growth rate $\gamma / \omega_{pe}$")
    ax2.set_ylim(0, 0.45)
    ax2.set_xticks([0.10, 0.15, 0.20])
    ax2.tick_params(axis="x", pad=10)
    ax2.legend(loc="upper right", ncol=2)
    panel_label(ax2, "(b)")
    savefig(fig, "two_stream_scan")


if __name__ == "__main__":
    if "--replot" not in sys.argv:
        populations, wpe, length = measure()
        if "--measure" in sys.argv:
            sys.exit(0)
        fine = [round(v, 4) for v in np.linspace(0.10, 0.21, 45)]
        cpu = json.loads(MEASUREMENTS.read_text())["drift_scan_cpu"]
        at_scan = theory_curve(populations, wpe, length, cpu["drifts_over_c"])
        deviation = {}
        for n, rates in zip(cpu["particles"], cpu["gamma_measured"]):
            rates = np.array(rates, dtype=float)
            ok = np.isfinite(rates)
            deviation[str(n)] = float(np.mean(np.abs(rates[ok] - at_scan[ok]) / at_scan[ok])) if ok.any() else None
        record(drift_scan_theory_drifts_over_c=fine,
               drift_scan_theory_gamma=[float(g) for g in theory_curve(populations, wpe, length, fine)],
               drift_scan_theory_gamma_at_scan=[float(g) for g in at_scan],
               drift_scan_mean_relative_deviation=deviation,
               drift_scan_deviation_percent_largest=100 * deviation[str(cpu["particles"][-1])],
               drift_scan_deviation_percent_smallest_shown=100 * deviation[str(SHOWN[0])],
               drift_scan_drifts_count=len(cpu["drifts_over_c"]), drift_scan_shown_smallest=SHOWN[0],
               drift_scan_particles_largest=cpu["particles"][-1])
    draw(json.loads(MEASUREMENTS.read_text()))
