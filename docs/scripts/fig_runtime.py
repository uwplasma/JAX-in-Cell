"""Runtime and resolution of the quiet two-stream problem: the wall-clock time of a fixed run
against the number of pseudo-electrons on a CPU and on a GPU, and the growth rate against the
beam drift for several particle counts.

On a GPU the script only times the run and records the numbers. On a CPU it times the run,
scans the drift, and draws the figure with the GPU numbers recorded earlier, so the figure
can be redrawn on a machine without a GPU; the GPU curve is left out if none were recorded."""
import json
import os
import platform
import subprocess
import time

import jax
import numpy as np
from common import COLORS, C_THEORY, MEASUREMENTS, figure, panel_label, record, savefig
from two_stream_setup import LENGTH, OMEGA_PE, build, measure, theory

STEPS = 900
COUNTS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
SCAN_COUNTS = [1000, 4000, 16000]
DRIFTS = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]) * 1e7
BACKEND = jax.default_backend()


def seconds(n, repeats=5):
    """Best wall-clock time of the fixed run with ``n`` pseudo-electrons, compilation excluded."""
    simulation = build(n=n)

    def run():
        simulation.run(STEPS, seed=0, store_particles=False).E.block_until_ready()

    run()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times)


def device_name():
    if BACKEND != "cpu":
        return jax.devices()[0].device_kind
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    with open("/proc/cpuinfo") as info:
        return next((line.split(":", 1)[1].strip() for line in info if line.startswith("model name")), "CPU")


timings = [seconds(n) for n in COUNTS]
for n, value in zip(COUNTS, timings):
    print(f"  {n:7d} pseudo-electrons: {value:.3f} s")
kind = "cpu" if BACKEND == "cpu" else "gpu"
record(runtime_steps=STEPS, runtime_counts=COUNTS,
       **{f"runtime_{kind}_seconds": [round(value, 4) for value in timings],
          f"runtime_{kind}_device": device_name(), f"runtime_{kind}_jax_version": jax.__version__,
          f"runtime_{kind}_platform": f"{platform.system()} {platform.machine()}",
          f"runtime_{kind}_load": round(os.getloadavg()[0], 1)})

if kind == "cpu":
    recorded = json.loads(MEASUREMENTS.read_text())
    gpu = recorded.get("runtime_gpu_seconds") if recorded.get("runtime_counts") == COUNTS else None

    measured = {}
    for n in SCAN_COUNTS:
        measured[n] = []
        for drift in DRIFTS:
            _, _, fit = measure(build(drift, n=n).run(STEPS, seed=0, store_particles=False))
            measured[n].append(fit["gamma"] if fit else np.nan)
        print(f"  {n} pseudo-electrons: " + ", ".join(f"{g:.4f}" for g in measured[n]))
    predicted = np.array([theory(drift) for drift in DRIFTS])

    fig, axes = figure(2)
    axes[0].loglog(COUNTS, timings, "o-", color=COLORS["blue"], label=f"CPU ({device_name()})")
    if gpu:
        axes[0].loglog(COUNTS, gpu, "s-", color=COLORS["vermillion"], label=f"GPU ({recorded['runtime_gpu_device']})")
    axes[0].set(xlabel="pseudo-electrons", ylabel="runtime (s)",
                title=f"two-stream, {STEPS} steps on {build().domain.cells} cells")
    axes[0].legend(loc="upper left")
    panel_label(axes[0], "a")

    scale = 2 * np.pi / LENGTH / OMEGA_PE                       # k v_0 / omega_pe per unit drift
    fine = np.linspace(2.0e7, 6.2e7, 22)
    axes[1].plot(scale * fine, [theory(d) for d in fine], "-", color=C_THEORY, label="kinetic theory")
    for (n, rates), color, marker in zip(measured.items(), ("sky", "green", "purple"), "os^"):
        axes[1].plot(scale * DRIFTS, rates, marker + "--", color=COLORS[color], lw=2, label=f"{n} pseudo-electrons")
    axes[1].set(xlabel=r"$k v_0/\omega_{pe}$", ylabel=r"$\gamma/\omega_{pe}$", ylim=(0, 0.34),
                title="growth rate of the seeded mode")
    axes[1].legend(loc="lower center", fontsize=16)
    panel_label(axes[1], "b")
    fig.tight_layout()
    savefig(fig, "runtime_resolution")

    summary = {f"resolution_mean_deviation_percent_{n}":
               round(float(np.nanmean(100 * np.abs(np.array(rates) - predicted) / predicted)), 1)
               for n, rates in measured.items()}
    summary.update({f"resolution_max_deviation_percent_{n}":
                    round(float(np.nanmax(100 * np.abs(np.array(rates) - predicted) / predicted)), 1)
                    for n, rates in measured.items()})
    summary["runtime_cpu_seconds_largest"] = round(timings[-1], 2)
    if gpu:
        summary["runtime_gpu_seconds_largest"] = round(gpu[-1], 2)
        summary["runtime_gpu_speedup_largest"] = round(timings[-1] / gpu[-1], 1)
        faster = [n for n, cpu, g in zip(COUNTS, timings, gpu) if g < cpu]
        summary["runtime_gpu_faster_from"] = faster[0] if faster else "never"
    record(runtime_counts_largest=COUNTS[-1], resolution_counts=SCAN_COUNTS, **summary)
