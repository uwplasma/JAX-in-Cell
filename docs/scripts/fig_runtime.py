"""Runtime and resolution of the quiet two-stream problem: the wall-clock time of a fixed run
against the number of pseudo-electrons on a CPU and on a GPU, and the growth rate against the
beam drift for several particle counts.

Each run times the problem on the JAX backend it finds and records the timings; on a CPU it
also scans the drift. The figure is then drawn from everything recorded, so run the script
once with a GPU and once on a CPU (``JAX_PLATFORMS=cpu``), in either order; the GPU curve is
left out if no GPU timings were recorded. ``--skip-timing`` keeps the recorded timings and
only scans the drift, for a machine too busy to time; ``--plot-only`` redraws from the
recorded numbers without running anything."""
import json
import os
import platform
import subprocess
import sys
import time

import jax
import numpy as np
from common import COLORS, C_THEORY, MEASUREMENTS, figure, panel_label, record, savefig
from two_stream_setup import LENGTH, OMEGA_PE, build, measure, theory

STEPS = 900
COUNTS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
SCAN_COUNTS = [1000, 4000, 16000]
DRIFTS = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]) * 1e7
SCALE = 2 * np.pi / LENGTH / OMEGA_PE                  # k v_0 / omega_pe per unit drift


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


def device_name(backend):
    if backend != "cpu":
        return jax.devices()[0].device_kind
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    with open("/proc/cpuinfo") as info:
        name = next((line.split(":", 1)[1].strip() for line in info if line.startswith("model name")), "CPU")
    return name.replace("(R)", "").replace(" CPU", "").split(" @")[0]


def time_backend():
    """Time every count on the current backend and record it; return the kind of device."""
    backend = jax.default_backend()
    kind = "cpu" if backend == "cpu" else "gpu"
    timings = [seconds(n) for n in COUNTS]
    for n, value in zip(COUNTS, timings):
        print(f"  {n:7d} pseudo-electrons: {value:.3f} s")
    record(runtime_steps=STEPS, runtime_counts=COUNTS, runtime_counts_largest=COUNTS[-1],
           **{f"runtime_{kind}_seconds": [round(value, 4) for value in timings],
              f"runtime_{kind}_device": device_name(backend), f"runtime_{kind}_jax_version": jax.__version__,
              f"runtime_{kind}_platform": f"{platform.system()} {platform.machine()}",
              f"runtime_{kind}_threads": os.cpu_count(),
              f"runtime_{kind}_load": round(os.getloadavg()[0], 1)})
    return kind


def scan_drift():
    """Growth rate of the seeded mode at every drift for each of SCAN_COUNTS, recorded with
    its mean and largest deviation from the kinetic root."""
    predicted = np.array([theory(drift) for drift in DRIFTS])
    values = {"resolution_counts": SCAN_COUNTS, "resolution_drifts_kv0": [round(float(SCALE * d), 4) for d in DRIFTS]}
    for n in SCAN_COUNTS:
        rates = []
        for drift in DRIFTS:
            _, _, fit = measure(build(drift, n=n).run(STEPS, seed=0, store_particles=False))
            rates.append(fit["gamma"] if fit else np.nan)
        rates = np.array(rates)
        print(f"  {n} pseudo-electrons: " + ", ".join(f"{g:.4f}" for g in rates))
        deviation = 100 * np.abs(rates - predicted) / predicted
        values[f"resolution_gamma_{n}"] = [round(float(g), 4) for g in rates]
        values[f"resolution_mean_deviation_percent_{n}"] = round(float(np.nanmean(deviation)), 1)
        values[f"resolution_max_deviation_percent_{n}"] = round(float(np.nanmax(deviation)), 1)
    record(**values)


def compare(recorded):
    """The GPU against the CPU at the largest count, once both are recorded."""
    cpu, gpu = recorded.get("runtime_cpu_seconds"), recorded.get("runtime_gpu_seconds")
    if not (cpu and gpu):
        return
    faster = [n for n, c, g in zip(COUNTS, cpu, gpu) if g < c]
    record(runtime_cpu_seconds_largest=round(cpu[-1], 2), runtime_gpu_seconds_largest=round(gpu[-1], 2),
           runtime_gpu_speedup_largest=round(cpu[-1] / gpu[-1], 1),
           runtime_gpu_faster_from=faster[0] if faster else "never")


def draw(recorded):
    fig, (a, b) = figure(2)
    for kind, color, marker in (("cpu", "blue", "o"), ("gpu", "vermillion", "s")):
        if recorded.get(f"runtime_{kind}_seconds") and recorded.get("runtime_counts") == COUNTS:
            a.loglog(COUNTS, recorded[f"runtime_{kind}_seconds"], marker + "-", color=COLORS[color],
                     label=f"{kind.upper()}, {recorded[f'runtime_{kind}_device']}")
    a.set(xlabel="pseudo-electrons", ylabel="runtime (s)",
          title=f"two-stream, {STEPS} steps on {build().domain.cells} cells")
    a.legend(loc="upper left", fontsize=15)
    panel_label(a, "a")

    fine = np.linspace(2.0e7, 6.2e7, 43)
    b.plot(SCALE * fine, [theory(d) for d in fine], "-", color=C_THEORY, label="kinetic theory")
    for n, color, marker in zip(SCAN_COUNTS, ("sky", "green", "purple"), "os^"):
        if f"resolution_gamma_{n}" in recorded:
            b.plot(SCALE * DRIFTS, recorded[f"resolution_gamma_{n}"], marker + "--", color=COLORS[color], lw=2,
                   label=f"{n} pseudo-electrons")
    b.set(xlabel=r"$k v_0/\omega_{pe}$", ylabel=r"$\gamma/\omega_{pe}$", ylim=(0, 0.36),
          title="growth rate of the seeded mode")
    b.legend(loc="lower center", fontsize=15, ncol=2)
    panel_label(b, "b")
    fig.tight_layout()
    savefig(fig, "runtime_resolution")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        kind = jax.default_backend() if "--skip-timing" in sys.argv else time_backend()
        if kind == "cpu":
            scan_drift()
        compare(json.loads(MEASUREMENTS.read_text()))
    draw(json.loads(MEASUREMENTS.read_text()))
