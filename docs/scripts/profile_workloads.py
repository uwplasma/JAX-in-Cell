"""Time one real workload, warm, in a fresh process; the JSON line goes to standard output.

Run from the checkout: WORKLOAD=beam_laser|rf_metrics|rf_jacfwd|sheath python docs/scripts/profile_workloads.py.
The two RF/laser workloads execute the definitions at the top of their example, up to the validation
that follows them, so they time the example's own ``final_mean_gamma(a0)``, ``metrics(a, steps)`` and
four-coefficient ``jacfwd``. ``sheath`` is the maintained-sheath test of tests/test_sources.py
(SHEATH_STEPS=6000 steps, SHEATH_CAPACITY=60000 slots per species). REPEATS sets the warm repetitions and
TRACE=<directory> records one extra traced run. Set JAX_ENABLE_X64=1: the examples' reference gates need float64.
"""
import json
import os
from pathlib import Path
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
root, workload = Path(__file__).resolve().parents[2], os.environ["WORKLOAD"]
repeats = int(os.environ.get("REPEATS", 5))
examples = root / "examples/3_advanced"


def definitions(name, stop):
    """The module-level code of an example that precedes the text ``stop``."""
    path = examples / name
    namespace = {"__file__": str(path), "__name__": "profile"}
    exec(compile(path.read_text().split(stop)[0], str(path), "exec"), namespace)
    return namespace


if workload == "beam_laser":
    ns = definitions("beam_laser.py", "errors = []")
    steps, argument = ns["steps"], ns["a0"]

    def function(a):
        eta = ns["simulate"](a).state.u / ns["c"]
        return jnp.mean(jnp.sqrt(1 + jnp.sum(eta**2, axis=1)))
elif workload.startswith("rf"):
    ns = definitions("rf_source_control.py", "report['refinement'] = []")
    steps, argument = ns["design_steps"], jnp.array([.8, 1.1, .9, 1.2])

    def function(a):
        return ns["metrics"](a, steps)
    if workload == "rf_jacfwd":
        function = jax.jacfwd(function)
else:
    sys.path.insert(0, str(root / "tests"))
    import test_sources as T
    from jaxincell import Simulation, Solver, Source, Species, mass_electron
    steps, capacity = int(os.environ.get("SHEATH_STEPS", 6000)), int(os.environ.get("SHEATH_CAPACITY", 60000))
    emit, speed, drift = 60, .2, (.2 * T.SIGMA, 0, 0)
    wall = float(T.floating_potential(speed))
    thermal = (np.sqrt(2) * T.SIGMA, 0, 0)
    source = T.maxwellian_source(emit, density=float(T.source_density(wall)) * T.DENSITY)
    electrons = Species("electrons", capacity, -1.0, mass_electron, T.DENSITY, thermal, active=capacity // 4,
                        sampling="low_noise", source=source)
    ions = Species("ions", capacity, 1.0, 1836.0 * mass_electron, T.DENSITY, 0.0, drift, active=capacity // 4,
                   sampling="low_noise", source=Source(density=T.DENSITY, vth=0.0, drift=drift, emit=emit))
    simulation = Simulation(T.box(cells=100, particle_bc="absorbing", field_bc=("open", "absorbing")),
                            [electrons, ions], Solver(model="electrostatic"))
    argument = jnp.asarray(1.0)

    def function(scale):
        out = simulation.run(steps, seed=0, store_every=max(steps // 60, 1), store_particles=False, moments=True)
        return out.state.x.sum() * scale
start = time.perf_counter()
executable = jax.jit(function).lower(argument).compile()
compile_seconds = time.perf_counter() - start
start = time.perf_counter()
jax.block_until_ready(executable(argument))
first = time.perf_counter() - start
times = []
for _ in range(repeats):
    start = time.perf_counter()
    result = jax.block_until_ready(executable(argument))
    times.append(time.perf_counter() - start)
if os.environ.get("TRACE"):
    with jax.profiler.trace(os.environ["TRACE"]):
        jax.block_until_ready(executable(argument))
print(json.dumps(dict(workload=workload, backend=jax.default_backend(), steps=steps, compile_s=compile_seconds,
                      first_s=first, median_s=statistics.median(times), min_s=min(times), max_s=max(times),
                      value=np.ravel(np.asarray(result))[:4].tolist())))
