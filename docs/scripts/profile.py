"""Profile one real PIC program; all output stays in a chosen local directory.

Run from the checkout: PROFILE_DIR=/tmp/jic-profile python docs/scripts/profile.py.
Environment controls: PROFILE_CASE=explicit|implicit|collisions|radiation,
PROFILE_MODE=forward|reverse|jvp, PROFILE_N=2048, PROFILE_STEPS=32,
PROFILE_CHECKPOINT=0|1, PROFILE_TRACE=0|1, PROFILE_COMPILE_ONLY=0|1.
Use a fresh process for each case, precision and device. The default explicit
case also runs unchanged against the parent research-release checkout.
"""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxincell
from jaxincell import Collisions, Domain, Simulation, Solver, Species, provenance
from jaxincell import elementary_charge as e, epsilon_0, speed_of_light as c

case = os.environ.get("PROFILE_CASE", "explicit")
mode = os.environ.get("PROFILE_MODE", "forward")
n, steps = int(os.environ.get("PROFILE_N", 2048)), int(os.environ.get("PROFILE_STEPS", 32))
checkpoint = os.environ.get("PROFILE_CHECKPOINT", "0") == "1"
destination = Path(os.environ.get("PROFILE_DIR", "/tmp/jic-profile")).resolve()
destination.mkdir(parents=True, exist_ok=True)
if case not in ("explicit", "implicit", "collisions", "radiation") or mode not in ("forward", "reverse", "jvp"):
    raise ValueError("choose a documented PROFILE_CASE and PROFILE_MODE")

density, amplitude = 4.37e17, 5e-7
electrons = Species.electrons(n, density, vth=(.02 * c, 0, 0), drift=(.6 * c, 0, 0),
                              plus_minus=True, sampling="low_noise", perturbation_amplitude=amplitude,
                              perturbation_mode=1)
ions = Species.ions(n // 2, density, mass_ratio=1836, vth=(0, 0, 0), sampling="low_noise")
options = {"collisions": Collisions(coulomb_log=15)} if case == "collisions" else {}
if case == "radiation":
    options = {"radiation": jaxincell.Radiation(species="electrons"),
               "external_B": jnp.broadcast_to(jnp.asarray([0., 0., 3.]), (64, 3))}
simulation = Simulation(Domain(length=.01, cells=64, dt_over_dx_c=.8), (electrons, ions),
                        Solver(algorithm="implicit" if case == "implicit" else "explicit",
                               relativistic=True, picard_iterations=8), **options)
scale = e * density * amplitude / epsilon_0


def objective(beta):
    sim = simulation.replace(species=(electrons.replace(drift=(beta * c, 0., 0.)), ions))
    settings = {"checkpoint": True} if checkpoint else {}
    out = sim.run(steps, seed=3, store_every=steps, store_particles=False, **settings)
    return jnp.mean((out.state.E / scale)**2)


def tangent(beta):
    return jax.jvp(objective, (beta,), (jnp.ones_like(beta),))


function = {"forward": objective, "reverse": jax.value_and_grad(objective), "jvp": tangent}[mode]
beta = jnp.asarray(.6)
start = time.perf_counter()
executable = jax.jit(function).lower(beta).compile()
compile_seconds = time.perf_counter() - start
memory = executable.memory_analysis()
report = provenance(case=case, mode=mode, checkpoint=checkpoint, particles=n + n // 2,
                    steps=steps, cells=64, source=str(Path(jaxincell.__file__).resolve()),
                    devices=[str(d) for d in jax.devices()], compile_seconds=compile_seconds,
                    memory={name: getattr(memory, name) for name in
                            ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                             "temp_size_in_bytes")},
                    memory_note="Compiler buffer estimates; not peak runtime resident memory.",
                    host_note="Shared-host timings; compare the same workload, precision and device.")
(destination / "program.hlo.txt").write_text(executable.as_text())
if os.environ.get("PROFILE_COMPILE_ONLY", "0") == "1":
    report["status"] = "compiled; not executed"
else:
    start = time.perf_counter()
    result = jax.block_until_ready(executable(beta))
    report["first_seconds"] = time.perf_counter() - start
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        result = jax.block_until_ready(executable(beta))
        timings.append(time.perf_counter() - start)
    report.update(status="executed", warm_seconds=timings, median_seconds=float(np.median(timings)),
                  values=[np.asarray(x).tolist() for x in jax.tree.leaves(result)])
    if mode != "forward":
        value, derivative = jax.block_until_ready(jax.jit(tangent)(beta))
        h = 1e-4
        finite_difference = float((objective(beta + h) - objective(beta - h)) / (2 * h))
        report.update(jvp=float(derivative), finite_difference=finite_difference,
                      derivative_relative_difference=abs(float(result[1]) - float(derivative))
                      / max(abs(float(derivative)), np.finfo(float).tiny))
    if os.environ.get("PROFILE_TRACE", "0") == "1":
        with jax.profiler.trace(str(destination / "trace")):
            jax.block_until_ready(executable(beta))
(destination / "profile.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
