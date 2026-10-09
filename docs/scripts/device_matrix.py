"""Device and precision coverage: the quiet two-stream run of fig_runtime.py, 64000
pseudo-electrons and 900 steps, in single and double precision on the backend JAX finds.
Records the warm wall-clock time, the growth rate of the seeded mode against the kinetic
root, and the largest relative energy error. Run once on a CPU (``JAX_PLATFORMS=cpu``) and
once on a GPU. Each precision runs in its own process, since it is fixed at import."""
import json
import os
import subprocess
import sys
import time

N, STEPS = 64000, 900

if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    import jax
    import numpy as np
    from two_stream_setup import build, measure, theory
    from jaxincell import diagnostics
    simulation = build(n=N)
    out = simulation.run(STEPS, seed=0, store_every=9)          # particles every 9th step, for the energy
    out.E.block_until_ready()
    simulation.run(STEPS, seed=0, store_particles=False).E.block_until_ready()
    times = []
    for _ in range(3):
        start = time.perf_counter()
        simulation.run(STEPS, seed=0, store_particles=False).E.block_until_ready()
        times.append(time.perf_counter() - start)
    total = np.asarray(diagnostics(out)["total"], dtype=float)
    fit = measure(simulation.run(STEPS, seed=0, store_particles=False))[2]
    print(json.dumps(dict(seconds=min(times), gamma=fit["gamma"] if fit else None, theory=theory(5e7),
                          energy_error=float(np.max(np.abs(total / total[0] - 1))),
                          dtype=str(out.E.dtype), device=jax.devices()[0].device_kind,
                          backend=jax.default_backend())))
elif __name__ == "__main__":
    from common import record
    results = {}
    for bits, flag in ((32, "0"), (64, "1")):
        env = dict(os.environ, JAX_ENABLE_X64=flag)
        line = subprocess.run([sys.executable, __file__, "--child"], env=env, check=True, capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.splitlines()[-1]
        results[bits] = json.loads(line)
        print(f"  float{bits}: {results[bits]}")
    kind = "cpu" if results[64]["backend"] == "cpu" else "gpu"
    values = {f"device_{kind}_name": results[64]["device"], f"device_{kind}_load": round(os.getloadavg()[0], 1)}
    for bits, r in results.items():
        values.update({f"device_{kind}_f{bits}_seconds": round(r["seconds"], 3),
                       f"device_{kind}_f{bits}_gamma": round(r["gamma"], 4) if r["gamma"] else None,
                       f"device_{kind}_f{bits}_energy_error": float(f"{r['energy_error']:.2e}")})
    values.update(device_theory_gamma=round(results[64]["theory"], 4), device_particles=N)
    record(**values)
