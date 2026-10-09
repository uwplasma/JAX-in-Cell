"""The discrete invariants of every field model, integrator, filter and collision setting,
each measured on its own rather than claimed together: the largest relative energy error,
the largest momentum error relative to the momentum scale, and the largest Gauss residual,
over a two-stream run in a periodic box and between reflecting walls."""
import numpy as np
from common import record

from jaxincell import Collisions, Domain, Simulation, Solver, Species, diagnostics, speed_of_light as c

CASES = {
    "explicit_em": dict(algorithm="explicit"),
    "explicit_em_filter": dict(algorithm="explicit", filter_passes=2),
    "explicit_gauss": dict(algorithm="explicit", field_solver="gauss"),
    "explicit_es": dict(algorithm="explicit", model="electrostatic"),
    "explicit_es_collisions": dict(algorithm="explicit", model="electrostatic"),
    "implicit_em": dict(algorithm="implicit"),
    "implicit_es": dict(algorithm="implicit", model="electrostatic"),
    "implicit_es_collisions": dict(algorithm="implicit", model="electrostatic"),
}


def run(name, boundary, steps=150, n=2000):
    e = Species.electrons(n=n, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0), plus_minus=True,
                          sampling="lattice", perturbation_amplitude=5e-7, perturbation_mode=1)
    i = Species.ions(n=n, density=4.37e17, electrons=e, sampling="lattice")
    domain = Domain(length=0.01, cells=64, dt_over_dx_c=4.5 if "implicit" in name else 0.9,
                    particle_bc=boundary, field_bc=boundary)
    # self-collisions only, which conserve each pair's momentum and energy exactly
    collisions = (Collisions(coulomb_log=1e4, pairs=(("electrons", "electrons"), ("ions", "ions")))
                  if name.endswith("collisions") else None)
    out = Simulation(domain, [e, i], Solver(**CASES[name]), collisions).run(steps, seed=3)
    d = diagnostics(out)
    return {metric: float(np.max(np.abs(np.asarray(d[metric]))))
            for metric in ("energy_error", "momentum_error", "gauss_residual")}


values = {}
for name in CASES:
    for boundary in ("periodic", "reflective"):
        result = run(name, boundary)
        print(f"  {name:24s} {boundary:10s} " + "  ".join(f"{k} {v:.1e}" for k, v in result.items()))
        values.update({f"audit_{name}_{boundary}_{k}": f"{v:.1e}" for k, v in result.items()})
record(**values)
