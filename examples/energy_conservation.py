"""Explicit leapfrog against the implicit Crank-Nicolson scheme.

The explicit scheme is fast and its energy error stays bounded but does not
vanish. The implicit scheme of Chen, Chacon and Barnes (J. Comput. Phys. 230,
7018, 2011) conserves the discrete total energy exactly once the Picard
iteration has converged, which this shows by sweeping the iteration count.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, speed_of_light as c


def run(steps, **solver):
    electrons = Species.electrons(n=4000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0),
                                  plus_minus=True, perturbation_amplitude=5e-7, perturbation_mode=1)
    ions = Species.ions(n=4000, density=4.37e17, electrons=electrons)
    simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [electrons, ions],
                            Solver(**solver))
    start = time.perf_counter()
    output = simulation.run(steps, seed=3)
    output.E.block_until_ready()
    return output, time.perf_counter() - start


steps = 300
drift = lambda out: float(np.max(np.abs(np.asarray(diagnostics(out)["total"]) / diagnostics(out)["total"][0] - 1)))
explicit, wall = run(steps, algorithm="explicit")
print(f"explicit                        energy drift {drift(explicit):.2e}   {wall:.2f} s")
implicit = {}
for iterations in (1, 2, 4, 8):
    implicit[iterations], wall = run(steps, algorithm="implicit", picard_iterations=iterations)
    print(f"implicit, {iterations} Picard iterations    energy drift {drift(implicit[iterations]):.2e}   {wall:.2f} s")

plt.figure(figsize=(6, 4))
error = lambda out: np.abs(np.asarray(diagnostics(out)["total"]) / diagnostics(out)["total"][0] - 1)
plt.semilogy(np.asarray(explicit.t) * 1e9, error(explicit) + 1e-17, "k", label="explicit")
for iterations, out in implicit.items():
    plt.semilogy(np.asarray(out.t) * 1e9, error(out) + 1e-17, label=f"implicit, {iterations} iterations")
plt.xlabel("t (ns)"); plt.ylabel(r"$|W(t)/W(0)-1|$"); plt.legend(frameon=False)
plt.tight_layout(); plt.show()
