"""Energy, momentum and charge in the explicit and the implicit scheme.

The explicit leapfrog holds the discrete Gauss law to round-off and, in a periodic box,
the momentum; its energy error stays bounded but does not vanish. The implicit
Crank-Nicolson scheme holds the energy and the Gauss law to round-off once its Picard
iteration has converged, and gives up the momentum instead (Chen, Chacon and Barnes, J.
Comput. Phys. 230, 7018, 2011; Kormann and Sonnendruecker, J. Comput. Phys. 425, 109890,
2021). Between absorbing walls the walls take energy and momentum out of the box, and the
Gauss law is what is left to check: the charge the walls collect has to show in the field.

Every error is relative: the energy to W(0); the momentum to the sum of |p| over the
particles, because two counter-streaming beams carry no net momentum; and the Gauss law to
e n/eps0, the density of one sign of charge, because a neutral plasma has no net density.
Round-off sets the floor: about 1e-16 in double precision, and about 1e-7 in single.

This is the setup of docs/scripts/fig_conservation.py, whose numbers the documentation
quotes: a quiet two-stream run of 400 steps.
"""

import os

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, speed_of_light as c

ERRORS = {"energy_error": "energy", "momentum_error": "momentum", "gauss_residual": "charge"}


def run(algorithm, walls):
    electrons = Species.electrons(n=4000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(5e7, 0, 0),
                                  plus_minus=True, quiet=True, perturbation_amplitude=5e-7, perturbation_mode=1)
    ions = Species.ions(n=4000, density=4.37e17, electrons=electrons, quiet=True)
    domain = Domain(length=0.01, cells=64, dt_over_dx_c=4.5, particle_bc=walls, field_bc=walls)
    start = time.perf_counter()
    output = Simulation(domain, [electrons, ions], Solver(algorithm=algorithm)).run(400, seed=3)
    d = diagnostics(output)
    largest = "".join(f"   {name} {float(np.max(d[key])):.1e}" for key, name in ERRORS.items())
    print(f"{algorithm:8s} {walls:9s} largest errors:{largest}   {time.perf_counter() - start:.1f} s")
    return np.asarray(output.t), d


fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for algorithm, color in (("explicit", "tab:blue"), ("implicit", "tab:red")):
    for walls, style in (("periodic", "-"), ("absorbing", "--")):
        t, d = run(algorithm, walls)
        for ax, (key, name) in zip(axes, ERRORS.items()):
            if walls == "periodic" or key == "gauss_residual":      # the walls take energy and momentum away
                ax.semilogy(t * 1e9, np.asarray(d[key]) + 1e-17, style, color=color, label=f"{algorithm}, {walls}")
            ax.set(xlabel="t (ns)", title=f"{name} error")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
