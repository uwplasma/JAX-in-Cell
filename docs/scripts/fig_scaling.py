"""Performance: cost per particle and per step, and how it scales."""
import platform
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_IONS, WIDE, panel_label, record, savefig

from jaxincell import Domain, Simulation, Solver, Species, speed_of_light as c

STEPS = 200


def timed(particles, cells, algorithm="explicit", repeats=3):
    """Seconds per step per particle, excluding compilation."""
    electrons = Species.electrons(n=particles, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(5e7, 0, 0),
                                  plus_minus=True, quiet=True, perturbation_amplitude=5e-7,
                                  perturbation_mode=1)
    ions = Species.ions(n=particles, density=4.37e17, electrons=electrons, quiet=True)
    simulation = Simulation(Domain(length=0.01, cells=cells, dt_over_dx_c=4.5), [electrons, ions],
                            Solver(algorithm=algorithm, filter_passes=2))
    run = lambda: simulation.run(STEPS, seed=0, store_particles=False).E.block_until_ready()
    run()                                              # compile
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return min(times) / (STEPS * 2 * particles), min(times)


counts = np.array([2000, 5000, 10000, 20000, 50000, 100000])
per_particle = np.array([timed(n, 64)[0] for n in counts])
implicit = np.array([timed(n, 64, algorithm="implicit")[0] for n in counts])
grids = np.array([32, 64, 128, 256, 512, 1024])
per_grid = np.array([timed(20000, g)[1] for g in grids])
for n, value in zip(counts, per_particle):
    print(f"  {2 * n:7d} particles: {1e9 * value:6.1f} ns per particle per step")

fig, axes = plt.subplots(1, 2, figsize=WIDE)
axes[0].loglog(2 * counts, 1e9 * per_particle, "o-", color=C_ELECTRONS, label="explicit")
axes[0].loglog(2 * counts, 1e9 * implicit, "s-", color=C_IONS, label="implicit, 8 Picard")
axes[0].set(xlabel="pseudo-particles", ylabel="ns per particle per step",
            title="cost per particle is flat once the device is busy")
axes[0].legend()
panel_label(axes[0], "a")

axes[1].loglog(grids, per_grid / STEPS * 1e3, "o-", color=C_ELECTRONS)
axes[1].set(xlabel="grid cells", ylabel="ms per step", title="40 000 particles, grid refined")
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "scaling")

record(scaling_steps=STEPS, scaling_device=str(jax.devices()[0].device_kind),
       scaling_platform=f"{platform.system()} {platform.machine()}",
       scaling_jax_version=jax.__version__,
       scaling_particles_max=int(2 * counts[-1]),
       scaling_ns_per_particle_step=round(float(1e9 * per_particle[-1]), 2),
       scaling_ns_per_particle_step_implicit=round(float(1e9 * implicit[-1]), 2),
       scaling_implicit_over_explicit=round(float(implicit[-1] / per_particle[-1]), 1),
       scaling_ms_per_step_1024_cells=round(float(per_grid[-1] / STEPS * 1e3), 2))
