"""Two-stream instability (Buneman, Phys. Rev. 115, 503, 1959).

Two counter-streaming electron beams on a background of protons. The unstable
mode grows exponentially until the beams trap each other and the phase space
rolls up into the vortex that closes the growth. Run it and watch the animation:

    python two_stream.py
"""

import os
from pathlib import Path

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, plot, save_run, speed_of_light as c

electrons = Species.electrons(n=8000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0),
                              plus_minus=True, perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=8000, density=4.37e17, electrons=electrons)
simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [electrons, ions],
                        Solver(filter_passes=2))

output = simulation.run(1200, seed=0, store_every=2)   # 600 frames; the particle history is the bulk of the memory
energy = diagnostics(output)
drift = abs(float(energy['total'][-1] / energy['total'][0]) - 1)
growth = float(energy['electric'].max() / energy['electric'][0])
print(f"energy drift over the run: {drift:.2e}")
print(f"electric energy grew by a factor {growth:.3g}")
save_run(Path.cwd() / "two_stream", "two_stream", dict(particles=8000, cells=64, steps=1200, filter_passes=2),
         dict(energy_drift=drift, electric_energy_growth=growth), t=output.t, electric=energy["electric"],
         total=energy["total"])

omega_pe = float(simulation.plasma_frequency())
plot(output, direction="x", omega=omega_pe)
# plot(output, direction="x", omega=omega_pe, save="two_stream.mp4", show=False)
