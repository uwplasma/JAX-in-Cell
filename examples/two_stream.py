"""Two-stream instability (Buneman, Phys. Rev. 115, 503, 1959).

Two counter-streaming electron beams on a background of protons. The unstable
mode grows exponentially until the beams trap each other and the phase space
rolls up into the vortex that closes the growth. Run it and watch the animation:

    python two_stream.py
"""
from jaxincell import Domain, Simulation, Solver, Species, diagnostics, plot, speed_of_light as c

electrons = Species.electrons(n=8000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0),
                              plus_minus=True, perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=8000, density=4.37e17, electrons=electrons)
simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5), [electrons, ions],
                        Solver(filter_passes=2))

output = simulation.run(1200, seed=0, store_every=2)   # 600 frames; the particle history is the bulk of the memory
energy = diagnostics(output)
print(f"energy drift over the run: {abs(float(energy['total'][-1] / energy['total'][0]) - 1):.2e}")
print(f"electric energy grew by a factor {float(energy['electric'].max() / energy['electric'][0]):.3g}")

omega_pe = float(simulation.plasma_frequency())
plot(output, direction="x", omega=omega_pe)
# plot(output, direction="x", omega=omega_pe, save="two_stream.mp4", show=False)
