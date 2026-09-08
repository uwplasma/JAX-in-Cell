"""Bump-on-tail instability and quasilinear flattening.

A weak beam on the tail of a Maxwellian makes df/dv positive there, and every wave
whose phase velocity sits in that window grows (Vedenov, Velikhov and Sagdeev, Nucl.
Fusion 1, 82, 1961; O'Neil, Phys. Fluids 8, 2255, 1965). The waves saturate by
flattening the bump into a plateau, which the second panel shows: the positive slope
has gone by the end of the run.

A wave resonates with the beam when omega_pe / k = v_beam, so the thermal speed is
chosen to put the fastest-growing mode at MODE, comfortably inside the box and well
resolved by the grid. Picking it any other way risks seeding a mode that is not
unstable at all.
"""
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

length, cells, mode, steps = 1.0, 128, 5, 2000
omega_pe = 0.05 * c * cells / length
density = omega_pe ** 2 * epsilon_0 * mass_electron / e_charge ** 2
beam_fraction, beam_drift_over_vth, beam_width = 0.03, 5.0, 0.7
v_th = omega_pe * length / (2 * np.pi * mode * beam_drift_over_vth)
beam_drift, beam_vth = beam_drift_over_vth * v_th, beam_width * v_th

bulk = Species.electrons(n=80000, density=(1 - beam_fraction) * density, vth=(v_th, 0, 0), quiet=True,
                         name="bulk", perturbation_mode=mode,
                         perturbation_amplitude=2e-3 * length / (2 * np.pi * mode))
beam = Species.electrons(n=40000, density=beam_fraction * density, vth=(beam_vth, 0, 0), quiet=True,
                         drift=(beam_drift, 0, 0), name="beam")
ions = Species.ions(n=10000, density=density, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0), [bulk, beam, ions],
                        Solver(filter_passes=0))
output = simulation.run(steps, seed=0, store_every=16)

t = np.asarray(output.t) * omega_pe
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, mode]) / cells
peak = int(np.argmax(amplitude))
window = ((amplitude > 1.2 * amplitude[:20].max()) & (amplitude < 0.3 * amplitude[peak])
          & (np.arange(t.size) < peak))
gamma = np.polyfit(t[window], np.log(amplitude[window]), 1)[0]
print(f"mode {mode} grows at gamma = {gamma:.4f} omega_pe")
print("kinetic theory gives 0.1463 omega_pe for these parameters")

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
left.semilogy(t, amplitude, lw=1)
left.semilogy(t[window], np.exp(np.polyval(np.polyfit(t[window], np.log(amplitude[window]), 1), t[window])),
              "k--", label=fr"$\gamma={gamma:.3f}\,\omega_{{pe}}$")
left.set(xlabel=r"$t\,\omega_{pe}$", ylabel=fr"$|E_{{k={mode}}}|$ (V/m)")
left.legend(frameon=False)

electrons = np.asarray(output.species) < 2
edges = np.linspace(-4 * v_th, 9 * v_th, 220)
centres = 0.5 * (edges[1:] + edges[:-1])
for step, style, label in ((0, "--", "initial"), (-1, "-", "final")):
    counts, _ = np.histogram(np.asarray(output.v[step, electrons, 0]), edges, density=True)
    right.plot(centres / v_th, counts, style, label=label)
right.set(xlabel=r"$v_x/v_{th}$", ylabel=r"$f(v_x)$", yscale="log", ylim=(1e-9, None),
          title="the bump flattens into a plateau")
right.legend(frameon=False)
plt.tight_layout()
plt.show()
