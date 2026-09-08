"""Landau damping (Landau, J. Phys. USSR 10, 25, 1946).

A small-amplitude Langmuir wave at k lambda_D = 0.5 decays because the electrons
travelling at the phase velocity absorb it. The least damped root of the kinetic
dispersion relation is omega = (1.4157 - 0.1533 i) omega_pe (Canosa, J. Plasma
Phys. 8, 187, 1972); this measures both parts from the decaying mode amplitude.

A quiet start (equally spaced particles, velocities at the quantiles of the
Maxwellian) is what makes the discrete-particle noise low enough to follow the
decay over three e-foldings with 150 000 particles.
"""
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

length, cells, k_lambda_d = 1.0, 64, 0.5
k = 2 * np.pi / length
omega_pe = 0.05 * c * cells / length                 # gives omega_pe * dt = 0.05 at dt = dx / c
density = omega_pe ** 2 * epsilon_0 * mass_electron / e_charge ** 2
v_th = k_lambda_d / k * np.sqrt(2) * omega_pe

electrons = Species.electrons(n=150000, density=density, vth=(v_th, 0, 0), quiet=True,
                              perturbation_amplitude=0.01 / k, perturbation_mode=1)
ions = Species.ions(n=20000, density=density, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0), [electrons, ions],
                        Solver(filter_passes=0))
output = simulation.run(500, seed=0, store_particles=False)

t = np.asarray(output.t) * omega_pe
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1]) / cells
floor = amplitude[400:].mean()                        # discrete-particle noise
i = np.arange(1, t.size - 1)
peaks = i[(amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] > amplitude[2:]) & (amplitude[1:-1] > 5 * floor)]
gamma = np.polyfit(t[peaks], np.log(amplitude[peaks]), 1)[0]
omega = np.pi / np.mean(np.diff(t[peaks]))            # maxima of |E_k| are half a period apart
print(f"measured  gamma/omega_pe = {gamma:+.4f}   omega/omega_pe = {omega:.4f}")
print("kinetic   gamma/omega_pe = -0.1533   omega/omega_pe = 1.4157")

plt.figure(figsize=(6, 4))
plt.semilogy(t, amplitude, lw=1, label=r"$|E_k(t)|$")
plt.semilogy(t[peaks], np.exp(np.polyval(np.polyfit(t[peaks], np.log(amplitude[peaks]), 1), t[peaks])),
             "k--", label=fr"fit $\gamma={gamma:.4f}\,\omega_{{pe}}$")
plt.axhline(floor, color="0.6", lw=0.8, label="noise floor")
plt.xlabel(r"$t\,\omega_{pe}$"); plt.ylabel("mode amplitude (V/m)")
plt.title(r"Landau damping at $k\lambda_D=0.5$"); plt.legend(frameon=False); plt.tight_layout()
plt.show()
