"""Landau damping (Landau, J. Phys. USSR 10, 25, 1946).

A small-amplitude Langmuir wave at k lambda_D = 0.5 decays because the electrons
travelling at the phase velocity absorb it. The least damped root of the kinetic
dispersion relation is omega = (1.4157 - 0.1533 i) omega_pe (Canosa, J. Plasma
Phys. 8, 187, 1972), which `jaxincell.theory.landau_root` solves for; this measures both
parts from the decaying mode amplitude.

A quiet start (equally spaced particles, velocities at the quantiles of the
Maxwellian) is what makes the discrete-particle noise low enough to follow the
decay over three e-foldings with 150 000 particles. The wave then sinks into that
noise; `jaxincell.theory.damped_mode` measures the floor where the amplitude stops
falling and fits only the maxima well above it, so running longer does not change
the answer (at 800 and 1200 steps it is the same to four digits).
"""

import os
from pathlib import Path

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, figure, mass_electron, save_run,
                       elementary_charge as e_charge, speed_of_light as c)
from jaxincell.theory import damped_mode, landau_root

length, cells, k_lambda_d, particles, steps = 1.0, 64, 0.5, 150000, 800
k = 2 * np.pi / length
omega_pe = 0.05 * c * cells / length                 # gives omega_pe * dt = 0.05 at dt = dx / c
density = omega_pe ** 2 * epsilon_0 * mass_electron / e_charge ** 2
v_th = k_lambda_d / k * np.sqrt(2) * omega_pe

electrons = Species.electrons(n=particles, density=density, vth=(v_th, 0, 0), sampling="quiet",
                              perturbation_amplitude=0.01 / k, perturbation_mode=1)
ions = Species.ions(n=particles // 8, density=density, mass_ratio=1e9, vth=(0, 0, 0), sampling="quiet")
simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0), [electrons, ions],
                        Solver(filter_passes=0))
output = simulation.run(steps, seed=0, store_particles=False)

t = np.asarray(output.t) * omega_pe
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, 1]) / cells
gamma, omega, peaks, floor = damped_mode(t, amplitude)
root = landau_root(k_lambda_d)
print(f"measured  gamma/omega_pe = {gamma:+.4f}   omega/omega_pe = {omega:.4f}   ({peaks.size} maxima)")
print(f"kinetic   gamma/omega_pe = {root.imag:+.4f}   omega/omega_pe = {root.real:.4f}")

fig, ax = figure()
ax.semilogy(t, amplitude, lw=2, label=r"$|E_k(t)|$")
ax.semilogy(t[peaks], amplitude[peaks], "o", ms=9, label="maxima fitted")
ax.semilogy(t[peaks], amplitude[peaks][0] * np.exp(root.imag * (t[peaks] - t[peaks][0])), "k--",
            label=fr"kinetic $\gamma={root.imag:.4f}\,\omega_{{pe}}$")
ax.axhline(floor, color="0.6", lw=2, label="noise floor")
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_k|$ (V/m)", title=r"Landau damping at $k\lambda_D=0.5$")
ax.legend()
fig.tight_layout()

save_run(Path.cwd() / "landau_damping", "landau_damping",
         dict(length=length, cells=cells, k_lambda_d=k_lambda_d, particles=particles, steps=steps),
         dict(gamma=gamma, omega=omega, maxima=int(peaks.size), floor=floor,
              gamma_kinetic=root.imag, omega_kinetic=root.real),
         figure=fig, t=t, amplitude=amplitude)
plt.show()
