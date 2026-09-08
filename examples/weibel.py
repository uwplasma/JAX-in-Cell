"""Weibel instability (Weibel, Phys. Rev. Lett. 2, 83, 1959).

A plasma hotter across the simulation axis than along it is unstable to purely
growing transverse magnetic perturbations. Setting omega = 0 in the transverse
dispersion relation gives the marginal wavenumber

    k_c c = omega_pe sqrt(T_z/T_x - 1),

so in a box holding several wavelengths the modes below k_c grow and those above
it do not. This run seeds nothing: every mode starts from the particle noise.
"""
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

ratio, density, n = 25.0, 1e15, 40000
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
k_c = np.sqrt(ratio - 1) * omega_pe / c
length = 4.0 * 2 * np.pi / k_c                      # four marginal wavelengths
v_th = 0.02 * c

rng = np.random.default_rng(0)
x = np.linspace(-length / 2, length / 2, n, endpoint=False) + length / (2 * n)
v = np.stack([v_th / np.sqrt(2) * rng.standard_normal(n), np.zeros(n),
              v_th * np.sqrt(ratio / 2) * rng.standard_normal(n)], axis=1)
electrons = Species.electrons(n=n, density=density, vth=(v_th, 0, v_th * np.sqrt(ratio)))
electrons = electrons.replace(x=np.stack([x, np.zeros(n), np.zeros(n)], axis=1), v=v)
ions = Species.ions(n=n // 4, density=density, mass_ratio=1e6, vth=(0, 0, 0), quiet=True)
simulation = Simulation(Domain(length=length, cells=128, dt_over_dx_c=0.5), [electrons, ions],
                        Solver(filter_passes=0))
output = simulation.run(4000, seed=0, store_every=20, store_particles=False)

t = np.asarray(output.t) * omega_pe
B_k = np.abs(np.fft.rfft(np.asarray(output.B[:, :, 1]), axis=1))
plt.figure(figsize=(6, 4))
for mode in range(1, 9):
    k = 2 * np.pi * mode / length
    plt.semilogy(t, B_k[:, mode], color=plt.cm.viridis(mode / 8),
                 ls="-" if k < k_c else ":", label=fr"$k/k_c={k / k_c:.2f}$")
    print(f"mode {mode}: k/k_c = {k / k_c:.2f}  {'unstable' if k < k_c else 'stable  '}"
          f"  |B_y| grew by {B_k[-1, mode] / B_k[0, mode]:8.2f}")
plt.xlabel(r"$t\,\omega_{pe}$"); plt.ylabel(r"$|B_{y,k}|$ (T)")
plt.title("solid: below the cutoff, dotted: above it")
plt.legend(frameon=False, fontsize=7, ncol=2); plt.tight_layout(); plt.show()
