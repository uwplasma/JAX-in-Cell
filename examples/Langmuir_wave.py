"""Bohm-Gross dispersion relation (Bohm and Gross, Phys. Rev. 75, 1851, 1949).

A warm plasma oscillates at omega^2 = omega_pe^2 (1 + 3 k^2 lambda_D^2). The
frequency is measured from the spacing of the maxima of |E_k(t)|, which are half
a period apart, for a handful of k lambda_D.
"""
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

length, cells = 1.0, 32
k = 2 * np.pi / length
omega_pe = 0.05 * c * cells / length
density = omega_pe ** 2 * epsilon_0 * mass_electron / e_charge ** 2

measured = []
k_lambda_d = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
for kld in k_lambda_d:
    electrons = Species.electrons(n=20000, density=density, vth=(kld / k * np.sqrt(2) * omega_pe, 0, 0),
                                  quiet=True, perturbation_amplitude=1e-3 / k, perturbation_mode=1)
    ions = Species.ions(n=5000, density=density, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
    out = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0), [electrons, ions],
                     Solver(filter_passes=0)).run(800, seed=0, store_particles=False)
    t = np.asarray(out.t) * omega_pe
    a = np.abs(np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1])
    i = np.arange(1, t.size - 1)
    peaks = i[(a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])]
    measured.append(np.pi / np.mean(np.diff(t[peaks])))
    print(f"k lambda_D = {kld:.2f}:  omega/omega_pe measured {measured[-1]:.4f}, "
          f"theory {np.sqrt(1 + 3 * kld ** 2):.4f}")

fine = np.linspace(0, 0.32, 100)
plt.figure(figsize=(5.5, 4))
plt.plot(fine, np.sqrt(1 + 3 * fine ** 2), "k-", label=r"$\omega^2=\omega_{pe}^2(1+3k^2\lambda_D^2)$")
plt.plot(k_lambda_d, measured, "o", label="JAX-in-Cell")
plt.xlabel(r"$k\lambda_D$"); plt.ylabel(r"$\omega/\omega_{pe}$")
plt.legend(frameon=False); plt.tight_layout(); plt.show()
