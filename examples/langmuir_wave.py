"""Langmuir waves and the kinetic dispersion relation.

A small density perturbation of a Maxwellian oscillates at the real part of the
least-damped root of the kinetic dispersion relation (Landau, J. Phys. USSR 10, 25, 1946)

    1 + [1 + zeta Z(zeta)] / (k lambda_D)^2 = 0,        zeta = omega / (k v_th).

At small k lambda_D it reduces to the fluid result of Bohm and Gross (Phys. Rev. 75,
1851, 1949), omega^2 = omega_pe^2 (1 + 3 k^2 lambda_D^2), and the two part as k lambda_D
grows: the kinetic root is 1.5 % above the fluid frequency at 0.25 and 2.9 % above it at
0.3. This scans k lambda_D and prints the measured frequency against both.

The grid shifts the frequency by a known amount. Depositing and gathering with the
quadratic spline, S(k) = sinc^3(k dx/2), and the staggered Gauss law, which replaces k by
K = (2/dx) sin(k dx/2), make a cold plasma oscillate at omega_pe S(k) sqrt(k/K): 0.4 %
low with 32 cells per wavelength. The comparison includes that factor.

The frequency is measured from the spacing of the maxima of |E_k(t)|, which are half a
period apart. A seed of a k = 1e-2 followed for 1600 steps keeps the wave far enough
above the particle noise for that; with a tenth of the seed the noise moves the maxima by
a per cent or more at k lambda_D >= 0.25.
"""

import os

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

# Real part of the least-damped kinetic root, omega/omega_pe, from the Faddeeva function
# (docs/scripts/dispersion.py, landau_root)
KINETIC = {0.05: 1.0038, 0.10: 1.0152, 0.15: 1.0348, 0.20: 1.0640, 0.25: 1.1057, 0.30: 1.1598}

length, cells, seed_ak, steps = 1.0, 32, 1e-2, 1600
k = 2 * np.pi / length
omega_pe = 0.05 * c * cells / length                 # gives omega_pe * dt = 0.05 at dt = dx / c
density = omega_pe ** 2 * epsilon_0 * mass_electron / e_charge ** 2
theta = k * (length / cells) / 2
grid = np.sqrt((np.sin(theta) / theta) ** 6 * theta / np.sin(theta))

print(f"grid factor S(k) sqrt(k/K) = {grid:.4f}")
print("k lambda_D   measured   kinetic x grid   deviation   Bohm-Gross x grid")
measured = []
for kld, kinetic in KINETIC.items():
    electrons = Species.electrons(n=20000, density=density, vth=(kld / k * np.sqrt(2) * omega_pe, 0, 0),
                                  quiet=True, perturbation_amplitude=seed_ak / k, perturbation_mode=1)
    ions = Species.ions(n=5000, density=density, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
    out = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=1.0), [electrons, ions],
                     Solver(filter_passes=0)).run(steps, seed=0, store_particles=False)
    t = np.asarray(out.t) * omega_pe
    a = np.abs(np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1])
    i = np.arange(1, t.size - 1)
    peaks = i[(a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])]
    measured.append(np.pi / np.mean(np.diff(t[peaks])))
    print(f"   {kld:.2f}       {measured[-1]:.4f}       {kinetic * grid:.4f}        "
          f"{100 * (measured[-1] / (kinetic * grid) - 1):+.2f} %        {np.sqrt(1 + 3 * kld ** 2) * grid:.4f}")

fine = np.linspace(0, 0.32, 100)
plt.figure(figsize=(5.5, 4))
plt.plot(fine, np.sqrt(1 + 3 * fine ** 2), "k--", label=r"Bohm-Gross, $\omega^2=\omega_{pe}^2(1+3k^2\lambda_D^2)$")
plt.plot(list(KINETIC), list(KINETIC.values()), "ks", mfc="none", label="kinetic root")
plt.plot(list(KINETIC), np.array(measured) / grid, "o", label="JAX-in-Cell, grid factor divided out")
plt.xlabel(r"$k\lambda_D$")
plt.ylabel(r"$\omega/\omega_{pe}$")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
