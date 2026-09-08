"""Landau damping at k lambda_D = 0.5 and the Bohm-Gross dispersion relation."""
import matplotlib.pyplot as plt
import numpy as np
from common import (C_ELECTRONS, C_FIT, C_THEORY, WIDE, maxima, panel_label,
                    rate_and_frequency, record, savefig)
from dispersion import landau_root

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

LENGTH, CELLS, PARTICLES, SEED_AK = 1.0, 64, 150000, 0.01
K = 2 * np.pi / LENGTH
OMEGA_PE = 0.05 * c * CELLS / LENGTH                     # gives omega_pe dt = 0.05 at dt = dx/c
DENSITY = OMEGA_PE ** 2 * epsilon_0 * mass_electron / e_charge ** 2


def run(k_lambda_d, particles=PARTICLES, seed_ak=SEED_AK, steps=500):
    electrons = Species.electrons(n=particles, density=DENSITY, quiet=True,
                                  vth=(k_lambda_d / K * np.sqrt(2) * OMEGA_PE, 0, 0),
                                  perturbation_amplitude=seed_ak / K, perturbation_mode=1)
    ions = Species.ions(n=particles // 8, density=DENSITY, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
    out = Simulation(Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=1.0), [electrons, ions],
                     Solver(filter_passes=0)).run(steps, seed=0, store_particles=False)
    amplitude = np.abs(np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1]) / CELLS
    return np.asarray(out.t) * OMEGA_PE, amplitude


t, amplitude = run(0.5)
floor = amplitude[int(0.8 * amplitude.size):].mean()
gamma, omega, peaks = rate_and_frequency(t, amplitude, above=5 * floor)
root = landau_root(0.5)

fig, axes = plt.subplots(1, 2, figsize=WIDE)
axes[0].semilogy(t, amplitude, color=C_ELECTRONS, lw=1.0, label=r"$|E_{k}(t)|$")
axes[0].semilogy(t[peaks], amplitude[peaks], "o", ms=4, color=C_FIT, label="maxima used")
span = np.linspace(t[peaks][0], t[peaks][-1], 2)
axes[0].semilogy(span, np.exp(np.polyval(np.polyfit(t[peaks], np.log(amplitude[peaks]), 1), span)),
                 "--", color=C_FIT, label=fr"fit: $\gamma={gamma:.4f}\,\omega_{{pe}}$")
axes[0].semilogy(span, amplitude[peaks][0] * np.exp(root.imag * (span - span[0])), ":", color=C_THEORY,
                 label=fr"kinetic: $\gamma={root.imag:.4f}\,\omega_{{pe}}$")
axes[0].axhline(floor, color="0.6", lw=0.8, label="noise floor")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_k|$ (V/m)",
            title=r"Landau damping, $k\lambda_D=0.5$", ylim=(0.3 * floor, 3 * amplitude.max()))
axes[0].legend(loc="lower left", ncol=2)
panel_label(axes[0], "a")

scan = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
measured = []
for k_lambda_d in scan:
    ts, amp = run(k_lambda_d)
    # successive maxima of |E_k| are half a period apart; keep the leading run of
    # them that still stands well above the noise the wave eventually decays into
    tops = maxima(amp)
    alive = amp[tops] > 0.05 * amp[tops][0]
    tops = tops[: int(np.argmin(alive))] if not alive.all() else tops
    omega_k = np.pi / np.mean(np.diff(ts[tops]))
    measured.append(omega_k)
    print(f"  k lambda_D {k_lambda_d:.2f}: omega/omega_pe {omega_k:.4f}, "
          f"Bohm-Gross {np.sqrt(1 + 3 * k_lambda_d ** 2):.4f}, kinetic {landau_root(k_lambda_d).real:.4f}")
fine = np.linspace(0.02, 0.52, 60)
axes[1].plot(fine, np.sqrt(1 + 3 * fine ** 2), "--", color="0.5", label=r"Bohm-Gross")
axes[1].plot(fine, [landau_root(k).real for k in fine], "-", color=C_THEORY, label="kinetic root")
axes[1].plot(scan, measured, "o", color=C_ELECTRONS, label="JAX-in-Cell")
axes[1].set(xlabel=r"$k\lambda_D$", ylabel=r"$\omega/\omega_{pe}$", title="Langmuir wave frequency")
axes[1].legend(loc="upper left")
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "landau_damping")

kinetic = np.array([landau_root(k).real for k in scan])
record(landau_k_lambda_D=0.5,
       landau_gamma_measured=round(float(gamma), 4), landau_gamma_theory=round(float(root.imag), 4),
       landau_gamma_deviation_percent=round(float(100 * abs(gamma - root.imag) / abs(root.imag)), 1),
       landau_omega_measured=round(float(omega), 4), landau_omega_theory=round(float(root.real), 4),
       landau_omega_deviation_percent=round(float(100 * abs(omega - root.real) / root.real), 1),
       landau_particles=PARTICLES, landau_seed_ak=SEED_AK, landau_cells=CELLS,
       landau_peaks_used=int(peaks.size), landau_omega_pe_dt=0.05,
       landau_dispersion_max_deviation_percent=round(float(np.max(100 * np.abs(np.array(measured) - kinetic) / kinetic)), 1))
