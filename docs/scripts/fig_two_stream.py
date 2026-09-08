"""Two-stream instability: growth and saturation, the electron phase space, and
a scan of the growth rate against the kinetic dispersion relation."""
import matplotlib.pyplot as plt
import numpy as np
from common import (C_ELECTRONS, C_FIT, C_THEORY, SINGLE, WIDE, maxwellian_populations, panel_label,
                    phase_space_hist, record, savefig)
from dispersion import electrostatic_epsilon, purely_growing_roots

from jaxincell import (Domain, Simulation, Solver, Species, diagnostics, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

LENGTH, CELLS, DENSITY, DRIFT, VTH = 0.01, 64, 4.37e17, 5e7, 0.05 * c
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
SEED_AK = 1e-4


def build(drift=DRIFT, n=20000):
    """Two counter-streaming quiet beams of electrons on mobile protons, with
    mode 1 seeded by a displacement of amplitude a k = 1e-4."""
    electrons = Species.electrons(n=n, density=DENSITY, vth=(VTH, 0, 0), drift=(drift, 0, 0),
                                  plus_minus=True, quiet=True, perturbation_mode=1,
                                  perturbation_amplitude=SEED_AK * LENGTH / (2 * np.pi))
    ions = Species.ions(n=n // 2, density=DENSITY, electrons=electrons, quiet=True)
    return Simulation(Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=4.5), [electrons, ions],
                      Solver(filter_passes=0))


def theory(drift):
    """Growth rate of mode 1 from the kinetic dispersion relation, in units of
    omega_pe. Empty when the mode is stable."""
    populations = maxwellian_populations(build(drift))
    roots = purely_growing_roots(lambda w: electrostatic_epsilon(w, 2 * np.pi / LENGTH, populations), OMEGA_PE)
    return max(roots) / OMEGA_PE if roots else np.nan


def measure(out):
    """Rate of the seeded mode, fitted between ten times the seed amplitude and a
    tenth of saturation. The window is set by amplitude rather than by time so
    that it follows the same part of the growth at every drift."""
    t = np.asarray(out.t) * OMEGA_PE
    amplitude = np.abs(np.fft.rfft(np.asarray(out.E[:, :, 0]), axis=1)[:, 1]) / CELLS
    peak = int(np.argmax(amplitude))
    window = ((amplitude > 10 * amplitude[0]) & (amplitude < 0.1 * amplitude[peak])
              & (np.arange(t.size) < peak))
    if window.sum() < 10:
        return t, amplitude, None
    slope, intercept = np.polyfit(t[window], np.log(amplitude[window]), 1)
    fit = np.polyval((slope, intercept), t[window])
    r2 = 1 - np.var(np.log(amplitude[window]) - fit) / np.var(np.log(amplitude[window]))
    return t, amplitude, {"gamma": slope, "intercept": intercept, "r2": r2,
                          "t0": t[window][0], "t1": t[window][-1]}


simulation = build()
output = simulation.run(900, seed=0, store_every=2)
t, amplitude, fit = measure(output)
gamma_theory = theory(DRIFT)

fig, axes = plt.subplots(1, 2, figsize=WIDE)
axes[0].semilogy(t, amplitude, color=C_ELECTRONS, label=r"$|E_{k=1}(t)|$")
span = np.linspace(fit["t0"], fit["t1"], 2)
axes[0].semilogy(span, np.exp(fit["intercept"] + fit["gamma"] * span), "--", color=C_FIT,
                 label=fr"fit: $\gamma={fit['gamma']:.3f}\,\omega_{{pe}}$")
axes[0].semilogy(span, np.exp(fit["intercept"] + gamma_theory * span), ":", color=C_THEORY,
                 label=fr"kinetic: $\gamma={gamma_theory:.3f}\,\omega_{{pe}}$")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_{k=1}|$ (V/m)", title="the seeded mode")
axes[0].legend(loc="lower right")
panel_label(axes[0], "a")

x_e, v_e = output.particles("electrons")
image = phase_space_hist(axes[1], np.asarray(x_e[-1, :, 0]), np.asarray(v_e[-1, :, 0]), LENGTH, 2.4 * DRIFT)
fig.colorbar(image, ax=axes[1], pad=0.02).set_label("pseudo-particles per bin")
axes[1].set(xlabel="$x/L$", ylabel="$v_x$ (m/s)",
            title=fr"electron phase space, $t\,\omega_{{pe}}={t[-1]:.0f}$")
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "two_stream")

drifts = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]) * 1e7
measured, predicted = [], []
for drift in drifts:
    _, _, result = measure(build(drift).run(900, seed=0, store_particles=False))
    measured.append(result["gamma"] if result else np.nan)
    predicted.append(theory(drift))
    print(f"  drift {drift:.2e} m/s (k v0/omega_pe {2 * np.pi / LENGTH * drift / OMEGA_PE:.3f}): "
          f"measured {measured[-1]:.4f}, kinetic {predicted[-1]:.4f}, "
          f"R2 {result['r2'] if result else float('nan'):.4f}")
measured, predicted = np.array(measured), np.array(predicted)

fine = np.linspace(2.0e7, 6.2e7, 22)
fig, ax = plt.subplots(figsize=SINGLE)
ax.plot(2 * np.pi / LENGTH * fine / OMEGA_PE, [theory(d) for d in fine], "-", color=C_THEORY,
        label="kinetic theory")
ax.plot(2 * np.pi / LENGTH * drifts / OMEGA_PE, measured, "o", color=C_ELECTRONS, label="JAX-in-Cell")
ax.axvline(1.0, color="0.7", lw=0.8)
ax.text(1.01, 0.05, "cold-beam cutoff", rotation=90, fontsize=7.5, color="0.4", transform=ax.get_xaxis_transform())
ax.set(xlabel=r"$k v_0/\omega_{pe}$", ylabel=r"$\gamma/\omega_{pe}$", title="growth rate of the seeded mode")
ax.legend()
fig.tight_layout()
savefig(fig, "two_stream_scan")

deviation = 100 * np.abs(measured - predicted) / predicted
energy = np.asarray(diagnostics(output)["total"])
record(two_stream_gamma_measured=round(float(fit["gamma"]), 4),
       two_stream_gamma_theory=round(float(gamma_theory), 4),
       two_stream_gamma_deviation_percent=round(float(100 * abs(fit["gamma"] - gamma_theory) / gamma_theory), 1),
       two_stream_fit_r2=round(float(fit["r2"]), 4),
       two_stream_fit_window=f"{fit['t0']:.0f}-{fit['t1']:.0f}",
       two_stream_particles=int(2 * 20000 + 10000),
       two_stream_seed_ak=SEED_AK,
       two_stream_cells=CELLS,
       two_stream_k_v0_over_wpe=round(float(DRIFT * 2 * np.pi / LENGTH / OMEGA_PE), 3),
       two_stream_omega_pe=f"{OMEGA_PE:.3e}",
       two_stream_omega_pe_dt=round(float(OMEGA_PE * simulation.domain.dt), 4),
       two_stream_dx_over_debye=round(float(simulation.domain.dx / simulation.debye_length()), 3),
       two_stream_energy_error=f"{float(np.max(np.abs(energy / energy[0] - 1))):.1e}",
       two_stream_scan_mean_deviation_percent=round(float(np.nanmean(deviation)), 1),
       two_stream_scan_max_deviation_percent=round(float(np.nanmax(deviation)), 1))
