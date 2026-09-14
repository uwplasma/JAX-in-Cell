"""Weibel instability: the marginal wavenumber in a box that holds several
wavelengths, and the growth rate against the transverse kinetic dispersion
relation from single-mode runs."""
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_THEORY, WIDE, panel_label, record, savefig
from dispersion import plasma_frequency, purely_growing_roots, weibel_dispersion

from jaxincell import (Domain, Simulation, Solver, Species, diagnostics, mass_electron, quiet_start,
                       elementary_charge as e_charge, speed_of_light as c)

RATIO, DENSITY, VTH_X = 25.0, 1e15, 0.02 * c
OMEGA_PE = plasma_frequency(DENSITY, e_charge, mass_electron)
K_C = np.sqrt(RATIO - 1) * OMEGA_PE / c          # k_c c = omega_pe sqrt(T_z/T_x - 1)
VTH = (VTH_X, 0.0, VTH_X * np.sqrt(RATIO))
POPULATIONS = [{"wp": OMEGA_PE, "vthx": VTH_X, "A": RATIO}]


def theory(k):
    roots = purely_growing_roots(lambda w: weibel_dispersion(w, k, POPULATIONS), OMEGA_PE, gamma_max=0.3)
    return max(roots) / OMEGA_PE if roots else 0.0


def simulate(length, cells, steps, n=20000, seed_amplitude=0.0, seed_mode=1, store_every=10,
             store_particles=False, quiet=True):
    """A bi-Maxwellian, optionally with a coherent transverse current
    v_z += d v_thz sin(k x) that seeds one magnetic mode.

    A quiet start suppresses the noise so that a seeded mode can be followed for
    many e-foldings, but it leaves no noise floor for the unseeded modes to grow
    out of, so the survey of the cutoff uses random velocities instead.
    """
    x, v = quiet_start(n, length, vth=VTH)
    if not quiet:
        v = np.random.default_rng(0).standard_normal((n, 3)) * np.asarray(VTH) / np.sqrt(2)
    v[:, 2] += seed_amplitude * VTH[2] * np.sin(2 * np.pi * seed_mode * x[:, 0] / length)
    electrons = Species.electrons(n=n, density=DENSITY, vth=VTH).replace(x=x, v=v)
    ions = Species.ions(n=n // 4, density=DENSITY, mass_ratio=1e6, vth=(0, 0, 0), quiet=True)
    return Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.5), [electrons, ions],
                      Solver(filter_passes=0)).run(steps, seed=0, store_every=store_every,
                                                   store_particles=store_particles)


def fit(t, amplitude):
    """Rate between three times the seed and a fifth of saturation, with the
    coefficient of determination that decides whether the mode grew cleanly."""
    peak = int(np.argmax(amplitude))
    window = ((amplitude > 3 * amplitude[0]) & (amplitude < 0.2 * amplitude[peak])
              & (np.arange(t.size) < peak))
    if window.sum() < 8:
        return np.nan, 0.0
    slope, intercept = np.polyfit(t[window], np.log(amplitude[window]), 1)
    residual = np.log(amplitude[window]) - np.polyval((slope, intercept), t[window])
    return slope, 1 - np.var(residual) / np.var(np.log(amplitude[window]))


# (a) several wavelengths in one box, nothing seeded: the cutoff sorts the modes
length = 4.0 * 2 * np.pi / K_C
output = simulate(length, cells=128, steps=4000, n=40000, store_every=40,
                  store_particles=True, quiet=False)
t = np.asarray(output.t) * OMEGA_PE
B_k = np.abs(np.fft.rfft(np.asarray(output.B[:, :, 1]), axis=1))
modes = np.arange(1, 9)
# against the median of the first samples: a single sample can land on a zero
gain = B_k[-1, modes] / np.median(B_k[:5, modes], axis=0)
unstable = 2 * np.pi * modes / length < K_C

fig, axes = plt.subplots(1, 2, figsize=WIDE)
for mode in modes:
    k = 2 * np.pi * mode / length
    axes[0].semilogy(t, B_k[:, mode], color=plt.cm.viridis(0.1 + 0.8 * mode / 8),
                     ls="-" if k < K_C else ":", lw=1.2, label=fr"$k/k_c={k / K_C:.2f}$")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|B_{y,k}|$ (T)",
            title="unseeded: solid below the cutoff, dotted above")
axes[0].legend(ncol=2, fontsize=7)
panel_label(axes[0], "a")

# (b) one wavelength per box, one mode seeded: the rate can be measured
fractions = np.array([0.15, 0.2, 0.3, 0.35, 0.5, 0.65, 0.8])
measured, quality = [], []
for fraction in fractions:
    out = simulate(2 * np.pi / (fraction * K_C), cells=64, steps=6000, seed_amplitude=1e-3)
    amplitude = np.abs(np.fft.rfft(np.asarray(out.B[:, :, 1]), axis=1)[:, 1])
    rate, r2 = fit(np.asarray(out.t) * OMEGA_PE, amplitude)
    measured.append(rate)
    quality.append(r2)
    print(f"  k/k_c {fraction:.2f}: measured {rate:.4f}, kinetic {theory(fraction * K_C):.4f}, R2 {r2:.3f}")
measured, quality = np.array(measured), np.array(quality)
predicted = np.array([theory(f * K_C) for f in fractions])
clean = quality > 0.85

fine = np.linspace(0.05, 1.6, 60)
axes[1].plot(fine, [theory(f * K_C) for f in fine], "-", color=C_THEORY, label="kinetic theory")
axes[1].plot(fractions[clean], measured[clean], "o", color=C_ELECTRONS, label="JAX-in-Cell")
axes[1].plot(fractions[~clean], measured[~clean], "o", mfc="none", color=C_ELECTRONS,
             label=r"$R^2<0.85$, excluded")
axes[1].axvline(1.0, color="0.7", lw=0.8)
axes[1].text(1.02, 0.55, r"$k_c c=\omega_{pe}\sqrt{T_z/T_x-1}$", rotation=90, fontsize=7.5,
             color="0.4", transform=axes[1].get_xaxis_transform())
axes[1].set(xlabel=r"$k/k_c$", ylabel=r"$\gamma/\omega_{pe}$", title="seeded single-mode runs")
axes[1].legend()
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "weibel")

deviation = 100 * np.abs(measured[clean] - predicted[clean]) / predicted[clean]
energy = np.asarray(diagnostics(output)["total"])
record(weibel_anisotropy=RATIO, weibel_particles=40000, weibel_cells=128, weibel_steps=4000,
       weibel_courant=0.5, weibel_t_end=round(float(t[-1]), 0),
       weibel_kc_c_over_wpe=round(float(np.sqrt(RATIO - 1)), 3),
       weibel_seed_amplitude=1e-3,
       weibel_modes_compared=int(clean.sum()), weibel_modes_run=len(fractions),
       weibel_mean_deviation_percent=round(float(np.mean(deviation)), 1),
       weibel_max_deviation_percent=round(float(np.max(deviation)), 1),
       weibel_gain_min_unstable=round(float(gain[unstable].min()), 1),
       weibel_gain_max_stable=round(float(gain[~unstable].max()), 2),
       weibel_gamma_max_theory=round(float(max(predicted)), 4),
       weibel_energy_error=f"{float(np.max(np.abs(energy / energy[0] - 1))):.1e}")
