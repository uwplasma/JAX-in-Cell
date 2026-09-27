"""Weibel instability (Weibel, Phys. Rev. Lett. 2, 83, 1959).

A plasma hotter across the simulation axis than along it is unstable to purely
growing transverse magnetic perturbations. Setting omega = 0 in the transverse
dispersion relation gives the marginal wavenumber

    k_c c = omega_pe sqrt(T_z/T_x - 1),

and below it every mode grows at the rate of the kinetic root that
`jaxincell.theory.weibel_rate` solves for. Nothing is seeded: every mode starts from
the particle noise of random velocities on equally spaced positions. Two boxes:

* four marginal wavelengths, the threshold test: modes below k_c grow, those above do not;
* twelve marginal wavelengths and a long run (the full preset), where the growth of
  each mode is fitted over the linear phase and compared mode by mode with the kinetic
  root, and the run goes on to saturation.

The linear phase is one window for every mode: from t omega_pe = 20, once the noise
has settled into its growing root, to when the total magnetic energy reaches 5 % of its
maximum. A gain against the start is no growth rate once a mode has saturated, which in
the wide box the long wavelengths do before the ones near the cutoff have grown. Modes
whose fit leaves R^2 < 0.8 are drawn open and not compared: near the cutoff the growth is
too slow to leave the noise in the window.

Run with `--quick` for a smoke preset (fewer particles, a six-wavelength box, a shorter
run): same physics, more noise, not a measurement.
"""

import os
import sys
from pathlib import Path

# Double precision, like the rest of the examples and the documentation figure, which runs this
# script. The answers are fitted slopes and ratios of amplitudes rather than differences of large
# numbers, so single precision (JAX_ENABLE_X64=0) gives the same gains to five digits and is faster.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, figure, mass_electron, save_run,
                       quiet_start, elementary_charge as e_charge, speed_of_light as c)
from jaxincell.theory import weibel_rate

# --- what to change ------------------------------------------------------------------
quick = "--quick" in sys.argv
ratio = 25.0                          # anisotropy T_z / T_x
density = 1e15                        # m^-3
vth_x = 0.02 * c                      # thermal speed along x, v_th = sqrt(2 T / m)
narrow = dict(wavelengths=4, particles=12000 if quick else 40000, cells=128, steps=4000)
wide = dict(wavelengths=6 if quick else 12, particles=24000 if quick else 120000,
            cells=142 if quick else 284, steps=3000 if quick else 12000)
store_every = 20                      # field samples; the particle history is not kept (memory)
fit_start, fit_energy = 20.0, 0.05    # the linear window: t omega_pe > 20, until |B|^2 = 5 % of its max
good_fit = 0.8                        # R^2 below which a mode is not compared

# --- derived scales ------------------------------------------------------------------
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
k_c = np.sqrt(ratio - 1) * omega_pe / c
vth = (vth_x, 0.0, vth_x * np.sqrt(ratio))
if quick:
    print("--quick is a smoke preset: fewer particles and a shorter, six-wavelength wide box;\n"
          "the rates it prints are noisier and are not the documented measurement.")


def run(wavelengths, particles, cells, steps):
    """A bi-Maxwellian of electrons on immobile ions, velocities at random; |B_{y,k}(t)|."""
    length = wavelengths * 2 * np.pi / k_c
    x, _ = quiet_start(particles, length, vth=vth)
    v = np.random.default_rng(0).standard_normal((particles, 3)) * np.asarray(vth) / np.sqrt(2)
    electrons = Species.electrons(n=particles, density=density, vth=vth).replace(x=x, v=v)
    ions = Species.ions(n=particles // 4, density=density, mass_ratio=1e6, vth=(0, 0, 0), sampling="quiet")
    simulation = Simulation(Domain(length=length, cells=cells, dt_over_dx_c=0.5), [electrons, ions],
                            Solver(filter_passes=0))
    output = simulation.run(steps, seed=0, store_every=store_every, store_particles=False)
    t = np.asarray(output.t) * omega_pe
    B_k = np.abs(np.fft.rfft(np.asarray(output.B[:, :, 1]), axis=1))
    return t, B_k, 2 * np.pi * np.arange(B_k.shape[1]) / length


# (a) the threshold: four wavelengths, gain of the last sample over the median of the first five
t_a, B_a, k_a = run(**narrow)
modes_a = np.arange(1, 9)
gain = B_a[-1, modes_a] / np.median(B_a[:5, modes_a], axis=0)
below = k_a[modes_a] < k_c
print(f"four wavelengths: smallest gain below k_c {gain[below].min():.2f}, largest above {gain[~below].max():.2f}")

# (b) mode by mode: twelve wavelengths, each mode fitted over the same linear window
t_b, B_b, k_b = run(**wide)
energy = (B_b[:, 1:] ** 2).sum(axis=1)
window = (t_b > fit_start) & (t_b < t_b[np.argmax(energy > fit_energy * energy.max())])
modes_b = np.arange(1, int(1.2 * wide["wavelengths"]) + 1)
measured, r2, theory = [], [], []
for mode in modes_b:
    y = np.log(B_b[window, mode])
    slope, intercept = np.polyfit(t_b[window], y, 1)
    measured.append(slope)
    r2.append(1 - np.var(y - np.polyval((slope, intercept), t_b[window])) / np.var(y))
    theory.append(weibel_rate(k_b[mode], omega_pe, vth_x, ratio) / omega_pe)
measured, r2, theory = map(np.array, (measured, r2, theory))
compared = (r2 >= good_fit) & (theory > 0)
deviation = 100 * np.abs(measured - theory)[compared] / theory[compared]
print(f"linear window t omega_pe = {t_b[window][0]:.0f}-{t_b[window][-1]:.0f}")
for mode, k, g, g0, q in zip(modes_b, k_b[modes_b] / k_c, measured, theory, r2):
    print(f"  mode {mode:2d}  k/k_c = {k:.3f}  measured {g:+.4f}  kinetic {g0:.4f}  R2 {q:.3f}")
print(f"measured against kinetic: {deviation.mean():.1f} % mean, {deviation.max():.1f} % worst over "
      f"{compared.sum()} of {int((theory > 0).sum())} unstable modes")

# --- the figure ----------------------------------------------------------------------------------
fig, axes = figure(3)
for mode in modes_a:
    axes[0].semilogy(t_a, B_a[:, mode], color=plt.cm.viridis(0.1 + 0.8 * mode / 8), lw=2,
                     ls="-" if k_a[mode] < k_c else ":", label=fr"$k/k_c={k_a[mode] / k_c:.2f}$")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|B_{y,k}|$ (T)", title="four wavelengths: the cutoff")
axes[0].set_ylim(top=3e3 * B_a[:, modes_a].max())       # room for the legend above the curves
axes[0].legend(ncol=2, fontsize=13, loc="upper left")
shown = [m for m in modes_b if compared[m - 1]][::2]
for i, mode in enumerate(shown):
    color = plt.cm.viridis(0.1 + 0.8 * i / max(len(shown) - 1, 1))
    axes[1].semilogy(t_b, B_b[:, mode], color=color, lw=2, label=fr"$k/k_c={k_b[mode] / k_c:.2f}$")
    axes[1].semilogy(t_b[window], np.exp(np.polyval(np.polyfit(t_b[window], np.log(B_b[window, mode]), 1),
                                                    t_b[window])), "k--", lw=1.5)
axes[1].axvspan(t_b[window][0], t_b[window][-1], color="0.9", zorder=0)
axes[1].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|B_{y,k}|$ (T)",
            title=f"{wide['wavelengths']} wavelengths: fits in the linear window")
axes[1].set_ylim(top=30 * B_b[:, shown].max())
axes[1].legend(ncol=2, fontsize="small", loc="upper left")
fine = np.linspace(0.02, 1.3, 80)
axes[2].plot(fine, [weibel_rate(f * k_c, omega_pe, vth_x, ratio) / omega_pe for f in fine], "k-",
             label="kinetic root")
kk = k_b[modes_b] / k_c
axes[2].plot(kk[compared], measured[compared], "o", ms=9, label="fitted, $R^2\\geq0.8$")
axes[2].plot(kk[~compared], measured[~compared], "o", ms=9, mfc="none", label="not compared")
axes[2].axvline(1.0, color="0.7", lw=2)
axes[2].set(xlabel=r"$k/k_c$", ylabel=r"$\gamma/\omega_{pe}$", title="growth rate, mode by mode")
axes[2].legend()
fig.tight_layout()

# --- the record --------------------------------------------------------------------------------
settings = dict(ratio=ratio, density=density, vth_x=vth_x, narrow=narrow, wide=wide, store_every=store_every,
                fit_start=fit_start, fit_energy=fit_energy, good_fit=good_fit, quick=quick)
summary = dict(gain_min_below_cutoff=float(gain[below].min()), gain_max_above_cutoff=float(gain[~below].max()),
               window=[float(t_b[window][0]), float(t_b[window][-1])],
               k_over_kc=kk.tolist(), measured=measured.tolist(), kinetic=theory.tolist(), r2=r2.tolist(),
               modes_compared=int(compared.sum()), mean_deviation_percent=float(deviation.mean()),
               max_deviation_percent=float(deviation.max()))
save_run(Path.cwd() / ("weibel_quick" if quick else "weibel"), "weibel", settings, summary, figure=fig,
         t_narrow=t_a, B_narrow=B_a, t_wide=t_b, B_wide=B_b)
plt.show()
