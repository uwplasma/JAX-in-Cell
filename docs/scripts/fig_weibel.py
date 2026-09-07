"""Weibel instability driven by a temperature anisotropy T_z / T_x = 100.

``weibel.png`` is a linear-theory check: the configuration of
examples/Weibel_instability.py with ten times more pseudo-particles, the digital
filter switched off, half the Courant number and a longer run, so that the growth of
individual Fourier modes of B_y can be compared with the transverse kinetic
dispersion relation.
``weibel_example.png`` shows the example as written (3000 particles per species,
filter on); ``--example-only`` regenerates just that figure."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready

from common import (CMAP_SIGNED, C_ELECTRONS, C_FIT, C_THEORY, fit_growth_rate, panel_label, record,
                    savefig, silence_progress_bars, species_for_linear_theory)
from dispersion import most_unstable_root, weibel_dispersion
from jaxincell import Simulation, diagnostics, speed_of_light

silence_progress_bars()


def parameters_for(n_particles, steps, filter_passes, courant=1.0):
    return {
        "domain_parameters": {"length": 3e-1, "timestep_over_spatialstep_times_c": courant,
                              "number_grid_points": 150, "total_steps": steps},
        "species_parameters": {
            "electrons": {"electrons0": {
                "number_pseudoparticles": n_particles, "grid_points_per_Debye_length": 1.1,
                "perturbation_amplitude_x": 0, "perturbation_wavenumber_x": 0,
                "random_positions_x": True, "random_positions_y": True, "random_positions_z": True,
                "vth_over_c_x": 0.01, "vth_over_c_z": 0.10, "drift_speed_x": 0, "drift_speed_z": 0,
                "velocity_plus_minus_x": False, "velocity_plus_minus_z": False}},
            "ions": {"ions0": {
                "number_pseudoparticles": n_particles, "grid_points_per_Debye_length": 1.1,
                "random_positions_x": True, "random_positions_y": True, "random_positions_z": True,
                "vth_over_c_x": "_electrons0", "vth_over_c_y": "_electrons0", "vth_over_c_z": "_electrons0",
                "ion_temperature_over_electron_temperature_x": 1}}},
        "solver_parameters": {"field_solver": 0, "time_evolution_algorithm": 0, "relativistic": False,
                              "filter_passes": filter_passes, "print_info": False},
    }


def mode_window(t, mode_energy, noise_factor=30.0, top=0.03):
    i_peak = int(np.argmax(mode_energy))
    noise = mode_energy[:20].mean()
    lo = np.where(mode_energy[:i_peak] < noise_factor * noise)[0]
    hi = np.where(mode_energy[:i_peak] < top * mode_energy[i_peak])[0]
    t0 = t[lo[-1]] if lo.size else t[0]
    t1 = t[hi[-1]] if hi.size else t[max(i_peak - 1, 1)]
    return t0, t1


# --- 1. Linear-theory check (skipped with --example-only) ------------------------
N_CHECK, STEPS_CHECK, COURANT_CHECK = 30000, 6000, 0.5
if "--example-only" in sys.argv:
    output = None
else:
    output = block_until_ready(Simulation(parameters_for(N_CHECK, STEPS_CHECK, 0, COURANT_CHECK)).run())
if output is not None:
  diagnostics(output)
  wpe = float(output["plasma_frequency"])
  t = np.asarray(output["time_array"]) * wpe
  L = float(output["length"])
  x = np.asarray(output["grid"])
  By = np.asarray(output["magnetic_field"][:, :, 1])
  energy_B = np.asarray(output["magnetic_field_energy"])

  populations = species_for_linear_theory(output)
  theory_species = [{"wp": s["wp"], "vthx": s["vthx"], "A": (s["vthz"] / s["vthx"]) ** 2} for s in populations]
  modes = np.arange(1, 17)
  gamma_theory = []
  for m in modes:
      k = 2 * np.pi * m / L
      root = most_unstable_root(lambda w, k=k: weibel_dispersion(w, k, theory_species),
                                (-0.02, 0.02), (1e-4, 0.2), n_real=9, n_imag=30, scale=wpe)
      gamma_theory.append(root.imag / wpe if root is not None else np.nan)
  gamma_theory = np.array(gamma_theory)
  m_fastest = int(modes[np.nanargmax(gamma_theory)])

  By_k = np.abs(np.fft.rfft(By, axis=1)) / By.shape[1]
  gamma_measured, windows = [], []
  for m in modes:
      mode_energy = By_k[:, m] ** 2
      t0, t1 = mode_window(t, mode_energy)
      e_folds = 0.5 * np.log(mode_energy.max() / mode_energy[:20].mean())
      if t1 - t0 < 10.0 or e_folds < 2.0:
          gamma_measured.append(np.nan)
          windows.append((np.nan, np.nan))
          continue
      g, _, _ = fit_growth_rate(t, mode_energy, t0, t1)
      gamma_measured.append(g)
      windows.append((t0, t1))
      print(f"m = {m}: theory {gamma_theory[m - 1]:.4f}, measured {g:.4f}, window [{t0:.0f}, {t1:.0f}], {e_folds:.1f} e-folds")
  gamma_measured = np.array(gamma_measured)
  t0, t1 = windows[m_fastest - 1]
  if not np.isfinite(t0):
      t0, t1 = mode_window(t, energy_B)
  gamma_energy, intercept, slope = fit_growth_rate(t, energy_B, t0, t1)
  ok = np.isfinite(gamma_measured)
  record(weibel_fastest_mode=m_fastest, weibel_gamma_theory_max=float(np.nanmax(gamma_theory)),
         weibel_gamma_measured_fastest=float(gamma_measured[m_fastest - 1]),
         weibel_gamma_energy_fit=gamma_energy, weibel_fit_window=[float(t0), float(t1)],
         weibel_kc_over_wpe_fastest=2 * np.pi * m_fastest / L * speed_of_light / wpe,
         weibel_check_particles=N_CHECK, weibel_check_steps=STEPS_CHECK, weibel_check_courant=COURANT_CHECK,
         weibel_modes_compared=int(ok.sum()),
         weibel_max_relative_deviation=float(np.max(np.abs(gamma_measured[ok] - gamma_theory[ok]) / gamma_theory[ok])),
         weibel_mean_relative_deviation=float(np.mean(np.abs(gamma_measured[ok] - gamma_theory[ok]) / gamma_theory[ok])),
         weibel_anisotropy=float(theory_species[0]["A"]),
         weibel_energy_error=float(np.max(np.abs(output["total_energy"] / output["total_energy"][0] - 1))))

  fig = plt.figure(figsize=(7.4, 5.8))
  gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.5, wspace=0.35)
  ax = fig.add_subplot(gs[0, :])
  vlim = np.percentile(np.abs(By), 99.5)
  im = ax.pcolormesh(x, t, By, cmap=CMAP_SIGNED, vmin=-vlim, vmax=vlim, rasterized=True, shading="nearest")
  ax.grid(False)
  ax.set_xlabel("x (m)")
  ax.set_ylabel(r"$t\,\omega_{pe}$")
  cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
  cb.set_label(r"$B_y$ (T)")
  panel_label(ax, "(a)", x=-0.07)

  ax = fig.add_subplot(gs[1, 0])
  ax.semilogy(t, energy_B, color=C_ELECTRONS, label="simulation")
  tt = np.linspace(t0, t1, 40)
  ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=2.2, alpha=0.85,
              label=rf"fit: $\gamma = {gamma_energy:.4f}\,\omega_{{pe}}$")
  ax.semilogy(tt, np.exp(intercept + slope * t0) * np.exp(2 * np.nanmax(gamma_theory) * (tt - t0)),
              ls="--", color=C_THEORY, label=rf"theory, fastest mode: $\gamma = {np.nanmax(gamma_theory):.4f}\,\omega_{{pe}}$")
  ax.axvspan(t0, t1, color="#EEEEEE", zorder=0)
  ax.set_xlabel(r"$t\,\omega_{pe}$")
  ax.set_ylabel(r"$\frac{1}{2\mu_0}\int B^2\,dx$  (J/m$^2$)")
  ax.legend(loc="lower right", fontsize=7.5)
  panel_label(ax, "(b)")

  ax = fig.add_subplot(gs[1, 1])
  kc = 2 * np.pi * modes / L * speed_of_light / wpe
  ax.plot(kc, gamma_theory, ls="--", color=C_THEORY, label="linear theory")
  ax.plot(kc[ok], gamma_measured[ok], "o", ms=4, color=C_ELECTRONS, label="simulation, per mode")
  ax.set_xlabel(r"$k c / \omega_{pe}$")
  ax.set_ylabel(r"$\gamma / \omega_{pe}$")
  ax.set_ylim(bottom=0)
  ax.legend(loc="upper right", fontsize=8)
  panel_label(ax, "(c)")
  savefig(fig, "weibel")

# --- 2. The example as written --------------------------------------------------
output = block_until_ready(Simulation(parameters_for(3000, 2500, 5)).run())
diagnostics(output)
wpe = float(output["plasma_frequency"])
t = np.asarray(output["time_array"]) * wpe
x = np.asarray(output["grid"])
By = np.asarray(output["magnetic_field"][:, :, 1])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
ax = axes[0]
vlim = np.percentile(np.abs(By), 99.5)
im = ax.pcolormesh(x, t, By, cmap=CMAP_SIGNED, vmin=-vlim, vmax=vlim, rasterized=True, shading="nearest")
ax.grid(False)
ax.set_xlabel("x (m)")
ax.set_ylabel(r"$t\,\omega_{pe}$")
cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05)
cb.ax.set_title(r"$B_y$ (T)", fontsize=9)
panel_label(ax, "(a)", x=-0.12)
ax = axes[1]
ax.semilogy(t, np.asarray(output["magnetic_field_energy"]), color=C_ELECTRONS, label="magnetic")
ax.semilogy(t, np.asarray(output["electric_field_energy"]), color=C_FIT, label="electric")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"field energy (J/m$^2$)")
ax.legend(loc="lower right", fontsize=8)
panel_label(ax, "(b)")
savefig(fig, "weibel_example")
