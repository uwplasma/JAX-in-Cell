"""Two-stream instability: growth of the electrostatic energy against the kinetic
dispersion relation, convergence of the measured growth rate with the number of
pseudo-particles, and the phase-space evolution. Uses examples/input.toml with the
particle count as the only change."""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready

from common import (C_ELECTRONS, C_FIT, C_THEORY, EXAMPLES_DIR, fit_growth_rate, panel_label,
                    phase_space_scatter, quiet_parameters, record, savefig, silence_progress_bars,
                    species_for_linear_theory)
from dispersion import electrostatic_epsilon, most_unstable_root
from jaxincell import Simulation, diagnostics, load_parameters, speed_of_light

silence_progress_bars()
parameters = quiet_parameters(load_parameters(EXAMPLES_DIR / "input.toml"))
N_EXAMPLE = parameters["species_parameters"]["electrons"]["electrons0"]["number_pseudoparticles"]
N_MAIN = 4 * N_EXAMPLE
DRIFTS_OVER_C = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26]


def mode_window(t, mode_energy, noise_factor=30.0, top=0.03):
    """Exponential-growth window of one Fourier mode: from the last time the mode
    energy was below ``noise_factor`` times its initial level to the last time it was
    below ``top`` times its peak."""
    i_peak = int(np.argmax(mode_energy))
    noise = mode_energy[:20].mean()
    lo = np.where(mode_energy[:i_peak] < noise_factor * noise)[0]
    hi = np.where(mode_energy[:i_peak] < top * mode_energy[i_peak])[0]
    t0 = t[lo[-1]] if lo.size else t[0]
    t1 = t[hi[-1]] if hi.size else t[max(i_peak - 1, 1)]
    if t1 <= t0:
        t0, t1 = t[max(i_peak // 4, 1)], t[max(i_peak - 1, 2)]
    return t0, t1


for species_type, label in (("electrons", "electrons0"), ("ions", "ions0")):
    parameters["species_parameters"][species_type][label]["number_pseudoparticles"] = N_MAIN
sim = Simulation(parameters)


def analyse(output):
    """Growth rate of mode 1 from its Fourier amplitude, and the theory root."""
    wpe = float(output["plasma_frequency"])
    t = np.asarray(output["time_array"]) * wpe
    L = float(output["length"])
    Ex = np.asarray(output["electric_field"][:, :, 0])
    mode_energy = (np.abs(np.fft.rfft(Ex, axis=1))[:, 1] / Ex.shape[1]) ** 2
    t0, t1 = mode_window(t, mode_energy)
    gamma_fit, intercept, slope = fit_growth_rate(t, mode_energy, t0, t1)
    populations = species_for_linear_theory(output)
    root = most_unstable_root(lambda w: electrostatic_epsilon(w, 2 * np.pi / L, populations),
                              (-0.5, 0.5), (0.01, 1.0), n_real=21, n_imag=20, scale=wpe)
    gamma_th = root.imag / wpe if root is not None else np.nan
    e_folds = 0.5 * np.log(mode_energy.max() / mode_energy[:20].mean())
    return t, gamma_fit, gamma_th, (t0, t1, intercept, slope), e_folds


# Drift-speed scan through the runtime inputs: one compiled program for all runs.
scan_measured, scan_theory = [], []
for v_over_c in DRIFTS_OVER_C:
    output = block_until_ready(sim.run({"electrons": {"electrons0": {"drift_speed_x": v_over_c * speed_of_light}}}))
    t, gamma_fit, gamma_th, window, e_folds = analyse(output)
    usable = np.isfinite(gamma_th) and gamma_th > 0.02 and e_folds > 2.0 and window[1] - window[0] > 5.0
    scan_measured.append(gamma_fit if usable else np.nan)
    scan_theory.append(gamma_th)
    print(f"v_d/c = {v_over_c:.2f}: theory {gamma_th:.4f}, measured {gamma_fit:.4f}, "
          f"window [{window[0]:.1f}, {window[1]:.1f}], {e_folds:.1f} e-folds{'' if usable else ' (not used)'}")

output = block_until_ready(sim.run())   # the example drift, 0.2 c
t, main_gamma, gamma_theory_check, main_window, _ = analyse(output)
diagnostics(output)
wpe = float(output["plasma_frequency"])
energy = np.asarray(output["electric_field_energy"])
L = float(output["length"])
k = 2 * np.pi / L
populations = species_for_linear_theory(output)
root = most_unstable_root(lambda w: electrostatic_epsilon(w, k, populations), (-0.5, 0.5), (0.02, 1.0), scale=wpe)
gamma_theory = root.imag / wpe
t0, t1, _, _ = main_window
gamma_energy, intercept, slope = fit_growth_rate(t, energy, t0, t1)
scan_measured, scan_theory = np.array(scan_measured), np.array(scan_theory)
ok = np.isfinite(scan_measured)
record(two_stream_gamma_theory=gamma_theory, two_stream_gamma_measured=main_gamma,
       two_stream_gamma_measured_energy=gamma_energy,
       two_stream_particles_main=N_MAIN, two_stream_particles_example=N_EXAMPLE,
       two_stream_scan_drifts_over_c=DRIFTS_OVER_C,
       two_stream_scan_gamma_theory=[float(g) for g in scan_theory],
       two_stream_scan_gamma_measured=[float(g) for g in scan_measured],
       two_stream_scan_max_relative_deviation=float(np.max(np.abs(scan_measured[ok] - scan_theory[ok]) / scan_theory[ok])),
       two_stream_scan_mean_relative_deviation=float(np.mean(np.abs(scan_measured[ok] - scan_theory[ok]) / scan_theory[ok])),
       two_stream_k_lambda_D=k * float(output["dx"]) / float(np.asarray(
           parameters["species_parameters"]["electrons"]["electrons0"]["grid_points_per_Debye_length"])),
       two_stream_k_vd_over_wpe=k * 6e7 / wpe,
       two_stream_fit_window=[float(t0), float(t1)], two_stream_omega_pe=wpe,
       two_stream_omega_pe_dt=float(output["dt"]) * wpe)

fig = plt.figure(figsize=(7.4, 5.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9], hspace=0.55, wspace=0.4)
ax = fig.add_subplot(gs[0, :2])
ax.semilogy(t, energy, color=C_ELECTRONS, label=f"simulation, {N_MAIN} particles per species")
tt = np.linspace(t0, t1, 50)
ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=2.4, alpha=0.85,
            label=rf"fit: $\gamma = {gamma_energy:.3f}\,\omega_{{pe}}$")
anchor = np.exp(intercept + slope * t0)
ax.semilogy(tt, anchor * np.exp(2 * gamma_theory * (tt - t0)), ls="--", color=C_THEORY,
            label=rf"kinetic theory: $\gamma = {gamma_theory:.3f}\,\omega_{{pe}}$")
ax.axvspan(t0, t1, color="#EEEEEE", zorder=0)
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.legend(loc="lower right", fontsize=8)
panel_label(ax, "(a)", x=-0.1)

ax = fig.add_subplot(gs[0, 2])
drifts = np.array(DRIFTS_OVER_C)
ax.plot(drifts, scan_theory, ls="--", color=C_THEORY, label="kinetic theory")
ax.plot(drifts[ok], scan_measured[ok], "o", ms=4.5, color=C_ELECTRONS, label="simulation")
ax.set_xlabel(r"drift speed $v_d / c$")
ax.set_ylabel(r"$\gamma / \omega_{pe}$")
ax.set_ylim(bottom=0)
ax.legend(loc="lower left", fontsize=8)
panel_label(ax, "(b)", x=-0.3)

vmax = 0.45
x_e = np.asarray(output["position_electrons"][:, :, 0])
v_e = np.asarray(output["velocity_electrons"][:, :, 0]) / speed_of_light
i_lin = int(np.argmin(np.abs(t - t1)))
snapshots = [(0, "initial"), (i_lin, "end of linear phase"), (len(t) - 1, "saturated")]
for col, (i, title) in enumerate(snapshots):
    axp = fig.add_subplot(gs[1, col])
    phase_space_scatter(axp, x_e[i], v_e[i], L, vmax, size=0.8)
    axp.set_title(f"{title}\n" + rf"$t\,\omega_{{pe}} = {t[i]:.0f}$", fontsize=9)
    axp.set_xlabel("x / L")
    if col == 0:
        axp.set_ylabel(r"$v_x / c$")
    else:
        axp.set_yticklabels([])
    panel_label(axp, f"({'cde'[col]})", x=-0.22)
savefig(fig, "two_stream")
