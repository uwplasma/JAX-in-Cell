"""Bump-on-tail instability with two electron populations. Uses
examples/bump-on-tail.toml (explicit scheme, periodic boundaries)."""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready

from common import (C_ELECTRONS, C_FIT, C_THEORY, C_IONS, EXAMPLES_DIR, fit_growth_rate,
                    linear_window, panel_label, phase_space_hist, quiet_parameters, record,
                    savefig, silence_progress_bars, species_for_linear_theory)
from dispersion import electrostatic_epsilon, most_unstable_root
from jaxincell import Simulation, diagnostics, load_parameters, speed_of_light

silence_progress_bars()
parameters = quiet_parameters(load_parameters(EXAMPLES_DIR / "bump-on-tail.toml"))
sim = Simulation(parameters)
output = block_until_ready(sim.run())
diagnostics(output)

wpe = float(output["plasma_frequency"])
t = np.asarray(output["time_array"]) * wpe
L = float(output["length"])
energy = np.asarray(output["electric_field_energy"])
populations = species_for_linear_theory(output)
modes = np.arange(1, 16)
gamma_theory, omega_theory = [], []
for m in modes:
    k = 2 * np.pi * m / L
    func = lambda w, k=k: electrostatic_epsilon(w, k, populations)
    root = most_unstable_root(func, (0.5, 1.5), (0.005, 0.3), n_real=25, n_imag=15, scale=wpe)
    gamma_theory.append(root.imag / wpe if root is not None else np.nan)
    omega_theory.append(root.real / wpe if root is not None else np.nan)
gamma_theory = np.array(gamma_theory)
m_fastest = int(modes[np.nanargmax(gamma_theory)])
Ex_k = np.abs(np.fft.rfft(np.asarray(output["electric_field"][:, :, 0]), axis=1))
mode_energy = Ex_k[:, m_fastest] ** 2
t0, t1 = linear_window(t, mode_energy, lower=1e-3, upper=2e-1)
gamma_fit, intercept, slope = fit_growth_rate(t, energy, t0, t1)
gamma_mode, _, _ = fit_growth_rate(t, mode_energy, t0, t1)
beam_fraction = populations[1]["density"] / populations[0]["density"]
record(bump_on_tail_fastest_mode=m_fastest, bump_on_tail_gamma_theory=float(np.nanmax(gamma_theory)),
       bump_on_tail_omega_theory=float(omega_theory[m_fastest - 1]),
       bump_on_tail_gamma_energy_fit=gamma_fit, bump_on_tail_gamma_mode_fit=gamma_mode,
       bump_on_tail_fit_window=[float(t0), float(t1)], bump_on_tail_beam_fraction=beam_fraction,
       bump_on_tail_kc_over_wpe=2 * np.pi * m_fastest / L * speed_of_light / wpe)

x_e = np.asarray(output["position_electrons"][:, :, 0])
v_e = np.asarray(output["velocity_electrons"][:, :, 0]) / speed_of_light
q_all = np.asarray(output["charge_integer_lookup"])[np.asarray(output["species_integer_index"])]
w_e = np.asarray(output["weights"]).reshape(-1)[q_all < 0]

fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(-0.3, 0.5, 161)
for i, label, color, ls in ((0, "t = 0", C_THEORY, "--"), (len(t) - 1, rf"$t\,\omega_{{pe}} = {t[-1]:.0f}$", C_ELECTRONS, "-")):
    f, edges = np.histogram(v_e[i], bins=bins, weights=w_e, density=True)
    ax.semilogy(0.5 * (edges[1:] + edges[:-1]), f, color=color, ls=ls, label=label)
ax.set_xlabel(r"$v_x / c$")
ax.set_ylabel(r"$f_e(v_x)$ (arbitrary units)")
ax.set_ylim(bottom=1e-2)
ax.legend(loc="upper right")
panel_label(ax, "(a)")

ax = fig.add_subplot(gs[0, 1])
ax.semilogy(t, energy, color=C_ELECTRONS, label="simulation")
tt = np.linspace(t0, t1, 40)
ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=2.2, alpha=0.85,
            label=rf"fit: $\gamma = {gamma_fit:.3f}\,\omega_{{pe}}$")
ax.semilogy(tt, np.exp(intercept + slope * t0) * np.exp(2 * np.nanmax(gamma_theory) * (tt - t0)),
            ls="--", color=C_THEORY, label=rf"theory, mode {m_fastest}: $\gamma = {np.nanmax(gamma_theory):.3f}\,\omega_{{pe}}$")
ax.axvspan(t0, t1, color="#EEEEEE", zorder=0)
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.legend(loc="lower right", fontsize=7.5)
panel_label(ax, "(b)")

for col, (i, title) in enumerate(((int(np.argmin(np.abs(t - t1))), "end of linear phase"), (len(t) - 1, "nonlinear phase"))):
    ax = fig.add_subplot(gs[1, col])
    im = phase_space_hist(ax, x_e[i], v_e[i], L, 0.45, weights=w_e, bins=(70, 90))
    ax.set_title(rf"{title}, $t\,\omega_{{pe}} = {t[i]:.0f}$")
    ax.set_xlabel("x / L")
    ax.set_ylabel(r"$v_x / c$")
    panel_label(ax, f"({'cd'[col]})")
cb = fig.colorbar(im, ax=fig.axes[2:], pad=0.02, fraction=0.03)
cb.set_label("electron density (weighted counts)")
savefig(fig, "bump_on_tail")
