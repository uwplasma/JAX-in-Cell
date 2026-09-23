"""Landau damping of a Langmuir wave.

Two figures. ``landau_damping.png`` is a linear-theory check with a quiet start
(deterministic Maxwellian velocities in bit-reversed order, supplied through
``initial_velocities``; see ``landau.py``) and a small perturbation, so that the
damping rate and the frequency can be compared with the exact kinetic root.
``landau_damping_example.png`` runs examples/Landau_damping.py as written, where the
perturbation is large enough for particle trapping to modify the damping."""
import numpy as np
from jax import block_until_ready

from common import (C_ELECTRONS, C_FIT, C_THEORY, figure, fit_growth_rate, panel_label, record,
                    savefig, silence_progress_bars)
from landau import (MODE, base_parameters, gamma_theory, k, k_lambda_D, maxima_above_floor,
                    mode_one, omega_theory, quiet_parameters)
from jaxincell import Simulation, diagnostics

silence_progress_bars()

# --- 1. Linear regime with a quiet start ---------------------------------------
N_QUIET = 300000
A_K = 0.01
output = block_until_ready(Simulation(quiet_parameters(N_QUIET, A_K, 600)).run())
diagnostics(output)
wpe = float(output["plasma_frequency"])
t = np.asarray(output["time_array"]) * wpe
energy = np.asarray(output["electric_field_energy"])
E1 = mode_one(output)
amplitude = np.abs(E1)
peaks, floor = maxima_above_floor(amplitude)
t_pk = t[peaks]
gamma_fit, intercept, slope = fit_growth_rate(t_pk, energy[peaks], t_pk[0], t_pk[-1])
signal = E1.real if np.abs(E1.real).max() > np.abs(E1.imag).max() else E1.imag
mask = (t >= 0.2) & (t <= t_pk[-1])
crossings = np.where(np.diff(np.sign(signal[mask])) != 0)[0]
omega_fit = np.pi / np.mean(np.diff(t[mask][crossings]))
record(landau_k_lambda_D=k_lambda_D, landau_gamma_theory=gamma_theory, landau_gamma_measured=gamma_fit,
       landau_omega_theory=omega_theory, landau_omega_measured=omega_fit,
       landau_quiet_particles=N_QUIET, landau_quiet_a_k=A_K, landau_quiet_bounce_over_wpe=float(np.sqrt(A_K)),
       landau_quiet_peaks_used=len(peaks), landau_quiet_efolds=float(np.log(amplitude[0] / floor)),
       landau_quiet_energy_error=float(np.max(np.abs(output["total_energy"] / output["total_energy"][0] - 1))))

fig, axes = figure(2, 1, gridspec_kw={"wspace": 0.3})
ax = axes[0]
ax.semilogy(t, energy, color=C_ELECTRONS, label="simulation")
ax.semilogy(t_pk, energy[peaks], "o", color=C_FIT, label="maxima used for the fit")
tt = np.linspace(t_pk[0], t_pk[-1], 50)
ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=6, alpha=0.6,
            label=rf"fit: $\gamma = {gamma_fit:.3f}\,\omega_{{pe}}$")
ax.semilogy(tt, energy[peaks[0]] * np.exp(2 * gamma_theory * (tt - t_pk[0])), ls="--", color=C_THEORY,
            label=rf"theory: $\gamma = {gamma_theory:.3f}\,\omega_{{pe}}$")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.set_xlim(0, 40)
ax.set_ylim(top=30 * energy.max())
ax.legend(loc="upper right")
panel_label(ax, "(a)")
ax = axes[1]
norm = np.abs(signal[:5]).max()
ax.plot(t, signal / norm, color=C_ELECTRONS, label=r"mode 1 of $E_x$")
envelope = np.exp(gamma_theory * t)
ax.plot(t, envelope, ls="--", color=C_THEORY, label=r"theory, $e^{\gamma t}$")
ax.plot(t, -envelope, ls="--", color=C_THEORY)
ax.set_xlim(0, 25)
ax.set_ylim(-1.45, 1.5)
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\hat E_1(t) / \hat E_1(0)$")
ax.legend(loc="upper right", ncol=2)
ax.text(0.03, 0.04, rf"$\omega_r/\omega_{{pe}}$: fit {omega_fit:.3f}, theory {omega_theory:.3f}",
        transform=ax.transAxes)
panel_label(ax, "(b)")
savefig(fig, "landau_damping")

# --- 2. The example as written (large amplitude) --------------------------------
example = base_parameters(300, {"number_pseudoparticles": 40000, "perturbation_amplitude_x": 0.025,
                                "perturbation_wavenumber_x": MODE, "drift_speed_x": 0.0,
                                "velocity_plus_minus_x": False})
example["solver_parameters"]["filter_passes"] = 5
output = block_until_ready(Simulation(example).run())
diagnostics(output)
wpe = float(output["plasma_frequency"])
t = np.asarray(output["time_array"]) * wpe
energy = np.asarray(output["electric_field_energy"])
a_k_example = 0.025 * k
peaks = [i for i in range(1, len(t) - 1) if energy[i] > energy[i - 1] and energy[i] > energy[i + 1] and 1.0 <= t[i] <= 12.0]
gamma_example, intercept, slope = fit_growth_rate(t[peaks], energy[peaks], t[peaks][0], t[peaks][-1])
record(landau_example_a_k=a_k_example, landau_example_bounce_over_wpe=float(np.sqrt(a_k_example)),
       landau_example_gamma_measured=gamma_example,
       landau_example_dominant_frequency_over_wpe=float(output["dominant_frequency"]) / wpe)

fig, ax = figure()
ax.semilogy(t, energy, color=C_ELECTRONS, label="examples/Landau_damping.py")
tt = np.linspace(t[peaks][0], t[peaks][-1], 50)
ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=6, alpha=0.6,
            label=rf"fit through maxima: $\gamma = {gamma_example:.3f}\,\omega_{{pe}}$")
ax.semilogy(tt, energy[peaks][0] * np.exp(2 * gamma_theory * (tt - tt[0])), ls="--", color=C_THEORY,
            label=rf"linear theory: $\gamma = {gamma_theory:.3f}\,\omega_{{pe}}$")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.set_ylim(top=100 * energy.max())
ax.legend(loc="upper right")
savefig(fig, "landau_damping_example")
