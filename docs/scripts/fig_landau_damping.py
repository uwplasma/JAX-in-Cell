"""Landau damping of a Langmuir wave.

Two figures. ``landau_damping.png`` is a linear-theory check with a quiet start
(deterministic Maxwellian velocities in bit-reversed order, supplied through
``initial_velocities``) and a small perturbation, so that the damping rate and the
frequency can be compared with the exact kinetic root. ``landau_damping_example.png``
runs examples/Landau_damping.py as written, where the perturbation is large enough
for particle trapping to modify the damping."""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready
from scipy.special import erfinv

from common import (C_ELECTRONS, C_FIT, C_THEORY, WIDE, fit_growth_rate, panel_label, record,
                    savefig, silence_progress_bars)
from dispersion import landau_root
from jaxincell import Simulation, diagnostics, speed_of_light

silence_progress_bars()

LENGTH = 1.0
GRID = 32
MODE = 1.02
GRID_POINTS_PER_DEBYE = 0.4
VTH_OVER_C = 0.35
k = 2 * np.pi * MODE / LENGTH
lambda_D = LENGTH / GRID / GRID_POINTS_PER_DEBYE
k_lambda_D = k * lambda_D
root = landau_root(k_lambda_D)
gamma_theory, omega_theory = root.imag, root.real


def base_parameters(steps, electrons):
    return {
        "domain_parameters": {"length": LENGTH, "timestep_over_spatialstep_times_c": 1.0,
                              "number_grid_points": GRID, "total_steps": steps},
        "species_parameters": {
            "electrons": {"electrons0": {"grid_points_per_Debye_length": GRID_POINTS_PER_DEBYE,
                                         "vth_over_c_x": VTH_OVER_C, **electrons}},
            "ions": {"ions0": {"number_pseudoparticles": 40000, "grid_points_per_Debye_length": GRID_POINTS_PER_DEBYE,
                               "mass_over_proton_mass": 1e9, "vth_over_c_x": "_electrons0",
                               "vth_over_c_y": "_electrons0", "vth_over_c_z": "_electrons0",
                               "ion_temperature_over_electron_temperature_x": 1e-9}}},
        "solver_parameters": {"field_solver": 0, "time_evolution_algorithm": 0, "print_info": False,
                              "filter_passes": 0},
    }


def van_der_corput(n, base=2):
    """First n terms of the base-2 van der Corput sequence, in (0, 1)."""
    q = np.zeros(n)
    denominator = np.ones(n)
    i = np.arange(1, n + 1)
    while i.any():
        denominator *= base
        q += (i % base) / denominator
        i //= base
    return q


def quiet_start(n, a_k):
    """Equally spaced positions with a sinusoidal displacement of relative
    amplitude a k, and Maxwellian velocities taken at the quantiles of a bit-reversed
    sequence so that neighbouring particles have very different velocities."""
    a = a_k / k
    x0 = np.linspace(-LENGTH / 2, LENGTH / 2, n, endpoint=False) + LENGTH / (2 * n)
    x = x0 + a * np.sin(k * x0)
    sigma = VTH_OVER_C * speed_of_light / np.sqrt(2)
    v = sigma * np.sqrt(2) * erfinv(2 * van_der_corput(n) - 1)
    positions = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    velocities = np.stack([v, np.zeros(n), np.zeros(n)], axis=1)
    return positions, velocities


def mode_one(output):
    Ex = np.asarray(output["electric_field"][:, :, 0])
    return np.fft.rfft(Ex, axis=1)[:, 1] / Ex.shape[1]


# --- 1. Linear regime with a quiet start ---------------------------------------
N_QUIET = 300000
A_K = 0.01
positions, velocities = quiet_start(N_QUIET, A_K)
parameters = base_parameters(600, {"number_pseudoparticles": N_QUIET,
                                   "initial_positions": positions, "initial_velocities": velocities})
output = block_until_ready(Simulation(parameters).run())
diagnostics(output)
wpe = float(output["plasma_frequency"])
t = np.asarray(output["time_array"]) * wpe
energy = np.asarray(output["electric_field_energy"])
E1 = mode_one(output)
amplitude = np.abs(E1)
floor = amplitude[-100:].mean()
peaks = [i for i in range(1, len(t) - 1) if amplitude[i] > amplitude[i - 1] and amplitude[i] > amplitude[i + 1]]
peaks = [i for i in peaks if amplitude[i] > 10 * floor]
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

fig, axes = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.32})
ax = axes[0]
ax.semilogy(t, energy, color=C_ELECTRONS, label="simulation")
ax.semilogy(t_pk, energy[peaks], "o", ms=3.5, color=C_FIT, label="maxima used for the fit")
tt = np.linspace(t_pk[0], t_pk[-1], 50)
ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=2.2, alpha=0.85,
            label=rf"fit: $\gamma = {gamma_fit:.3f}\,\omega_{{pe}}$")
ax.semilogy(tt, energy[peaks[0]] * np.exp(2 * gamma_theory * (tt - t_pk[0])), ls="--", color=C_THEORY,
            label=rf"theory: $\gamma = {gamma_theory:.3f}\,\omega_{{pe}}$")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.set_xlim(0, 40)
ax.legend(loc="upper right", fontsize=8)
panel_label(ax, "(a)")
ax = axes[1]
norm = np.abs(signal[:5]).max()
ax.plot(t, signal / norm, color=C_ELECTRONS, label=r"mode-1 amplitude of $E_x$")
envelope = np.exp(gamma_theory * t)
ax.plot(t, envelope, ls="--", color=C_THEORY, label=r"theory envelope $e^{\gamma t}$")
ax.plot(t, -envelope, ls="--", color=C_THEORY)
ax.set_xlim(0, 25)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\hat E_1(t) / \hat E_1(0)$")
ax.legend(loc="upper right", fontsize=8)
ax.text(0.03, 0.05, rf"$\omega_r/\omega_{{pe}}$: fit {omega_fit:.3f}, theory {omega_theory:.3f}",
        transform=ax.transAxes, fontsize=8.5)
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

fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.semilogy(t, energy, color=C_ELECTRONS, label="simulation (examples/Landau_damping.py)")
tt = np.linspace(t[peaks][0], t[peaks][-1], 50)
ax.semilogy(tt, np.exp(intercept + slope * tt), color=C_FIT, lw=2.2, alpha=0.85,
            label=rf"fit through maxima: $\gamma = {gamma_example:.3f}\,\omega_{{pe}}$")
ax.semilogy(tt, energy[peaks][0] * np.exp(2 * gamma_theory * (tt - tt[0])), ls="--", color=C_THEORY,
            label=rf"linear theory: $\gamma = {gamma_theory:.3f}\,\omega_{{pe}}$")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.legend(loc="upper right", fontsize=8)
savefig(fig, "landau_damping_example")
