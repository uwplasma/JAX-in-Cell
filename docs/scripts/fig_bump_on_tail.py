"""Bump-on-tail instability: growth of the fastest mode against linear theory,
and the flattening of the beam into a plateau.

The four populations of examples/bump-on-tail.toml are loaded with a quiet start
(equally spaced positions, Maxwellian velocity components taken at quantiles of a
bit-reversed sequence) and the bulk electrons are displaced at the wavenumber of
the fastest-growing mode. Started from particle noise instead, as the input file
does, every mode emerges from the noise floor and saturates about two e-foldings
later, which leaves no exponential phase to fit; the quiet start lowers that floor
by orders of magnitude and gives the seeded mode room to grow."""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready
from scipy.special import erfinv

from common import (C_ELECTRONS, C_FIT, C_THEORY, panel_label, phase_space_scatter, record,
                    robust_growth_fit, savefig, silence_progress_bars, species_for_linear_theory)
from dispersion import electrostatic_epsilon, most_unstable_root
from jaxincell import (Simulation, diagnostics, mass_electron, mass_proton, speed_of_light)

silence_progress_bars()

L, NX, STEPS, COURANT, N, AK = 1.0, 70, 500, 3.0, 12000, 1e-3
MODE = 7                                    # fastest growing mode of this box
VTH_E, VTH_BN = 0.07071067812, 0.0016
VTH_ION = VTH_E * np.sqrt(mass_electron / mass_proton)      # T_i = T_e
DRIFT_BULK, DRIFT_BEAM = -2.25e6, 7.5e7
k = 2 * np.pi * MODE / L


def van_der_corput(n, base):
    """First n terms of the base-b van der Corput sequence, in (0, 1)."""
    q, denominator, i = np.zeros(n), 1.0, np.arange(1, n + 1)
    while i.any():
        denominator *= base
        q += (i % base) / denominator
        i //= base
    return q


def phase_space(n, drift, vth_over_c, base, offset=0.0, seed_ak=0.0):
    """Quiet start: equally spaced positions, Maxwellian velocities at quantiles."""
    x = np.linspace(-L / 2, L / 2, n, endpoint=False) + L / (2 * n) + offset
    if seed_ak:
        x = x + (seed_ak / k) * np.sin(k * x)
    v = drift + vth_over_c * speed_of_light * erfinv(2 * van_der_corput(n, base) - 1)
    return (np.stack([x, np.zeros(n), np.zeros(n)], 1),
            np.stack([v, np.zeros(n), np.zeros(n)], 1))


# The bulk pair shares positions so that it starts locally neutral, and the seed
# displaces its electrons only; the beam pair is offset by half a spacing.
xe, ve = phase_space(N, DRIFT_BULK, VTH_E, 2, seed_ak=AK)
xi, vi = phase_space(N, 0.0, VTH_ION, 3)
xb, vb = phase_space(N, DRIFT_BEAM, VTH_E, 5, offset=L / (2 * N))
xn, vn = phase_space(N, 0.0, VTH_BN, 7, offset=L / (2 * N))

parameters = {
    "domain_parameters": {"length": L, "timestep_over_spatialstep_times_c": COURANT,
                          "number_grid_points": NX, "total_steps": STEPS},
    "species_parameters": {
        "electrons": {
            "electrons0": {"number_pseudoparticles": N, "grid_points_per_Debye_length": 2.565,
                           "vth_over_c_x": VTH_E, "drift_speed_x": DRIFT_BULK,
                           # the first electron population defaults to a +/- split;
                           # the loaded velocities are a single drifting Maxwellian
                           "velocity_plus_minus_x": False,
                           "initial_positions": xe, "initial_velocities": ve},
            "beam": {"number_pseudoparticles": N, "grid_points_per_Debye_length": 0.44427103214,
                     "vth_over_c_x": VTH_E, "drift_speed_x": DRIFT_BEAM,
                     "initial_positions": xb, "initial_velocities": vb}},
        "ions": {
            "ions0": {"number_pseudoparticles": N, "grid_points_per_Debye_length": 2.565,
                      "mass_over_proton_mass": 1, "vth_over_c_x": "_electrons0",
                      "ion_temperature_over_electron_temperature_x": 1.0,
                      "initial_positions": xi, "initial_velocities": vi},
            "beam_neutralizer": {"number_pseudoparticles": N,
                                 "grid_points_per_Debye_length": 0.44427103214,
                                 "mass_over_proton_mass": 1, "vth_over_c_x": VTH_BN,
                                 "initial_positions": xn, "initial_velocities": vn}}},
    "solver_parameters": {"field_solver": 0, "filter_passes": 0, "print_info": False},
}

output = block_until_ready(Simulation(parameters).run())
diagnostics(output)
wpe = float(output["plasma_frequency"])
t = np.asarray(output["time_array"]) * wpe
Ex = np.asarray(output["electric_field"][:, :, 0])
mode_amplitude = np.abs(np.fft.rfft(Ex, axis=1))[:, MODE] / Ex.shape[1]
mode_real = (np.fft.rfft(Ex, axis=1) / Ex.shape[1])[:, MODE].real

populations = species_for_linear_theory(output)
root = most_unstable_root(lambda w: electrostatic_epsilon(w, k, populations),
                          (0.5, 1.5), (0.005, 0.3), n_real=25, n_imag=15, scale=wpe)
gamma_theory, omega_theory = root.imag / wpe, root.real / wpe

fit = robust_growth_fit(t, mode_amplitude ** 2)
if fit is None:
    raise SystemExit("no window met the fitting criterion; inspect the run before publishing")

window = (t >= fit["t0"]) & (t <= fit["t1"])
crossings = np.where(np.diff(np.sign(mode_real[window])) != 0)[0]
omega_measured = (np.pi / np.mean(np.diff(t[window][crossings]))
                  if len(crossings) > 2 else np.nan)

record(bump_on_tail_mode=MODE,
       bump_on_tail_kc_over_wpe=float(k * speed_of_light / wpe),
       bump_on_tail_gamma_theory=gamma_theory, bump_on_tail_gamma_measured=fit["gamma"],
       bump_on_tail_gamma_deviation_percent=100 * abs(fit["gamma"] - gamma_theory) / gamma_theory,
       bump_on_tail_omega_deviation_percent=100 * abs(omega_measured - omega_theory) / omega_theory,
       bump_on_tail_beam_percent=100 * (
           sum(pop["density"] for pop in populations if pop["name"].endswith("beam"))
           / sum(pop["density"] for pop in populations if pop["name"].endswith("electrons0"))),
       bump_on_tail_omega_theory=omega_theory, bump_on_tail_omega_measured=float(omega_measured),
       bump_on_tail_fit_r2=fit["r2"], bump_on_tail_fit_efolds=fit["efolds"],
       bump_on_tail_fit_window=[fit["t0"], fit["t1"]],
       bump_on_tail_particles=N, bump_on_tail_seed_ak=AK,
       bump_on_tail_beam_fraction=(
           sum(pop["density"] for pop in populations if pop["name"].endswith("beam"))
           / sum(pop["density"] for pop in populations if pop["name"].endswith("electrons0"))),
       bump_on_tail_energy_error=float(np.max(np.abs(
           output["total_energy"] / output["total_energy"][0] - 1))))

# ---------------------------------------------------------------------------
x_e = np.asarray(output["position_electrons"][:, :, 0])
v_e = np.asarray(output["velocity_electrons"][:, :, 0]) / speed_of_light
charges = np.asarray(output["charge_integer_lookup"])[np.asarray(output["species_integer_index"])]
weights = np.asarray(output["weights"]).reshape(-1)[charges < 0]

fig = plt.figure(figsize=(7.4, 5.6))
gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)

ax = fig.add_subplot(gs[0, 0])
bins = np.linspace(-0.3, 0.45, 161)
centres = 0.5 * (bins[1:] + bins[:-1])
for index, label, colour, style in ((0, "initial", C_THEORY, "--"),
                                    (len(t) - 1, rf"$t\,\omega_{{pe}} = {t[-1]:.0f}$", C_ELECTRONS, "-")):
    f, _ = np.histogram(v_e[index], bins=bins, weights=weights, density=True)
    ax.semilogy(centres, f, color=colour, ls=style, label=label)
ax.set_xlabel(r"$v_x / c$")
ax.set_ylabel(r"$f_e(v_x)$  (arbitrary units)")
ax.set_ylim(bottom=1e-2)
ax.legend(loc="upper right", fontsize=8)
panel_label(ax, "(a)")

ax = fig.add_subplot(gs[0, 1])
ax.semilogy(t, mode_amplitude, color=C_ELECTRONS, label=rf"mode {MODE} of $E_x$")
tt = np.linspace(fit["t0"], fit["t1"], 40)
ax.semilogy(tt, np.exp(0.5 * (fit["intercept"] + fit["slope"] * tt)), color=C_FIT, lw=2.4,
            alpha=0.85, label=rf"fit: $\gamma = {fit['gamma']:.4f}\,\omega_{{pe}}$")
anchor = np.exp(0.5 * (fit["intercept"] + fit["slope"] * fit["t0"]))
ax.semilogy(tt, anchor * np.exp(gamma_theory * (tt - fit["t0"])), ls="--", color=C_THEORY,
            label=rf"theory: $\gamma = {gamma_theory:.4f}\,\omega_{{pe}}$")
ax.axvspan(fit["t0"], fit["t1"], color="#EEEEEE", zorder=0)
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$|\hat E_x(k)|$  (V/m)")
ax.legend(loc="lower right", fontsize=7.5)
panel_label(ax, "(b)")

ax = fig.add_subplot(gs[1, 0])
i_end = int(np.argmin(np.abs(t - fit["t1"])))
phase_space_scatter(ax, x_e[i_end], v_e[i_end], L, 0.45, size=0.6)
ax.set_title(rf"end of linear phase, $t\,\omega_{{pe}} = {t[i_end]:.0f}$", fontsize=9)
ax.set_xlabel("x / L")
ax.set_ylabel(r"$v_x / c$")
panel_label(ax, "(c)")

ax = fig.add_subplot(gs[1, 1])
phase_space_scatter(ax, x_e[-1], v_e[-1], L, 0.45, size=0.6)
ax.set_title(rf"saturated, $t\,\omega_{{pe}} = {t[-1]:.0f}$", fontsize=9)
ax.set_xlabel("x / L")
ax.set_ylabel(r"$v_x / c$")
panel_label(ax, "(d)")
savefig(fig, "bump_on_tail")
