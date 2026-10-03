"""Relativistic two-stream instability with the relativistic Boris pusher on and off.

Two cold electron beams drift through each other at +-v0 = +-0.8 c (Lorentz factor
5/3) over a cold proton background. The same input is run twice, with
``solver_parameters["relativistic"]`` set to True and to False, and everything else
identical. The box holds one wavelength of the fastest-growing mode of the cold
relativistic dispersion relation.

The pusher stores velocities v (not momenta p/m) in both cases, so the Lorentz
factor of each pseudo-particle is 1 / sqrt(1 - |v|^2 / c^2) and the pseudo-particle
mass (``output["masses"]``) already carries the weight. From the stored output the
script forms two total energies, both with the electromagnetic field energy:

* relativistic, sum (gamma - 1) m c^2, which the relativistic equations conserve;
* Newtonian, sum m |v|^2 / 2, which the non-relativistic equations conserve.

Panels: (a) electrostatic energy with the linear growth rates of the cold
relativistic and non-relativistic dispersion relations, (b) relative change of both
total energies for both pushers, (c), (d) electron phase space at saturation.
"""
import time

import numpy as np
from jax import block_until_ready

from common import (C_THEORY, COLORS, figure, fit_growth_rate, mode_window, panel_label, record,
                    savefig, silence_progress_bars)
from dispersion import electrostatic_epsilon, most_unstable_root, plasma_frequency
from jaxincell import Simulation, diagnostics, elementary_charge, mass_electron, speed_of_light

silence_progress_bars()
c = speed_of_light
V0_OVER_C = 0.8                 # beam drift speed
VTH_OVER_C = 0.01               # beam thermal speed, v_th = sqrt(2 T / m)
DENSITY = 1e18                  # total electron density, m^-3 (both beams)
N_PER_SPECIES = 10000           # pseudo-electrons (both beams) and pseudo-protons
GRID_POINTS = 128
C_DT_OVER_DX = 0.9
T_END = 150.0                   # in units of 1 / omega_pe
PERTURBATION = 1e-6             # initial displacement of the electrons, in units of L
RUNS = (("relativistic", True, "relativistic Boris", COLORS["vermillion"]),
        ("newtonian", False, "non-relativistic Boris", COLORS["blue"]))

gamma0 = 1.0 / np.sqrt(1.0 - V0_OVER_C**2)
v0 = V0_OVER_C * c
wpe = plasma_frequency(DENSITY, elementary_charge, mass_electron)
wb = wpe / np.sqrt(2.0)         # plasma frequency of one beam


def cold_roots(k, beam_frequency):
    """All roots of 1 = wb^2 [(w - k v0)^-2 + (w + k v0)^-2]: the quartic
    (w^2 - a^2)^2 - 2 wb^2 (w^2 + a^2) = 0 with a = k v0."""
    a = k * v0
    b = beam_frequency**2
    return np.roots([1.0, 0.0, -2.0 * (a**2 + b), 0.0, a**4 - 2.0 * b * a**2])


# Cold relativistic theory: for motion along the drift the longitudinal mass is
# gamma^3 m, so wb^2 -> wb^2 / gamma0^3. With x = w^2, the growing branch is
# x = a^2 + wb^2 - wb sqrt(wb^2 + 4 a^2), most negative at a^2 = 3 wb^2 / 4.
wb_rel = wb / gamma0**1.5
k = np.sqrt(3.0) / 2.0 * wb_rel / v0
L = 2.0 * np.pi / k
dx = L / GRID_POINTS
dt = C_DT_OVER_DX * dx / c
steps = int(round(T_END / (wpe * dt)))
# Debye length of the beams, lambda_D = v_th / (sqrt(2) omega_pe) with v_th = sqrt(2 T / m):
# the convention of the code (dx_over_Debye_length) and of the kinetic
# two-stream dispersion relation 1 + [2 + xi_1 Z(xi_1) + xi_2 Z(xi_2)] / (2 k^2 lambda_D^2) = 0.
debye_length = VTH_OVER_C * c / (np.sqrt(2.0) * wpe)

# Growth rate of every box mode k_m = m k, for both sets of equations. Only the
# first mode is unstable for the relativistic beams (a^2 < 2 wb^2 / gamma0^3);
# without the gamma0^3 the first three are, and the second grows fastest.
MODES = np.arange(1, 6)
theory = {}
for key, beam_frequency in (("relativistic", wb_rel), ("newtonian", wb)):
    cold = np.array([max(cold_roots(m * k, beam_frequency).imag.max(), 0.0) / wpe for m in MODES])
    m_fast = int(MODES[np.argmax(cold)])
    # Warm check: the kinetic dielectric of drifting Maxwellians in v. For a narrow
    # beam, d v / d p = 1 / (gamma^3 m), so to leading order in v_th / c the
    # relativistic beam is a Maxwellian in v with wp^2 -> wp^2 / gamma0^3.
    populations = [{"wp": beam_frequency, "u": s * v0, "vth": VTH_OVER_C * c} for s in (+1, -1)]
    root = most_unstable_root(lambda w: electrostatic_epsilon(w, m_fast * k, populations),
                              (-0.3, 0.3), (0.02, 0.5), n_real=13, n_imag=12, scale=wpe)
    theory[key] = {"modes": cold, "mode": m_fast, "cold": cold[m_fast - 1],
                   "warm": root.imag / wpe if root is not None else np.nan}
assert theory["relativistic"]["mode"] == 1
assert abs(theory["relativistic"]["cold"] - 0.5 * wb_rel / wpe) < 1e-8


def parameters(relativistic):
    beams = {"number_pseudoparticles": N_PER_SPECIES, "weight": DENSITY * L / N_PER_SPECIES,
             "charge_over_elementary_charge": -1, "perturbation_amplitude_x": PERTURBATION * L,
             "perturbation_wavenumber_x": 1, "vth_over_c_x": VTH_OVER_C, "drift_speed_x": v0,
             "velocity_plus_minus_x": True}
    protons = {"number_pseudoparticles": N_PER_SPECIES, "weight": DENSITY * L / N_PER_SPECIES,
               "charge_over_elementary_charge": 1, "mass_over_proton_mass": 1, "vth_over_c_x": 1e-4}
    return {"domain_parameters": {"length": L, "number_grid_points": GRID_POINTS,
                                  "timestep_over_spatialstep_times_c": C_DT_OVER_DX,
                                  "total_steps": steps},
            "solver_parameters": {"field_solver": 0, "time_evolution_algorithm": 0,
                                  "relativistic": relativistic, "print_info": False},
            "species_parameters": {"electrons": {"electrons0": beams}, "ions": {"ions0": protons}}}


results = {}
for key, relativistic, *_ in RUNS:
    start = time.perf_counter()
    output = block_until_ready(Simulation(parameters(relativistic)).run())
    seconds = time.perf_counter() - start
    mass = np.asarray(output["masses"]).reshape(-1)          # pseudo-particle masses, weights included
    electrons = np.asarray(output["charges"]).reshape(-1) < 0
    diagnostics(output)
    t = np.asarray(output["time_array"]) * wpe
    field = np.asarray(output["electric_field_energy"]) + np.asarray(output["magnetic_field_energy"])
    v = np.concatenate([np.asarray(output["velocity_electrons"]), np.asarray(output["velocity_ions"])], axis=1)
    mass = np.concatenate([mass[electrons], mass[~electrons]])
    v2 = np.sum(v**2, axis=-1)
    superluminal = v2 >= c**2
    with np.errstate(invalid="ignore", divide="ignore"):
        lorentz = 1.0 / np.sqrt(1.0 - v2 / c**2)
    # (gamma - 1) m c^2 written as m v^2 gamma^2 / (gamma + 1), which does not lose
    # digits to cancellation for the slow protons.
    kinetic_rel = np.sum(mass * v2 * lorentz**2 / (lorentz + 1.0), axis=1)
    kinetic_rel[superluminal.any(axis=1)] = np.nan              # undefined once any |v| >= c
    kinetic_newton = 0.5 * np.sum(mass * v2, axis=1)
    total_rel = kinetic_rel + field
    total_newton = kinetic_newton + field
    ve = np.asarray(output["velocity_electrons"][:, :, 0]) / c
    Ex = np.asarray(output["electric_field"][:, :, 0])
    # Energy of the fastest-growing box mode of the corresponding theory.
    mode_energy = (np.abs(np.fft.rfft(Ex, axis=1))[:, theory[key]["mode"]] / Ex.shape[1]) ** 2
    energy = np.asarray(output["electric_field_energy"])
    # Linear window: from the time the mode energy has grown by three decades above
    # its initial level (the non-growing roots seeded by the initial displacement no
    # longer matter) to the time it reaches a tenth of its first peak.
    t0, t1 = mode_window(t, mode_energy, noise_factor=1e3, top=0.1)
    gamma_fit, intercept, slope = fit_growth_rate(t, mode_energy, t0, t1)
    fit = {"gamma": gamma_fit, "intercept": intercept, "slope": slope, "t0": float(t0), "t1": float(t1)}
    i_first = np.flatnonzero(superluminal.any(axis=1))
    results[key] = {
        "t": t, "energy": energy, "fit": fit, "seconds": seconds,
        "error_rel": np.abs(total_rel / total_rel[0] - 1.0),
        "error_newton": np.abs(total_newton / total_newton[0] - 1.0),
        "t_superluminal": float(t[i_first[0]]) if i_first.size else None,
        "superluminal_fraction": float(superluminal[:, :ve.shape[1]].mean(axis=1).max()),
        "x": np.asarray(output["position_electrons"][:, :, 0]) / L, "ve": ve,
        "lorentz_max": float(np.nanmax(lorentz[:, :ve.shape[1]])),
        "field_peak_over_kinetic": float(field.max() / kinetic_rel[0]),
        "t_peak": float(t[np.argmax(energy)]),
        "field_energy": field,
    }
    del output, v, v2, lorentz, superluminal

rel, newt = results["relativistic"], results["newtonian"]
for key, r in results.items():
    f = r["fit"]
    print(f"{key}: mode {theory[key]['mode']}, fit {f['gamma']:.4f} on [{f['t0']:.1f}, {f['t1']:.1f}], cold theory {theory[key]['cold']:.4f}, warm {theory[key]['warm']:.4f}, "
          f"{r['seconds']:.1f} s")

record(relativistic_v0_over_c=V0_OVER_C, relativistic_gamma0=gamma0, relativistic_vth_over_c=VTH_OVER_C,
       relativistic_particles=N_PER_SPECIES, relativistic_grid_points=GRID_POINTS,
       relativistic_c_dt_over_dx=C_DT_OVER_DX, relativistic_omega_pe_dt=wpe * dt, relativistic_steps=steps,
       relativistic_k_c_over_wpe=k * c / wpe, relativistic_k_v0_over_wpe=k * v0 / wpe,
       relativistic_length_c_over_wpe=L * wpe / c, relativistic_dx_wpe_over_c=dx * wpe / c,
       relativistic_t_end=T_END, relativistic_length_over_debye=int(round(L / debye_length)),
       relativistic_dx_over_debye=dx / debye_length, relativistic_debye_c_over_wpe=debye_length * wpe / c,
       **{f"relativistic_gamma_cold_{key}": theory[key]["cold"] for key in theory},
       **{f"relativistic_gamma_warm_{key}": theory[key]["warm"] for key in theory},
       **{f"relativistic_fastest_mode_{key}": theory[key]["mode"] for key in theory},
       **{f"relativistic_gamma_modes_{key}": [float(g) for g in theory[key]["modes"]] for key in theory},
       **{f"relativistic_gamma_fit_{key}": results[key]["fit"]["gamma"] for key in results},
       **{f"relativistic_gamma_deviation_percent_{key}":
          100 * abs(results[key]["fit"]["gamma"] / theory[key]["cold"] - 1) for key in results},
       relativistic_perturbation_over_L=PERTURBATION, relativistic_density=DENSITY,
       relativistic_electron_lorentz_ratio_cubed=gamma0**3,
       **{f"relativistic_fit_window_{key}": [results[key]["fit"]["t0"], results[key]["fit"]["t1"]]
          for key in results},
       relativistic_error_rel_max_relativistic=float(np.nanmax(rel["error_rel"])),
       relativistic_error_rel_final_relativistic=float(rel["error_rel"][-1]),
       relativistic_error_newton_max_relativistic=float(np.nanmax(rel["error_newton"])),
       relativistic_error_newton_max_newtonian=float(np.nanmax(newt["error_newton"])),
       relativistic_error_newton_final_newtonian=float(newt["error_newton"][-1]),
       relativistic_error_rel_max_newtonian_before_superluminal=float(np.nanmax(newt["error_rel"])),
       relativistic_t_superluminal_newtonian=newt["t_superluminal"],
       relativistic_superluminal_percent_newtonian=100 * newt["superluminal_fraction"],
       relativistic_lorentz_max_relativistic=rel["lorentz_max"],
       relativistic_field_peak_over_kinetic_relativistic=rel["field_peak_over_kinetic"],
       relativistic_field_peak_over_kinetic_newtonian=newt["field_peak_over_kinetic"],
       **{f"relativistic_seconds_{key}": results[key]["seconds"] for key in results})

fig, axes = figure(2, 2, gridspec_kw={"wspace": 0.3, "hspace": 0.34})
# (a) Electrostatic energy and linear growth
ax = axes[0, 0]
for key, _, label, colour in RUNS:
    ax.semilogy(results[key]["t"], results[key]["energy"], color=colour, label=label)
# Fastest box mode of each theory: mode 1 with the gamma0^3, mode 2 without (see the page).
THEORY_LABELS = {"relativistic": "relativistic cold theory", "newtonian": "non-rel. cold theory"}
for key, _, _, colour in RUNS:
    f = results[key]["fit"]
    tt = np.linspace(f["t0"], f["t1"], 50)
    # Theory is drawn from the electrostatic energy at the start of the fit window,
    # shifted up by a factor 5 so that it does not hide the simulation.
    anchor = 5 * results[key]["energy"][int(np.argmin(np.abs(results[key]["t"] - f["t0"])))]
    ax.semilogy(tt, anchor * np.exp(2 * theory[key]["cold"] * (tt - tt[0])), ls="--", color=colour,
                label=rf"{THEORY_LABELS[key]}, $\gamma = {theory[key]['cold']:.3f}\,\omega_{{pe}}$")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)")
ax.set_xlim(0, T_END)
ax.set_ylim(1e-9, 30 * max(r["energy"].max() for r in results.values()))
ax.legend(loc="lower right", fontsize=16, handlelength=1.8)
panel_label(ax, "(a)")

# (b) Relative change of the two total energies
ax = axes[0, 1]
for key, _, label, colour in RUNS:
    r = results[key]
    ax.semilogy(r["t"][1:], np.maximum(r["error_rel"][1:], 1e-12), color=colour)
    ax.semilogy(r["t"][1:], np.maximum(r["error_newton"][1:], 1e-12), color=colour, ls=":")
# Colour gives the pusher, as in (a); line style gives the energy that is tested.
ax.plot([], [], color=C_THEORY, label=r"$\sum(\gamma-1)mc^2$ + field")
ax.plot([], [], color=C_THEORY, ls=":", label=r"$\sum mv^2/2$ + field")
if newt["t_superluminal"] is not None:
    ax.axvline(newt["t_superluminal"], color=COLORS["grey"], lw=2, ls="-.")
    ax.text(newt["t_superluminal"] + 2, 3e3, "first $|v| \\geq c$: $\\gamma$ undefined,\n"
            r"so is $\sum(\gamma-1)mc^2$", color=COLORS["grey"], fontsize=18, va="top")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$|\,\mathcal{E}(t) - \mathcal{E}(0)\,| / \mathcal{E}(0)$")
ax.set_xlim(0, T_END)
ax.set_ylim(1e-9, 1e4)
ax.legend(loc="lower right")
panel_label(ax, "(b)")

# (c), (d) Electron phase space at the first saturation of each run
v_lim = 1.1 * max(np.abs(r["ve"][int(np.argmax(r["energy"]))]).max() for r in results.values())
for ax, (key, _, label, colour), letter in zip(axes[1], RUNS, ("(c)", "(d)")):
    r = results[key]
    i = int(np.argmax(r["energy"]))
    ax.scatter(r["x"][i] * L / debye_length, r["ve"][i], s=1.5, color=colour, alpha=0.5, linewidths=0, rasterized=True)
    for sign in (+1, -1):
        ax.axhline(sign, color=C_THEORY, ls="--", lw=2)
        ax.axhspan(sign, sign * v_lim, color="#EEEEEE", zorder=0)
    ax.set_xlim(-0.5 * L / debye_length, 0.5 * L / debye_length)
    ax.set_ylim(-v_lim, v_lim)
    ax.set_xlabel(r"$x / \lambda_D$")
    ax.set_ylabel(r"$v_x / c$")
    ax.set_title(rf"{label}, $t\,\omega_{{pe}} = {r['t'][i]:.0f}$")
    panel_label(ax, letter)
savefig(fig, "relativistic_two_stream")
