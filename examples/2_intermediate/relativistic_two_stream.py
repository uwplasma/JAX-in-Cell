"""Relativistic two-stream instability with the relativistic Boris pusher on and off.

Two cold electron beams drift through each other at +-v0 = +-0.8 c (Lorentz factor 5/3)
over a cold proton background. The same input is run twice, with
``Solver(relativistic=True)`` and ``Solver(relativistic=False)``, everything else
identical. The box holds one wavelength of the fastest-growing mode of the cold
relativistic dispersion relation.

Both runs store the velocity v, so the Lorentz factor of each particle is
1/sqrt(1 - |v|^2/c^2). From the stored output the script forms two total energies, both
with the field energy:

* relativistic, sum (gamma - 1) m c^2, which the relativistic equations conserve;
* Newtonian, sum m |v|^2 / 2, which the non-relativistic equations conserve.

The documentation's figure is this script's own (docs/scripts/fig_relativistic.py runs it).
About a minute on a CPU.

Panels: (a) electric energy with the linear growth rates of the cold relativistic and
non-relativistic dispersion relations, (b) relative change of both total energies for both
pushers, (c), (d) electron phase space at saturation.
"""
import os
import time
from pathlib import Path

# Double precision: the energy errors below go down to 1e-9.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, elementary_charge, epsilon_0, figure, mass_electron,
                       mu_0, save_run, speed_of_light as c)
from jaxincell.theory import electrostatic_epsilon, most_unstable_root, plasma_frequency

# Okabe-Ito colours, as in the documentation's figures
COLORS = {"blue": "#0072B2", "vermillion": "#D55E00", "grey": "#7F7F7F"}
C_THEORY = "#000000"


def panel_label(ax, text):
    ax.text(-0.16, 1.02, f"({text})", transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")


V0_OVER_C = 0.8                 # beam drift speed
VTH_OVER_C = 0.01               # beam thermal speed, v_th = sqrt(2 T / m)
DENSITY = 1e18                  # total electron density, m^-3 (both beams)
N_PER_SPECIES = 10000           # electrons (both beams) and protons
CELLS = 128
C_DT_OVER_DX = 0.9
T_END = 150.0                   # in units of 1 / omega_pe
PERTURBATION = 1e-6             # initial displacement of the electrons, in units of L
PARTICLE_FRAMES = 300           # stored particle snapshots; the fields are stored every step
RUNS = (("relativistic", True, "relativistic Boris", COLORS["vermillion"]),
        ("newtonian", False, "non-relativistic Boris", COLORS["blue"]))

gamma0 = 1.0 / np.sqrt(1.0 - V0_OVER_C ** 2)
v0 = V0_OVER_C * c
wpe = plasma_frequency(DENSITY, elementary_charge, mass_electron)
wb = wpe / np.sqrt(2.0)         # plasma frequency of one beam


def cold_roots(k, beam_frequency):
    """All roots of 1 = wb^2 [(w - k v0)^-2 + (w + k v0)^-2]: the quartic
    (w^2 - a^2)^2 - 2 wb^2 (w^2 + a^2) = 0 with a = k v0."""
    a, b = k * v0, beam_frequency ** 2
    return np.roots([1.0, 0.0, -2.0 * (a ** 2 + b), 0.0, a ** 4 - 2.0 * b * a ** 2])


# Cold relativistic theory: for motion along the drift the longitudinal mass is gamma^3 m, so
# wb^2 -> wb^2 / gamma0^3. With x = w^2 the growing branch is x = a^2 + wb^2 - wb sqrt(wb^2 + 4 a^2),
# most negative at a^2 = 3 wb^2 / 4.
wb_rel = wb / gamma0 ** 1.5
k = np.sqrt(3.0) / 2.0 * wb_rel / v0
L = 2.0 * np.pi / k
dx = L / CELLS
dt = C_DT_OVER_DX * dx / c
steps = int(round(T_END / (wpe * dt)))
every = max(1, steps // PARTICLE_FRAMES)
steps -= steps % every
# Debye length of the beams, lambda_D = v_th / (sqrt(2) omega_pe) with v_th = sqrt(2 T / m), the
# convention of Species.vth and of the kinetic dispersion relation
debye_length = VTH_OVER_C * c / (np.sqrt(2.0) * wpe)

# Growth rate of every box mode k_m = m k for both sets of equations. Only the first mode is
# unstable for the relativistic beams (a^2 < 2 wb^2 / gamma0^3); without the gamma0^3 the first
# three are, and the second grows fastest.
MODES = np.arange(1, 6)
theory = {}
for key, beam_frequency in (("relativistic", wb_rel), ("newtonian", wb)):
    cold = np.array([max(cold_roots(m * k, beam_frequency).imag.max(), 0.0) / wpe for m in MODES])
    m_fast = int(MODES[np.argmax(cold)])
    # Warm check: for a narrow beam dv/dp = 1/(gamma^3 m), so to leading order in v_th/c the
    # relativistic beam is a Maxwellian in v with wp^2 -> wp^2 / gamma0^3.
    populations = [{"wp": beam_frequency, "u": s * v0, "vth": VTH_OVER_C * c} for s in (+1, -1)]
    root = most_unstable_root(lambda w: electrostatic_epsilon(w, m_fast * k, populations),
                              (-0.3, 0.3), (0.02, 0.5), n_real=13, n_imag=12, scale=wpe)
    theory[key] = {"modes": cold, "mode": m_fast, "cold": cold[m_fast - 1],
                   "warm": root.imag / wpe if root is not None else np.nan}
assert theory["relativistic"]["mode"] == 1
assert abs(theory["relativistic"]["cold"] - 0.5 * wb_rel / wpe) < 1e-8


def simulation(relativistic):
    """Both species on the same lattice of positions, so that the start is neutral cell by cell
    and the displacement, not the particle noise, seeds the growth; the velocities are drawn."""
    beams = Species.electrons(n=N_PER_SPECIES, density=DENSITY, vth=(VTH_OVER_C * c, 0, 0), drift=(v0, 0, 0),
                              plus_minus=True, perturbation_mode=1,
                              perturbation_amplitude=PERTURBATION * L)
    protons = Species.ions(n=N_PER_SPECIES, density=DENSITY, mass_ratio=1.0, vth=(1e-4 * c, 0, 0))
    return Simulation(Domain(length=L, cells=CELLS, dt_over_dx_c=C_DT_OVER_DX), [beams, protons],
                      Solver(relativistic=relativistic))


def growth_window(energy, noise_factor=1e3, top=0.1):
    """Indices bounding the exponential growth of one mode energy, before its first saturation
    (the first time it reaches half its largest value): from the last time it was below
    ``noise_factor`` times its initial level, where the non-growing roots the displacement seeds
    no longer matter, to the last time it was below ``top`` of its largest value."""
    peak = int(np.argmax(energy >= 0.5 * energy.max()))
    low = np.flatnonzero(energy[:peak] < noise_factor * energy[:20].mean())
    high = np.flatnonzero(energy[:peak] < top * energy.max())
    start, stop = (low[-1] if low.size else 0), (high[-1] if high.size else peak - 1)
    return (start, stop) if stop > start + 2 else (max(peak // 4, 1), max(peak - 1, 2))


results = {}
for key, relativistic, *_ in RUNS:
    sim = simulation(relativistic)
    start = time.perf_counter()
    fields = sim.run(steps, seed=0, store_particles=False)
    fields.E.block_until_ready()
    seconds = time.perf_counter() - start
    out = sim.run(steps, seed=0, store_every=every)
    t = np.asarray(fields.t) * wpe
    Ex = np.asarray(fields.E[:, :, 0])
    energy = 0.5 * epsilon_0 * np.sum(Ex ** 2, axis=1) * dx
    field_all = 0.5 * epsilon_0 * np.sum(np.asarray(out.E) ** 2, axis=(1, 2)) * dx
    field_all += 0.5 / mu_0 * np.sum(np.asarray(out.B) ** 2, axis=(1, 2)) * dx
    mass = np.asarray(out.mass)[None, :] * np.asarray(out.weight)
    v = np.asarray(out.v)
    v2 = np.sum(v ** 2, axis=-1)
    superluminal = v2 >= c ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lorentz = 1.0 / np.sqrt(1.0 - v2 / c ** 2)
    # (gamma - 1) m c^2 written as m v^2 gamma^2 / (gamma + 1), which keeps its digits for the slow protons
    kinetic_rel = np.sum(mass * v2 * lorentz ** 2 / (lorentz + 1.0), axis=1)
    kinetic_rel[superluminal.any(axis=1)] = np.nan              # undefined once any |v| >= c
    kinetic_newton = 0.5 * np.sum(mass * v2, axis=1)
    total_rel, total_newton = kinetic_rel + field_all, kinetic_newton + field_all
    electrons = slice(0, N_PER_SPECIES)
    t_out = np.asarray(out.t) * wpe
    mode_energy = (np.abs(np.fft.rfft(Ex, axis=1))[:, theory[key]["mode"]] / CELLS) ** 2
    i0, i1 = growth_window(mode_energy)
    slope, intercept = np.polyfit(t[i0:i1], np.log(mode_energy[i0:i1]), 1)
    first = np.flatnonzero(superluminal.any(axis=1))
    i_peak = int(np.argmin(np.abs(t_out - t[int(np.argmax(energy))])))
    results[key] = {
        "t": t, "energy": energy, "t_out": t_out, "seconds": seconds,
        "fit": {"gamma": slope / 2, "t0": float(t[i0]), "t1": float(t[i1])},
        "error_rel": np.abs(total_rel / total_rel[0] - 1.0),
        "error_newton": np.abs(total_newton / total_newton[0] - 1.0),
        "t_superluminal": float(t_out[first[0]]) if first.size else None,
        "superluminal_fraction": float(superluminal[:, electrons].mean(axis=1).max()),
        "lorentz_max": float(np.nanmax(lorentz[:, electrons])),
        "x": np.asarray(out.x[i_peak, electrons, 0]), "ve": v[i_peak, electrons, 0] / c,
        "t_peak": float(t_out[i_peak]),
    }
    del out, v, v2, lorentz, superluminal

rel, newt = results["relativistic"], results["newtonian"]
for key, r in results.items():
    f = r["fit"]
    print(f"  {key}: mode {theory[key]['mode']}, fit {f['gamma']:.4f} on [{f['t0']:.1f}, {f['t1']:.1f}], "
          f"cold theory {theory[key]['cold']:.4f}, warm {theory[key]['warm']:.4f}, {r['seconds']:.1f} s")

measurements = dict(relativistic_v0_over_c=V0_OVER_C, relativistic_gamma0=gamma0, relativistic_vth_over_c=VTH_OVER_C,
                    relativistic_particles=N_PER_SPECIES, relativistic_grid_points=CELLS,
                    relativistic_c_dt_over_dx=C_DT_OVER_DX, relativistic_omega_pe_dt=wpe * dt, relativistic_steps=steps,
                    relativistic_k_c_over_wpe=k * c / wpe, relativistic_k_v0_over_wpe=k * v0 / wpe,
                    relativistic_length_c_over_wpe=L * wpe / c, relativistic_t_end=T_END, relativistic_density=DENSITY,
                    relativistic_dx_wpe_over_c=dx * wpe / c, relativistic_debye_c_over_wpe=debye_length * wpe / c,
                    relativistic_perturbation_over_L=PERTURBATION, relativistic_particle_frames_every=every,
                    relativistic_length_over_debye=int(round(L / debye_length)),
                    relativistic_dx_over_debye=dx / debye_length,
                    relativistic_electron_lorentz_ratio_cubed=gamma0 ** 3,
                    **{f"relativistic_gamma_cold_{key}": theory[key]["cold"] for key in theory},
                    **{f"relativistic_gamma_warm_{key}": theory[key]["warm"] for key in theory},
                    **{f"relativistic_fastest_mode_{key}": theory[key]["mode"] for key in theory},
                    **{f"relativistic_gamma_fit_{key}": results[key]["fit"]["gamma"] for key in results},
                    **{f"relativistic_gamma_deviation_percent_{key}":
                       100 * abs(results[key]["fit"]["gamma"] / theory[key]["cold"] - 1) for key in results},
                    relativistic_error_rel_max_relativistic=float(np.nanmax(rel["error_rel"])),
                    relativistic_error_newton_max_relativistic=float(np.nanmax(rel["error_newton"])),
                    relativistic_error_newton_max_newtonian=float(np.nanmax(newt["error_newton"])),
                    relativistic_error_rel_max_newtonian_before_superluminal=float(np.nanmax(newt["error_rel"])),
                    relativistic_t_superluminal_newtonian=newt["t_superluminal"],
                    relativistic_superluminal_percent_newtonian=100 * newt["superluminal_fraction"],
                    relativistic_superluminal_percent_relativistic=100 * rel["superluminal_fraction"],
                    relativistic_lorentz_max_relativistic=rel["lorentz_max"],
                    **{f"relativistic_seconds_{key}": results[key]["seconds"] for key in results})

fig, axes = figure(2, 2, gridspec_kw={"wspace": 0.3, "hspace": 0.34})
ax = axes[0, 0]
THEORY_LABELS = {"relativistic": "relativistic cold theory", "newtonian": "non-rel. cold theory"}
for key, _, label, colour in RUNS:
    r = results[key]
    ax.semilogy(r["t"], r["energy"], color=colour, label=label)
for key, _, _, colour in RUNS:
    r, f = results[key], results[key]["fit"]
    tt = np.linspace(f["t0"], f["t1"], 50)
    # drawn from five times the energy at the start of the window, so as not to hide the run
    anchor = 5 * r["energy"][int(np.argmin(np.abs(r["t"] - f["t0"])))]
    ax.semilogy(tt, anchor * np.exp(2 * theory[key]["cold"] * (tt - tt[0])), ls="--", color=colour,
                label=rf"{THEORY_LABELS[key]}, $\gamma = {theory[key]['cold']:.3f}\,\omega_{{pe}}$")
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$\frac{\epsilon_0}{2}\int E_x^2\,dx$  (J/m$^2$)", xlim=(0, T_END),
       ylim=(1e-9, 30 * max(r["energy"].max() for r in results.values())))
ax.legend(loc="lower right", fontsize=16, handlelength=1.8)
panel_label(ax, "a")

ax = axes[0, 1]
for key, _, label, colour in RUNS:
    r = results[key]
    ax.semilogy(r["t_out"][1:], np.maximum(r["error_rel"][1:], 1e-12), color=colour)
    ax.semilogy(r["t_out"][1:], np.maximum(r["error_newton"][1:], 1e-12), color=colour, ls=":")
ax.plot([], [], color=C_THEORY, label=r"$\sum(\gamma-1)mc^2$ + field")
ax.plot([], [], color=C_THEORY, ls=":", label=r"$\sum mv^2/2$ + field")
if newt["t_superluminal"] is not None:
    ax.axvline(newt["t_superluminal"], color=COLORS["grey"], lw=2, ls="-.")
    ax.text(newt["t_superluminal"] + 2, 3e3, "first $|v| \\geq c$: $\\gamma$ undefined,\n"
            r"so is $\sum(\gamma-1)mc^2$", color=COLORS["grey"], fontsize=18, va="top")
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|\,\mathcal{E}(t) - \mathcal{E}(0)\,| / \mathcal{E}(0)$",
       xlim=(0, T_END), ylim=(1e-9, 1e4))
ax.legend(loc="lower right")
panel_label(ax, "b")

v_lim = 1.1 * max(np.abs(r["ve"]).max() for r in results.values())
for ax, (key, _, label, colour), letter in zip(axes[1], RUNS, "cd"):
    r = results[key]
    ax.scatter(r["x"] / debye_length, r["ve"], s=1.5, color=colour, alpha=0.5, linewidths=0, rasterized=True)
    for sign in (+1, -1):
        ax.axhline(sign, color=C_THEORY, ls="--", lw=2)
        ax.axhspan(sign, sign * v_lim, color="#EEEEEE", zorder=0)
    ax.set(xlim=(-0.5 * L / debye_length, 0.5 * L / debye_length), ylim=(-v_lim, v_lim),
           xlabel=r"$x / \lambda_D$", ylabel=r"$v_x / c$", title=rf"{label}, $t\,\omega_{{pe}} = {r['t_peak']:.0f}$")
    panel_label(ax, letter)
save_run(Path.cwd() / "relativistic_two_stream", "relativistic_two_stream",
         dict(v0_over_c=V0_OVER_C, vth_over_c=VTH_OVER_C, density=DENSITY, particles=N_PER_SPECIES, cells=CELLS,
              c_dt_over_dx=C_DT_OVER_DX, t_end=T_END, perturbation=PERTURBATION),
         dict(measurements={k: (v if v is None else int(v) if isinstance(v, (int, np.integer)) else float(v))
                            for k, v in measurements.items()}),
         figure=fig, **{f"{key}_{q}": results[key][q] for key in results for q in ("t", "energy")})
plt.show()
