"""Weibel instability: growth rate against the transverse dispersion relation.

The electrons carry the temperature anisotropy of examples/Weibel_instability.py
(T_z/T_x = 100). One simulation is run per wavenumber: it starts quiet (equally
spaced positions, Maxwellian velocity components at quantiles of bit-reversed
sequences) and the electrons get a small coherent modulation
v_z -> v_z + delta sin(k x), whose current seeds B_y at that one wavenumber.
Started from particle noise instead, the modes rise only about two e-foldings
above the floor before saturating and no rate can be fitted.

This script runs one simulation per mode and takes a few minutes.
"""
import numpy as np
import matplotlib.pyplot as plt
from jax import block_until_ready
from scipy.special import erfinv

from common import (CMAP_SIGNED, C_ELECTRONS, C_FIT, C_THEORY, COLORS, panel_label, record,
                    robust_growth_fit, savefig, silence_progress_bars, species_for_linear_theory)
from dispersion import most_unstable_root, weibel_dispersion
from jaxincell import (Simulation, diagnostics, mass_electron, mass_proton, speed_of_light)

silence_progress_bars()

L, NX, STEPS, COURANT, N = 3e-1, 150, 1800, 0.5, 8000
VTH_X, VTH_Z = 0.01, 0.10
MASS_RATIO = np.sqrt(mass_electron / mass_proton)          # T_i = T_e on every axis
VTH_IX, VTH_IZ = VTH_X * MASS_RATIO, VTH_Z * MASS_RATIO
SEED = 1e-3
MODES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
SHOWCASE = 4                                               # fastest growing mode


def van_der_corput(n, base):
    q, denominator, i = np.zeros(n), 1.0, np.arange(1, n + 1)
    while i.any():
        denominator *= base
        q += (i % base) / denominator
        i //= base
    return q


def quantiles(n, base):
    return erfinv(2 * van_der_corput(n, base) - 1)


def phase_space(n, vthx, vthz, bases, k=None, seed=0.0, offset=0.0):
    x = np.linspace(-L / 2, L / 2, n, endpoint=False) + L / (2 * n) + offset
    vx = vthx * speed_of_light * quantiles(n, bases[0])
    vz = vthz * speed_of_light * quantiles(n, bases[1])
    if seed:
        vz = vz + seed * vthz * speed_of_light * np.sin(k * x)
    return (np.stack([x, np.zeros(n), np.zeros(n)], 1),
            np.stack([vx, np.zeros(n), vz], 1))


def run(mode):
    k = 2 * np.pi * mode / L
    xe, ve = phase_space(N, VTH_X, VTH_Z, (2, 3), k=k, seed=SEED)
    xi, vi = phase_space(N, VTH_IX, VTH_IZ, (5, 7), offset=L / (2 * N))
    parameters = {
        "domain_parameters": {"length": L, "timestep_over_spatialstep_times_c": COURANT,
                              "number_grid_points": NX, "total_steps": STEPS},
        "species_parameters": {
            "electrons": {"electrons0": {
                "number_pseudoparticles": N, "grid_points_per_Debye_length": 1.1,
                "vth_over_c_x": VTH_X, "vth_over_c_z": VTH_Z,
                "initial_positions": xe, "initial_velocities": ve}},
            "ions": {"ions0": {
                "number_pseudoparticles": N, "grid_points_per_Debye_length": 1.1,
                "mass_over_proton_mass": 1, "vth_over_c_x": VTH_IX, "vth_over_c_z": VTH_IZ,
                "initial_positions": xi, "initial_velocities": vi}}},
        "solver_parameters": {"field_solver": 0, "filter_passes": 0, "print_info": False},
    }
    output = block_until_ready(Simulation(parameters).run())
    diagnostics(output)
    wpe = float(output["plasma_frequency"])
    t = np.asarray(output["time_array"]) * wpe
    By = np.asarray(output["magnetic_field"][:, :, 1])
    amplitude = np.abs(np.fft.rfft(By, axis=1))[:, mode] / By.shape[1]
    populations = species_for_linear_theory(output)
    species = [{"wp": s["wp"], "vthx": s["vthx"], "A": (s["vthz"] / s["vthx"]) ** 2}
               for s in populations]
    root = most_unstable_root(lambda w: weibel_dispersion(w, k, species),
                              (-0.02, 0.02), (1e-4, 0.2), n_real=9, n_imag=30, scale=wpe)
    return {"t": t, "amplitude": amplitude, "wpe": wpe, "By": By,
            "grid": np.asarray(output["grid"]), "anisotropy": species[0]["A"],
            "kc": k * speed_of_light / wpe,
            "theory": root.imag / wpe if root is not None else np.nan,
            "energy_error": float(np.max(np.abs(
                output["total_energy"] / output["total_energy"][0] - 1)))}


results, showcase = {}, None
for mode in MODES:
    r = run(mode)
    fit = robust_growth_fit(r["t"], r["amplitude"] ** 2)
    r["fit"] = fit
    results[mode] = r
    if mode == SHOWCASE:
        showcase = r
    status = (f"measured {fit['gamma']:.4f} (r2 {fit['r2']:.3f})" if fit else "rejected")
    print(f"  mode {mode:2d}: theory {r['theory']:.4f}  {status}", flush=True)

kept = [m for m in MODES if results[m]["fit"] is not None]
deviations = np.array([abs(results[m]["fit"]["gamma"] - results[m]["theory"]) / results[m]["theory"]
                       for m in kept])
wpe = showcase["wpe"]
record(weibel_modes_run=len(MODES), weibel_modes_compared=len(kept),
       weibel_mean_deviation_percent=float(100 * deviations.mean()),
       weibel_max_deviation_percent=float(100 * deviations.max()),
       weibel_anisotropy=float(showcase["anisotropy"]),
       weibel_particles=N, weibel_steps=STEPS, weibel_courant=COURANT, weibel_seed=SEED,
       weibel_grid_points=NX, weibel_t_end=float(showcase["t"][-1]),
       weibel_energy_error=float(max(results[m]["energy_error"] for m in MODES)),
       weibel_fastest_mode=int(MODES[int(np.nanargmax([results[m]["theory"] for m in MODES]))]),
       weibel_gamma_theory_max=float(np.nanmax([results[m]["theory"] for m in MODES])),
       weibel_kc_over_wpe_fastest=float(results[SHOWCASE]["kc"]))

fig = plt.figure(figsize=(7.4, 5.8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.5, wspace=0.35)

ax = fig.add_subplot(gs[0, :])
By, grid, t = showcase["By"], showcase["grid"], showcase["t"]
limit = np.percentile(np.abs(By), 99.5)
mesh = ax.pcolormesh(grid, t, By, cmap=CMAP_SIGNED, vmin=-limit, vmax=limit,
                     rasterized=True, shading="nearest")
ax.grid(False)
ax.set_xlabel("x (m)")
ax.set_ylabel(r"$t\,\omega_{pe}$")
ax.set_title(rf"seeded mode {SHOWCASE},  $kc/\omega_{{pe}} = {showcase['kc']:.2f}$", fontsize=9)
bar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.04)
bar.set_label(r"$B_y$ (T)")
panel_label(ax, "(a)", x=-0.07)

ax = fig.add_subplot(gs[1, 0])
for mode, colour in zip((2, 4, 8), (COLORS["sky"], C_ELECTRONS, COLORS["purple"])):
    r = results[mode]
    ax.semilogy(r["t"], r["amplitude"], color=colour, lw=1.2, label=f"mode {mode}")
    if r["fit"]:
        f = r["fit"]
        tt = np.linspace(f["t0"], f["t1"], 30)
        ax.semilogy(tt, np.exp(0.5 * (f["intercept"] + f["slope"] * tt)), color=C_FIT,
                    lw=2.2, alpha=0.8)
ax.plot([], [], color=C_FIT, lw=2.2, label="fitted window")
ax.set_xlabel(r"$t\,\omega_{pe}$")
ax.set_ylabel(r"$|\hat B_y(k)|$  (T)")
ax.legend(loc="lower right", fontsize=7.5)
panel_label(ax, "(b)")

ax = fig.add_subplot(gs[1, 1])
kc_fine = np.array([results[m]["kc"] for m in MODES])
ax.plot(kc_fine, [results[m]["theory"] for m in MODES], ls="--", color=C_THEORY,
        label="linear theory")
ax.plot([results[m]["kc"] for m in kept], [results[m]["fit"]["gamma"] for m in kept],
        "o", ms=5, color=C_ELECTRONS, label="simulation")
ax.set_xlabel(r"$k c / \omega_{pe}$")
ax.set_ylabel(r"$\gamma / \omega_{pe}$")
ax.set_ylim(bottom=0)
ax.legend(loc="lower left", fontsize=8)
panel_label(ax, "(c)")
savefig(fig, "weibel")
