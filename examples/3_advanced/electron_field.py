"""Electron-field instability, and the controls that decide what drives it.

Beving, Hopkins and Baalrud (Phys. Plasmas 30, 112105, 2023) put a periodic helium plasma
in a static, uniform field and see electron plasma waves grow, at a rate they compare with
Fried's dielectric function of electrons in a field (their eq. 7; `jaxincell.theory.field_epsilon`).
Their case: n = 3e14 m^-3, T_e = 3 eV, T_i = 0.026 eV, E_0 = -800 V/m (kappa lambda_De =
e E_0 lambda_De / T_e = 0.2), a box of 1200 lambda_De with five cells per lambda_De, 400
particles per cell per species, positions at random, sixteen realisations. They fit the
fluctuation energy <E> between t omega_pe = 35 and 170 and quote 7.9e-3 omega_pe against 1.5e-2
from eq. 10 at the mean electron drift of that window.

The indispensable control is a change of frame. For collisionless Vlasov-Poisson with an
exactly uniform, immobile background, x' = x - a t^2/2 and v' = v - a t (a = -e E_0/m_e)
remove the field: a driven and an undriven run with the same particles must show the same
fluctuations, shifted by a t^2/2. So the ions are taken three ways, and every case is run:

* "uniform": no ions, the exactly uniform background the periodic field solve provides,
  with and without the field -- here the equations say the two are one run;
* "frozen": ion macro-particles of 1e9 proton masses, at random positions, with and
  without the field -- the noisy immobile ions a PIC code has when it "freezes" them;
* "helium": mobile He+ at 0.026 eV, the paper's case.

The field is `Simulation(external_E=...)`, uniform on the faces, not a potential
difference on the grid; its work, E_0 times the integrated current, is compared with the
change of kinetic plus field energy. The time step keeps an electron at 40 v_Te (the drift
at t omega_pe = 200) within one cell per step, as the paper's did. Nothing asserts growth:
the numbers are printed and saved, and the documentation says what they mean.

Run with `--quick` for a smoke preset (a 240 lambda_De box, 40 particles per cell, one
realisation, to t omega_pe = 100 at twice the step): same controls, not a measurement.
"""

import os
import sys
import time
from pathlib import Path

# Double precision: the driven and undriven uniform runs are compared to many digits.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, figure, mass_electron, mass_proton,
                       save_run, elementary_charge as e_charge)
from jaxincell.theory import field_rate

# --- what to change ------------------------------------------------------------------
quick = "--quick" in sys.argv
density, T_e, T_i, E_0 = 3e14, 3.0, 0.026, -800.0          # m^-3, eV, eV, V/m
helium = 4.002602 * 1.66053907e-27 - mass_electron           # He+, kg
debye_lengths = 240 if quick else 1200                       # box length / lambda_De
cells_per_debye = 5
per_cell = 40 if quick else 400                              # particles per cell per species
t_end = 100.0 if quick else 200.0                            # in 1/omega_pe
dt_wpe = 0.01 if quick else 0.005                            # omega_pe dt
realisations = 1 if quick else 4
store_every = 20
fit = (35.0, 80.0 if quick else 170.0)                       # the paper's window, t omega_pe
cases = {  # name: (ions, driven)
    "uniform": (None, True), "uniform, no field": (None, False),
    "frozen": ("frozen", True), "frozen, no field": ("frozen", False),
    "helium": ("helium", True),
}
if quick:
    print("--quick is a smoke preset: a smaller box, fewer particles, one realisation;\n"
          "the rates it prints are not the documented measurement.")

# --- derived scales, in the paper's convention v_Te = sqrt(T_e/m_e) -------------------
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
lambda_D = np.sqrt(epsilon_0 * T_e / (density * e_charge))
v_Te = lambda_D * omega_pe
kappa = abs(E_0) * lambda_D / T_e                             # kappa_e lambda_De
cells = debye_lengths * cells_per_debye
length, n = debye_lengths * lambda_D, per_cell * cells
steps = int(round(t_end / dt_wpe / store_every)) * store_every
accel = -e_charge * E_0 / mass_electron                       # electron acceleration, m/s^2
print(f"lambda_De = {lambda_D:.4e} m, omega_pe = {omega_pe:.4e} rad/s, kappa lambda_De = {kappa:.4f}; "
      f"{cells} cells, {n} particles per species, {steps} steps")
vth_e = np.sqrt(2) * v_Te                                     # JAX-in-Cell's sqrt(2T/m)
vth_i = np.sqrt(2 * T_i * e_charge / helium)
ion_species = {
    "frozen": Species.ions(n=n, density=density, mass_ratio=1e9, vth=(0, 0, 0), sampling="random"),
    "helium": Species.ions(n=n, density=density, mass_ratio=helium / mass_proton, vth=(vth_i,) * 3,
                           sampling="random"),
}
electrons = Species.electrons(n=n, density=density, vth=(vth_e,) * 3, sampling="random")
field = np.zeros((cells, 3))
field[:, 0] = E_0


def run(ions, driven, seed):
    """One realisation: times, E_x on the faces, the total current and the kinetic energies at both ends."""
    species = [electrons] + ([ion_species[ions]] if ions else [])
    simulation = Simulation(Domain(length=length, cells=cells, time_step=dt_wpe / omega_pe), species,
                            Solver(model="electrostatic"), external_E=field * driven)
    mass = np.asarray(simulation.per_particle[0])
    first = simulation.run(store_every, seed=seed, store_every=store_every, store_particles=False)
    rest = simulation.run(steps - store_every, seed=seed, store_every=store_every, store_particles=False,
                          state=first.state)
    kinetic = [0.5 * np.sum(mass * np.asarray(s.w) * np.sum(np.asarray(s.u) ** 2, axis=1)) for s in
               (first.state, rest.state)]
    E = np.concatenate([np.asarray(first.E[:, :, 0]), np.asarray(rest.E[:, :, 0])])
    J = np.concatenate([np.asarray(first.J[:, :, 0]), np.asarray(rest.J[:, :, 0])]).sum(axis=1) * length / cells
    return np.concatenate([np.asarray(first.t), np.asarray(rest.t)]) * omega_pe, E, J, kinetic


def fluctuation(E):
    """The paper's eq. 2 averaged over the box: eps_0/2 (E - time mean at each x)^2, in J/m^3."""
    return 0.5 * epsilon_0 * np.mean((E - E.mean(axis=0)) ** 2, axis=1)


modes = np.arange(1, int(0.3 * debye_lengths / (2 * np.pi)) + 1)      # k lambda_De up to 0.3
k_modes = 2 * np.pi * modes / debye_lengths
energy, spectrum, keep, seconds = {}, {}, {}, {}
for name, (ions, driven) in cases.items():
    start = time.perf_counter()
    for seed in range(realisations):
        t, E, J, kinetic = run(ions, driven, seed)
        dE = E - E.mean(axis=0)
        energy[name] = energy.get(name, 0) + fluctuation(E) / realisations
        power = np.abs(np.fft.rfft(dE, axis=1)[:, modes]) ** 2
        spectrum[name] = spectrum.get(name, 0) + power / realisations
        if seed == 0:
            keep[name] = (E, J, kinetic)
    seconds[name] = time.perf_counter() - start
    print(f"{name:18s} {realisations} realisation(s) in {seconds[name]:.0f} s")

# --- the growth, fitted where the paper fitted it ----------------------------------------
# <E> beats at twice the plasma frequency; its mean over one plasma period is what is fitted and drawn
period = int(round(2 * np.pi / (dt_wpe * store_every)))
smooth = {name: np.convolve(energy[name], np.ones(period) / period, mode="same") for name in cases}
window = (t >= fit[0]) & (t <= fit[1])
rates = {}
for name in cases:
    slope = np.polyfit(t[window], np.log(smooth[name][window]), 1)[0]
    rates[name] = dict(energy_rate=slope, gamma=slope / 2, gain=smooth[name][window][-1] / smooth[name][window][0])
t_mid = 0.5 * sum(fit)
k_star = 1 / (kappa * t_mid)                                   # eq. 12: where omega_r = 0 at the mean drift
gamma_theory = float(field_rate(k_star * 0.999, kappa, drift=kappa * t_mid))   # eq. 10 just inside the cutoff
paper_fit = 7.9e-3
for name, r in rates.items():
    print(f"{name:18s} <E> rises x{r['gain']:.3g} over t omega_pe {fit[0]:.0f}-{fit[1]:.0f}: "
          f"d ln<E>/dt = {r['energy_rate']:+.2e}, gamma = {r['gamma']:+.2e} omega_pe")
print(f"eq. 10 at k* = 1/(kappa t) = {k_star:.4f}/lambda_De, t omega_pe = {t_mid:.0f}: gamma = {gamma_theory:.2e} "
      f"omega_pe (energy rate {2 * gamma_theory:.2e}); the paper fitted {paper_fit:.1e} to <E>")

# --- the change of frame: a driven uniform run is the undriven one, moved by a t^2/2 ----
E_driven, E_still = keep["uniform"][0], keep["uniform, no field"][0]
k_grid = 2 * np.pi * np.fft.rfftfreq(cells, d=length / cells)
shift = 0.5 * accel * (t / omega_pe) ** 2
moved = np.fft.rfft(E_driven, axis=1) * np.exp(1j * np.outer(shift, k_grid))
still = np.fft.rfft(E_still, axis=1)
frame = np.linalg.norm(moved - still, axis=1) / np.linalg.norm(still, axis=1)
# the grid is fixed in the laboratory, so the shift is exact only on the continuum's scales:
# the same difference over the wavelengths the instability is about, k lambda_De <= 0.3
long = k_grid * lambda_D <= 0.3
frame_long = np.linalg.norm((moved - still)[:, long], axis=1) / np.linalg.norm(still[:, long], axis=1)
print(f"uniform background: |E_driven(x + a t^2/2) - E_undriven(x)| / |E_undriven| = {frame[window].max():.2e} "
      f"at most in the fit window ({frame_long[window].max():.2e} for k lambda_De <= 0.3)")

# --- the energy ledger of the driven uniform run: work of E_0 against kinetic plus field energy ---
_, J, kinetic = keep["uniform"]
work = E_0 * np.sum(0.5 * (J[1:] + J[:-1]) * np.diff(t / omega_pe))   # from the first stored step on
field_energy = [0.5 * epsilon_0 * np.sum(E_driven[i] ** 2) * length / cells for i in (0, -1)]
change = kinetic[1] - kinetic[0] + field_energy[1] - field_energy[0]
ledger = abs(change - work) / abs(work)
print(f"external work {work:.4e} J/m^2, kinetic plus field energy gained {change:.4e} J/m^2: "
      f"relative difference {ledger:.1e}")

# --- the peak of the spectrum against eq. 12 ---------------------------------------------
peak = {name: k_modes[np.argmax(spectrum[name], axis=1)] for name in ("frozen", "helium")}
late = t > fit[0]
for name, k_peak in peak.items():
    ratio = np.median(k_peak[late] * kappa * t[late])
    print(f"{name}: peak k lambda_De times kappa t omega_pe, median after t omega_pe = {fit[0]:.0f}: {ratio:.2f} "
          "(eq. 12 gives 1)")

# --- the figure ----------------------------------------------------------------------------------
fig, axes = figure(2, 2)
colors = {"uniform": "C0", "uniform, no field": "C0", "frozen": "C1", "frozen, no field": "C1", "helium": "C2"}
ax = axes[0, 0]
inside = slice(period // 2, t.size - period // 2)                 # where the running mean is a full period
for name in cases:
    ax.semilogy(t[inside], smooth[name][inside] / smooth[name][inside][0], color=colors[name], lw=2,
                ls=":" if "no field" in name else "-", label=name)
ax.axvspan(*fit, color="0.9", zorder=0)
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$\langle\mathcal{E}\rangle/\langle\mathcal{E}_0\rangle$",
       title="fluctuation energy, mean over a plasma period")
ax.legend(fontsize="small")
ax = axes[0, 1]
ax.semilogy(t[1:], frame[1:], lw=2, label="all wavelengths")
ax.semilogy(t[1:], frame_long[1:], lw=2, label=r"$k\lambda_{De}\leq0.3$")
ax.legend(fontsize="small")
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel="relative difference",
       title="uniform: driven, moved by $at^2/2$, vs undriven")
ax = axes[1, 0]
image = spectrum["helium"] / spectrum["helium"].max()
ax.pcolormesh(t, k_modes, np.log10(image.T), vmin=-4, vmax=0, shading="auto", cmap="viridis")
ax.plot(t[late], 1 / (kappa * t[late]), "w--", lw=2,
        label=r"eq. 12, $k^*\lambda_{De}=1/(\kappa_e\lambda_{De}t\omega_{pe})$")
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$k\lambda_{De}$", ylim=(k_modes[0], k_modes[-1]),
       title="helium: fluctuation spectrum")
ax.legend(fontsize="small", loc="upper right")
ax = axes[1, 1]
names = list(cases)
ax.bar(range(len(names)), [rates[m]["energy_rate"] for m in names], color=[colors[m] for m in names])
ax.axhline(2 * gamma_theory, color="k", ls="--", label=r"$2\gamma$, eq. 10 at the mean drift")
ax.axhline(paper_fit, color="0.5", ls=":", label="the paper's fit to $\\langle\\mathcal{E}\\rangle$")
ax.set_xticks(range(len(names)), [m.replace(", ", "\n") for m in names], fontsize="small")
ax.set(ylabel=r"$d\ln\langle\mathcal{E}\rangle/dt\;/\;\omega_{pe}$",
       title=f"growth of the energy, $t\\omega_{{pe}}$ {fit[0]:.0f}-{fit[1]:.0f}",
       ylim=(min(0, *(r["energy_rate"] for r in rates.values())), 1.4 * max(2 * gamma_theory, paper_fit)))
ax.legend(fontsize="small", loc="upper left")
fig.tight_layout()

# --- the record --------------------------------------------------------------------------------
settings = dict(density=density, T_e=T_e, T_i=T_i, E_0=E_0, ion_mass=helium, debye_lengths=debye_lengths,
                cells_per_debye=cells_per_debye, per_cell=per_cell, t_end=t_end, dt_wpe=dt_wpe, steps=steps,
                realisations=realisations, store_every=store_every, fit=fit, quick=quick)
results = dict(lambda_D=lambda_D, omega_pe=omega_pe, kappa_lambda_D=kappa, rates=rates, seconds=seconds,
               k_star=k_star, gamma_theory=gamma_theory, paper_fit=paper_fit,
               frame_difference_window=float(frame[window].max()),
               frame_difference_long=float(frame_long[window].max()),
               external_work=work, energy_gained=change, ledger_difference=ledger,
               peak_over_eq12={m: float(np.median(p[late] * kappa * t[late])) for m, p in peak.items()})
save_run(Path.cwd() / ("electron_field_quick" if quick else "electron_field"), "electron_field", settings,
         results, figure=fig, t=t, k=k_modes, frame=frame, frame_long=frame_long,
         **{f"energy_{m.replace(', ', '_').replace(' ', '_')}": energy[m] for m in cases},
         spectrum_helium=spectrum["helium"], spectrum_frozen=spectrum["frozen"])
plt.show()
