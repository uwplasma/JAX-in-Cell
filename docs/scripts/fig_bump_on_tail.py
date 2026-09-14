"""Bump-on-tail instability: the growth of the resonant mode against kinetic
theory, and the quasilinear plateau it leaves behind."""
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_FIT, C_IONS, C_THEORY, WIDE, panel_label, record, savefig
from dispersion import electrostatic_epsilon, newton, plasma_frequency

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

LENGTH, CELLS, MODE, STEPS = 1.0, 128, 5, 2000
OMEGA_PE = 0.05 * c * CELLS / LENGTH
DENSITY = OMEGA_PE ** 2 * epsilon_0 * mass_electron / e_charge ** 2
BEAM_FRACTION, BEAM_DRIFT_OVER_VTH, BEAM_WIDTH = 0.03, 5.0, 0.7
# A wave resonates with the beam when omega_pe / k = v_beam, so the mode that
# grows fastest is fixed by the thermal speed: choosing v_th this way puts it at
# MODE, comfortably inside the box and well resolved by the grid.
V_TH = OMEGA_PE * LENGTH / (2 * np.pi * MODE * BEAM_DRIFT_OVER_VTH)
BEAM_DRIFT, BEAM_VTH = BEAM_DRIFT_OVER_VTH * V_TH, BEAM_WIDTH * V_TH
SEED_AK = 2e-3

populations = [
    {"wp": plasma_frequency((1 - BEAM_FRACTION) * DENSITY, e_charge, mass_electron), "u": 0.0, "vth": V_TH},
    {"wp": plasma_frequency(BEAM_FRACTION * DENSITY, e_charge, mass_electron), "u": BEAM_DRIFT, "vth": BEAM_VTH},
]


def theory(mode):
    """Root of the electrostatic dispersion relation nearest the Langmuir wave.

    Newton is started from the real Langmuir root rather than from a grid of
    guesses, which keeps it on the physical Riemann sheet of Z."""
    k = 2 * np.pi * mode / LENGTH
    guess = OMEGA_PE * np.sqrt(1 + 3 * (k * V_TH / (np.sqrt(2) * OMEGA_PE)) ** 2) + 0.02j * OMEGA_PE
    root = newton(lambda w: electrostatic_epsilon(w, k, populations), guess)
    return (root.real / OMEGA_PE, root.imag / OMEGA_PE) if root is not None else (np.nan, np.nan)


bulk = Species.electrons(n=80000, density=(1 - BEAM_FRACTION) * DENSITY, vth=(V_TH, 0, 0), quiet=True,
                         name="bulk", perturbation_mode=MODE,
                         perturbation_amplitude=SEED_AK * LENGTH / (2 * np.pi * MODE))
beam = Species.electrons(n=40000, density=BEAM_FRACTION * DENSITY, vth=(BEAM_VTH, 0, 0), quiet=True,
                         drift=(BEAM_DRIFT, 0, 0), name="beam")
ions = Species.ions(n=10000, density=DENSITY, mass_ratio=1e9, vth=(0, 0, 0), quiet=True)
output = Simulation(Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=1.0), [bulk, beam, ions],
                    Solver(filter_passes=0)).run(STEPS, seed=0, store_every=16)

t = np.asarray(output.t) * OMEGA_PE
amplitude = np.abs(np.fft.rfft(np.asarray(output.E[:, :, 0]), axis=1)[:, MODE]) / CELLS
peak = int(np.argmax(amplitude))
# above the settling seed transient, below the turn towards saturation
window = ((amplitude > 1.2 * amplitude[:20].max()) & (amplitude < 0.3 * amplitude[peak])
          & (np.arange(t.size) < peak))
slope, intercept = np.polyfit(t[window], np.log(amplitude[window]), 1)
omega_theory, gamma_theory = theory(MODE)

fig, axes = plt.subplots(1, 2, figsize=WIDE)
axes[0].semilogy(t, amplitude, color=C_ELECTRONS, label=fr"$|E_{{k={MODE}}}(t)|$")
span = np.linspace(t[window][0], t[window][-1], 2)
axes[0].semilogy(span, np.exp(intercept + slope * span), "--", color=C_FIT,
                 label=fr"fit: $\gamma={slope:.4f}\,\omega_{{pe}}$")
axes[0].semilogy(span, np.exp(intercept + gamma_theory * span), ":", color=C_THEORY,
                 label=fr"kinetic: $\gamma={gamma_theory:.4f}\,\omega_{{pe}}$")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|E_k|$ (V/m)", title="the resonant mode")
axes[0].legend(loc="lower right")
panel_label(axes[0], "a")

electrons = np.asarray(output.species) < 2
edges = np.linspace(-4 * V_TH, 9 * V_TH, 220)
centres = 0.5 * (edges[1:] + edges[:-1])
for step, style, label in ((0, "--", "initial"), (-1, "-", "final")):
    counts, _ = np.histogram(np.asarray(output.v[step, electrons, 0]), edges, density=True)
    axes[1].semilogy(centres / V_TH, counts, style, color=C_ELECTRONS if step else C_IONS, label=label)
phase_velocity = omega_theory * OMEGA_PE / (2 * np.pi * MODE / LENGTH) / V_TH
axes[1].axvline(phase_velocity, color="0.5", lw=1.0, ls="-.",
                label=fr"$v_\varphi$ of mode {MODE}")
axes[1].set(xlabel=r"$v_x/v_{th}$", ylabel=r"$f(v_x)$", title="the bump flattens into a plateau",
            ylim=(1e-9, None))
axes[1].legend(loc="lower left")
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "bump_on_tail")

print(f"  mode {MODE}: measured gamma {slope:.4f}, kinetic {gamma_theory:.4f}")
record(bump_on_tail_mode=MODE, bump_on_tail_beam_fraction=BEAM_FRACTION,
       bump_on_tail_beam_drift_over_vth=BEAM_DRIFT_OVER_VTH,
       bump_on_tail_beam_width_over_vth=BEAM_WIDTH,
       bump_on_tail_dx_over_debye=round(float(LENGTH / CELLS / (V_TH / (np.sqrt(2) * OMEGA_PE))), 2),
       bump_on_tail_gamma_measured=round(float(slope), 4),
       bump_on_tail_gamma_theory=round(float(gamma_theory), 4),
       bump_on_tail_gamma_deviation_percent=round(float(100 * abs(slope - gamma_theory) / gamma_theory), 1),
       bump_on_tail_omega_theory=round(float(omega_theory), 4),
       bump_on_tail_particles=130000, bump_on_tail_cells=CELLS, bump_on_tail_seed_ak=SEED_AK,
       bump_on_tail_bulk_k_lambda_D=round(float(2 * np.pi * MODE / LENGTH * V_TH / (np.sqrt(2) * OMEGA_PE)), 3),
       bump_on_tail_phase_velocity_over_vth=round(float(phase_velocity), 2))
