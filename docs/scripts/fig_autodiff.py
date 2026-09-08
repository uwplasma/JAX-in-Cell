"""Differentiability: reverse-mode gradients against central finite differences,
and gradient ascent finding the fastest-growing two-stream drift."""
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_THEORY, WIDE, maxwellian_populations, panel_label, record, savefig
from dispersion import electrostatic_epsilon, purely_growing_roots

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron,
                       elementary_charge as e_charge, speed_of_light as c)

LENGTH, CELLS, DENSITY, STEP = 0.01, 64, 4.37e17, 160
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
K = 2 * np.pi / LENGTH
REFERENCE = 4.0e7


def build(drift, n=8000):
    electrons = Species.electrons(n=n, density=DENSITY, vth=(0.05 * c, 0, 0), drift=(drift, 0, 0),
                                  plus_minus=True, quiet=True, perturbation_amplitude=5e-7,
                                  perturbation_mode=1)
    ions = Species.ions(n=n, density=DENSITY, electrons=electrons, quiet=True)
    return Simulation(Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=4.5), [electrons, ions],
                      Solver(filter_passes=0))


def amplification(drift):
    """ln of the seeded mode amplitude at a fixed time, which is gamma t plus a
    constant while the mode grows exponentially. Fixing the time rather than
    fitting a window keeps the objective a smooth function of the drift."""
    out = build(drift).run(STEP, seed=3, store_particles=False)
    return jnp.log(jnp.abs(jnp.fft.rfft(out.E[:, :, 0], axis=1)[-1, 1]))


value_and_grad = jax.jit(jax.value_and_grad(amplification))
evaluate = jax.jit(amplification)
start = time.perf_counter()
value, gradient = value_and_grad(REFERENCE)
gradient.block_until_ready()
first_call = time.perf_counter() - start
start = time.perf_counter()
value_and_grad(REFERENCE * 1.01)[1].block_until_ready()
warm_call = time.perf_counter() - start
gradient = float(gradient)

steps = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
differences = np.array([float((evaluate(REFERENCE + h) - evaluate(REFERENCE - h)) / (2 * h)) for h in steps])
relative = np.abs(differences / gradient - 1)
print(f"  reverse mode {gradient:.6e} per (m/s); best central difference "
      f"{differences[np.argmin(relative)]:.6e} at h = {steps[np.argmin(relative)]:.0e} "
      f"(relative error {relative.min():.1e})")

fig, axes = plt.subplots(1, 2, figsize=WIDE)
axes[0].loglog(steps, relative, "o-", color=C_ELECTRONS)
axes[0].set(xlabel=r"central-difference step $h$ (m/s)", ylabel=r"$|\,\mathrm{FD}/\nabla_\mathrm{AD}-1|$",
            title="one gradient, checked against differences")
axes[0].text(0.04, 0.9, "round-off\ndominates", transform=axes[0].transAxes, fontsize=7.5, color="0.4")
axes[0].text(0.72, 0.9, "truncation\ndominates", transform=axes[0].transAxes, fontsize=7.5, color="0.4")
panel_label(axes[0], "a")

drift, history = 2.5e7, []
for iteration in range(12):
    objective, slope = value_and_grad(drift)
    history.append((drift, float(objective)))
    drift = float(drift + 1.5e14 * slope)
drifts, objectives = np.array(history).T

scan = np.linspace(2.4e7, 6.0e7, 19)
axes[1].plot(K * scan / OMEGA_PE, [float(evaluate(d)) for d in scan], "-", color="0.6", label="scan")
axes[1].plot(K * drifts / OMEGA_PE, objectives, "o-", ms=4, color=C_ELECTRONS, label="gradient ascent")


def kinetic_rate(drift):
    populations = maxwellian_populations(build(drift))
    roots = purely_growing_roots(lambda w: electrostatic_epsilon(w, K, populations), OMEGA_PE)
    return max(roots) / OMEGA_PE if roots else 0.0


fine = np.linspace(2.4e7, 6.0e7, 40)
optimum = K * fine[int(np.argmax([kinetic_rate(d) for d in fine]))] / OMEGA_PE
axes[1].axvline(optimum, color=C_THEORY, ls="--", label=fr"fastest kinetic mode, {optimum:.2f}")
time_label = STEP * 4.5 * LENGTH / CELLS / c * OMEGA_PE
axes[1].set(xlabel=r"$k v_0/\omega_{pe}$", title="ascent finds the fastest-growing beam",
            ylabel=fr"$\ln|E_{{k=1}}|$ at $t\,\omega_{{pe}}={time_label:.0f}$")
axes[1].legend(loc="lower center")
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "autodiff")

found = K * history[-1][0] / OMEGA_PE
print(f"  gradient ascent reached k v0/omega_pe = {found:.3f}; fastest kinetic mode at {optimum:.3f}")
record(autodiff_gradient=f"{gradient:.4e}",
       autodiff_best_relative_error=f"{float(relative.min()):.1e}",
       autodiff_best_step=f"{float(steps[np.argmin(relative)]):.0e}",
       autodiff_grad_time_first_s=round(first_call, 2), autodiff_grad_time_warm_s=round(warm_call, 3),
       autodiff_ascent_iterations=len(history),
       autodiff_ascent_k_v0_over_wpe=round(float(found), 3),
       autodiff_kinetic_optimum_k_v0_over_wpe=round(float(optimum), 3),
       autodiff_cold_optimum_k_v0_over_wpe=round(float(np.sqrt(3 / 8)), 3),
       autodiff_ascent_deviation_percent=round(float(100 * abs(found / optimum - 1)), 1))
