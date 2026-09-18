"""A maintained source-to-collector sheath against kinetic theory, and the gradient
through it against forward mode and finite differences."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import COLORS, C_ELECTRONS, C_IONS, C_THEORY, panel_label, record, savefig

from jaxincell import (Domain, Simulation, Solver, Source, Species, epsilon_0, mass_electron, potential,
                       elementary_charge as e_charge, speed_of_light as c)
from jaxincell.sheath import densities, floating_potential, source_density

T_E, DENSITY, MASS_RATIO, BEAM, BOX = 1.0, 1e16, 1836.0, 0.2, 10.0
CELLS, CAPACITY, EMIT, STORED, TRANSITS = 120, 120000, 120, 60, 6.0
SIGMA = np.sqrt(T_E * e_charge / mass_electron)
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
DEBYE = SIGMA / OMEGA_PE
LENGTH, DT = BOX * DEBYE, 0.1 / OMEGA_PE
PHI_WALL = float(floating_potential(BEAM))
AMPLITUDE = float(source_density(PHI_WALL))


def sheath(cells=CELLS, capacity=CAPACITY, emit=EMIT, dt=DT, reflection=0.0):
    domain = Domain(length=LENGTH, cells=cells, dt_over_dx_c=dt * c / (LENGTH / cells),
                    particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", capacity, -1.0, mass_electron, DENSITY, (np.sqrt(2) * SIGMA, 0, 0),
                        active=capacity // 4, quiet=True, reflection=(0.0, reflection),
                        source=Source(density=AMPLITUDE * DENSITY, vth=(np.sqrt(2) * SIGMA,) * 3,
                                      emit=emit, model="maxwellian"))
    ions = Species("ions", capacity, 1.0, MASS_RATIO * mass_electron, DENSITY, 0.0, (BEAM * SIGMA, 0, 0),
                   active=capacity // 4, quiet=True,
                   source=Source(density=DENSITY, vth=0.0, drift=(BEAM * SIGMA, 0, 0), emit=emit, model="beam"))
    return Simulation(domain, [electrons, ions], Solver(model="electrostatic"))


steps = STORED * (int(TRANSITS * LENGTH / (BEAM * SIGMA) / DT) // STORED)
simulation = sheath()
out = simulation.run(steps, seed=0, store_every=steps // STORED, store_particles=False, moments=True)
late = STORED // 2
faces = np.asarray(simulation.domain.faces)
phi = np.asarray(potential(out)) / T_E
profile = phi[late:].mean(axis=0)
window = np.asarray(out.moments[-1] - out.moments[late]) / ((STORED - late) * steps // STORED)
n_e, n_i = window[0, 0] / DENSITY, window[1, 0] / DENSITY
centre_phi = np.concatenate([[profile[0]], 0.5 * (profile[:-1] + profile[1:])])
reference_e, reference_i = densities(np.minimum(centre_phi, 0.0), PHI_WALL, BEAM, MASS_RATIO)
measured = phi[late:, -1]
collected = np.asarray(out.wall.collected)
late_charge = collected[-1, :, 1] - collected[late, :, 1]

# the gradient of a short response with respect to the collector's reflectivity, at three horizons
prepared = jax.block_until_ready(sheath(cells=48, capacity=48000, emit=48, dt=0.15 / OMEGA_PE,
                                        reflection=0.25).run(1200, seed=0, store_every=1200,
                                                             store_particles=False).state)
sensor = np.exp(-0.5 * ((np.asarray(Domain(length=LENGTH, cells=48).grid) + LENGTH / 96
                         - (LENGTH / 2 - 2.5 * DEBYE)) / (1.2 * DEBYE)) ** 2)
SENSOR = jnp.asarray(sensor / (sensor.sum() * LENGTH / 48))
STEPS_OF_H = np.logspace(-1, -7, 7)
curves = {}
for horizon in (5, 25, 100):
    def measure(r, horizon=horizon):
        run = sheath(cells=48, capacity=48000, emit=48, dt=0.15 / OMEGA_PE, reflection=r).run(
            horizon, store_every=horizon, store_particles=False, state=prepared)
        return jnp.sum(SENSOR * potential(run)[-1]) * LENGTH / 48 / T_E
    f = jax.jit(measure)
    reverse = float(jax.jit(jax.grad(measure))(0.25))
    forward = float(jax.jit(lambda r: jax.jvp(measure, (r,), (1.0,))[1])(0.25))
    mismatch = [abs(float((f(0.25 + h) - f(0.25 - h)) / (2 * h)) / reverse - 1) for h in STEPS_OF_H]
    curves[horizon] = (reverse, forward, np.maximum(mismatch, 1e-17))

fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.3))
distance = (LENGTH / 2 - faces) / DEBYE
axes[0].plot(distance, profile, color=C_ELECTRONS, label="measured")
axes[0].axhline(PHI_WALL, ls="--", lw=1.0, color=C_THEORY, label="kinetic reference")
axes[0].plot(0.0, measured.mean(), "o", ms=5, color=C_ELECTRONS)
axes[0].set(xlabel=r"distance from the collector ($\lambda_D$)", ylabel=r"$e\phi/T_e$",
            xlim=(distance.max(), -0.3), title="the sheath a source holds up")
axes[0].legend(loc="lower left")
panel_label(axes[0], "a")

centres = (LENGTH / 2 - np.asarray(simulation.domain.grid)) / DEBYE
axes[1].plot(centres, n_e, color=C_ELECTRONS, label=r"$n_e$")
axes[1].plot(centres, n_i, color=C_IONS, label=r"$n_i$")
axes[1].plot(centres, reference_e, "--", lw=1.0, color=C_THEORY, label=r"$n(\phi)$, kinetic")
axes[1].plot(centres, reference_i, "--", lw=1.0, color=C_THEORY)
axes[1].set(xlabel=r"distance from the collector ($\lambda_D$)", ylabel=r"$n/n_0$",
            ylim=(0, 1.35), xlim=(centres.max(), 0), title="the electrons are pushed out")
axes[1].legend(loc="lower right")
panel_label(axes[1], "b")

for horizon, colour in zip((5, 25, 100), (COLORS["blue"], COLORS["green"], COLORS["vermillion"])):
    axes[2].plot(STEPS_OF_H, curves[horizon][2], "o-", ms=3.5, color=colour, label=f"{horizon} steps")
axes[2].axhline(2e-16, ls=":", lw=1.0, color=C_THEORY)
axes[2].text(3e-2, 3e-16, "forward vs reverse mode", fontsize=7.5, color=C_THEORY)
axes[2].set(xscale="log", yscale="log", xlabel=r"finite-difference step in $r$",
            ylabel=r"$|\,\mathrm{FD}/\mathrm{AD} - 1\,|$", ylim=(1e-17, 5),
            title="the gradient, and how far it is smooth")
axes[2].legend(loc="lower left")
panel_label(axes[2], "c")
plt.tight_layout()
savefig(fig, "sheath_source")

record(
    source_sheath_phi_wall=round(float(measured.mean()), 4),
    source_sheath_phi_wall_error=round(float(measured.std() / np.sqrt(len(measured))), 4),
    source_sheath_phi_wall_reference=round(PHI_WALL, 5),
    source_sheath_phi_wall_deviation_percent=round(abs(measured.mean() / PHI_WALL - 1) * 100, 1),
    source_sheath_amplitude=round(AMPLITUDE, 5),
    source_sheath_net_current_percent=round(float((late_charge[1] - late_charge[0]) / late_charge[1] * 100), 2),
    source_sheath_density_error_electrons=round(float(np.abs(n_e - reference_e)[2:-2].max()), 3),
    source_sheath_density_error_ions=round(float(np.abs(n_i - reference_i)[2:-2].max()), 3),
    source_sheath_cells=CELLS, source_sheath_capacity=CAPACITY, source_sheath_emit=EMIT,
    source_sheath_steps=steps, source_sheath_mass_ratio=int(MASS_RATIO), source_sheath_beam_speed=BEAM,
    source_sheath_mach=round(BEAM * np.sqrt(MASS_RATIO), 2), source_sheath_box_debye=int(BOX),
    source_sheath_live_electrons=int((np.asarray(out.state.w)[:CAPACITY] > 0).sum()),
    source_sheath_live_ions=int((np.asarray(out.state.w)[CAPACITY:] > 0).sum()),
    **{f"gradient_modes_agree_{h}": float(f"{abs(curves[h][1] / curves[h][0] - 1):.1e}") for h in curves},
    **{f"gradient_best_mismatch_{h}": float(f"{min(curves[h][2]):.1e}") for h in curves},
    **{f"gradient_best_step_{h}": float(f"{STEPS_OF_H[int(np.argmin(curves[h][2]))]:.0e}") for h in curves},
)
