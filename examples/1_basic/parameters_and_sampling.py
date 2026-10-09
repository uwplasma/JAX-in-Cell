"""Physical inputs, numerical inputs, and how the particles are loaded.

A run is set by two kinds of number. The physical ones describe the plasma: a
temperature, a density, a mass. The numerical ones describe how it is resolved: cells
per Debye length, steps per plasma period, particles per cell. The code takes SI units
and derives the scales that connect the two; this prints them, builds the same plasma
three ways, and measures what the loading does to it.

Thermal speeds follow v_th = sqrt(2 T / m), so the Debye length is
lambda_D = v_th / (sqrt(2) omega_pe) = sqrt(eps0 T / (n e^2)).

`Species(sampling=...)` has three loadings:

* "random": positions and velocities drawn at random. The number of particles in a cell
  is then Poisson-distributed, so the density varies from cell to cell by
  1/sqrt(particles per cell), and the sampled temperature misses the requested one by
  about sqrt(2/N): noise that the field sees at once.
* "lattice", the default: equally spaced positions and random velocities. The density is
  uniform to round-off; the velocities keep their sampling noise.
* "low_noise": equally spaced positions and velocities at the quantiles of the Maxwellian.
  Both moments are exact up to the discreteness of the quantiles, and the field starts
  orders of magnitude quieter: the loading for following small, linear signals.

Each line below prints the measured spread against the prediction.
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, diagnostics, figure, mass_electron, save_run,
                       elementary_charge as e_charge, epsilon_0)

# --- physical inputs --------------------------------------------------------------------
temperature_ev = 10.0                  # electron temperature, eV
density = 1e18                         # m^-3

# --- numerical inputs -------------------------------------------------------------------
debye_lengths = 64                     # box length in Debye lengths
cells_per_debye = 1.0                  # dx = lambda_D / cells_per_debye
particles_per_cell = 100
dt_omega_pe = 0.1                      # omega_pe dt
periods = 5                            # how long to run, in plasma periods

# --- derived scales ---------------------------------------------------------------------
v_th = np.sqrt(2 * temperature_ev * e_charge / mass_electron)
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye = v_th / (np.sqrt(2) * omega_pe)
cells = int(round(debye_lengths * cells_per_debye))
n = cells * particles_per_cell
steps = 2 * int(round(periods * np.pi / dt_omega_pe))   # even: every second step is stored
print(f"v_th = sqrt(2T/m) = {v_th:.4e} m/s,  omega_pe = {omega_pe:.4e} rad/s,  lambda_D = {debye:.4e} m")
print(f"{cells} cells, {n} particles, {steps} steps of {dt_omega_pe / omega_pe:.3e} s\n")

results, energy = {}, {}
for sampling in ("random", "lattice", "low_noise"):
    electrons = Species.electrons(n=n, density=density, vth=(v_th, 0, 0), sampling=sampling)
    ions = Species.ions(n=n, density=density, mass_ratio=1e9, vth=(0, 0, 0), sampling="lattice")
    simulation = Simulation(Domain(length=debye_lengths * debye, cells=cells, time_step=dt_omega_pe / omega_pe),
                            [electrons, ions], Solver(model="electrostatic"))
    if sampling == "random":   # the code reports the same scales it was given
        assert np.isclose(float(simulation.plasma_frequency()), omega_pe)
        assert np.isclose(float(simulation.debye_length()), debye)
    # the state the run starts from: the leapfrog keeps positions half a step ahead, so step them back
    loaded, _ = simulation.initial_state(jax.random.PRNGKey(0))
    v = np.asarray(loaded.u[:n, 0])
    x = np.asarray(loaded.x[:n, 0]) - v * simulation.domain.dt / 2
    out = simulation.run(steps, seed=0, store_every=2, store_particles=False)
    counts, _ = np.histogram(x, cells, range=(-debye_lengths * debye / 2, debye_lengths * debye / 2))
    density_spread = float(np.std(counts / particles_per_cell))
    temperature_error = float(np.mean(v ** 2) / (v_th ** 2 / 2) - 1)
    energy[sampling] = (np.asarray(out.t) * omega_pe, np.asarray(diagnostics(out)["electric"]))
    results[sampling] = dict(density_spread=density_spread, temperature_error=temperature_error,
                             electric_energy_end=float(energy[sampling][1][-1]))
    print(f"{sampling:>8}: density spread {density_spread:.2e} "
          f"(random: 1/sqrt(ppc) = {particles_per_cell ** -0.5:.2e}), "
          f"temperature error {temperature_error:+.1e} (random: +-{np.sqrt(2 / n):.1e})")

fig, ax = figure()
for sampling, (t, W) in energy.items():
    ax.semilogy(t, np.maximum(W, 1e-30), lw=2, label=sampling)
ax.set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"electric energy (J/m$^2$)", title="field noise of each loading")
ax.legend()
fig.tight_layout()
save_run(Path.cwd() / "parameters_and_sampling", "parameters_and_sampling",
         dict(temperature_ev=temperature_ev, density=density, debye_lengths=debye_lengths,
              cells_per_debye=cells_per_debye, particles_per_cell=particles_per_cell, dt_omega_pe=dt_omega_pe,
              steps=steps),
         dict(v_th=v_th, omega_pe=omega_pe, debye_length=debye, predicted_density_spread=particles_per_cell ** -0.5,
              predicted_temperature_error=np.sqrt(2 / n), samplings=results), figure=fig)
plt.show()
