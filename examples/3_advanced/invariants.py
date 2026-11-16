"""What the time integrators conserve: collisions placed at the integer time, and the implicit
electrostatic scheme.

Two checks of the step itself, each against what the discrete scheme promises rather than
against another run:

* **Collisions at the integer time.** The explicit step kicks the velocity u^n -> u^{n+1} in the
  field at x^{n+1/2}; collisions act at x^{n+1} = x^{n+1/2} + dt u^{n+1}/2, between the two
  half drifts. Electrons and a 25 m_e species of the same charge share one harmonic well, all
  inside one cell so that every collision conserves its pair's energy exactly, with a collision
  rate of about a tenth of the well frequency. The total energy then keeps the leapfrog's own
  second-order error, that of the collisionless run, at every step size.
* **The implicit electrostatic scheme.** Solver(algorithm="implicit", model="electrostatic")
  advances E_x by Ampere's law with the continuity current, with no Poisson solve: the energy
  is conserved to round-off once the Picard iteration has converged, and the Gauss law holds
  at every step. The explicit electrostatic leapfrog on the same two-stream run is shown for
  scale.

Run with `--quick` for two step sizes and fewer Picard counts.
"""

import os
import sys
from pathlib import Path

# Double precision: the implicit energy error goes down to 1e-16.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Collisions, Domain, Simulation, Solver, Species, diagnostics, figure, mass_electron,
                       save_run, elementary_charge as e_charge, speed_of_light as c)

# --- what to change ---------------------------------------------------------------------
quick = "--quick" in sys.argv
steps_per_period = (0.4, 0.2) if quick else (0.4, 0.2, 0.1)   # omega dt of the well
picard = (2, 4, 8) if quick else (2, 3, 4, 6, 8, 12)           # Picard iterations per implicit step
heavy = 25.0                                                    # mass of the second species / m_e


# --- collisions at the integer time -----------------------------------------------------------
def oscillators(omega_dt, collide, n=1000, periods=16.0):
    """Largest relative error of sum(m v^2/2 + e k (x - x0)^2/2) over sixteen well periods."""
    length, omega = 0.5, 1e9
    k = omega ** 2 * mass_electron / e_charge
    rng = np.random.default_rng(0)
    x, v = np.zeros((n, 3)), rng.normal(0.0, 5e5, (n, 3))
    x0, x[:, 0] = length / 128, length / 128 + rng.uniform(-1e-3, 1e-3, n)
    species = [Species("electrons", n, -1.0, mass_electron, 1e10, x=x, v=v),
               Species("heavy", n, -1.0, heavy * mass_electron, 1e10, x=x, v=v / np.sqrt(heavy))]
    domain = Domain(length=length, cells=64, time_step=omega_dt / omega)
    well = np.zeros((64, 3))
    well[:, 0] = k * (np.asarray(domain.grid) + domain.dx / 2 - x0)
    pairs = Collisions(coulomb_log=1e11, pairs=(("electrons", "heavy"),)) if collide else None
    steps = 40 * int(round(2 * np.pi * periods / omega_dt / 40))
    out = Simulation(domain, species, Solver(model="electrostatic"), pairs, external_E=well).run(
        steps, store_every=steps // 40)
    mass = np.r_[np.full(n, mass_electron), np.full(n, heavy * mass_electron)]
    energy = np.sum(np.asarray(out.weight) * (0.5 * mass * np.sum(np.asarray(out.v) ** 2, axis=-1)
                                              + 0.5 * e_charge * k * (np.asarray(out.x)[:, :, 0] - x0) ** 2), axis=1)
    return float(np.max(np.abs(energy / energy[0] - 1)))


collisionless = [oscillators(w, False) for w in steps_per_period]
collisional = [oscillators(w, True) for w in steps_per_period]
for w, a, b in zip(steps_per_period, collisionless, collisional):
    print(f"well, omega dt = {w}: energy error collisionless {a:.2e}, with collisions {b:.2e}")

# --- the implicit electrostatic scheme ---------------------------------------------------------
electrons = Species.electrons(n=2000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(6e7, 0, 0), plus_minus=True,
                              sampling="lattice", perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=2000, density=4.37e17, electrons=electrons, sampling="lattice")
domain = Domain(length=0.01, cells=64, dt_over_dx_c=4.5)


def two_stream(solver):
    d = diagnostics(Simulation(domain, [electrons, ions], solver).run(150, seed=3))
    return np.asarray(d["energy_error"]), np.asarray(d["gauss_residual"])


picard_error = []
for iterations in picard:
    energy, gauss = two_stream(Solver(algorithm="implicit", model="electrostatic", picard_iterations=iterations))
    picard_error.append(float(energy.max()))
    print(f"implicit electrostatic, {iterations:2d} Picard iterations: energy error {energy.max():.1e}, "
          f"Gauss residual {gauss.max():.1e}")
implicit_energy, implicit_gauss = two_stream(Solver(algorithm="implicit", model="electrostatic"))
explicit_energy, explicit_gauss = two_stream(Solver(model="electrostatic"))
print(f"explicit electrostatic: energy error {explicit_energy.max():.1e}, Gauss residual {explicit_gauss.max():.1e}")

# --- the figure -------------------------------------------------------------------------------
fig, axes = figure(3)
w = np.array(steps_per_period)
axes[0].loglog(w, collisionless, "o-", label="collisionless")
axes[0].loglog(w, collisional, "s--", label="with collisions")
axes[0].loglog(w, collisionless[0] * (w / w[0]) ** 2, "k:", lw=1.5, label=r"$\propto\Delta t^2$")
axes[0].set_xticks(w, [str(x) for x in w])
axes[0].minorticks_off()
axes[0].set(xlabel=r"$\omega\,\Delta t$", ylabel="max relative energy error", title="collisions at the integer time")
axes[0].legend(frameon=False)
axes[1].semilogy(picard, np.maximum(picard_error, 1e-17), "o-", color="C3")
axes[1].set(xlabel="Picard iterations per step", ylabel="max relative energy error",
            title="implicit electrostatic: energy")
steps = np.arange(implicit_energy.size)
axes[2].semilogy(steps, np.maximum(explicit_energy, 1e-17), label="explicit, energy")
axes[2].semilogy(steps, np.maximum(implicit_energy, 1e-17), color="C3", label="implicit, energy")
axes[2].semilogy(steps, np.maximum(implicit_gauss, 1e-17), color="C3", ls=":", label="implicit, Gauss law")
axes[2].set(xlabel="step", ylabel="relative error", title="electrostatic two-stream", ylim=(1e-17, 1e-2))
axes[2].legend(frameon=False, loc="center right")
plt.tight_layout()

save_run(Path.cwd() / ("invariants_quick" if quick else "invariants"), "invariants",
         dict(omega_dt=list(steps_per_period), picard=list(picard), heavy=heavy, quick=quick),
         dict(collisionless=collisionless, collisional=collisional, picard_energy_error=picard_error,
              implicit_energy_error=float(implicit_energy.max()), implicit_gauss=float(implicit_gauss.max()),
              explicit_energy_error=float(explicit_energy.max()), explicit_gauss=float(explicit_gauss.max())),
         figure=fig, implicit_energy=implicit_energy, explicit_energy=explicit_energy, implicit_gauss=implicit_gauss)
plt.show()
