"""Conservation properties: the energy error of the two schemes as the Picard
iteration converges, and the discrete Gauss law and charge budget."""
import matplotlib.pyplot as plt
import numpy as np
from common import C_EXPLICIT, C_IMPLICIT, WIDE, panel_label, record, savefig

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, speed_of_light as c

STEPS, COURANT = 400, 4.5


def run(**solver):
    electrons = Species.electrons(n=4000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(5e7, 0, 0),
                                  plus_minus=True, quiet=True, perturbation_amplitude=5e-7,
                                  perturbation_mode=1)
    ions = Species.ions(n=4000, density=4.37e17, electrons=electrons, quiet=True)
    simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=COURANT), [electrons, ions],
                            Solver(**solver))
    return simulation, simulation.run(STEPS, seed=3)


error = lambda out: np.abs(np.asarray(diagnostics(out)["total"]) / diagnostics(out)["total"][0] - 1)
simulation, explicit = run(algorithm="explicit")
implicit = {n: run(algorithm="implicit", picard_iterations=n)[1] for n in (1, 2, 4, 8)}

fig, axes = plt.subplots(1, 2, figsize=WIDE)
t = np.asarray(explicit.t) * float(simulation.plasma_frequency())
axes[0].semilogy(t, error(explicit) + 1e-17, color=C_EXPLICIT, label="explicit (Boris leapfrog)")
for iterations, out in implicit.items():
    axes[0].semilogy(t, error(out) + 1e-17, color=C_IMPLICIT, alpha=0.3 + 0.7 * np.log2(iterations) / 3,
                     label=f"implicit, {iterations} Picard")
axes[0].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$|W(t)/W(0)-1|$", title="total energy error")
axes[0].legend(fontsize=7.5)
panel_label(axes[0], "a")

axes[1].semilogy(t, np.asarray(diagnostics(explicit)["gauss_residual"]) + 1e-18, color=C_EXPLICIT,
                 label="explicit")
axes[1].semilogy(t, np.asarray(diagnostics(implicit[8])["gauss_residual"]) + 1e-18, color=C_IMPLICIT,
                 label="implicit, 8 Picard")
axes[1].set(xlabel=r"$t\,\omega_{pe}$", title="Gauss law residual",
            ylabel=r"$\max_i|\nabla\!\cdot\!E-\rho/\epsilon_0| \,/\, \max_i|\rho/\epsilon_0|$")
axes[1].legend()
panel_label(axes[1], "b")
fig.tight_layout()
savefig(fig, "conservation")

# the Gauss law has to hold at every wall, not only in a periodic box
walls = {}
for wall in ("periodic", "reflective", "absorbing"):
    e = Species.electrons(n=2000, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0), quiet=True)
    i = Species.ions(n=2000, density=1e17, electrons=e, quiet=True)
    domain = Domain(length=1e-2, cells=32, dt_over_dx_c=1.0, particle_bc=wall, field_bc=wall)
    out = Simulation(domain, [e, i], Solver(filter_passes=2, filter_strides=(1, 2))).run(120, seed=0)
    walls[wall] = float(np.asarray(diagnostics(out)["gauss_residual"]).max())
    print(f"  {wall:11s}: gauss residual {walls[wall]:.1e}")

charge = np.asarray(explicit.charge)
on_grid = np.asarray(explicit.rho).sum(axis=1) * explicit.dx
momentum = np.asarray(diagnostics(explicit)["momentum"])[:, 0]
content = float(np.sum(np.asarray(explicit.mass) * np.abs(np.asarray(explicit.v[0, :, 0]))))
record(energy_courant=COURANT, energy_steps=STEPS,
       energy_omega_pe_dt=round(float(simulation.plasma_frequency() * simulation.domain.dt), 4),
       energy_error_max_explicit=f"{float(error(explicit).max()):.1e}",
       **{f"energy_error_max_implicit_{n}": f"{float(error(out).max()):.1e}" for n, out in implicit.items()},
       gauss_residual_max_explicit=f"{float(np.asarray(diagnostics(explicit)['gauss_residual']).max()):.1e}",
       charge_error_relative=f"{float(np.abs(on_grid - charge.sum()).max() / np.abs(charge).sum()):.1e}",
       momentum_error_relative=f"{float(np.abs(momentum - momentum[0]).max() / content):.1e}",
       **{f"gauss_residual_{wall}_wall": f"{value:.1e}" for wall, value in walls.items()})
