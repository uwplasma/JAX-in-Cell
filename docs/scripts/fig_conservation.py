"""Conservation of energy, momentum and charge by the two schemes, in the setup of
examples/conservation.py: the energy error as the Picard iteration converges, the momentum,
and the discrete Gauss law in a periodic box, between absorbing walls and at every wall."""
import matplotlib.pyplot as plt
import numpy as np
from common import C_EXPLICIT, C_IMPLICIT, panel_label, record, savefig

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, speed_of_light as c

STEPS, COURANT = 400, 4.5


def run(walls="periodic", **solver):
    electrons = Species.electrons(n=4000, density=4.37e17, vth=(0.05 * c, 0, 0), drift=(5e7, 0, 0),
                                  plus_minus=True, sampling="quiet", perturbation_amplitude=5e-7,
                                  perturbation_mode=1)
    ions = Species.ions(n=4000, density=4.37e17, electrons=electrons, sampling="quiet")
    domain = Domain(length=0.01, cells=64, dt_over_dx_c=COURANT, particle_bc=walls, field_bc=walls)
    simulation = Simulation(domain, [electrons, ions], Solver(**solver))
    out = simulation.run(STEPS, seed=3)
    return simulation, out, diagnostics(out)


def largest(d, key):
    return f"{float(np.max(np.asarray(d[key]))):.1e}"


simulation, explicit, d_explicit = run(algorithm="explicit")
implicit = {n: run(algorithm="implicit", picard_iterations=n)[2] for n in (1, 2, 4, 8)}
walled = {scheme: run("absorbing", algorithm=scheme)[2] for scheme in ("explicit", "implicit")}
t = np.asarray(explicit.t) * float(simulation.plasma_frequency())

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
axes[0].semilogy(t, np.asarray(d_explicit["energy_error"]) + 1e-17, color=C_EXPLICIT, label="explicit (Boris leapfrog)")
for n, d in implicit.items():
    axes[0].semilogy(t, np.asarray(d["energy_error"]) + 1e-17, color=C_IMPLICIT, alpha=0.3 + 0.7 * np.log2(n) / 3,
                     label=f"implicit, {n} Picard")
for scheme, d, color in (("explicit", d_explicit, C_EXPLICIT), ("implicit, 8 Picard", implicit[8], C_IMPLICIT)):
    axes[1].semilogy(t, np.asarray(d["momentum_error"]) + 1e-17, color=color, label=scheme)
    axes[2].semilogy(t, np.asarray(d["gauss_residual"]) + 1e-17, color=color, label=f"{scheme}, periodic")
for scheme, color in (("explicit", C_EXPLICIT), ("implicit", C_IMPLICIT)):
    axes[2].semilogy(t, np.asarray(walled[scheme]["gauss_residual"]) + 1e-17, "--", color=color,
                     label=f"{scheme}, absorbing walls")
axes[0].set(ylabel=r"$|W(t)-W(0)|\,/\,W(0)$", title="energy")
axes[1].set(ylabel=r"$|P(t)-P(0)|\,/\,\sum_p |p_p(0)|$", title="momentum")
axes[2].set(ylabel=r"$\max_i|\nabla\!\cdot\!E-\rho/\epsilon_0|\,/\,(en/\epsilon_0)$", title="charge (Gauss law)")
for ax, label in zip(axes, "abc"):
    ax.set_xlabel(r"$t\,\omega_{pe}$")
    ax.legend(fontsize=7)
    panel_label(ax, label)
fig.tight_layout()
savefig(fig, "conservation")

# the Gauss law has to hold at every wall, not only in a periodic box, and in both schemes
walls = {}
for wall, particle_bc, field_bc, reflection in (
        ("periodic", "periodic", "periodic", 0.0), ("reflective", "reflective", "reflective", 0.0),
        ("absorbing", "absorbing", "absorbing", 0.0), ("reflecting", "absorbing", "absorbing", 0.5),
        ("thermal", ("thermal", "absorbing"), ("reflective", "absorbing"), 0.0)):
    e = Species.electrons(n=2000, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0), sampling="quiet",
                          reflection=reflection)
    i = Species.ions(n=2000, density=1e17, electrons=e, sampling="quiet")
    domain = Domain(length=1e-2, cells=32, dt_over_dx_c=1.0, particle_bc=particle_bc, field_bc=field_bc)
    for suffix, solver in (("", Solver(filter_passes=2, filter_strides=(1, 2))), ("_implicit", Solver("implicit"))):
        out = Simulation(domain, [e, i], solver).run(120, seed=0)
        walls[f"gauss_residual_{wall}_wall{suffix}"] = largest(diagnostics(out), "gauss_residual")

charge = np.asarray(explicit.charge * explicit.weight[-1])
on_grid = np.asarray(explicit.rho).sum(axis=1) * explicit.dx
record(energy_courant=COURANT, energy_steps=STEPS,
       energy_omega_pe_dt=round(float(simulation.plasma_frequency() * simulation.domain.dt), 4),
       energy_error_max_explicit=largest(d_explicit, "energy_error"),
       **{f"energy_error_max_implicit_{n}": largest(d, "energy_error") for n, d in implicit.items()},
       gauss_residual_max_explicit=largest(d_explicit, "gauss_residual"),
       gauss_residual_max_implicit=largest(implicit[8], "gauss_residual"),
       charge_error_relative=f"{float(np.abs(on_grid - charge.sum()).max() / np.abs(charge).sum()):.1e}",
       momentum_error_relative=largest(d_explicit, "momentum_error"),
       momentum_error_implicit=largest(implicit[8], "momentum_error"), **walls)
