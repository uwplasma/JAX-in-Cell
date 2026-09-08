"""What the three wall types do to a drifting plasma."""
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, record, savefig

from jaxincell import Domain, Simulation, Solver, Species, diagnostics, speed_of_light as c

LENGTH, STEPS = 1e-2, 400


def run(kind):
    electrons = Species.electrons(n=4000, density=1e17, vth=(0.02 * c, 0, 0), drift=(0.05 * c, 0, 0),
                                  quiet=True)
    ions = Species.ions(n=4000, density=1e17, electrons=electrons, quiet=True).replace(drift=(0.05 * c, 0, 0))
    domain = Domain(length=LENGTH, cells=64, dt_over_dx_c=1.0, particle_bc=kind, field_bc=kind)
    return Simulation(domain, [electrons, ions], Solver(filter_passes=0)).run(STEPS, seed=0)


kinds = ("periodic", "reflective", "absorbing")
outputs = {kind: run(kind) for kind in kinds}

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
for ax, kind in zip(axes, kinds):
    out = outputs[kind]
    x, v = out.particles("electrons")
    alive = np.asarray(out.charge)[: out.counts[0]] != 0
    ax.plot(np.asarray(x[-1, alive, 0]) / LENGTH, np.asarray(v[-1, alive, 0]), ".", ms=1.2,
            color=C_ELECTRONS, rasterized=True)
    ax.set(xlabel="$x/L$", title=kind, xlim=(-0.55, 0.55))
    ax.grid(False)
    kept = 100 * alive.mean()
    ax.text(0.03, 0.94, f"{kept:.0f}% of the electrons remain", transform=ax.transAxes, fontsize=8,
            va="top")
axes[0].set_ylabel(r"$v_x$ (m/s)")
fig.suptitle(r"electron phase space after 400 steps of a plasma drifting to the right", fontsize=10)
fig.tight_layout()
savefig(fig, "boundaries")

kept = {kind: float((np.asarray(outputs[kind].charge) != 0).mean()) for kind in kinds}
energy = {kind: float(np.max(np.abs(np.asarray(diagnostics(outputs[kind])["total"])
                                    / diagnostics(outputs[kind])["total"][0] - 1))) for kind in kinds}
for kind in kinds:
    print(f"  {kind:11s}: {100 * kept[kind]:5.1f}% of the particles kept, energy error {energy[kind]:.1e}")
record(boundary_steps=STEPS,
       **{f"boundary_kept_percent_{kind}": round(100 * kept[kind], 1) for kind in kinds},
       **{f"boundary_energy_error_{kind}": f"{energy[kind]:.1e}" for kind in kinds})
