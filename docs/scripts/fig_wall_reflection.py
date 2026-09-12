"""What a partly reflecting wall sends back: the flux average of its reflection law."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_IONS, C_THEORY, panel_label, record, savefig

from jaxincell import Domain, Simulation, Solver, Species, quiet_start

SIGMA, LENGTH, N, RESTITUTION = 1e6, 1e-2, 200_000, 0.8
WIDTHS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
domain = Domain(length=LENGTH, cells=64, dt_over_dx_c=50.0, particle_bc="absorbing", field_bc="absorbing",
                restitution=RESTITUTION)
steps = int(round(0.1 * LENGTH / SIGMA / domain.dt))             # a tenth of a transit: nothing arrives twice
x, v = quiet_start(N, LENGTH, vth=(np.sqrt(2) * SIGMA, 0, 0))
w0 = 1e6 * LENGTH / N
returned, energy = [], []
for u in WIDTHS * SIGMA:
    law = lambda speed, u=u: jnp.exp(-speed ** 2 / (2 * u ** 2))
    electrons = Species.electrons(n=N, density=1e6, vth=(np.sqrt(2) * SIGMA, 0, 0), reflection=law).replace(x=x, v=v)
    w = np.asarray(Simulation(domain, [electrons], Solver()).run(steps, store_every=steps).weight[-1])
    hit = ~np.isclose(w, w0, rtol=1e-12, atol=0)
    returned.append(w[hit].sum() / (w0 * hit.sum()))
    energy.append(RESTITUTION ** 2 * (w[hit] * v[hit, 0] ** 2).sum() / (w0 * v[hit, 0] ** 2).sum())
    if u == SIGMA:
        speed, kept, hits = np.abs(v[hit, 0]) / SIGMA, w[hit] / w0, int(hit.sum())
returned, energy, flux = np.array(returned), np.array(energy), WIDTHS ** 2 / (WIDTHS ** 2 + 1)
print(f"  returned {returned.round(4)} against the flux average {flux.round(4)}")

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
fine = np.linspace(0.1, 4.5, 200)
axes[0].plot(fine, fine ** 2 / (fine ** 2 + 1), color=C_THEORY, label=r"flux average $u^2/(u^2+\sigma^2)$")
axes[0].plot(fine, fine / np.sqrt(fine ** 2 + 1), ":", color=C_THEORY, label="distribution average")
axes[0].plot(WIDTHS, returned, "o", color=C_ELECTRONS, label="simulation")
axes[0].set(xlabel=r"width of the law $u/\sigma$", ylabel="fraction returned", title="particles")
axes[0].legend(loc="lower right")
panel_label(axes[0], "a")

axes[1].plot(fine, RESTITUTION ** 2 * (fine ** 2 / (fine ** 2 + 1)) ** 2, color=C_THEORY,
             label=rf"$e^2 R_{{\rm eff}}^2$, $e={RESTITUTION}$")
axes[1].plot(WIDTHS, energy, "o", color=C_ELECTRONS, label="simulation")
axes[1].set(xlabel=r"$u/\sigma$", ylabel="fraction returned", title="normal energy")
axes[1].legend(loc="lower right")
panel_label(axes[1], "b")

bins = np.linspace(0, 4, 33)
centres, width = 0.5 * (bins[1:] + bins[:-1]), bins[1] - bins[0]
arrived, back = np.histogram(speed, bins)[0], np.histogram(speed, bins, weights=kept)[0]
arriving = hits * width * centres * np.exp(-centres ** 2 / 2)     # the flux distribution, in units of sigma
axes[2].bar(centres, back, width, color=C_ELECTRONS, alpha=0.6, label="returned")
axes[2].bar(centres, arrived - back, width, bottom=back, color=C_IONS, alpha=0.6, label="collected")
axes[2].plot(centres, arriving, "--", color=C_THEORY, label=r"$v f(v)$")
axes[2].plot(centres, arriving * np.exp(-centres ** 2 / 2), color=C_THEORY, label=r"$R(v)\,v f(v)$")
axes[2].set(xlabel=r"impact speed $|v_x|/\sigma$", ylabel="particles", title=r"at $u=\sigma$")
axes[2].legend()
panel_label(axes[2], "c")
fig.tight_layout()
savefig(fig, "wall_reflection")

record(reflection_restitution=RESTITUTION, reflection_particles=N, reflection_hits=hits,
       reflection_returned_sigma=round(float(returned[WIDTHS == 1.0][0]), 4),
       reflection_distribution_average_sigma=round(float(1 / np.sqrt(2)), 2),
       reflection_max_error=f"{float(np.abs(returned - flux).max()):.0e}",
       reflection_energy_max_error=f"{float(np.abs(energy - RESTITUTION ** 2 * flux ** 2).max()):.0e}")
