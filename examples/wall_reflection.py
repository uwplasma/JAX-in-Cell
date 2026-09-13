"""What a partly reflecting wall sends back.

A wall does not sample the velocity distribution of a plasma, it samples the flux.
From a uniform Maxwellian of spread sigma, the particles that reach it with normal
speed near v are in proportion to v f(v), so a reflection law R(v) returns its flux
average

    R_eff = (1/sigma^2) int_0^inf R(v) v exp(-v^2 / 2 sigma^2) dv .

For a Gaussian law R = exp(-v^2 / 2u^2) that is u^2 / (u^2 + sigma^2), not the
average over the distribution, u / sqrt(u^2 + sigma^2). Restitution e then scales the
speed of what returns, so the wall hands back e^2 [u^2 / (u^2 + sigma^2)]^2 of the
normal energy flux.

R_eff is the coefficient that enters the floating potential of a wall (Hobbs and
Wesson, Plasma Phys. 9, 85, 1967): only 1 - R_eff of the electron flux counts, which
sheath.py measures.

Here the electrons are too tenuous for any field to act. They run into two absorbing
walls for a tenth of a thermal transit, so nothing arrives twice, and the weight and
energy that come back are compared with the two formulas for several widths u.
"""

import os

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below to "0", for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import Domain, Simulation, Solver, Species, quiet_start

sigma, length, n, restitution = 1e6, 1e-2, 200_000, 0.8          # m/s, m, particles, e
domain = Domain(length=length, cells=64, dt_over_dx_c=50.0, particle_bc="absorbing", field_bc="absorbing",
                restitution=restitution)
steps = int(round(0.1 * length / sigma / domain.dt))
x, v = quiet_start(n, length, vth=(np.sqrt(2) * sigma, 0, 0))

widths = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
returned, energy = [], []
for u in widths * sigma:
    def law(speed, u=u):
        return jnp.exp(-speed ** 2 / (2 * u ** 2))

    electrons = Species.electrons(n=n, density=1e6, vth=(np.sqrt(2) * sigma, 0, 0), reflection=law).replace(x=x, v=v)
    w = np.asarray(Simulation(domain, [electrons], Solver()).run(steps, store_every=steps).weight[-1])
    w0 = w.max()                                  # the weight of a particle that met no wall
    hit = w < w0
    returned.append(w[hit].sum() / (w0 * hit.sum()))
    energy.append(restitution ** 2 * (w[hit] * v[hit, 0] ** 2).sum() / (w0 * v[hit, 0] ** 2).sum())
    if u == sigma:
        speed, kept = np.abs(v[hit, 0]), w[hit] / w0

flux_average = widths ** 2 / (widths ** 2 + 1)
for s, r, e, theory in zip(widths, returned, energy, flux_average):
    print(f"u = {s:4.2f} sigma: returns {r:.4f} of the particles (flux average {theory:.4f}, "
          f"distribution average {s / np.sqrt(s ** 2 + 1):.4f}) and {e:.4f} of the energy "
          f"(theory {restitution ** 2 * theory ** 2:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
fine = np.linspace(0.1, 4.5, 200)
axes[0].plot(fine, fine ** 2 / (fine ** 2 + 1), "k-", label=r"flux average $u^2/(u^2+\sigma^2)$")
axes[0].plot(fine, fine / np.sqrt(fine ** 2 + 1), "k:", label=r"distribution average")
axes[0].plot(widths, returned, "o", label="simulation")
axes[0].set(xlabel=r"width of the law $u/\sigma$", ylabel="fraction of the particles returned",
            title="the wall samples the flux")
axes[0].legend(frameon=False)

axes[1].plot(fine, restitution ** 2 * (fine ** 2 / (fine ** 2 + 1)) ** 2, "k-",
             label=rf"$e^2[u^2/(u^2+\sigma^2)]^2$, $e={restitution}$")
axes[1].plot(widths, energy, "o", label="simulation")
axes[1].set(xlabel=r"$u/\sigma$", ylabel="fraction of the normal energy flux returned",
            title="restitution takes the rest")
axes[1].legend(frameon=False)

bins = np.linspace(0, 4, 41)
arrived, _ = np.histogram(speed / sigma, bins)
back, _ = np.histogram(speed / sigma, bins, weights=kept)
centres, width = 0.5 * (bins[1:] + bins[:-1]), bins[1] - bins[0]
scale = arrived.sum() * width
axes[2].bar(centres, back, width, alpha=0.5, label="returned")
axes[2].bar(centres, arrived - back, width, bottom=back, alpha=0.5, label="collected")
axes[2].plot(centres, scale * centres * np.exp(-centres ** 2 / 2) * np.exp(-centres ** 2 / 2), "k-",
             label=r"$R(v)\,v f(v)$")
axes[2].plot(centres, scale * centres * np.exp(-centres ** 2 / 2), "k--", label=r"$v f(v)$, all that arrives")
axes[2].set(xlabel=r"impact speed $|v_x|/\sigma$", ylabel="particles", title=r"slow ones come back ($u=\sigma$)")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
