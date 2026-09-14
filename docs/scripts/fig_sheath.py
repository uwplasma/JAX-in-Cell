"""The edge of a plasma against a floating wall, with and without electron reflection."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from common import COLORS, C_THEORY, panel_label, record, savefig

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron, potential,
                       quiet_start, elementary_charge as e_charge, speed_of_light as c)

T_E, DENSITY, MASS_RATIO, PARTICLES, CELLS, BOX, STEPS = 1.0, 1e16, 400.0, 40000, 120, 60, 6000
SIGMA = np.sqrt(T_E * e_charge / mass_electron)
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
DEBYE, C_S = SIGMA / OMEGA_PE, SIGMA / np.sqrt(MASS_RATIO)
LENGTH = BOX * DEBYE
V_E, V_I = np.sqrt(2) * SIGMA, np.sqrt(2) * SIGMA / np.sqrt(40 * MASS_RATIO)       # T_i = T_e / 40

domain = Domain(length=LENGTH, cells=CELLS, dt_over_dx_c=(0.2 / OMEGA_PE) * c / (LENGTH / CELLS),
                particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
WALLS = {"absorbing": (0.0, 0.0, COLORS["blue"]), "returns half": (0.5, 0.5, COLORS["vermillion"]),
         "returns the slow ones": (lambda s: jnp.exp(-s ** 2 / (2 * SIGMA ** 2)), 0.5, COLORS["green"])}
x, v = quiet_start(PARTICLES, LENGTH, vth=(V_E, 0, 0))
x_i, v_i = quiet_start(PARTICLES, LENGTH, vth=(V_I, 0, 0))
ions = Species("ions", PARTICLES, 1.0, MASS_RATIO * mass_electron, DENSITY, (V_I, 0, 0)).replace(x=x_i, v=v_i)

late = slice(STEPS // 200, None)
distance = (LENGTH / 2 - np.asarray(domain.grid) - domain.dx / 2) / DEBYE
bins = np.linspace(-LENGTH / 2, LENGTH / 2, CELLS // 4 + 1)
centres = (LENGTH / 2 - 0.5 * (bins[:-1] + bins[1:])) / DEBYE
theory = 0.5 * np.log(MASS_RATIO / (2 * np.pi))
results = {}
for name, (reflection, R_eff, color) in WALLS.items():
    electrons = Species.electrons(n=PARTICLES, density=DENSITY, vth=(V_E, 0, 0),
                                  reflection=(0.0, reflection)).replace(x=x, v=v)
    out = Simulation(domain, [electrons, ions], Solver(filter_passes=4)).run(STEPS, store_every=100)
    phi = np.asarray(potential(out))[late] / T_E
    profile = phi.mean(axis=0) - phi[:, -1].mean()
    position, speed, weight = (np.asarray(a)[late, PARTICLES:] for a in (out.x[..., 0], out.v[..., 0], out.weight))
    flow = (np.histogram(position, bins, weights=weight * speed)[0]
            / np.maximum(np.histogram(position, bins, weights=weight)[0], 1e-300) / C_S)
    # where the flow crosses c_s, between the first bin that reaches it and the one before; a bin
    # centre alone would move the edge by half a bin, where the potential falls 0.1 T_e/e per lambda_D
    k = int(np.argmax(flow >= 1))
    edge = float(np.interp(1.0, flow[k - 1:k + 1], centres[k - 1:k + 1])) if k > 0 else float(centres[0])
    results[name] = dict(profile=profile, flow=flow, edge=edge, color=color, expected=theory + np.log(1 - R_eff),
                         sheath=float(np.interp(edge, distance[::-1], profile[::-1])),
                         rho=np.asarray(out.rho)[late].mean(axis=0) / (DENSITY * e_charge),
                         ions_left=float(np.asarray(out.weight)[-1, PARTICLES:].sum() / weight[0].sum()))
    print(f"  {name:22s} edge {edge:4.1f} lambda_D, sheath drop {results[name]['sheath']:.2f} "
          f"(Hobbs-Wesson {results[name]['expected']:.2f})")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for name, r in results.items():
    axes[0].plot(distance, r["profile"], color=r["color"], label=name)
    axes[0].plot(r["edge"], r["sheath"], "o", color=r["color"], ms=4)
axes[0].set(xlabel=r"distance from the wall ($\lambda_D$)", ylabel=r"$(\phi-\phi_{wall})/T_e$", xlim=(0, 30),
            title="the sheath, and where ions reach $c_s$")
axes[0].legend(loc="lower right")
panel_label(axes[0], "a")

r = results["absorbing"]
axes[1].plot(centres, r["flow"], color=COLORS["orange"], label=r"ion flow $v_i/c_s$")
axes[1].plot(distance, 10 * r["rho"], color=COLORS["blue"], label=r"$10\,\rho/en_0$")
axes[1].axhline(1.0, ls="--", color=C_THEORY, lw=0.8)
axes[1].set(xlabel=r"distance from the wall ($\lambda_D$)", xlim=(0, 30),
            title="the charge builds where ions reach $c_s$")
axes[1].legend(loc="upper right")
panel_label(axes[1], "b")

names = list(results)
axes[2].bar(range(3), [results[k]["sheath"] for k in names], color=[results[k]["color"] for k in names],
            alpha=0.75, label="measured")
axes[2].plot(range(3), [results[k]["expected"] for k in names], "_", color=C_THEORY, ms=38, mew=2,
             label="Hobbs and Wesson")
axes[2].set_xticks(range(3), ["absorbing", "returns\nhalf", "returns the\nslow ones"])
axes[2].set(ylabel=r"sheath drop ($T_e/e$)", title=r"same $R_{\rm eff}$, same sheath")
axes[2].legend(loc="upper right")
panel_label(axes[2], "c")
fig.tight_layout()
savefig(fig, "sheath")

deviation = max(abs(r["sheath"] / r["expected"] - 1) for r in results.values())
record(sheath_mass_ratio=MASS_RATIO, sheath_box_debye=BOX, sheath_cells=CELLS, sheath_particles=2 * PARTICLES,
       sheath_steps=STEPS, sheath_drop_theory=round(float(theory), 2),
       sheath_drop_theory_reflecting=round(float(theory - np.log(2)), 2),
       sheath_drop_absorbing=round(results["absorbing"]["sheath"], 2),
       sheath_drop_half=round(results["returns half"]["sheath"], 2),
       sheath_drop_slow=round(results["returns the slow ones"]["sheath"], 2),
       sheath_drop_deviation_percent=round(float(100 * deviation), 0),
       sheath_edge_debye=round(float(results["absorbing"]["edge"]), 0),
       sheath_ions_left_percent=round(100 * results["absorbing"]["ions_left"], 0))
