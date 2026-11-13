"""The edge of a plasma against a floating wall, with and without electron reflection."""
import jax.numpy as jnp
import numpy as np
from common import COLORS, C_THEORY, figure, panel_label, record, savefig

from jaxincell import (Domain, Simulation, Solver, Species, bohm_edge, epsilon_0, mass_electron,
                       potential, quiet_start, elementary_charge as e_charge, speed_of_light as c)

# the same preset as examples/2_intermediate/sheath_reflection.py, so that the figure and the
# example are one run and not two that happen to look alike
T_E, DENSITY, MASS_RATIO, PARTICLES, CELLS, BOX, STEPS = 1.0, 1e16, 400.0, 30000, 120, 60, 6000
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

first = STEPS // 200
late = slice(first, None)
distance = (LENGTH / 2 - np.asarray(domain.faces)) / DEBYE
centres = (LENGTH / 2 - np.asarray(domain.grid)) / DEBYE
theory = 0.5 * np.log(MASS_RATIO / (2 * np.pi))
results = {}
for name, (reflection, R_eff, color) in WALLS.items():
    electrons = Species.electrons(n=PARTICLES, density=DENSITY, vth=(V_E, 0, 0),
                                  reflection=(0.0, reflection)).replace(x=x, v=v)
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic", filter_passes=4)).run(
        STEPS, store_every=100, store_particles=False, moments="flux").validate()
    phi = np.asarray(potential(out))[late] / T_E
    profile = phi.mean(axis=0) - phi[:, -1].mean()
    window = np.asarray(out.moments[-1] - out.moments[first]) / float(out.steps[-1] - out.steps[first])
    n_i = window[1, 0]
    flow = np.divide(window[1, 1], n_i, out=np.zeros(CELLS), where=n_i > 0) / C_S
    # where the flow crosses c_s, interpolated, with the number of crossings: a bare argmax returns
    # zero when there is none and invents an edge at the first bin
    crossing, count = bohm_edge(np.asarray(domain.grid), flow, 1.0)
    edge = float("nan") if count == 0 else (LENGTH / 2 - float(crossing)) / DEBYE
    # R_eff as the wall applied it, not as it was meant to
    arrived = float(np.asarray(out.wall.arrived)[-1, 0, 1])
    measured_R = 1.0 - float(np.asarray(out.wall.collected)[-1, 0, 1]) / arrived
    held = np.asarray(out.moments)[:, 1, 0].sum(axis=1)
    content = np.diff(held)
    results[name] = dict(profile=profile, flow=flow, edge=edge, color=color, crossings=int(count),
                         expected=theory + np.log(1 - measured_R), measured_R=measured_R,
                         sheath=float(np.interp(edge, distance[::-1], profile[::-1])),
                         rho=np.asarray(out.rho)[late].mean(axis=0) / (DENSITY * e_charge),
                         ions_left=float(content[-1] / content[0]))
    print(f"  {name:22s} edge {edge:4.1f} lambda_D ({count} crossing), R_eff {measured_R:.3f}, "
          f"sheath drop {results[name]['sheath']:.2f} (Hobbs-Wesson {results[name]['expected']:.2f})")

fig, axes = figure(3)
for name, r in results.items():
    axes[0].plot(distance, r["profile"], color=r["color"], label=name)
    axes[0].plot(r["edge"], r["sheath"], "o", color=r["color"], ms=9)
axes[0].set(xlabel=r"distance from the wall ($\lambda_D$)", ylabel=r"$(\phi-\phi_{wall})/T_e$", xlim=(0, 30),
            title="the sheath, and where ions reach $c_s$")
axes[0].legend(loc="lower right")
panel_label(axes[0], "a")

r = results["absorbing"]
axes[1].plot(centres, r["flow"], color=COLORS["orange"], label=r"ion flow $v_i/c_s$")
axes[1].plot(distance, 10 * r["rho"], color=COLORS["blue"], label=r"$10\,\rho/en_0$")
axes[1].axhline(1.0, ls="--", color=C_THEORY, lw=2)
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
       sheath_ions_left_percent=round(100 * results["absorbing"]["ions_left"], 0),
       sheath_reff_measured=round(results["returns half"]["measured_R"], 3))
