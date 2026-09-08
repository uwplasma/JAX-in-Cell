"""The sheath that forms between a plasma and two absorbing walls."""
import matplotlib.pyplot as plt
import numpy as np
from common import C_ELECTRONS, C_IONS, C_THEORY, panel_label, record, savefig

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron, potential,
                       quiet_start, elementary_charge as e_charge, speed_of_light as c)

T_E, DENSITY, MASS_RATIO, PARTICLES, CELLS, BOX = 1.0, 1e16, 400.0, 60000, 240, 120
V_TH = np.sqrt(2 * T_E * e_charge / mass_electron)
OMEGA_PE = np.sqrt(DENSITY * e_charge ** 2 / (epsilon_0 * mass_electron))
DEBYE = V_TH / (np.sqrt(2) * OMEGA_PE)
LENGTH = BOX * DEBYE
V_TH_ION = V_TH * np.sqrt(1 / (40 * MASS_RATIO))                  # T_i = T_e / 40

x, v = quiet_start(PARTICLES, LENGTH, vth=(V_TH, 0, 0))
electrons = Species.electrons(n=PARTICLES, density=DENSITY, vth=(V_TH, 0, 0)).replace(x=x, v=v)
x, v = quiet_start(PARTICLES, LENGTH, vth=(V_TH_ION, 0, 0))
ions = Species("ions", PARTICLES, 1.0, MASS_RATIO * mass_electron, DENSITY,
               (V_TH_ION, 0, 0)).replace(x=x, v=v)
domain = Domain(length=LENGTH, cells=CELLS, particle_bc="absorbing", field_bc="absorbing",
                dt_over_dx_c=(0.2 / OMEGA_PE) * c / (LENGTH / CELLS))
simulation = Simulation(domain, [electrons, ions], Solver(filter_passes=4))
STEPS = 6000                                                       # about one ion transit
output = simulation.run(STEPS, seed=0, store_every=100)

t = np.asarray(output.t) * OMEGA_PE
phi = np.asarray(potential(output))
v_x = np.asarray(output.v[..., 0])
position = np.asarray(output.x[..., 0])
inside = np.abs(position) <= LENGTH / 2
bulk = np.abs(position[:, :PARTICLES]) < LENGTH / 5
T_bulk = np.array([mass_electron * np.var(v_x[k, :PARTICLES][bulk[k]]) / e_charge
                   for k in range(t.size)])
late = t > 0.6 * t[-1]
drop = (phi[:, 2 * CELLS // 5:3 * CELLS // 5].mean(axis=1) / T_bulk)[late]
theory = 0.5 * np.log(MASS_RATIO / (2 * np.pi))
c_s = np.sqrt(T_bulk * e_charge / (MASS_RATIO * mass_electron))
edge = (position[:, PARTICLES:] > 0.30 * LENGTH) & (position[:, PARTICLES:] < 0.40 * LENGTH)
flow = np.array([v_x[k, PARTICLES:][edge[k]].mean() for k in range(t.size)]) / c_s
print(f"  drop {drop.mean():.2f} +- {drop.std():.2f} T_e/e (theory {theory:.2f}); "
      f"ion flow {flow[-1]:.2f} c_s")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
grid = (np.asarray(output.grid) + output.dx / 2) / DEBYE
axes[0].plot(grid, (phi[late] / T_bulk[late, None]).mean(axis=0), color=C_ELECTRONS,
             label=r"$\phi/T_e$")
axes[0].axhline(theory, ls="--", color=C_THEORY, label=r"$\frac{1}{2}\ln(m_i/2\pi m_e)$")
axes[0].set(xlabel=r"$x/\lambda_D$", ylabel=r"$\phi/T_e$", title="bulk, pre-sheath and sheath")
axes[0].legend(loc="center left")
charge = axes[0].twinx()
charge.plot(np.asarray(output.grid) / DEBYE,
            np.asarray(output.rho)[late].mean(axis=0) / (DENSITY * e_charge), color=C_IONS, lw=1)
charge.axhline(0, color="0.8", lw=0.6)
charge.set_ylabel(r"$\rho/en_0$", color=C_IONS)
charge.grid(False)
panel_label(axes[0], "a")

excess = 100 * (inside[:, PARTICLES:].sum(axis=1) - inside[:, :PARTICLES].sum(axis=1)) / PARTICLES
axes[1].plot(t, excess, color=C_ELECTRONS)
axes[1].set(xlabel=r"$t\,\omega_{pe}$", ylabel="excess electrons absorbed (%)",
            title="the sheath throttles the electron flux")
panel_label(axes[1], "b")

axes[2].plot(t, flow, color=C_ELECTRONS)
axes[2].axhline(1.0, ls="--", color=C_THEORY, label="Bohm speed")
axes[2].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$v_i/c_s$ at the sheath edge",
            title="the pre-sheath accelerates the ions")
axes[2].legend(loc="lower right")
panel_label(axes[2], "c")
fig.tight_layout()
savefig(fig, "sheath")

record(sheath_mass_ratio=MASS_RATIO, sheath_box_debye=BOX, sheath_cells=CELLS,
       sheath_particles=2 * PARTICLES, sheath_steps=STEPS,
       sheath_drop_measured=round(float(drop.mean()), 2),
       sheath_drop_spread=round(float(drop.std()), 2),
       sheath_drop_theory=round(float(theory), 2),
       sheath_drop_deviation_percent=round(float(100 * (drop.mean() / theory - 1)), 0),
       sheath_bohm_ratio=round(float(flow[-1]), 2),
       sheath_wall_potential=f"{float(np.abs(phi[:, -1]).max() / T_bulk.max()):.0e}",
       sheath_electrons_left_percent=round(float(100 * inside[-1, :PARTICLES].mean()), 0),
       sheath_temperature_final=round(float(T_bulk[-1]), 2))
