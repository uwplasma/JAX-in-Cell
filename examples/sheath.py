"""A plasma floats above the wall it touches, and less so when the wall reflects electrons.

The box is the edge of a large plasma. Its left wall is thermal: whatever reaches it
comes back with a fresh velocity from a Maxwellian at the starting temperature, as if
from the plasma behind it -- the source boundary of Schwager and Birdsall (Phys.
Fluids B 2, 1057, 1990). Its right wall is a floating conductor that collects what
reaches it. Electrons outrun the ions, charge the conductor negative, and are held
back by its field until no more electrons arrive than ions. Two closed-form results
follow, and this reproduces both.

* Ions enter the sheath at the Bohm speed c_s = sqrt(T_e/m_i) (Bohm 1949), and from
  there the potential falls to the wall by (Hobbs and Wesson, Plasma Phys. 9, 85, 1967)

      e dphi / T_e = (1/2) ln(m_i / 2 pi m_e) + ln(1 - R_eff),

  where R_eff is the fraction of the electron flux the wall sends back.
* R_eff is the flux average of the wall's reflection law (wall_reflection.py). A wall
  that returns half of every electron and one that returns the slow electrons with a
  Gaussian of width sigma both have R_eff = 1/2, so both hold a sheath ln 2 T_e/e
  shallower than a wall that keeps everything, although they return quite different
  electrons.

The thermal wall is what makes the comparison sharp: it keeps the electrons that
reach the conductor Maxwellian. Between two absorbing walls nothing would, and the
walls would strip the tail of the distribution the formula is derived from.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron, potential, quiet_start,
                       elementary_charge as e_charge, speed_of_light as c)

T_e, density, mass_ratio, particles, cells = 1.0, 1e16, 400.0, 30000, 120
sigma = np.sqrt(T_e * e_charge / mass_electron)                  # electron thermal spread
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye, c_s = sigma / omega_pe, sigma / np.sqrt(mass_ratio)
length, steps = 60 * debye, 6000                                  # about one ion transit
v_th_e, v_th_i = np.sqrt(2) * sigma, np.sqrt(2) * sigma / np.sqrt(40 * mass_ratio)   # T_i = T_e / 40

# electrostatic, so the time step follows the plasma frequency and not the speed of light
domain = Domain(length=length, cells=cells, dt_over_dx_c=(0.2 / omega_pe) * c / (length / cells),
                particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
walls = {"absorbing": (0.0, 0.0), "returns half": (0.5, 0.5),
         "returns the slow ones": (lambda speed: jnp.exp(-speed ** 2 / (2 * sigma ** 2)), 0.5)}
x, v = quiet_start(particles, length, vth=(v_th_e, 0, 0))
x_i, v_i = quiet_start(particles, length, vth=(v_th_i, 0, 0))
ions = Species("ions", particles, 1.0, mass_ratio * mass_electron, density, (v_th_i, 0, 0)).replace(x=x_i, v=v_i)

late = slice(steps // 200, None)                                  # the second half of the stored steps
distance = (length / 2 - np.asarray(domain.grid) - domain.dx / 2) / debye   # of each face from the conductor
bins = np.linspace(-length / 2, length / 2, cells // 4 + 1)
results = {}
for name, (reflection, R_eff) in walls.items():
    electrons = Species.electrons(n=particles, density=density, vth=(v_th_e, 0, 0),
                                  reflection=(0.0, reflection)).replace(x=x, v=v)
    out = Simulation(domain, [electrons, ions], Solver(filter_passes=4)).run(steps, store_every=100)
    phi = np.asarray(potential(out))[late] / T_e
    profile = phi.mean(axis=0) - phi[:, -1].mean()                # above the conductor
    position, speed, weight = (np.asarray(a)[late, particles:] for a in (out.x[..., 0], out.v[..., 0], out.weight))
    ions_per_bin = np.histogram(position, bins, weights=weight)[0]
    flow = np.histogram(position, bins, weights=weight * speed)[0] / np.maximum(ions_per_bin, 1e-300) / c_s
    edge = (length / 2 - 0.5 * (bins[:-1] + bins[1:])[np.argmax(flow >= 1)]) / debye   # ions reach c_s
    results[name] = dict(profile=profile, flow=flow, edge=edge, sheath=np.interp(edge, distance[::-1], profile[::-1]),
                         theory=0.5 * np.log(mass_ratio / (2 * np.pi)) + np.log(1 - R_eff),
                         rho=np.asarray(out.rho)[late].mean(axis=0) / (density * e_charge))
    print(f"{name:22s} sheath edge {edge:4.1f} Debye lengths from the wall, drop {results[name]['sheath']:.2f} T_e/e, "
          f"Hobbs-Wesson {results[name]['theory']:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for name, r in results.items():
    line, = axes[0].plot(distance, r["profile"], label=name)
    axes[0].plot(r["edge"], r["sheath"], "o", color=line.get_color())
axes[0].set(xlabel=r"distance from the wall ($\lambda_D$)", ylabel=r"$(\phi - \phi_{wall})/T_e$", xlim=(0, 30),
            title="the sheath, and where the ions reach $c_s$")
axes[0].legend(frameon=False)

centres = (length / 2 - 0.5 * (bins[:-1] + bins[1:])) / debye
axes[1].plot(centres, results["absorbing"]["flow"], label=r"ion flow $v_i/c_s$")
axes[1].plot(distance, 10 * results["absorbing"]["rho"], label=r"$10\,\rho/en_0$")
axes[1].axhline(1.0, ls="--", color="k", lw=0.8)
axes[1].set(xlabel=r"distance from the wall ($\lambda_D$)", xlim=(0, 30),
            title="the charge builds where the ions reach $c_s$")
axes[1].legend(frameon=False)

names = list(results)
axes[2].bar(range(3), [results[k]["sheath"] for k in names], color=["C0", "C1", "C2"], alpha=0.7, label="measured")
axes[2].plot(range(3), [results[k]["theory"] for k in names], "k_", ms=40, mew=2, label="Hobbs-Wesson")
axes[2].set_xticks(range(3), names, rotation=12)
axes[2].set(ylabel=r"sheath drop ($T_e/e$)", title=r"same $R_{\rm eff}$, same sheath")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
