"""A plasma between two absorbing walls forms a sheath.

Electrons are far faster than ions, so they reach the walls first and charge them
negative. The plasma floats positive until the electric field it sets up holds
back enough electrons for the two fluxes to balance. What is left is a
quasi-neutral bulk joined to each wall by a thin positively charged layer a few
Debye lengths thick: the sheath.

Two closed-form results come out of it, and this reproduces both.

* The plasma floats above a floating wall by (Lieberman and Lichtenberg,
  *Principles of Plasma Discharges*, section 6.2)

      phi_plasma - phi_wall = (T_e / 2e) ln(m_i / 2 pi m_e).

* Ions must enter the sheath at least at the Bohm speed c_s = sqrt(T_e/m_i)
  (Bohm 1949), which the pre-sheath field accelerates them to.

Absorbing walls in JAX-in-Cell are conductors that keep the charge they collect,
short-circuited to one another, so both stay at the same potential; that is the
standard closure for a bounded plasma (Verboncoeur et al., J. Comput. Phys. 104,
321, 1993).

Nothing sustains the plasma here, and that shows. The walls take the fast
electrons preferentially, so the bulk cools and its distribution loses the tail
the flux balance is derived from -- by the end there is nothing left beyond about
two standard deviations. A truncated tail carries less flux than a Maxwellian one,
so a smaller barrier suffices and the measured drop comes out around fifteen per
cent below the formula. Closing that gap means sustaining the plasma: an
ionisation source to replace what is lost, or electron-electron collisions to
refill the tail from the bulk. Both belong in a study of a real discharge rather
than in a demonstration of where a sheath comes from.
"""
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Species, epsilon_0, mass_electron, potential,
                       quiet_start, elementary_charge as e_charge, speed_of_light as c)

T_e, density, mass_ratio, particles, cells = 1.0, 1e16, 400.0, 60000, 240
v_th = np.sqrt(2 * T_e * e_charge / mass_electron)
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye = v_th / (np.sqrt(2) * omega_pe)
length = 120 * debye
v_th_ion = v_th * np.sqrt(T_e / 40 / (mass_ratio * T_e))          # T_i = T_e / 40

x, v = quiet_start(particles, length, vth=(v_th, 0, 0))
electrons = Species.electrons(n=particles, density=density, vth=(v_th, 0, 0)).replace(x=x, v=v)
x, v = quiet_start(particles, length, vth=(v_th_ion, 0, 0))
ions = Species("ions", particles, 1.0, mass_ratio * mass_electron, density,
               (v_th_ion, 0, 0)).replace(x=x, v=v)

# electrostatic, so the time step follows the plasma frequency and not the speed of light
domain = Domain(length=length, cells=cells, dt_over_dx_c=(0.2 / omega_pe) * c / (length / cells),
                particle_bc="absorbing", field_bc="absorbing")
simulation = Simulation(domain, [electrons, ions], Solver(filter_passes=4))
steps = int(round(omega_pe * length / 2 / np.sqrt(T_e * e_charge / (mass_ratio * mass_electron))
                  / (omega_pe * domain.dt) / 100)) * 100          # about one ion transit
output = simulation.run(steps, seed=0, store_every=100)

t = np.asarray(output.t) * omega_pe
phi = np.asarray(potential(output))
grid = (np.asarray(output.grid) + output.dx / 2) / debye          # faces, in Debye lengths
inside = np.abs(np.asarray(output.x[..., 0])) <= length / 2
electron, ion = inside[:, :particles], inside[:, particles:]
v_x = np.asarray(output.v[..., 0])

# the bulk temperature falls as the walls take the tail, so measure it as we go
bulk = np.abs(np.asarray(output.x[:, :particles, 0])) < length / 5
T_bulk = np.array([mass_electron * np.var(v_x[k, :particles][bulk[k]]) / e_charge
                   for k in range(t.size)])
drop = phi[:, 2 * cells // 5:3 * cells // 5].mean(axis=1) / T_bulk
theory = 0.5 * np.log(mass_ratio / (2 * np.pi))
late = t > 0.6 * t[-1]
print(f"plasma-wall drop {drop[late].mean():.2f} +- {drop[late].std():.2f} T_e/e   "
      f"theory (1/2) ln(m_i / 2 pi m_e) = {theory:.2f}")
print(f"potential of the right wall {phi[-1, -1] / T_bulk[-1]:+.1e} T_e/e (short-circuited to the left)")

# Bohm criterion: the ion flow speed where the sheath begins
c_s = np.sqrt(T_bulk * e_charge / (mass_ratio * mass_electron))
edge = (np.asarray(output.x[:, particles:, 0]) > 0.30 * length) & \
       (np.asarray(output.x[:, particles:, 0]) < 0.40 * length)
flow = np.array([v_x[k, particles:][edge[k]].mean() for k in range(t.size)]) / c_s
print(f"ion flow at the sheath edge {flow[-1]:.2f} c_s   (Bohm criterion: at least 1)")
print(f"{100 * electron[-1].mean():.0f} % of the electrons and {100 * ion[-1].mean():.0f} % of the ions left")

# why the drop lands below the formula: the wall has taken the tail it assumes
final = v_x[-1, :particles][bulk[-1]]
beyond = (np.abs(final) > 2 * final.std()).mean()
print(f"electrons beyond two standard deviations: {beyond:.4f} of the bulk, "
      f"{0.0455:.4f} for a Maxwellian")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
# averaged over the late window, where the profiles are steady but noisy
axes[0].plot(grid, (phi[late] / T_bulk[late, None]).mean(axis=0), label=r"$\phi/T_e$")
axes[0].axhline(theory, ls="--", color="k", label=r"$\frac{1}{2}\ln(m_i/2\pi m_e)$")
axes[0].set(xlabel=r"$x/\lambda_D$", ylabel=r"$\phi/T_e$",
            title="a quasi-neutral bulk between two sheaths")
charge = axes[0].twinx()
# the sheath is the layer where quasi-neutrality fails: rho > 0, ions left behind
charge.plot(np.asarray(output.grid) / debye,
            np.asarray(output.rho)[late].mean(axis=0) / (density * e_charge), color="C1", lw=1)
charge.axhline(0, color="0.8", lw=0.6)
charge.set_ylabel(r"$\rho / e n_0$", color="C1")
axes[0].legend(loc="center left", frameon=False)

excess = (ion.sum(axis=1) - electron.sum(axis=1)) / particles
axes[1].plot(t, 100 * excess)
axes[1].set(xlabel=r"$t\,\omega_{pe}$", ylabel="excess electrons absorbed (%)",
            title="the sheath switches off the excess")

axes[2].plot(t, flow)
axes[2].axhline(1.0, ls="--", color="k", label="Bohm speed")
axes[2].set(xlabel=r"$t\,\omega_{pe}$", ylabel=r"$v_i/c_s$ at the sheath edge",
            title="ions reach the Bohm speed")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
