"""A maintained sheath, from a plasma source to a collector that floats.

The box is a slab of the edge of a plasma. Its left wall is an open plane with a
reservoir behind it: electrons cross it from a Maxwellian at rest and ions as a cold
beam, at fluxes that do not depend on what leaves, and whatever reaches the plane from
inside goes back into the reservoir. Its right wall is a conductor that collects
everything and is connected to nothing, so it charges until it draws no net current.

That is what a source is for. A thermal wall returns what reaches it and cannot replace
what the collector takes, so a plasma between the two drains; a reservoir holds the
plasma up and the sheath reaches a steady state that can be compared with theory.

Two closed-form results follow, and this reproduces both. Electrons conserve
:math:`\\tfrac12 v^2 - e\\phi/m_e`, so only those launched faster than
:math:`\\sqrt{-2e\\phi_w/m_e}` reach the wall, and equal particle currents at a floating
collector give the wall potential as the root of

    exp(phi_w) / [1 + erf(sqrt(-phi_w))] = sqrt(pi/2) v_0,

in units of T_e/e with v_0 the beam speed in units of the electron spread. The same
conservation law gives the densities as functions of the potential alone, a local
relation that does not care where in the box the potential took that value:

    n_e = (n_e0/2) exp(phi) [1 + erf(sqrt(phi - phi_w))],    n_i = v_0/sqrt(v_0^2 - 2 phi m_e/m_i).

`jaxincell.sheath` has both, in NumPy, sharing nothing with the deposit, the gather or
the field solver, so the comparison below is with mathematics and not with the code.

The parameters are those of the sheath benchmark of Konewko, Maestracci and Van Loo
(kobra, arXiv:2609.11563): m_i/m_e = 1836, v_0 = 0.2, a box ten Debye lengths across.
Their printed wall potential, +0.739, does not solve their printed equation; the root of
that equation at v_0 = 0.2 is -0.79926, which is what is used here. Run with
`--quick` for a smaller, faster version that shows the same structure with more noise.
"""

import os
import sys

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below, for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Source, Species, bohm_edge, epsilon_0, mass_electron,
                       potential, elementary_charge as e_charge)
from jaxincell.sheath import densities, floating_potential, source_density

# --- what to change ------------------------------------------------------------------
quick = "--quick" in sys.argv
electron_temperature = 1.0            # eV
density = 1e16                        # m^-3, the upstream plasma density
mass_ratio = 1836.0                   # m_i / m_e; 1836 is hydrogen
beam_speed = 0.2                      # ion drift at the source plane, in electron spreads
box_debye_lengths = 10.0              # box length in Debye lengths
cells = 48 if quick else 120          # dx = L / cells
steps_per_plasma_period = 10.0        # omega_pe dt = 0.1
transits = 1.0 if quick else 6.0      # how many ion transits of the box to run for
capacity = 12000 if quick else 120000  # particle slots per species: a pool, not a population
emit = 12 if quick else 120           # particles each source emits per step

# --- the setup -------------------------------------------------------------------------
spread = np.sqrt(electron_temperature * e_charge / mass_electron)          # sigma_e = sqrt(T_e/m_e)
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye = spread / omega_pe
length = box_debye_lengths * debye
dt = 1.0 / (steps_per_plasma_period * omega_pe)
stored = 60                                                     # states kept, for the averages below
steps = stored * max(int(transits * length / (beam_speed * spread) / dt) // stored, 1)
phi_wall = float(floating_potential(beam_speed))                # the reference, solved independently
amplitude = float(source_density(phi_wall))                     # makes the source plane neutral

domain = Domain(length=length, cells=cells, time_step=dt,
                particle_bc="absorbing",                 # both planes take back whatever reaches them
                field_bc=("open", "absorbing"))          # the source plane imposes nothing; the collector floats
electrons = Species("electrons", capacity, -1.0, mass_electron, density, (np.sqrt(2) * spread, 0, 0),
                    active=capacity // 4, quiet=True,
                    source=Source(density=amplitude * density, vth=(np.sqrt(2) * spread,) * 3, emit=emit))
ions = Species("ions", capacity, 1.0, mass_ratio * mass_electron, density, 0.0, (beam_speed * spread, 0, 0),
               active=capacity // 4, quiet=True,
               source=Source(density=density, vth=0.0, drift=(beam_speed * spread, 0, 0), emit=emit))
simulation = Simulation(domain, [electrons, ions], Solver(model="electrostatic"))

print(f"m_i/m_e {mass_ratio:.0f}   v_0/sigma_e {beam_speed}   Mach {beam_speed * np.sqrt(mass_ratio):.2f}   "
      f"L/lambda_D {box_debye_lengths:.0f}   dx/lambda_D {length / cells / debye:.3f}   "
      f"omega_pe dt {omega_pe * dt:.2f}")
print(f"{steps} steps = {transits:.0f} ion transits; {capacity} slots and {emit} emitted a step per species")
print(f"reference wall potential {phi_wall:.5f} T_e/e, source amplitude {amplitude:.4f} n_0\n")

out = simulation.run(steps, seed=0, store_every=steps // stored, store_particles=False,
                     moments=True).validate()     # a pool that overflowed invalidates everything below

# --- what came out -----------------------------------------------------------------------
late = stored // 2                                  # average over the second half of the run
faces = np.asarray(domain.faces)
phi = np.asarray(potential(out)) / electron_temperature
window = np.asarray(out.moments[-1] - out.moments[late]) / ((stored - late) * steps // stored)
n_e, n_i = window[0, 0] / density, window[1, 0] / density
flow = np.divide(window[1, 1], window[1, 0], out=np.zeros(cells), where=window[1, 0] > 0)
measured = phi[late:, -1]

print(f"wall potential  {measured.mean():+.4f} +- {measured.std() / np.sqrt(len(measured)):.4f} T_e/e "
      f"(reference {phi_wall:+.4f}, {abs(measured.mean() / phi_wall - 1) * 100:.1f} % off)")
collected = np.asarray(out.wall.collected)
late_charge = collected[-1, :, 1] - collected[late, :, 1]
print(f"collector current balance: electrons {late_charge[0]:.4e}, ions {late_charge[1]:.4e} m^-2, "
      f"net {(late_charge[1] - late_charge[0]) / late_charge[1] * 100:+.2f} % of the ion current")
print(f"pool: {int((np.asarray(out.state.w)[:capacity] > 0).sum())} electrons and "
      f"{int((np.asarray(out.state.w)[capacity:] > 0).sum())} ions live of {capacity} slots, "
      f"overflow {float(out.overflow[-1]):.3g}")
sound_speed = spread / np.sqrt(mass_ratio)
edge, crossings = bohm_edge(faces, flow, sound_speed)
print(f"ion flow reaches c_s at {crossings} place(s)" +
      (f", {(length / 2 - float(edge)) / debye:.1f} Debye lengths from the wall" if crossings else
       ": the beam enters at Mach {:.1f}, already far above it".format(beam_speed * np.sqrt(mass_ratio))))

# the densities against the local relation energy conservation gives, at the measured potential
phi_profile = 0.5 * (phi[late:].mean(axis=0)[:-1] + phi[late:].mean(axis=0)[1:])      # at the cell centres
phi_profile = np.concatenate([[phi[late:].mean(axis=0)[0]], phi_profile])
reference_e, reference_i = densities(np.minimum(phi_profile, 0.0), phi_wall, beam_speed, mass_ratio)
inside = slice(2, cells - 2)
print(f"densities against the kinetic relation n(phi): electrons {np.abs(n_e - reference_e)[inside].max():.3f}, "
      f"ions {np.abs(n_i - reference_i)[inside].max():.3f} at worst, in units of n_0")

# --- the figure ----------------------------------------------------------------------------
distance = (length / 2 - faces) / debye
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
axes[0].plot(distance, phi[late:].mean(axis=0), label="measured")
axes[0].axhline(phi_wall, ls="--", color="k", lw=0.9, label="kinetic reference")
axes[0].plot(0.0, measured.mean(), "o", color="C0")
axes[0].set(xlabel=r"distance from the collector ($\lambda_D$)", ylabel=r"$e\phi/T_e$",
            title="the sheath potential", xlim=(distance.max(), 0))
axes[0].legend(frameon=False)

centres = (length / 2 - np.asarray(domain.grid)) / debye
axes[1].plot(centres, n_e, label=r"$n_e$")
axes[1].plot(centres, n_i, label=r"$n_i$")
axes[1].plot(centres, reference_e, "k--", lw=0.9, label=r"$n(\phi)$, kinetic")
axes[1].plot(centres, reference_i, "k--", lw=0.9)
axes[1].set(xlabel=r"distance from the collector ($\lambda_D$)", ylabel=r"$n/n_0$", ylim=(0, 1.3),
            title="the electrons are pushed out, the beam is not", xlim=(centres.max(), 0))
axes[1].legend(frameon=False)

time = np.asarray(out.t) * omega_pe
axes[2].plot(time, phi[:, -1])
axes[2].axhline(phi_wall, ls="--", color="k", lw=0.9)
axes[2].axvspan(time[late], time[-1], color="0.9", zorder=0)
axes[2].set(xlabel=r"$\omega_{pe} t$", ylabel=r"$e\phi_{\rm wall}/T_e$",
            title="the collector charges and then floats")
plt.tight_layout()
plt.show()
