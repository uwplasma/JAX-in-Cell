"""The same sheath in a magnetic field that meets the wall at an angle.

Everything is as in `1_basic/sheath_unmagnetized.py` -- the same reservoir, the same
floating collector, the same electrostatic solve -- with one line added: a uniform
external field

    B_0 = B_0 (sin alpha, 0, cos alpha),

with `alpha` the angle to the **wall plane**, so that normal incidence is alpha = 90
degrees and grazing incidence approaches zero. The angle a particle makes with the wall
when it arrives is a different quantity and is reported separately, measured from the
outward normal.

What the field does: it is the component along x that the grid resolves, so the ions
still reach the wall along the normal, but between collisions with nothing they follow
the field. Where the gyro-radius of an ion is large compared with the Debye length there
are two layers rather than one -- a magnetic presheath a few ion gyro-radii deep, in
which the ions turn from following the field to crossing it, and inside that the Debye
sheath (Chodura, Phys. Fluids 25, 1628, 1982). Resolving both means a box many ion
gyro-radii long and cells a fraction of a Debye length wide, so the separation of scales
`rho_s/lambda_D = sqrt(m_i/m_e) omega_pe/Omega_e` is what this example is really about,
and it is printed below.

The ions here have a finite temperature, unlike the cold beam of the unmagnetized
example, and enter at the sound speed. There is no closed-form wall potential for this
problem, so what is checked is what can be: the field-free limit, which must reproduce
the unmagnetized run; normal incidence, which must not change the normal motion at all;
and the ion impact energies and angles, which are what a wall actually feels.

Run with `--quick` for a smaller, faster version.
"""

import os
import sys

# Double precision is the default, and what the conservation checks rely on. Run with
# JAX_ENABLE_X64=0, or change the "1" below, for single precision.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Impacts, Simulation, Solver, Source, Species, epsilon_0, mass_electron,
                       potential, elementary_charge as e_charge)

# --- what to change ---------------------------------------------------------------------
quick = "--quick" in sys.argv
electron_temperature = 1.0              # eV
temperature_ratio = 1.0                 # T_i / T_e
density = 1e16                          # m^-3
mass_ratio = 400.0                      # m_i/m_e, reduced so that the ion transit fits in a laptop run
angles = (90.0, 30.0) if quick else (90.0, 30.0, 15.0)   # to the wall plane; 90 is normal incidence
gyro_over_debye = 8.0                   # rho_s / lambda_D, which sets B_0
box_debye_lengths = 60.0
cells = 96 if quick else 240
steps_per_plasma_period = 10.0
transits = 0.15 if quick else 4.0
# `Species.n` is a capacity and has to hold every particle alive at once, which is `emit`
# times the residence time in steps. An ion entering at c_s takes a whole sound transit to
# cross, 12000 steps here; an electron takes about a tenth of that, so the two want
# different emission rates to end up with comparable numbers of markers in comparable pools.
# The pool sizes below are measured, and the run refuses to report anything if one overflows.
capacity = 30000 if quick else 80000
emit_ions = 10 if quick else 5
emit_electrons = 10 if quick else 40

# --- the setup ----------------------------------------------------------------------------
spread = np.sqrt(electron_temperature * e_charge / mass_electron)
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye = spread / omega_pe
sound_speed = spread / np.sqrt(mass_ratio)                       # c_s = sqrt(T_e/m_i)
ion_mass = mass_ratio * mass_electron
# rho_s = c_s/Omega_i = c_s m_i/(e B), so B follows from the separation of scales asked for
field = ion_mass * sound_speed / (e_charge * gyro_over_debye * debye)
omega_ce, omega_ci = e_charge * field / mass_electron, e_charge * field / ion_mass
length = box_debye_lengths * debye
dt = 1.0 / (steps_per_plasma_period * omega_pe)
stored = 40
steps = stored * max(int(transits * length / sound_speed / dt) // stored, 1)
ion_spread = spread * np.sqrt(temperature_ratio / mass_ratio)

print(f"m_i/m_e {mass_ratio:.0f}   T_i/T_e {temperature_ratio}   rho_s/lambda_D {gyro_over_debye:.1f}   "
      f"rho_e/lambda_D {omega_pe / omega_ce:.3f}   L/rho_s {box_debye_lengths / gyro_over_debye:.1f}")
print(f"dx/lambda_D {length / cells / debye:.2f}   omega_pe dt {omega_pe * dt:.2f}   "
      f"Omega_e dt {omega_ce * dt:.2f}   Omega_i dt {omega_ci * dt:.2e}   {steps} steps "
      f"= {transits:.2f} sound transits, one of which is {length / sound_speed / dt:.0f} steps")
print(f"pools of {capacity} slots per species, {emit_electrons} electrons and {emit_ions} ions "
      f"emitted a step\n")
if quick:
    print("--quick is a smoke preset: a fraction of a sound transit, so the sheath has not settled\n"
          "and none of the numbers below is a measurement. It checks that every step of this script\n"
          "runs and that the pools hold. The documentation quotes the full preset.\n")
if omega_ce * dt > 0.3:
    print(f"WARNING: Omega_e dt = {omega_ce * dt:.2f}; the electron gyro-phase is under-resolved.\n")

# What the collector is struck by, binned at the crossing. An ion falls through a few T_e of
# sheath and arrives with that plus its thermal energy, so a ceiling of twenty is generous;
# whatever passes it goes to the overflow bin and is reported rather than piled on the end.
energy_ceiling = 20.0 * electron_temperature                            # eV
impacts = Impacts(energy_max=energy_ceiling * e_charge, energy_bins=40, angle_bins=30)
angle_centres = 0.5 * (np.asarray(impacts.angle_edges)[:-1] + np.asarray(impacts.angle_edges)[1:])

results = {}
for angle in angles:
    radians = np.radians(angle)
    B = jnp.zeros((cells, 3)).at[:, 0].set(field * np.sin(radians)).at[:, 2].set(field * np.cos(radians))
    domain = Domain(length=length, cells=cells, time_step=dt,
                    particle_bc="absorbing", field_bc=("open", "absorbing"))
    electrons = Species("electrons", capacity, -1.0, mass_electron, density, (np.sqrt(2) * spread, 0, 0),
                        active=capacity // 4, sampling="quiet",
                        source=Source(density=density, vth=(np.sqrt(2) * spread,) * 3, emit=emit_electrons))
    ions = Species("ions", capacity, 1.0, ion_mass, density, (np.sqrt(2) * ion_spread, 0, 0),
                   (sound_speed, 0, 0), active=capacity // 4, sampling="quiet",
                   source=Source(density=density, vth=(np.sqrt(2) * ion_spread,) * 3, emit=emit_ions))
    out = Simulation(domain, [electrons, ions], Solver(model="electrostatic"), external_B=B,
                     impacts=impacts).run(steps, seed=0, store_every=steps // stored,
                                          store_particles=False, moments="flux").validate()

    late = stored // 2
    phi = np.asarray(potential(out))[late:].mean(axis=0) / electron_temperature
    elapsed = float(out.steps[-1] - out.steps[late])      # the steps the window really spans
    window = np.asarray(out.moments[-1] - out.moments[late]) / elapsed
    # What the collector was struck by during the late window: one entry per crossing, made
    # when the crossing happened and at the velocity that carried the ion there. A snapshot of
    # the ions near the wall is a different and wrong thing -- it repeats each ion across
    # frames, counts the ones on their way out, and weights by how many happen to be there
    # rather than by how many arrived.
    spectrum = np.asarray(out.wall.spectrum[-1, 1, 1] - out.wall.spectrum[late, 1, 1])
    fluence = float(out.wall.arrived[-1, 1, 1] - out.wall.arrived[late, 1, 1])
    mean_energy = float(out.wall.energy_in[-1, 1, 1] - out.wall.energy_in[late, 1, 1]) / fluence / e_charge
    incidence = spectrum.sum(axis=0) / spectrum.sum()
    results[angle] = dict(phi=phi, n_e=window[0, 0] / density, n_i=window[1, 0] / density,
                          flow=np.divide(window[1, 1], window[1, 0], out=np.zeros(cells), where=window[1, 0] > 0),
                          energy=spectrum[:-1].sum(axis=1), incidence=incidence, fluence=fluence,
                          above_range=spectrum[-1].sum() / spectrum.sum())
    live = np.asarray(out.state.w) > 0
    wall_phi = float(np.asarray(potential(out))[late:, -1].mean()) / electron_temperature
    print(f"    pool: {int(live[:capacity].sum())} electrons and {int(live[capacity:].sum())} ions "
          f"live of {capacity} slots each")
    print(f"alpha {angle:4.0f} deg to the wall: wall potential {wall_phi:+.2f} T_e/e, "
          f"ion fluence {fluence:.3e} m^-2, mean impact energy {mean_energy:.2f} eV, "
          f"mean incidence {np.degrees((incidence * angle_centres).sum()):.0f} deg from the normal"
          + (f"  [{results[angle]['above_range']:.1%} above {energy_ceiling:.0f} eV]"
             if results[angle]["above_range"] > 0.01 else ""))

normal_field = results[90.0]
print(f"\nnormal incidence is the field-free case for the motion along x: its wall potential is "
      f"{float(normal_field['phi'][-1]):+.2f} T_e/e, and the drop {np.ptp(normal_field['phi']):.2f} T_e/e.")

# --- the figure -------------------------------------------------------------------------------
distance = (length / 2 - (np.asarray(np.arange(cells)) + 0.5) * length / cells + length / 2) / debye
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
for angle in angles:
    r = results[angle]
    axes[0].plot(distance, r["phi"], label=rf"$\alpha = {angle:.0f}^\circ$")
    axes[1].plot(distance, r["flow"] / sound_speed, label=rf"$\alpha = {angle:.0f}^\circ$")
axes[0].set(xlabel=r"distance from the collector ($\lambda_D$)", ylabel=r"$e\phi/T_e$",
            title="the potential, at three field angles", xlim=(distance.max(), 0))
axes[0].legend(frameon=False)
axes[1].axhline(1.0, ls="--", color="k", lw=0.8)
axes[1].set(xlabel=r"distance from the collector ($\lambda_D$)", ylabel=r"$v_{i,x}/c_s$",
            title=r"the normal ion flow, and $c_s$", xlim=(distance.max(), 0))
axes[1].legend(frameon=False)
for angle in angles:
    r = results[angle]
    axes[2].step(np.degrees(np.asarray(impacts.angle_edges)),
                 np.append(r["incidence"], r["incidence"][-1]) / np.degrees(float(impacts.angle_edges[1])),
                 where="post", label=rf"$\alpha = {angle:.0f}^\circ$")
axes[2].set(xlabel="ion incidence from the wall normal (deg)", ylabel="fraction of the fluence (1/deg)",
            title="what the wall is struck by", xlim=(0, 90))
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
