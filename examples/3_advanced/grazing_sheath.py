"""A sheath in a magnetic field that grazes the wall, set up to be compared with GYRAZE.

Where the field meets the wall at a few degrees the plasma-wall transition has two layers:
a **magnetic presheath** a few ion sound gyroradii deep, in which the ions turn from
following the field to crossing it, and inside it a **Debye sheath** a few Debye lengths
deep (Chodura, Phys. Fluids 25, 1628, 1982). GYRAZE solves the two as separate asymptotic
systems in the limit `lambda_D/rho_s -> 0` and `alpha -> 0`, with gyrokinetic ions in the
presheath and gyrokinetic electrons of finite `rho_e/lambda_D` in the Debye sheath
(Geraldini, Ewart, Brunner and Parra, arXiv:2508.09067). This runs the same problem with
full orbits and no ordering in `alpha`, so that the two can be compared where both are
valid, and states plainly where they cannot be.

**The entrance condition is the reference's own.** GYRAZE prescribes the ion distribution
at the magnetic-presheath entrance as

    F(v_par, v_perp) ~ v_par^2 exp(-(v_par - u)^2/2 - v_perp^2/2),

in units of `sqrt(T_i/m_i)`, with `u` fixed by the kinetic Chodura condition, which at
`T_i = T_e` gives `u = 0`. The `v_par^2` is that condition: it empties the distribution at
zero parallel velocity, and a drifting Maxwellian, which does not, is a different problem
(Geraldini, Parra and Militello 2019). The electrons enter as a Maxwellian. Both are
sampled here and handed to `Source(samples=...)`, because the field angle mixes the
parallel and perpendicular directions into every Cartesian component and no product of
three one-dimensional draws makes that distribution.

**What this run is, in the units both codes have to agree on**, printed as a manifest
before anything else: `x` measured from the entrance plane towards the wall, the potential
measured from the entrance plane, `rho_s = c_s/Omega_i` with `c_s = sqrt((ZT_e+T_i)/m_i)`
-- not the Bohm gyroradius, which is smaller by `sqrt(1+tau)` -- velocities normalised to
`sqrt(T/m)` and not `sqrt(2T/m)`, `gamma = rho_e/lambda_D` at the entrance plane, a
floating wall rather than a prescribed potential, and the ion temperature as the width
parameter of the distribution above rather than a Maxwellian temperature.

Three presets, because the matched case is expensive:

* `--quick`: a smoke run, a few minutes. The angle is not grazing and the scales are not
  separated, so it checks that the script runs and holds its pools, and nothing else.
* the default: `m_i/m_e = 400` at 5 degrees, a few hours, which is where GYRAZE still
  converges at the smallest cost. Five degrees is the edge of the range its own README
  calls inaccurate, so it is a rehearsal.
* `--matched`: `m_i/m_e = 900` at 4 degrees, inside that range, and an overnight run.

`--markers=N` and `--transits=T` scale the cost of any of them: the first is the markers a
cell holds, the second how many entrance-speed crossings of the box the run lasts, and the
run says what it used. `--markers=40 --transits=2` is about a quarter of the default, which
is a noisier first look at the same physics rather than a different problem.

**The ions are emitted once every `k` steps, not every step** (`Source(every=k)`). An ion
stays in the box for hundreds of thousands of steps, and a source that emits at least one
marker a step would hold that many whatever the markers per cell asked for: 465 000 in the
default preset and 1.2 million in `--matched`. Emitting every `k = residence/markers` steps
makes the ion pool the markers asked for, each marker carrying `k` steps of flux and placed
where its orbit has taken it since it crossed, so the stream moves `dx/markers_per_cell` in a
window -- the marker spacing -- and turns through under a hundredth of a radian. The
electrons, whose gyro-angle is already a quarter radian a step, stay at `k = 1`. `--every=K`
sets the ions' `k` by hand; `--every=1` is the control that the result does not depend on it.

The cell is half a Debye length in all three, which resolves the sheath and the presheath
but **not** the electron gyroradius, `rho_e = 0.2 lambda_D`. The finite-`rho_e` electron
response is what the reference keeps in its Debye sheath, so the comparison there is
between a resolved kinetic electron and one whose orbit the grid averages over; the
presheath, whose scale is `rho_s = 5.7 lambda_D` and more, does not have that problem.

The reference's own figure -- `m_i/m_e = 3600` at 2.5 degrees -- is out of reach for full
orbits: the box is 719 Debye lengths, the ions cross it at `c_s sin(2.5 deg)`, and
resolving the electron gyro-phase caps the step at `omega_pe dt = 0.075`, which is 9.3e6
steps a transit. That is the asymptotic limit doing its job, and it is worth stating
rather than approximating quietly.

With `--reference DIR`, the profiles are compared against the GYRAZE output directory at
DIR. Nothing from that code is kept here: it is unlicensed, so it is run separately and
read, not vendored.
"""

import json
import os
import sys
from pathlib import Path

# Double precision is the default. The comparison below is at the per cent level and the
# potential is a difference of large numbers.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from jaxincell import (Domain, Impacts, Simulation, Solver, Source, Species, epsilon_0, figure, mass_electron,
                       potential, provenance, elementary_charge as e_charge)

# --- what to change ---------------------------------------------------------------------
quick = "--quick" in sys.argv
matched = "--matched" in sys.argv
reference = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--reference=")), None)
electron_temperature = 1.0                  # eV
density = 1e16                              # m^-3, at the entrance plane
temperature_ratio = 1.0                     # tau = Z T_i / T_e, the width parameter of F above
mass_ratio = 100.0 if quick else (900.0 if matched else 400.0)
angle_degrees = 15.0 if quick else (4.0 if matched else 5.0)
gyro_over_debye = 0.2                       # gamma = rho_e/lambda_D at the entrance plane
buffer_gyro = 2.0 if quick else 3.0         # rho_s of run-up before the presheath proper
presheath_gyro = 6.0 if quick else 15.0     # rho_s of magnetic presheath resolved
sheath_debye = 30.0 if quick else 60.0      # lambda_D of Debye sheath resolved
cells_per_debye = 2.0                       # dx = 0.5 lambda_D, which does not resolve rho_e
markers_per_cell = 60 if quick else 100
transits = 2.0 if quick else 3.0            # entrance-speed crossings of the box
# the two knobs that trade noise and settling against the run's cost, so that a first look
# need not be a whole night: --markers=40 --transits=2 is about a quarter of the default
for flag, name in (("--markers=", "markers_per_cell"), ("--transits=", "transits")):
    given = next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag)), None)
    if given is not None:
        globals()[name] = type(globals()[name])(given)
# the steps between ion emissions, worked out below unless given: --every=1 is the control
ion_every = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--every=")), None)
reservoir_samples = 200000                  # velocities drawn once to stand for each reservoir

# --- the scales, and the manifest they make ----------------------------------------------
spread = np.sqrt(electron_temperature * e_charge / mass_electron)     # sqrt(T_e/m_e)
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye = spread / omega_pe
ion_spread = spread * np.sqrt(temperature_ratio / mass_ratio)         # sqrt(T_i/m_i)
sound_speed = spread * np.sqrt((1 + temperature_ratio) / mass_ratio)  # c_s = sqrt((ZT_e+T_i)/m_i)
angle = np.radians(angle_degrees)
# gamma = rho_e/lambda_D = omega_pe/Omega_e fixes B, and rho_s/lambda_D follows from it
field = mass_electron * omega_pe / (e_charge * gyro_over_debye)
omega_ce, omega_ci = e_charge * field / mass_electron, e_charge * field / (mass_ratio * mass_electron)
gyro_radius = sound_speed / omega_ci                                  # rho_s
length = (buffer_gyro + presheath_gyro) * gyro_radius + sheath_debye * debye
cells = int(round(length / debye * cells_per_debye))
time_step = min(0.05, 0.25 * gyro_over_debye) / omega_pe              # Omega_e dt <= 0.25
# the mean parallel speed of the entrance distribution, 1.5958 sqrt(T_i/m_i) at u = 0, times
# sin(alpha): a magnetised particle's progress towards the wall is along the field
entrance_speed = 1.5958 * ion_spread * np.sin(angle)
stored = 40
steps = stored * max(int(transits * length / entrance_speed / time_step) // stored, 1)

print("grazing-incidence sheath: the manifest both codes have to agree on")
print(f"  alpha {angle_degrees:.1f} deg to the wall plane; x runs from the entrance plane to the wall "
      f"and phi is measured from the plane")
print(f"  m_i/m_e {mass_ratio:.0f}   tau = ZT_i/T_e {temperature_ratio:.1f} (the width of F, not a "
      f"Maxwellian temperature)   Z 1")
print(f"  rho_s/lambda_D {gyro_radius / debye:.2f} with rho_s = c_s/Omega_i and c_s = sqrt((ZT_e+T_i)/m_i); "
      f"the Bohm gyroradius is smaller by sqrt(1+tau) = {np.sqrt(1 + temperature_ratio):.3f}")
print(f"  rho_e/lambda_D {omega_pe / omega_ce:.2f}   L/rho_s {length / gyro_radius:.1f}   "
      f"L/lambda_D {length / debye:.0f}   dx/lambda_D {length / cells / debye:.2f}")
print(f"  omega_pe dt {omega_pe * time_step:.3f}   Omega_e dt {omega_ce * time_step:.2f}   "
      f"Omega_i dt {omega_ci * time_step:.1e}   {steps} steps = {transits:.0f} entrance crossings")
print("  velocities are normalised to sqrt(T/m), not sqrt(2T/m); the wall floats, and its potential "
      "is measured, not prescribed")
if quick:
    print("\n--quick is a smoke preset: the angle is not grazing and the scales are not separated, so\n"
          "nothing below is a measurement of the grazing-incidence problem. It checks that the script\n"
          "runs and that the pools hold.\n")

# --- the entrance distributions, sampled -------------------------------------------------


def chodura_drift(tau, step=1e-4):
    """The drift `u` of the entrance distribution, found the way GYRAZE finds it: the value
    at which the kinetic Chodura integral equals `tau`, to that code's own 0.05. At `tau = 1`
    it is zero, so the distribution is `v_par^2` times a Maxwellian at rest."""
    def integral(u):
        norm = (np.sqrt(2) * (1 + erf(u / np.sqrt(2))) * (1 + u * u)
                + np.sqrt(2 / np.pi) * u * np.exp(-0.5 * u * u))
        return np.sqrt(2) * (1 + erf(u / np.sqrt(2))) / norm
    u, value = 0.0, np.inf
    while (value > tau) or (value < tau - 0.05):
        u += step if value > tau else -step
        value = integral(u)
    return u


def entrance_velocities(count, drift, rng):
    """`count` ion velocities of the entrance distribution, in the box's frame.

    The parallel speed follows `p(v) ~ v^2 exp(-(v-u)^2/2)` on `v > 0`, drawn from its own
    quantile: rejection against a normal of the same width is not valid, because the ratio
    `v^2` is unbounded. The two perpendicular components are a Maxwellian, and the gyrophase
    is uniform because the two are drawn independently. The result is rotated into the box,
    where the field lies at `alpha` to the wall plane in the x-z plane.
    """
    grid = np.linspace(0.0, drift + 12.0, 200001)
    weight = grid ** 2 * np.exp(-0.5 * (grid - drift) ** 2)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(grid))])
    parallel = np.interp(rng.random(count) * cumulative[-1], cumulative, grid)
    perpendicular = rng.standard_normal((count, 2))
    along = np.array([np.sin(angle), 0.0, np.cos(angle)])
    across = np.array([np.cos(angle), 0.0, -np.sin(angle)])
    return ion_spread * (parallel[:, None] * along + perpendicular[:, 0:1] * across
                         + perpendicular[:, 1:2] * np.array([0.0, 1.0, 0.0]))


rng = np.random.default_rng(0)
drift = chodura_drift(temperature_ratio)
ion_reservoir = entrance_velocities(reservoir_samples, drift, rng)
electron_reservoir = spread * rng.standard_normal((reservoir_samples, 3))
field_direction = np.array([np.sin(angle), 0.0, np.cos(angle)])
sampled_parallel = ion_reservoir @ field_direction / ion_spread
print(f"entrance distribution: u = {drift:.4f}; the sample has <v_par> {sampled_parallel.mean():.4f} "
      f"and <v_par^2> {(sampled_parallel ** 2).mean():.4f} against the closed forms "
      f"{1.5958 if drift < 1e-3 else float('nan'):.4f} and {3.0:.4f} at u = 0, in sqrt(T_i/m_i)")
print(f"  and under {100 * (sampled_parallel < 0.1).mean():.3f} per cent of it below 0.1, where a "
      f"Maxwellian would have {100 * erf(0.1 / np.sqrt(2)):.1f} per cent: that hole is the Chodura "
      "condition")

# --- the run -----------------------------------------------------------------------------
# Both species are magnetised, so both make progress towards the wall at their parallel speed
# times sin(alpha), and the pool has to hold `emit / every` times that residence in steps. An
# ion's residence is hundreds of thousands of steps, far more than the markers the run asks
# for, so the ions emit once every `every` steps rather than on every one: the pool is then the
# markers asked for and not the residence. A window's markers are spread over the distance the
# stream covers in it, which at the entrance speed is dx / markers_per_cell -- the marker
# spacing itself -- and the gyro-angle over a window, Omega every dt, is held to the step's own
# 0.25, which keeps the electrons at every = 1.
residence_i = length / entrance_speed / time_step
residence_e = length / (spread * np.sqrt(np.pi / 2) * np.sin(angle)) / time_step
markers = markers_per_cell * cells


def schedule(residence, gyro_angle_per_step, every=None):
    """`(emit, every)` for a pool of about `markers`: one emission every `residence / markers`
    steps, as many at a time as a pool of that size needs, and no window longer than a quarter
    of a gyro-radian."""
    if every is None:
        every = min(max(int(round(residence / markers)), 1), max(int(0.25 / gyro_angle_per_step + 1e-9), 1))
    return max(int(round(markers * every / residence)), 1), every


emit_ions, every_ions = schedule(residence_i, omega_ci * time_step, ion_every)
emit_electrons, every_electrons = schedule(residence_e, omega_ce * time_step)
capacity_ions = int(1.4 * emit_ions * residence_i / every_ions)
capacity_electrons = int(1.4 * emit_electrons * residence_e / every_electrons)
print(f"residence: ions {residence_i:.0f} steps, electrons {residence_e:.0f}; emitting {emit_ions} every "
      f"{every_ions} steps and {emit_electrons} every {every_electrons} into pools of {capacity_ions} and "
      f"{capacity_electrons}")
print(f"  a window moves the ion stream {entrance_speed * every_ions * time_step / (length / cells):.3f} dx "
      f"and turns it {omega_ci * every_ions * time_step:.1e} rad")

external_B = np.zeros((cells, 3))
external_B[:, 0], external_B[:, 2] = field * np.sin(angle), field * np.cos(angle)
domain = Domain(length=length, cells=cells, time_step=time_step,
                particle_bc="absorbing", field_bc=("open", "absorbing"))
# an ion falls through a few T_e of sheath and arrives with that plus its thermal energy, so a
# ceiling of twenty is generous; what passes it goes to the overflow bin and is said rather than
# piled onto the end of the spectrum
impacts = Impacts(energy_max=20.0 * electron_temperature * e_charge, energy_bins=40, angle_bins=30)
energies = 0.5 * (np.asarray(impacts.energy_edges)[:-1] + np.asarray(impacts.energy_edges)[1:]) / e_charge
angles = 0.5 * (np.asarray(impacts.angle_edges)[:-1] + np.asarray(impacts.angle_edges)[1:])
electrons = Species("electrons", capacity_electrons, -1.0, mass_electron, density,
                    active=capacity_electrons // 4, sampling="quiet",
                    source=Source(density=density, samples=electron_reservoir, emit=emit_electrons,
                                  every=every_electrons))
ions = Species("ions", capacity_ions, 1.0, mass_ratio * mass_electron, density,
               active=capacity_ions // 4, sampling="quiet",
               source=Source(density=density, samples=ion_reservoir, emit=emit_ions, every=every_ions))
out = Simulation(domain, [electrons, ions], Solver(model="electrostatic"),
                 external_B=external_B, impacts=impacts).run(
    steps, seed=0, store_every=steps // stored, store_particles=False, moments="flux").validate()

# --- what came out -----------------------------------------------------------------------
late = stored // 2
faces = np.asarray(domain.faces)
phi = np.asarray(potential(out))[late:] / electron_temperature
profile = phi.mean(axis=0)
wall_potential = float(profile[-1])
window = np.asarray(out.moments[-1] - out.moments[late]) / float(out.steps[-1] - out.steps[late])
n_e, n_i = window[0, 0] / density, window[1, 0] / density
flow = np.divide(window[1, 1], window[1, 0], out=np.zeros(cells), where=window[1, 0] > 0) / sound_speed
centres = np.asarray(out.grid) + domain.length / 2                   # distance from the entrance plane
current = np.asarray(out.wall.collected)[-1, :, 1] - np.asarray(out.wall.collected)[late, :, 1]
# a short slice, such as a timing run, can end before any ion has reached the wall; every ratio to
# the ion current or fluence is then undefined, and is said and recorded as null, not divided by 0
none_arrived = "no ion reached the wall in the window"
net_current = float(100 * (current[1] - current[0]) / current[1]) if current[1] > 0 else None

print(f"\nwall potential {wall_potential:+.3f} T_e/e, measured from the entrance plane")
print(f"  net collector current {net_current:+.2f} per cent of the ion current: the wall floats, so it "
      "should be zero" if net_current is not None else f"  net collector current: {none_arrived}")
print(f"  ion flow at the wall {flow[-1]:.2f} c_s, and {flow[len(flow) // 2]:.2f} halfway in")
print(f"  densities: {n_e[0]:.3f} and {n_i[0]:.3f} n_0 at the plane, {n_e[-1]:.3f} and {n_i[-1]:.3f} "
      f"at the wall")
print(f"  pool: {int((np.asarray(out.state.w)[:capacity_electrons] > 0).sum())} electrons of "
      f"{capacity_electrons}, {int((np.asarray(out.state.w)[capacity_electrons:] > 0).sum())} ions of "
      f"{capacity_ions}; overflow {float(out.overflow[-1]):.3g}")
# the spectrum over the late window alone, binned at the crossing; the last energy bin holds
# everything above the ceiling and is reported as itself
spectrum = np.asarray(out.wall.spectrum[-1, 1, 1] - out.wall.spectrum[late, 1, 1])
fluence = float(out.wall.arrived[-1, 1, 1] - out.wall.arrived[late, 1, 1])
mean_energy = incidence = None
if fluence > 0 and spectrum.sum() > 0:
    mean_energy = float(out.wall.energy_in[-1, 1, 1] - out.wall.energy_in[late, 1, 1]) / fluence / e_charge
    incidence = float((spectrum.sum(axis=0) * angles).sum() / spectrum.sum())
    print(f"  ion impacts: fluence {fluence:.3e} m^-2, mean energy {mean_energy:.2f} eV, mean incidence "
          f"{incidence:.0f} deg from the normal, {100 * spectrum[-1].sum() / spectrum.sum():.2f} per cent "
          "above the energy ceiling")
else:
    print(f"  ion impacts: {none_arrived}")

# --- against the reference, if it is there -----------------------------------------------
comparison = {}
if reference is not None:
    folder = Path(reference)
    presheath = np.loadtxt(folder / "phi_n_MP.txt")                  # x/rho_s, phi, sum n_i, n_e
    sheath = np.loadtxt(folder / "phi_n_DS.txt")                     # x/lambda_D from the wall
    # their x runs from the Debye-sheath entrance outwards, and ours from the entrance plane in
    their_x = (presheath[:, 0] - presheath[0, 0]) * gyro_radius
    ours = length - centres                                          # distance from the wall
    # the far field of each file has one density column written as zeros; drop it
    good = (presheath[:, 2] > 0) & (presheath[:, 3] > 0)
    theirs = np.interp(their_x[good][::-1], their_x[good][::-1], presheath[good, 1][::-1])
    mine = np.interp(their_x[good][::-1], ours[::-1], np.interp(ours, faces + domain.length / 2, profile)[::-1])
    comparison = {"presheath_potential_max_difference": float(np.abs(mine - theirs).max()),
                  "their_wall_potential": float(sheath[0, 1] + presheath[0, 1])}
    print(f"\nagainst {folder}: the presheath potentials differ by at most "
          f"{comparison['presheath_potential_max_difference']:.3f} T_e/e, and their wall potential is "
          f"{comparison['their_wall_potential']:+.3f} against this run's {wall_potential:+.3f}")

# --- the figure and the record -----------------------------------------------------------
fig, axes = figure(3)
axes[0].plot(centres / gyro_radius, np.interp(centres, faces + domain.length / 2, profile), color="C0")
axes[0].set(xlabel=r"$x/\rho_s$ from the entrance plane", ylabel=r"$e\phi/T_e$", title="the potential")
axes[1].plot(centres / gyro_radius, n_i, label=r"$n_i$")
axes[1].plot(centres / gyro_radius, n_e, label=r"$n_e$")
axes[1].set(xlabel=r"$x/\rho_s$", ylabel=r"$n/n_0$", title="densities")
axes[1].legend(frameon=False)
axes[2].plot(centres / gyro_radius, flow, color="C2")
axes[2].axhline(1.0, color="0.6", lw=2)
axes[2].set(xlabel=r"$x/\rho_s$", ylabel=r"$\langle v_x\rangle/c_s$", title="ion flow towards the wall")
plt.tight_layout()

folder = Path.cwd() / ("grazing_sheath_quick" if quick else
                       ("grazing_sheath_matched" if matched else "grazing_sheath"))
folder.mkdir(exist_ok=True)
settings = dict(electron_temperature=electron_temperature, density=density, mass_ratio=mass_ratio,
                temperature_ratio=temperature_ratio, angle_degrees=angle_degrees,
                gyro_over_debye=gyro_over_debye, rho_s_over_debye=float(gyro_radius / debye),
                length_over_debye=float(length / debye), cells=cells, steps=steps,
                omega_pe_dt=float(omega_pe * time_step), omega_ce_dt=float(omega_ce * time_step),
                emit_ions=emit_ions, emit_electrons=emit_electrons, every_ions=every_ions,
                every_electrons=every_electrons, capacity_ions=capacity_ions,
                capacity_electrons=capacity_electrons, chodura_drift=float(drift),
                transits=transits, quick=quick, matched=matched)
results = dict(wall_potential=wall_potential, flow_at_the_wall=float(flow[-1]),
               ion_fluence=fluence, mean_impact_energy=mean_energy, mean_incidence=incidence,
               net_current_percent=net_current,
               density_at_the_plane=[float(n_e[0]), float(n_i[0])],
               overflow=float(out.overflow[-1]), **comparison)
(folder / "run.json").write_text(json.dumps(provenance(example="grazing_sheath", settings=settings,
                                                       results=results), indent=1))
np.savez(folder / "profiles.npz", centres=centres, faces=faces, phi=profile, n_e=n_e, n_i=n_i,
         flow=flow, spectrum=spectrum, energies=energies, angles=angles)
fig.savefig(folder / "figure.png")
print(f"\nwrote {folder}/run.json, profiles.npz and figure.png")
plt.show()
