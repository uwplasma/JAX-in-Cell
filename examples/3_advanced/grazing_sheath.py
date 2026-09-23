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

With `--reference=DIR`, the run is compared against the GYRAZE output directory at DIR and
the manifest beside it: the potential and both densities through the presheath, the
Debye-sheath drop, the wall potential, the mean ion impact energy and the ion flux, each
against a tolerance declared below, with the reference dashed on the figure. GYRAZE fixes
`gamma` at the Debye-sheath entrance rather than at the entrance plane, so a matched run
passes the entrance value its manifest names, `--gamma=G`; a run that does not match is said
to be a different problem. Nothing from that code is kept here: it is unlicensed, so it is
run separately and read, not vendored.
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
# need not be a whole night: --markers=40 --transits=2 is about a quarter of the default. The
# third, --gamma=G, is rho_e/lambda_D at the entrance plane: a GYRAZE reference fixes it at the
# Debye-sheath entrance instead, and its manifest names the entrance value that matches it
for flag, name in (("--markers=", "markers_per_cell"), ("--transits=", "transits"),
                   ("--gamma=", "gyro_over_debye")):
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

# --- the reference, read before the run so that a bad one fails now and not after hours ----
# GYRAZE is unlicensed, so it is run outside this repository (plan.md 8.1): only its output
# directory is read, with the manifest.json that has to sit beside it.
if reference is not None:
    reference = Path(reference).expanduser()
    manifest = json.loads((reference / "manifest.json").read_text())
    missing = [k for k in ("case", "reference", "run", "gamma_definition", "coordinates",
                           "potential_references", "normalisations", "conversion_to_jax_in_cell",
                           "incoming_distributions", "debye_reference_density", "asymptotics")
               if k not in manifest]
    if missing or manifest["run"]["exit_status"] != 0:
        sys.exit(f"{reference}: the manifest lacks {missing} or GYRAZE did not exit cleanly")
    case = manifest["case"]
    ours_vs_theirs = {"m_i/m_e": (mass_ratio, case["M"]), "tau": (temperature_ratio, case["tau"]),
                      "alpha": (angle_degrees, case["alpha_deg"]),
                      "gamma at the entrance plane": (gyro_over_debye, case["gamma_up_equivalent"])}
    unmatched = [f"{k} {a:g} against {b:g}" for k, (a, b) in ours_vs_theirs.items() if abs(a / b - 1) > 0.01]
    print(f"reference {reference.name}: {manifest['role']}, GYRAZE {manifest['reference']['commit'][:10]}, "
          f"gamma_DS {case['gamma_DS']:g} (= {case['gamma_up_equivalent']:g} at the entrance plane)")
    if unmatched:
        print("  NOT the same problem as this run -- " + "; ".join(unmatched) + ". The comparison below "
              "runs end to end, and its pass/fail says nothing about either code.")

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
# GYRAZE's files, in the C's write order and units (checked in the pinned source, plan.md 8.1):
#   phi_n_MP.txt   x/rho_B from the Debye-sheath entrance out, phi (zero at the presheath
#                  entrance), sum n_i, n_e (over the presheath-entrance density)
#   phi_n_DS.txt   x/rho_e from the wall out, phi - phi_DSE, sum n_i, n_e (over n_e at the DSE)
#   misc_output.txt  net current, |phi_wall|, Q_e, sum Q_i, flux_e, sum flux_i (along B)
# rho_B = sqrt(Z T_e m_i)/(ZeB) = rho_s/sqrt(1+tau) is the Bohm gyroradius, not rho_s, and
# rho_e = sqrt(T_e m_e)/(eB) = gamma lambda_D: both are fixed by B alone. The two layers are
# joined the matched-asymptotic way -- the potentials add, the densities multiply -- and past its
# own far end the Debye-sheath layer takes its matching value, 0 and 1.
#
# The tolerances are declared here, before any matched run. A number passes when
#     |this run - reference| <= SIGMAS * standard error + (max(epsilon, alpha) + (dx/l)^2) * scale.
SIGMAS = 3.0    # this run's noise: the standard error of BLOCKS block means of the late window
BLOCKS = 4      # four blocks of five stored frames: the fewest that still give a spread
# The reference is the limit epsilon = lambda_D,DSE/rho_B -> 0 at lowest order in alpha, so it
# drops terms of first order in both; with no computed coefficient, the allowance is that order
# times the layer's own drop (`scale`), coefficient one. (dx/l)^2 is the second-order deposit and
# field solve on the layer's scale l: lambda_D,DSE in the Debye sheath, rho_B in the presheath.
PRESHEATH_EDGE = 0.1   # the presheath is compared where the reference Debye-sheath drop is under 10 %
comparison = {}
if reference is not None:
    presheath = np.loadtxt(reference / "phi_n_MP.txt")
    sheath = np.loadtxt(reference / "phi_n_DS.txt")
    misc = np.loadtxt(reference / "misc_output.txt")
    rho_e = gyro_over_debye * debye
    rho_B = gyro_radius / np.sqrt(1 + temperature_ratio)
    n_dse = presheath[0, 3]
    epsilon = 1 / (gyro_over_debye * np.sqrt(mass_ratio * n_dse))   # lambda_D,DSE / rho_B in this run
    order = max(epsilon, angle)
    dx = length / cells

    def layer(table, column, unit, d, beyond):
        """One reference column at this run's distances from the wall. The far field that the C
        writes as exact zeros is an unfilled array, not a profile: it is cut off, and past the cut
        the column is `beyond`."""
        end = np.flatnonzero(table[:, column])[-1] + 1
        return np.interp(d, table[:end, 0] * unit, table[:end, column], right=beyond)

    def composite(d):
        """The reference's (phi, n_i, n_e) at distance d from the wall; NaN past the presheath file."""
        return (layer(presheath, 1, rho_B, d, np.nan) + layer(sheath, 1, rho_e, d, 0.0),
                layer(presheath, 2, rho_B, d, np.nan) * layer(sheath, 2, rho_e, d, 1.0),
                layer(presheath, 3, rho_B, d, np.nan) * layer(sheath, 3, rho_e, d, 1.0))

    # this run's late window in BLOCKS pieces, for the standard errors
    cut = np.linspace(late, len(out.steps) - 1, BLOCKS + 1).round().astype(int)
    span = np.diff(np.asarray(out.steps)[cut]).astype(float)
    moments = np.diff(np.asarray(out.moments)[cut], axis=0)[:, :, 0] / span[:, None, None] / density
    arrived = np.diff(np.asarray(out.wall.arrived)[cut, 1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = np.diff(np.asarray(out.wall.energy_in)[cut, 1, 1]) / arrived / e_charge / electron_temperature
    phi_blocks = np.array([block.mean(axis=0) for block in np.array_split(phi, BLOCKS)])

    def error(blocks):
        return np.asarray(blocks).std(axis=0, ddof=1) / np.sqrt(BLOCKS)

    def number(value):
        return float(value) if value is not None and np.isfinite(value) else None   # JSON has no NaN

    # where things are compared: out of the source's run-up, inside the presheath file, and, for the
    # profiles, outside the reference's Debye sheath
    d_face, d_cell = length - (faces + domain.length / 2), length - centres
    top = length - buffer_gyro * gyro_radius
    drop_ds = layer(sheath, 1, rho_e, 0.0, 0.0)
    d_grid = np.linspace(0, top, 20001)
    edge = d_grid[np.argmax(np.abs(layer(sheath, 1, rho_e, d_grid, 0.0)) < PRESHEATH_EDGE * abs(drop_ds))]
    ref_face, ref_cell = composite(d_face), composite(d_cell)
    out_face = (d_face >= edge) & (d_face <= top) & np.isfinite(ref_face[0])
    phi_at_edge = np.interp(edge, d_face[::-1], profile[::-1])
    phi_edge_blocks = [np.interp(edge, d_face[::-1], b[::-1]) for b in phi_blocks]
    ref_at_edge = composite(np.array([edge]))
    lam_dse = debye / np.sqrt(n_dse)
    # name: (this run, reference, standard error, scale, the layer's length); a profile's row holds
    # its largest difference in place of this run's number, and None for the reference's
    rows = {}
    rows["wall potential [T_e/e]"] = (wall_potential, ref_face[0][-1], error(phi_blocks[:, -1]),
                                      abs(ref_face[0][-1]), lam_dse)
    rows["Debye-sheath drop [T_e/e]"] = (wall_potential - phi_at_edge, ref_face[0][-1] - ref_at_edge[0][0],
                                         error(phi_blocks[:, -1] - phi_edge_blocks),
                                         abs(ref_face[0][-1] - ref_at_edge[0][0]), lam_dse)
    rows["presheath potential, max |diff| [T_e/e]"] = (
        float(np.abs(profile - ref_face[0])[out_face].max()), None, error(phi_blocks)[out_face].max(),
        abs(ref_at_edge[0][0]), rho_B)
    for name, index, ours in (("ion", 1, n_i), ("electron", 2, n_e)):
        inside = (d_cell >= edge) & (d_cell <= top) & np.isfinite(ref_cell[index])
        rows[f"{name} density, max |diff| [n_0]"] = (
            float(np.abs(ours - ref_cell[index])[inside].max()), None,
            error(moments[:, 1 if index == 1 else 0])[inside].max(),
            float(np.ptp(ref_cell[index][inside])), rho_B)
    # each ion's energy is conserved in the static fields, so its mean at the wall is the entrance
    # flux's own (sum Q_i / sum flux_i, which is 3 tau for this distribution) plus |phi_wall|
    rows["mean ion impact energy [T_e]"] = (
        mean_energy / electron_temperature if mean_energy is not None else np.nan,
        misc[3] / misc[5] + misc[1], error(energy), abs(ref_face[0][-1]), lam_dse)
    # the ion flux is fixed by the entrance distribution in both codes: only the noise is allowed
    rows["ion flux to the wall [n_0 sqrt(T_e/m_e)]"] = (
        fluence / (span.sum() * time_step) / (density * spread), misc[5] * np.sin(np.radians(case["alpha_deg"])),
        error(arrived / (span * time_step) / (density * spread)), 0.0, np.inf)

    print(f"\nagainst {reference}: epsilon = lambda_D,DSE/rho_B {epsilon:.3f} and alpha {angle:.3f} rad, "
          f"the orders the reference drops; the presheath is compared from {edge / rho_B:.2f} rho_B")
    quantities = {}
    for name, (mine, theirs, se, scale, ell) in rows.items():
        tolerance = SIGMAS * float(se) + (order + (dx / ell) ** 2) * scale
        difference = mine if theirs is None else abs(mine - theirs)
        verdict = bool(difference <= tolerance) if np.isfinite(difference + tolerance) else None
        quantities[name] = dict(this_run=number(None if theirs is None else mine), reference=number(theirs),
                                difference=number(difference), standard_error=number(se), scale=float(scale),
                                tolerance=number(tolerance), passed=verdict)
        pair = f"{'':8s}  {'':8s}" if theirs is None else f"{mine:+.4f}  {theirs:+.4f}"
        word = {True: "pass", False: "FAIL", None: "not measured"}[verdict]
        print(f"  {name:44s} {pair}  |diff| {difference:.4f} <= {tolerance:.4f}? {word}")
    comparison = dict(
        reference=str(reference), role=manifest["role"], case=case, matched=not unmatched,
        unmatched=unmatched, epsilon=float(epsilon), alpha=float(angle), sigmas=SIGMAS, blocks=BLOCKS,
        presheath_from_rho_B=float(edge / rho_B), quantities=quantities,
        impact_distribution="not compared: Fi_W.txt holds the wall orbits' invariants, and making an "
                            "energy-angle distribution of them is GYRAZE's own post-processing, which "
                            "would have to be re-derived; the mean impact energy needs none of it")
    if unmatched:
        print("  (not the same problem: see above)")

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
if reference is not None:                       # the reference, dashed, on the same axes
    d_plot = np.linspace(0.0, length, 4000)
    ref_phi, ref_ni, ref_ne = composite(d_plot)
    axes[0].plot((length - d_plot) / gyro_radius, ref_phi, "--", color="0.3", label="GYRAZE")
    axes[0].legend(frameon=False)
    axes[1].plot((length - d_plot) / gyro_radius, ref_ni, "--", color="C0")
    axes[1].plot((length - d_plot) / gyro_radius, ref_ne, "--", color="C1")
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
               overflow=float(out.overflow[-1]), comparison=comparison or None)
(folder / "run.json").write_text(json.dumps(provenance(example="grazing_sheath", settings=settings,
                                                       results=results), indent=1))
np.savez(folder / "profiles.npz", centres=centres, faces=faces, phi=profile, n_e=n_e, n_i=n_i,
         flow=flow, spectrum=spectrum, energies=energies, angles=angles)
fig.savefig(folder / "figure.png")
print(f"\nwrote {folder}/run.json, profiles.npz and figure.png")
plt.show()
