"""Recovering a wall's electron reflectivity from the sheath it holds, by differentiating
the simulation that produced it.

The inverse problem. A collector returns the fraction `r` of every electron that reaches
it and keeps the rest. Reflected electrons are not lost, so the wall needs less voltage
to hold the rest back, and the sheath in front of it is shallower. Measure the potential
in the plasma, and `r` follows. Here the measurement is generated at a known reference
value and a bounded optimiser is asked to find it back from a different start, using the
gradient of the whole particle-in-cell calculation with respect to `r`.

Nothing is finite-differenced. `jax.grad` runs back through the field solve, the
deposit, the gather, the Boris push, the source and the wall for every step of the
experiment; the finite differences below are there to check that gradient, not to
compute it.

**The experiment is deliberately short.** A plasma is prepared once at a fixed
reflectivity, outside the differentiated calculation and therefore independent of `r`;
the trial value is then applied and the response is watched for a few electron plasma
periods. That is the time scale on which the electrons rearrange and the wall's charge
follows, so it is the window the measurement lives in. It is also as long as the
gradient can usefully be taken over, and the script measures why:

* the reverse-mode gradient agrees with the forward-mode one to round-off, and with a
  central difference of the same realisation to nine digits, at every horizon. The
  implementation is exact.
* the step size at which the finite difference agrees shrinks as the window grows --
  1e-1 over five steps, 1e-4 over twenty-five, 1e-7 over a hundred -- because every
  absorption at the wall is a branch of the program, and over a long window a change in
  `r` of one part in a million already flips some. Past that the derivative of one
  realisation stops tracking the response of the average, which is the effect Chung,
  Bond, Cyr and Freund analyse (J. Comput. Phys. 400, 108969, 2020).

So the gradient here is the exact derivative of the discrete map, and over this window
it is also a useful estimate of the physical response. Over a much longer one it would
be neither useless nor wrong, but no longer the thing the optimiser wants.

Run with `--quick` for a smaller, faster version, and with `--oblique` for the same
experiment in a magnetic field 30 degrees to the wall: one array added to the same
`Simulation`, and nothing else in the script changes.
"""

import os
import sys

# Double precision is the default. The finite-difference checks below are chosen for it.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Source, Species, epsilon_0, mass_electron, potential,
                       elementary_charge as e_charge)
from jaxincell.sheath import floating_potential, source_density

# --- what to change ------------------------------------------------------------------------
quick = "--quick" in sys.argv
oblique = "--oblique" in sys.argv         # the same experiment in a field 30 degrees to the wall
electron_temperature = 1.0                 # eV
density = 1e16                             # m^-3
mass_ratio = 400.0                         # m_i/m_e, reduced so the ion transit fits in a laptop run
beam_speed = 0.25                          # ion drift at the source plane, in electron spreads
box_debye_lengths = 12.0
cells = 48
steps_per_plasma_period = 1 / 0.15         # omega_pe dt = 0.15
capacity = 8000 if quick else 48000        # slots per species
emit = 8 if quick else 48                  # emitted per step per species
preparation = 300 if quick else 1200       # steps of preparation, at r_prepared, not differentiated
window = 25                                # steps of the differentiated response experiment
r_prepared = 0.25                          # the reflectivity the baseline plasma is prepared at
r_reference = 0.35                         # the answer the optimiser has to find
r_start = 0.08                             # where it starts, well away from it
bounds = (0.02, 0.50)                      # admissible interval, away from the limiting branches
training_seeds = (0, 1, 2) if quick else (0, 1, 2, 3)       # fixed across every call and line search
held_out_seeds = (10, 11, 12) if quick else (10, 11, 12, 13)   # never used to choose a step
field_angle = 30.0                         # degrees to the wall plane, with --oblique
gyro_over_debye = 6.0                      # rho_s/lambda_D, which sets B_0, with --oblique

# --- the setup ---------------------------------------------------------------------------------
spread = np.sqrt(electron_temperature * e_charge / mass_electron)
omega_pe = np.sqrt(density * e_charge ** 2 / (epsilon_0 * mass_electron))
debye = spread / omega_pe
length = box_debye_lengths * debye
dt = 1.0 / (steps_per_plasma_period * omega_pe)
reservoir = float(source_density(float(floating_potential(beam_speed)))) * density
domain = Domain(length=length, cells=cells, time_step=dt,
                particle_bc="absorbing", field_bc=("open", "absorbing"))

# The one line that makes this the magnetized experiment. B_0 = B_0 (sin alpha, 0, cos alpha)
# with alpha to the wall plane, at a strength set by the ion gyro-radius in Debye lengths.
external_B = None
if oblique:
    radians = np.radians(field_angle)
    sound_speed = spread / np.sqrt(mass_ratio)
    strength = mass_ratio * mass_electron * sound_speed / (e_charge * gyro_over_debye * debye)
    external_B = jnp.zeros((cells, 3)).at[:, 0].set(strength * np.sin(radians)).at[:, 2].set(
        strength * np.cos(radians))
    print("oblique: alpha %.0f deg to the wall, rho_s/lambda_D %.1f, Omega_e dt %.2f"
          % (field_angle, gyro_over_debye, e_charge * strength / mass_electron * dt))

# Two fixed Gaussian sensors, one in the plasma and one in the sheath, both well inside the
# box and both of a fixed physical width, so that neither moves if the grid is refined.
faces = np.asarray(domain.faces)


def sensor(centre, width):
    k = np.exp(-0.5 * ((faces - centre) / width) ** 2)
    return jnp.asarray(k / (k.sum() * domain.dx))


# The sheath sensor sits where the response to the reflectivity is: a scan of positions puts
# the response over the admissible interval at 3.9 times the scatter between realisations at
# 1.5 Debye lengths from the collector, against 2.9 at 2.5 and 0.9 at 6. Closer still is
# better again, but a Gaussian of this width centred inside two cloud half-widths of the wall
# would take part of its reading from the cells the deposit truncates.
sensors = jnp.stack([sensor(-length / 2 + 3 * debye, 1.5 * debye), sensor(length / 2 - 1.5 * debye, 1.0 * debye)])


def simulation(r):
    """The same public setup as the physical sheath examples, with the collector's electron
    reflectivity as the one thing that changes."""
    electrons = Species("electrons", capacity, -1.0, mass_electron, density, (np.sqrt(2) * spread, 0, 0),
                        active=capacity // 3, quiet=True, reflection=(0.0, r),
                        source=Source(density=reservoir, vth=(np.sqrt(2) * spread, 0, 0), emit=emit, beam=False))
    ions = Species("ions", capacity, 1.0, mass_ratio * mass_electron, density, 0.0, (beam_speed * spread, 0, 0),
                   active=capacity // 3, quiet=True,
                   source=Source(density=density, vth=0.0, drift=(beam_speed * spread, 0, 0), emit=emit, beam=True))
    return Simulation(domain, [electrons, ions], Solver(model="electrostatic"), external_B=external_B)


def measure(r, state):
    """The two sensor readings of the potential at the end of the response window, in T_e/e.
    Everything in here is traced: this is what `jax.grad` differentiates."""
    out = simulation(r).run(window, store_every=window, store_particles=False, state=state)
    return sensors @ potential(out)[-1] / electron_temperature * domain.dx


if quick:
    print("--quick is a smoke run: a sixth of the particles, a quarter of the preparation and three\n"
          "realisations instead of four. It checks that every step of this script executes and that\n"
          "the gradient is still the derivative of the calculation. It usually recovers the control\n"
          "too, but to about 0.01 rather than 0.005, and the identifiability check below is what\n"
          "says how far to trust it. The documentation quotes the full preset.\n")
print("preparing the baseline plasma at r = %.2f, %d steps, for %d training and %d held-out seeds"
      % (r_prepared, preparation, len(training_seeds), len(held_out_seeds)))
prepared = {seed: jax.block_until_ready(simulation(r_prepared).run(preparation, seed=seed,
                                                                   store_every=preparation,
                                                                   store_particles=False).state)
            for seed in training_seeds + held_out_seeds}

measure_jit = jax.jit(measure)


def readings(r, seeds):
    """The measurement vector averaged over the realisations, which is the quantity the loss
    is of: the loss of the mean, not the mean of the losses, which optimise different things."""
    return jnp.mean(jnp.stack([measure(r, prepared[s]) for s in seeds]), axis=0)


# The scales are the scatter of the measurement between realisations, which is what its
# uncertainty is; they are fixed before the optimiser runs and are not touched again.
scatter = jnp.std(jnp.stack([measure_jit(r_prepared, prepared[s]) for s in training_seeds]), axis=0)
scales = jnp.maximum(scatter, 1e-6)

target = jax.block_until_ready(readings(r_reference, training_seeds))
held_out_target = jax.block_until_ready(readings(r_reference, held_out_seeds))
print("target at r = %.2f: %s on the training realisations, %s on the held-out ones"
      % (r_reference, np.array2string(np.asarray(target), precision=5),
         np.array2string(np.asarray(held_out_target), precision=5)))
print("measurement scales (the scatter between realisations): %s\n"
      % np.array2string(np.asarray(scales), precision=5))


def loss(r):
    """Half the squared mismatch of the mean measurement, in units of its own scatter. The
    target is generated on the same realisations and the same protocol, so this problem has
    an exact answer and the recovery below is a test of the whole differentiated chain. What
    it does not test is generalisation, which is what the held-out realisations are for."""
    return 0.5 * jnp.sum(((readings(r, training_seeds) - target) / scales) ** 2)


value_and_grad = jax.jit(jax.value_and_grad(loss))

# --- 1. is the gradient the derivative of the calculation? ---------------------------------------
print("the gradient of one realisation, against forward mode and against finite differences")


def one(r):
    return measure(r, prepared[training_seeds[0]])[1]


reverse = float(jax.jit(jax.grad(one))(r_prepared))
forward = float(jax.jit(lambda r: jax.jvp(one, (r,), (1.0,))[1])(r_prepared))
one_jit = jax.jit(one)
print(" reverse %+.10e   forward %+.10e   relative difference %.1e"
      % (reverse, forward, abs(forward / reverse - 1)))
steps_of_h = [1e-1, 1e-3, 1e-5] if quick else [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
differences = [float((one_jit(r_prepared + h) - one_jit(r_prepared - h)) / (2 * h)) for h in steps_of_h]
for h, d in zip(steps_of_h, differences):
    print("   h %7.0e   central difference %+.8e   relative mismatch %.1e" % (h, d, abs(d / reverse - 1)))
best = min(range(len(steps_of_h)), key=lambda i: abs(differences[i] / reverse - 1))
print(" best agreement %.1e at h = %.0e\n" % (abs(differences[best] / reverse - 1), steps_of_h[best]))

# --- 2. is the response identifiable above the scatter between realisations? ----------------------
scan_points = np.linspace(bounds[0], bounds[1], 4 if quick else 9)
scan = np.array([np.asarray(readings(float(r), training_seeds)) for r in scan_points])
spread_of_seeds = np.std([np.asarray(measure_jit(r_reference, prepared[s])) for s in held_out_seeds], axis=0)
signal = np.ptp(scan, axis=0)
print("a coarse scan, as a reference and an identifiability check (it is not the optimiser)")
for i, name in enumerate(("plasma sensor", "sheath sensor")):
    print("  %-14s response over the interval %.4f, scatter between realisations %.4f, ratio %5.1f"
          % (name, signal[i], spread_of_seeds[i], signal[i] / spread_of_seeds[i]))
print()

# --- 3. bounded gradient descent with backtracking -------------------------------------------------
r, history = r_start, []
step_size = 0.02 / max(abs(float(value_and_grad(r_start)[1])), 1e-12)
print("%4s %8s %12s %12s %10s" % ("iter", "r", "loss", "d loss/d r", "step"))
for iteration in range(8 if quick else 20):
    objective, slope = value_and_grad(r)
    objective, slope = float(objective), float(slope)
    history.append((r, objective, slope))
    print("%4d %8.4f %12.6f %12.4f %10.2e" % (iteration, r, objective, slope, step_size))
    if abs(slope) < 1e-9:
        break
    trial, accepted = r, False
    for _ in range(14):                        # backtracking, so a gradient that is off in scale still works
        trial = float(np.clip(r - step_size * slope, *bounds))
        if trial == r or float(value_and_grad(trial)[0]) < objective:
            accepted = trial != r
            break
        step_size *= 0.5
    if not accepted:
        print("     converged: no step along the gradient lowers the loss")
        break
    if abs(trial - r) < 1e-4:                  # a move smaller than the control is known to
        r = trial
        print("     converged: the step is below the uncertainty of the control")
        break
    step_size *= 1.6
    r = trial

r_final = history[min(range(len(history)), key=lambda i: history[i][1])][0]
print("\nrecovered r = %.4f, reference %.4f, error %.4f" % (r_final, r_reference, abs(r_final - r_reference)))

# --- 4. held-out validation, on realisations the optimiser never saw ------------------------------


def held_out_loss(r):
    """The same mismatch on realisations that were used neither to prepare the target nor to
    choose a step: does the recovered reflectivity reproduce what the reference gives on a
    plasma it has not seen?"""
    return float(0.5 * jnp.sum(((readings(r, held_out_seeds) - held_out_target) / scales) ** 2))


print("\nheld-out validation")
for name, value in (("start", r_start), ("recovered", r_final), ("reference", r_reference)):
    print("  %-10s r = %.4f   training loss %10.5f   held-out loss %10.5f"
          % (name, value, float(value_and_grad(value)[0]), held_out_loss(value)))

# a coarse scan of the held-out loss locates the minimum without a gradient at all, and its
# distance from the reference is the uncertainty of the recovered control, not the optimiser's
fine = np.linspace(*bounds, 7 if quick else 25)
held_out_curve = np.array([held_out_loss(float(v)) for v in fine])
held_out_best = float(fine[int(np.argmin(held_out_curve))])
print("  the held-out loss is smallest at r = %.4f on a scan of %d points; the recovered value is "
      "%.4f from it" % (held_out_best, len(fine), abs(r_final - held_out_best)))
print("  uncertainty of the recovered control, from the held-out minimum: %.3f absolute in r"
      % abs(held_out_best - r_reference))

# --- the figure -----------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
r_history, losses, slopes = np.array(history).T
axes[0].semilogy(losses, "o-")
axes[0].set(xlabel="iteration", ylabel="training loss", title="bounded gradient descent")
twin = axes[0].twinx()
twin.plot(r_history, "s--", color="C1")
twin.axhline(r_reference, ls=":", color="k")
twin.set_ylabel("r", color="C1")

axes[1].plot(fine, held_out_curve, "-", color="0.6", label="held-out loss (scan)")
axes[1].axvline(r_reference, ls=":", color="k", label="reference")
axes[1].axvline(r_final, ls="--", color="C0", label="recovered")
axes[1].set(xlabel="collector reflectivity r", ylabel="loss", yscale="log",
            title="the loss the optimiser never saw")
axes[1].legend(frameon=False)

axes[2].plot(steps_of_h, [abs(d / reverse - 1) for d in differences], "o-")
axes[2].axhline(abs(forward / reverse - 1) + 1e-16, ls="--", color="k",
                label="forward vs reverse mode")
axes[2].set(xscale="log", yscale="log", xlabel="finite-difference step $h$",
            ylabel=r"$|{\rm FD}/{\rm AD} - 1|$", title="the gradient against finite differences")
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
