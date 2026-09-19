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
the trial value is then applied and the response is watched for twenty-five steps at
`omega_pe dt = 0.15`. That is `3.75` inverse plasma frequencies, which is **0.60 of an
oscillation** and not four periods: the two are `2 pi` apart and the script prints both,
because a window quoted in periods when it is inverse frequencies is six times longer
than it sounds. It is the time scale on which the electrons rearrange and the wall's
charge begins to follow, so it is the window the measurement lives in -- not a settled
response, which is why the result is a response and not a steady state. It is also as
long as the gradient can usefully be taken over, and the script measures why:

* the reverse-mode gradient agrees with the forward-mode one to round-off, and with a
  central difference of the same realisation to between eight and ten digits, at every
  horizon. The implementation is exact.
* the step size at which the finite difference agrees shrinks as the window grows --
  1e-2 over five steps, 1e-4 over twenty-five, 1e-6 over a hundred, measured over the
  three horizons in `docs/scripts/fig_sheath_source.py` -- because every
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

import json
import os
import sys
from pathlib import Path

# Double precision is the default. The finite-difference checks below are chosen for it.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jaxincell import (Domain, Simulation, Solver, Source, Species, epsilon_0, mass_electron, potential,
                       provenance,
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
# A third set, used for neither. The target built on the training realisations is exactly zero at
# the reference by construction, which makes the recovery below a test of the differentiated chain
# and not an inference; a target measured on realisations the optimiser never touches is an
# inference, and its residual at the optimum is not zero because it should not be.
target_seeds = (20, 21, 22) if quick else (20, 21, 22, 23, 24, 25)
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


# The sheath sensor sits where the response to the reflectivity is. Scanning the position over
# the same runs puts the response over the admissible interval at 13.0 times the scatter between
# realisations at 1.5 Debye lengths from the collector, against 16.1 at 0.8, 9.1 at 2.5 and 1.9
# at 6. Closer still is better again, but a Gaussian of this width centred inside two cloud
# half-widths of the wall would take part of its reading from the cells the deposit truncates.
sensors = jnp.stack([sensor(-length / 2 + 3 * debye, 1.5 * debye), sensor(length / 2 - 1.5 * debye, 1.0 * debye)])


def simulation(r):
    """The same public setup as the physical sheath examples, with the collector's electron
    reflectivity as the one thing that changes."""
    electrons = Species("electrons", capacity, -1.0, mass_electron, density, (np.sqrt(2) * spread, 0, 0),
                        active=capacity // 3, sampling="quiet", reflection=(0.0, r),
                        source=Source(density=reservoir, vth=(np.sqrt(2) * spread,) * 3, emit=emit, model="maxwellian"))
    ions = Species("ions", capacity, 1.0, mass_ratio * mass_electron, density, 0.0, (beam_speed * spread, 0, 0),
                   active=capacity // 3, sampling="quiet",
                   source=Source(density=density, vth=0.0, drift=(beam_speed * spread, 0, 0), emit=emit, model="beam"))
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
          "too, but with three noisy realisations and eight scan points the error bar below comes\n"
          "out around 0.09 against the full preset's 0.02, which is that check doing its job.\n"
          "The documentation quotes the full preset.\n")
# The pool has to hold every particle alive at once, and an electron lives longest at the
# largest reflectivity the optimiser may try. Checking the worst case here, once, on the
# host, covers every trial inside the interval: the differentiated measurement below is
# traced and has nothing to read.
worst = simulation(bounds[1]).run(preparation, seed=training_seeds[0], store_every=preparation,
                                  store_particles=False)
if worst.problems:
    raise SystemExit("the pool is too small at the most reflective end of the admissible interval, "
                     "r = %.2f, so no trial in it can be trusted. %s" % (bounds[1], worst.problems[0]))
print("pool checked at r = %.2f, the longest electron lifetime the optimiser may ask for" % bounds[1])

print("the response window is %d steps = %.2f / omega_pe = %.2f electron plasma oscillations"
      % (window, window * omega_pe * dt, window * omega_pe * dt / (2 * np.pi)))
print("preparing the baseline plasma at r = %.2f, %d steps, for %d training and %d held-out seeds"
      % (r_prepared, preparation, len(training_seeds), len(held_out_seeds)))
prepared = {seed: jax.block_until_ready(simulation(r_prepared).run(preparation, seed=seed,
                                                                   store_every=preparation,
                                                                   store_particles=False).validate().state)
            for seed in training_seeds + held_out_seeds + target_seeds}

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
independent = jax.block_until_ready(readings(r_reference, target_seeds))
print("target at r = %.2f: %s on the training realisations, %s on the held-out ones"
      % (r_reference, np.array2string(np.asarray(target), precision=5),
         np.array2string(np.asarray(held_out_target), precision=5)))
print("an independent target on %d realisations the optimiser never sees: %s"
      % (len(target_seeds), np.array2string(np.asarray(independent), precision=5)))
print("measurement scales (the scatter between realisations): %s\n"
      % np.array2string(np.asarray(scales), precision=5))


def loss(r):
    """Half the squared mismatch of the mean measurement, in units of its own scatter, against
    the target built on the **same** realisations. That makes the minimum exactly zero at the
    reference by construction, so what the recovery tests is the differentiated chain -- the
    source, the wall, the electrode closure, the gradient -- and not the ability to infer
    anything. It is a paired-realisation self-test and is labelled as one."""
    return 0.5 * jnp.sum(((readings(r, training_seeds) - target) / scales) ** 2)


def inference(r):
    """The same mismatch against a target measured on realisations the optimiser never touches.
    Nothing here is zero by construction: what is recovered is an estimate, its residual at the
    optimum is the noise it could not fit, and the two together are what an inverse problem on
    a noisy simulation actually looks like."""
    return 0.5 * jnp.sum(((readings(r, training_seeds) - independent) / scales) ** 2)


value_and_grad = jax.jit(jax.value_and_grad(loss))
inference_value_and_grad = jax.jit(jax.value_and_grad(inference))

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
# Three named tolerances, and the loop says which one stopped it. "Stalled" is not "converged":
# a backtracking line search that runs out of halvings has found no step that lowers the loss,
# which may be a minimum or may be a gradient that is no longer informative about the average.
# Every accepted point is evaluated and recorded, including the last, so that the point returned
# is the best one the optimiser actually stood on.
slope_tolerance = 1e-9                     # a projected gradient this small is a stationary point
step_tolerance = 1e-4                      # a move smaller than this is below the scan's resolution
objective_tolerance = 1e-6                 # a fall smaller than this, relative, is not progress


def descend(objective_and_gradient, iterations):
    """Bounded gradient descent with backtracking. Returns every point it stood on and why it
    stopped."""
    r, history, outcome = r_start, [], "ran out of iterations"
    step_size = 0.02 / max(abs(float(objective_and_gradient(r_start)[1])), 1e-12)
    print("%4s %8s %12s %12s %10s" % ("iter", "r", "loss", "d loss/d r", "step"))
    for iteration in range(iterations):
        objective, slope = objective_and_gradient(r)
        objective, slope = float(objective), float(slope)
        history.append((r, objective, slope))
        print("%4d %8.4f %12.6f %12.4f %10.2e" % (iteration, r, objective, slope, step_size))
        # the gradient projected onto the admissible interval: at a bound, a slope pushing
        # outwards is not a direction the optimiser may take, and its size says nothing
        projected = (slope if bounds[0] < r < bounds[1]
                     else min(slope, 0.0) if r <= bounds[0] else max(slope, 0.0))
        if abs(projected) < slope_tolerance:
            outcome = "converged: the projected gradient is below %.0e" % slope_tolerance
            break
        trial, value, accepted = r, objective, False
        for _ in range(14):                    # so that a gradient off in scale still works
            trial = float(np.clip(r - step_size * slope, *bounds))
            value = objective if trial == r else float(objective_and_gradient(trial)[0])
            if trial != r and value < objective:
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            outcome = "stalled: no step along the gradient lowers the loss, after 14 halvings"
            break
        moved, fell = abs(trial - r), (objective - value) / max(abs(objective), 1e-30)
        r, step_size = trial, step_size * 1.6
        if moved < step_tolerance:
            outcome = "converged: the step %.2e is below the %.0e the scan can resolve" % (
                moved, step_tolerance)
            break
        if fell < objective_tolerance:
            outcome = "converged: the loss fell by %.1e, below %.0e" % (fell, objective_tolerance)
            break
    objective, slope = objective_and_gradient(r)   # the point it ended on is a point it stood on
    history.append((r, float(objective), float(slope)))
    print("     %s" % outcome)
    return history, outcome


iterations = 8 if quick else 20
print("against the paired target, which is exactly zero at the reference by construction:")
history, outcome = descend(value_and_grad, iterations)
r_final = history[min(range(len(history)), key=lambda i: history[i][1])][0]
print("\nrecovered r = %.4f, reference %.4f, error %.4f" % (r_final, r_reference, abs(r_final - r_reference)))
print("That the error is this small is a property of the problem, not knowledge about r: the\n"
      "target was generated on these realisations and this protocol, so the minimum is at the\n"
      "reference by construction. What it tests is the differentiated chain, end to end.\n")

print("against the independent target, measured on %d realisations the optimiser never sees:"
      % len(target_seeds))
independent_history, independent_outcome = descend(inference_value_and_grad, iterations)
r_inferred = independent_history[min(range(len(independent_history)),
                                     key=lambda i: independent_history[i][1])][0]
print("\ninferred r = %.4f, reference %.4f, error %.4f, residual loss %.5f"
      % (r_inferred, r_reference, abs(r_inferred - r_reference),
         min(h[1] for h in independent_history)))
print("The residual is the noise the model could not fit, and it is not zero because nothing\n"
      "made it so. That number and the error bar below are the inference; the line above is the\n"
      "self-test.\n")

# --- 4. held-out validation, on realisations the optimiser never saw ------------------------------


def held_out_loss(r):
    """The same mismatch on realisations that were used neither to prepare the target nor to
    choose a step: does the recovered reflectivity reproduce what the reference gives on a
    plasma it has not seen?"""
    return float(0.5 * jnp.sum(((readings(r, held_out_seeds) - held_out_target) / scales) ** 2))


print("\nheld-out validation")
for name, value in (("start", r_start), ("recovered", r_final), ("inferred", r_inferred),
                    ("reference", r_reference)):
    print("  %-10s r = %.4f   training loss %10.5f   held-out loss %10.5f"
          % (name, value, float(value_and_grad(value)[0]), held_out_loss(value)))

# A scan of the held-out loss locates the minimum without a gradient at all. Three things are
# separate and were not: where the scan can put a minimum (its spacing), where the minimum
# actually is (a parabola through the three lowest points, which the spacing does not limit),
# and how far it moves between realisations (the only one of the three that is an uncertainty).
# The grid is anchored on the reference, so that a scan that does not contain the answer cannot
# report the distance to its nearest node as an error bar -- the earlier one had spacing 0.02
# and no node at 0.35, so its "uncertainty 0.01" was the grid, every time it ran.
spacing = (bounds[1] - bounds[0]) / (6 if quick else 24)
fine = np.unique(np.clip(r_reference + spacing * np.arange(-24, 25), *bounds))


def refined_minimum(curve, grid):
    """Where a parabola through the lowest sample and its two neighbours has its vertex, which
    is not restricted to the grid. At an end there is no parabola and the node is all there is."""
    i = int(np.argmin(curve))
    if i in (0, len(curve) - 1):
        return float(grid[i])
    left, middle, right = curve[i - 1:i + 2]
    curvature = left - 2 * middle + right
    if curvature <= 0:
        return float(grid[i])
    return float(grid[i] + 0.5 * (left - right) / curvature * (grid[i + 1] - grid[i]))


def one_seed_loss(r, seed):
    """The same mismatch on one held-out realisation, so that the scatter of the minimum over
    realisations can be measured instead of assumed."""
    return float(0.5 * jnp.sum(((measure_jit(r, prepared[seed]) - held_out_target) / scales) ** 2))


curves = np.array([[one_seed_loss(float(v), seed) for v in fine] for seed in held_out_seeds])
held_out_curve = curves.mean(axis=0)
held_out_best = refined_minimum(held_out_curve, fine)
per_seed = np.array([refined_minimum(row, fine) for row in curves])
scatter = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed)))
print("  held-out scan: %d points spaced %.4f, anchored so that the reference is one of them"
      % (len(fine), spacing))
print("  its minimum, refined off the grid, is at r = %.4f; the recovered value is %.4f from it "
      "and the inferred one %.4f"
      % (held_out_best, abs(r_final - held_out_best), abs(r_inferred - held_out_best)))
print("  per realisation the minimum sits at %s" % np.array2string(per_seed, precision=4))
print("  so the control is recovered as %.4f +- %.4f (standard error over %d realisations) against "
      "the reference %.4f" % (held_out_best, scatter, len(per_seed), r_reference))
print("  which is %.1f standard errors out" % (abs(held_out_best - r_reference) / max(scatter, 1e-12)))

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

# --- the record --------------------------------------------------------------------------------
# One name per variant: --oblique is a different experiment and would otherwise overwrite
# the record of the run it is meant to be compared with.
folder = Path.cwd() / ("sheath_optimization" + ("_oblique" if oblique else "")
                       + ("_quick" if quick else ""))
folder.mkdir(exist_ok=True)
settings = dict(electron_temperature=electron_temperature, density=density, mass_ratio=mass_ratio,
                beam_speed=beam_speed, box_debye_lengths=box_debye_lengths, cells=cells,
                capacity=capacity, emit=emit, preparation=preparation, window=window,
                window_over_omega_pe=window * omega_pe * dt, r_prepared=r_prepared,
                r_reference=r_reference, r_start=r_start, bounds=list(bounds),
                training_seeds=list(training_seeds), held_out_seeds=list(held_out_seeds),
                target_seeds=list(target_seeds), oblique=oblique, quick=quick)
summary = dict(self_test_recovered=r_final, self_test_outcome=outcome,
               inferred=r_inferred, inference_outcome=independent_outcome,
               inference_residual=float(min(h[1] for h in independent_history)),
               held_out_minimum=held_out_best, held_out_scatter=scatter,
               per_seed_minima=[float(v) for v in per_seed],
               reverse_forward_mismatch=abs(forward / reverse - 1),
               best_finite_difference=float(min(abs(d / reverse - 1) for d in differences)))
(folder / "run.json").write_text(json.dumps(provenance(example="sheath_optimization", settings=settings,
                                                       results=summary), indent=1))
np.savez(folder / "curves.npz", fine=fine, held_out_curve=held_out_curve, per_seed_curves=curves,
         history=np.array(history), independent_history=np.array(independent_history))
fig.savefig(folder / "figure.png", dpi=150)
print(f"\nwrote {folder}/run.json, curves.npz and figure.png")
plt.show()
