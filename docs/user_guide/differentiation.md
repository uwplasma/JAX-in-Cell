# Gradients

A {class}`~jaxincell.Simulation` is a JAX pytree whose physical parameters are leaves, so
`jax.grad` differentiates through the whole run: the initial sampling, the deposition,
the field solve, the Boris rotation, the boundary conditions and the time loop. There is
no adjoint to write and no finite differences anywhere.

```python
import jax, jax.numpy as jnp

def objective(drift):
    electrons = base.replace(drift=(drift, 0.0, 0.0))
    out = simulation.replace(species=(electrons, ions)).run(200, seed=0, store_particles=False)
    return jnp.mean(out.E[:, :, 0] ** 2)

gradient = jax.grad(objective)(6e7)
```

## What can be differentiated

| kind | fields |
|---|---|
| differentiable (pytree leaves) | `length`, `dt_over_dx_c`, `restitution`, the species `charge`, `mass`, `density`, `vth`, `drift`, `perturbation_amplitude` and `reflection` (when it is a number), `filter_alpha`, and the external field arrays |
| static, so not differentiable | particle counts, cell counts, boundary types, `algorithm` |

Differentiating with respect to the whole object works too, and returns a matching
pytree:

```python
grads = jax.grad(lambda s: jnp.sum(s.run(100, seed=0).E ** 2))(simulation)
print(grads.domain.length, grads.species[0].density)
```

Reverse mode works through the implicit scheme as well as the explicit one, because the
Picard iteration is a `lax.scan` of fixed length rather than a `lax.while_loop`, which
has no reverse-mode rule — hence `picard_iterations` is a count, not a tolerance.

## Forward or reverse

| mode | call | cost |
|---|---|---|
| reverse | `jax.grad` | one backward pass gives the derivative with respect to every parameter at once, at the price of keeping what that pass needs from every step of the run |
| forward | `jax.jvp`, `jax.jacfwd` | one pass per parameter, and nothing kept; memory stays flat as the number of steps grows |

```python
value, derivative = jax.jvp(objective, (6e7,), (1.0,))
```

For the handful of parameters a physics optimisation usually has, forward mode is the
better tool. On the run in the figure, a forward pass takes
{{ autodiff_forward_time_warm_s }} s and a reverse one {{ autodiff_grad_time_warm_s }} s,
against {{ autodiff_run_time_warm_s }} s for the run alone, and the two derivatives agree
to {{ autodiff_forward_reverse_agreement }}.

## Accuracy

```{figure} ../_static/figures/autodiff.png
:width: 100%
:alt: Relative mismatch between finite differences and the reverse-mode gradient against step size, and gradient ascent on the beam drift

(a) The relative mismatch $|\mathrm{FD}/\mathrm{AD} - 1|$ against the central-difference
step $h$ in m/s, on log-log axes: a V whose left arm is dominated by round-off and whose
right arm is dominated by truncation. The floor of the V is
{{ autodiff_best_relative_error }} at $h = ${{ autodiff_best_step }} m/s. (b) The
objective $\ln|E_{k=1}|$ at a fixed time, scanned over the beam drift (grey line), with
the {{ autodiff_ascent_iterations }} gradient-ascent iterates on it (blue circles); the
black dashed vertical line marks the fastest-growing kinetic mode.
```

* The gradient is exact to floating point, so the comparison is really a test of the
  finite difference.
* The first call costs {{ autodiff_grad_time_first_s }} s including compilation and
  {{ autodiff_grad_time_warm_s }} s afterwards.

## An inverse problem with a known answer

Panel (b) checks that the gradient is not merely self-consistent but points somewhere
useful.

* Two cold counter-streaming beams are most unstable at
  $kv_0/\omega_{pe} = \sqrt{3/8} = ${{ autodiff_cold_optimum_k_v0_over_wpe }}, moving to
  {{ autodiff_kinetic_optimum_k_v0_over_wpe }} for beams this warm.
* Starting well off resonance, {{ autodiff_ascent_iterations }} steps of plain gradient
  ascent reach {{ autodiff_ascent_k_v0_over_wpe }}, within
  {{ autodiff_ascent_deviation_percent }} % of the kinetic optimum.
* The full script is `examples/3_advanced/optimize_two_stream.py`.

## Choosing an objective

* **Chaos.** A quantity measured after saturation — the saturated field energy, a
  late-time temperature — depends on the parameters through a chaotic trajectory, and its
  gradient is a large, noisy number that is the correct derivative of a function no
  optimiser can follow. Prefer an objective from the linear phase, or an ensemble
  average.
* **A data-dependent window.** Fitting a growth rate between "ten times the seed" and "a
  tenth of saturation" is the right way to *measure* a rate, but the window boundaries
  jump as the parameter changes and the objective is not smooth. The objective in the
  figure is instead $\ln|E_k|$ at a **fixed** time, which is $\gamma t$ plus a constant
  while the mode grows and is smooth in the drift.
* `grad` combined with a `vmap` over seeds ({doc}`running`) gives the gradient of an
  ensemble average, the practical way to optimise through a noisy simulation.

## Which derivative, and over how long

Four things get called "the derivative of the simulation", and they are not the same:

1. the derivative of a **fixed discretisation and a fixed realisation** — the number
   `jax.grad` returns;
2. the derivative of a **finite-time expectation** of an observable, which a finite
   number of particles estimates;
3. a **continuum** response, the limit of refining the discretisation;
4. a **long-time stationary** response.

Forward mode agreeing with reverse mode checks the first against itself, and so does a
finite difference of the same run. Neither says anything about the others, and the
difference between them is not small.

Where a wall absorbs particles this becomes concrete. Every absorption is a branch:
change a parameter enough to move one particle across the wall that did not cross
before, and the objective takes a small step. The gradient is exactly the slope between
those steps, and it is correct. Whether it is *useful* depends on how many branches a
realistic change flips, which grows with the length of the run. Measured on the sheath of
{doc}`../examples/sheath_optimization`, differentiating with respect to a collector's
electron reflectivity:

| horizon | forward against reverse | central difference agrees to | at step |
|---|---|---|---|
| 5 steps | $8\times10^{-15}$ | $1.5\times10^{-9}$ | $10^{-3}$ |
| 25 steps | $7\times10^{-16}$ | $1.1\times10^{-9}$ | $10^{-5}$ |
| 100 steps | $3\times10^{-14}$ | $1.9\times10^{-7}$ | $10^{-7}$ |

The implementation is exact at every horizon; what falls is the step over which the
objective looks smooth. Beyond a few hundred steps the derivative of one realisation
grows to tens of times the response of the average and changes sign from run to run: it
is still the derivative of the program, and no longer an estimate of the physical
response. Sensitivities that do not follow the plasma particles {cite}`chung2020` are a
different method, not a tolerance to be loosened.

The practical consequences, all of which the sheath example follows:

* **Keep the differentiated window short.** Prepare the state you want to perturb outside
  the differentiated calculation — it is then genuinely independent of the control — and
  differentiate only the response.
* **Average the measurement over realisations first**, and take the loss of that mean.
  The loss of the mean and the mean of the losses are different objectives.
* **Keep counts out of it.** A functional that counts particles, or bins them sharply,
  has a branchwise derivative of exactly zero almost everywhere, whatever its expectation
  does. Hence a {class}`~jaxincell.Source` emits a fixed number of particles with a
  continuous weight rather than a flux-dependent number of them, and the reflection at a
  wall is a fraction of each particle's weight rather than a hit-or-miss trial.
  `tests/test_gradients.py` has the counting functional as a negative control, with the
  zero it correctly returns.
* **Say what the measurement can resolve.** A response smaller than the scatter between
  realisations is not identifiable however good the gradient is; the scan the sheath
  example prints before optimising is there to say so.

## What each kind of derivative is checked against

| observable | what is claimed | test in `tests/` |
|---|---|---|
| a smooth sensor at a fixed time, one realisation | forward equals reverse; a small central difference agrees | `test_gradients.py::test_forward_and_reverse_mode_agree_through_a_maintained_sheath`, `::test_the_gradient_is_the_derivative_of_the_discrete_map_at_a_small_enough_step`, `test_api.py::test_gradient_matches_central_finite_difference` |
| the same sensor averaged over realisations | the gradient of the mean agrees with differences of the mean at steps 4 and 16 times larger, within the scatter between realisations | `test_gradients.py::test_the_mean_over_realisations_has_the_derivative_of_the_mean` |
| one particle crossing a wall | the derivative at the crossing, with the $d\tau/d\theta$ term, against the trajectory's own algebra | `test_gradients.py::test_the_impact_energy_a_prescribed_field_gives_and_its_derivative` |
| a hard count or a sharp bin | none: the branchwise derivative is zero and is not the expected flux | `test_gradients.py::test_a_functional_that_counts_particles_has_no_branchwise_derivative` |
| a long-time stationary average | none: the horizon table above | |
| continuum limit | none at a fixed particle count and grid | |

## Where the program branches

Each of these makes the run piecewise smooth in its parameters. The gradient is the derivative
of the branch the run took; the jumps between branches are not in it, and averaging over
realisations does not put them back.

| branch | derivative | test in `tests/` |
|---|---|---|
| a source's quantile inversion (a fixed number of bisections) | exact, by implicit differentiation of the inverted quantile | `test_sources.py::test_the_quantile_of_the_drifting_flux_is_inverted_and_differentiated_exactly` |
| the slots a source refills (`top_k` of the weights) | a relabelling of particles, which weighted moments do not see; a sheath fed through it is differentiable in its reservoir | `test_sources.py::test_a_sheath_fed_every_k_steps_is_differentiable_in_the_reservoir` |
| the sort, pairing and scattering angle of collisions | the derivative with the realised pairing; a small difference agrees | `test_gradients.py::test_collisions_have_the_derivative_of_the_realised_pairing` |
| identical velocities in a collision | finite, where the scattering frame is undefined | `test_collisions.py::test_gradients_through_collisions_stay_finite_for_identical_velocities` |
| a wall's weight floor, `Source.min_weight` | the truncated weight is reported, not differentiated; `Wall.truncated` bounds what it leaves out | `test_sources.py::test_the_cutoff_budget_falls_with_the_cutoff` |
| a particle crossing a wall inside a step | the crossing time is differentiated | `test_gradients.py::test_the_impact_energy_a_prescribed_field_gives_and_its_derivative` |

The collision operator has no accept-or-reject step: every pair scatters, by an angle whose
variance is the only parameter-dependent part.
