# Gradients

A {class}`~jaxincell.Simulation` is a JAX pytree whose physical parameters are leaves,
so `jax.grad` differentiates through the whole run: the initial sampling, the
deposition, the field solve, the Boris rotation, the boundary conditions and the time
loop. There is no adjoint to write and no finite differences anywhere.

```python
import jax, jax.numpy as jnp

def objective(drift):
    electrons = base.replace(drift=(drift, 0.0, 0.0))
    out = simulation.replace(species=(electrons, ions)).run(200, seed=0, store_particles=False)
    return jnp.mean(out.E[:, :, 0] ** 2)

gradient = jax.grad(objective)(6e7)
```

## What can be differentiated

Every pytree leaf: `length`, `dt_over_dx_c`, `restitution`, the species `charge`,
`mass`, `density`, `vth`, `drift`, `perturbation_amplitude`, `filter_alpha`, and the
external field arrays. Differentiating with respect to the whole object works too, and
returns a matching pytree:

```python
grads = jax.grad(lambda s: jnp.sum(s.run(100, seed=0).E ** 2))(simulation)
print(grads.domain.length, grads.species[0].density)
```

Static fields — particle counts, cell counts, boundary types, `algorithm` — are not
differentiable, which is what "static" means.

## Both integrators

Reverse-mode works through the implicit scheme as well as the explicit one, because
the Picard iteration is a `lax.scan` of fixed length rather than a `lax.while_loop`,
which has no reverse-mode rule. That is the reason `picard_iterations` is a count and
not a tolerance.

## Accuracy

```{figure} ../_static/figures/autodiff.png
:width: 100%
:alt: Reverse-mode gradient against central finite differences, and gradient ascent on the growth rate

(a) The reverse-mode gradient against central differences over five step sizes. The
best agreement is {{ autodiff_best_relative_error }} at
$h = ${{ autodiff_best_step }} m/s; larger steps are limited by truncation, smaller
ones by round-off. (b) Gradient ascent on the amplitude the seeded two-stream mode
reaches by a fixed time.
```

The gradient is exact to floating point, so the comparison is really a test of the
finite difference. The first call costs {{ autodiff_grad_time_first_s }} s including
compilation and {{ autodiff_grad_time_warm_s }} s afterwards — about twice a forward
run, the usual reverse-mode ratio.

## An inverse problem with a known answer

Panel (b) is a check that the gradient is not merely self-consistent but points
somewhere useful. Two cold counter-streaming beams are most unstable at
$kv_0/\omega_{pe} = \sqrt{3/8} = ${{ autodiff_cold_optimum_k_v0_over_wpe }}, moving to
{{ autodiff_kinetic_optimum_k_v0_over_wpe }} for beams this warm. Starting well off
resonance, {{ autodiff_ascent_iterations }} steps of plain gradient ascent reach
{{ autodiff_ascent_k_v0_over_wpe }}, within
{{ autodiff_ascent_deviation_percent }} % of the kinetic optimum. The full script is
`examples/optimisation.py`.

## Choosing an objective

The one real pitfall is chaos. A quantity measured after the instability has saturated
— the saturated field energy, a late-time temperature — depends on the parameters
through a trajectory that has become chaotic, and its gradient is a large, noisy number
that happens to be the correct derivative of a function no optimiser can follow. Prefer
an objective from the linear phase, or one averaged over an ensemble.

The second pitfall is a data-dependent window. Fitting a growth rate between "ten times
the seed" and "a tenth of saturation" is the right way to *measure* a rate, but the
window boundaries jump as the parameter changes, and the objective is not smooth. The
objective in the figure above is instead $\ln|E_k|$ at a **fixed** time, which is
$\gamma t$ plus a constant while the mode grows and is a smooth function of the drift.

## vmap and ensembles

`seed` is traced, so an ensemble is one `vmap` and one compilation:

```python
fields = jax.vmap(lambda s: simulation.run(300, seed=s).E[-1, :, 0])(jnp.arange(32))
```

Combining `vmap` with `grad` gives the gradient of an ensemble average, which is the
practical way to optimise through a noisy simulation.
