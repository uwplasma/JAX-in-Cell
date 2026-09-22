# Optimisation through the solver

An inverse problem with a known answer, solved by differentiating the entire simulation.
The objective is $\ln|E_{k=1}|$ at a fixed time, which is $\gamma t$ plus a constant while
the mode grows exponentially, so ascending it in the beam drift should find the
fastest-growing two-stream configuration.

```{figure} ../_static/figures/autodiff.png
:width: 100%
:alt: Gradient against finite differences, and gradient ascent finding the resonance

(a) The relative mismatch between a central difference and the reverse-mode gradient,
scanned over the step $h$: round-off dominates on the left, truncation on the right.
(b) The objective scanned over the beam drift (grey) with the {{ autodiff_ascent_iterations }}
ascent iterates on it (circles); the dashed line is the fastest-growing kinetic mode.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| optimum $kv_0/\omega_{pe}$ after {{ autodiff_ascent_iterations }} ascent steps | {{ autodiff_ascent_k_v0_over_wpe }} | {{ autodiff_kinetic_optimum_k_v0_over_wpe }} (kinetic root, beams this warm) | {{ autodiff_ascent_deviation_percent }} % |
| cold-beam optimum, for reference | — | $\sqrt{3/8} = {{ autodiff_cold_optimum_k_v0_over_wpe }}$ | — |
| reverse-mode gradient | {{ autodiff_gradient }} | central difference, best at $h = {{ autodiff_best_step }}$ m/s | {{ autodiff_best_relative_error }} |
| forward against reverse mode | — | each other | {{ autodiff_forward_reverse_agreement }} |

Nothing is finite-differenced to compute the gradient: `jax.grad` runs back through the
field solve, the deposit, the gather and the Boris push. The finite differences check it.

## What the gradient costs

| | seconds |
|---|---|
| a plain run, warm | {{ autodiff_run_time_warm_s }} |
| the objective, warm | {{ autodiff_forward_time_warm_s }} |
| the gradient, first call (compiling) | {{ autodiff_grad_time_first_s }} |
| the gradient, warm | {{ autodiff_grad_time_warm_s }} |

## Why the objective is what it is

Two obvious alternatives do not work:

* **The saturated field energy** is chaotic. Its gradient is a large, noisy number that is
  the correct derivative of a function no optimiser can follow. Use a quantity from the
  linear phase, or average over an ensemble with `vmap`.
* **A fitted growth rate** uses a window chosen from the data, and those boundaries jump
  as the parameter changes, so the objective is not smooth. Fixing the time instead makes
  it smooth, at the cost of having to know in advance that the mode is still growing then.

More on both in {doc}`../user_guide/differentiation`.

## How to run

```bash
python examples/3_advanced/optimize_two_stream.py
```

The figure and the numbers come from `docs/scripts/fig_autodiff.py`, which runs this setup
and adds a forward-mode check and a scan of the finite-difference step.

## Things to try

* Optimise something else: the box length, the density, the temperature ratio, the
  external field profile. They are all pytree leaves.
* Replace the hand-rolled ascent with `optax`; nothing in the objective cares.
* Use `jax.vmap` over `seed` inside the objective to optimise an ensemble average.
