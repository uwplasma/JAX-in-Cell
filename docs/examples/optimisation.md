# Optimisation through the solver

`examples/optimisation.py`

An inverse problem with a known answer, solved by differentiating the entire
simulation.

```{figure} ../_static/figures/autodiff.png
:width: 100%
:alt: Gradient against finite differences, and gradient ascent finding the resonance

(a) The reverse-mode gradient against central differences. (b) Gradient ascent on the
seeded mode's amplitude.
```

## What it does

The objective is $\ln|E_{k=1}|$ at a fixed time, which is $\gamma t$ plus a constant
while the mode grows exponentially. Ascending it in the beam drift should find the
fastest-growing two-stream configuration, which for cold beams is
$kv_0/\omega_{pe} = \sqrt{3/8} = {{ autodiff_cold_optimum_k_v0_over_wpe }}$ and for
beams this warm is {{ autodiff_kinetic_optimum_k_v0_over_wpe }}.

Starting well off resonance, {{ autodiff_ascent_iterations }} steps of plain gradient
ascent reach {{ autodiff_ascent_k_v0_over_wpe }} —
{{ autodiff_ascent_deviation_percent }} per cent away. The script also checks the
gradient against a central difference: {{ autodiff_best_relative_error }}.

## Why the objective is what it is

Two obvious alternatives do not work, and the reasons are worth knowing before
building an objective of your own.

**The saturated field energy** is chaotic. Its gradient is a large, noisy number that
is the correct derivative of a function no optimiser can follow. Use a quantity from
the linear phase, or average over an ensemble with `vmap`.

**A fitted growth rate** uses a window chosen from the data — "from ten times the seed
to a tenth of saturation" — and those boundaries jump as the parameter changes, so the
objective is not smooth. Fixing the time instead makes it smooth, at the cost of having
to know in advance that the mode is still growing then.

More on both in {doc}`../user_guide/differentiation`.

## Things to try

* Optimise something else: the box length, the density, the temperature ratio, the
  external field profile. They are all pytree leaves.
* Replace the hand-rolled ascent with `optax`; nothing in the objective cares.
* Use `jax.vmap` over `seed` inside the objective to optimise an ensemble average.
