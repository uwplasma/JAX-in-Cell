# Optimising a simulation

`examples/optimize_two_stream_saturation.py` searches for the ion temperature that
minimises the saturated electrostatic energy of the two-stream instability. It shows
the pattern for any gradient-based or derivative-free optimisation over simulation
inputs.

```{literalinclude} ../../examples/optimize_two_stream_saturation.py
:language: python
```

The objective is the mean of the field energy over the last 800 steps, a smooth-enough
function of $T_i/T_e$ for optimisation. Three things happen in the script:

1. one run at the initial guess, to plot the energy history and the value that will
   be minimised;
2. a scan over ten values of $T_i/T_e$ on a logarithmic grid, to see the landscape;
3. an optimisation with `scipy.optimize.least_squares` starting from $T_i/T_e = 3$.

The objective is wrapped in `jax.jit` and its gradient in `jit(grad(...))`; the
commented block at the end shows the same optimisation with Optax's Adam optimiser
using that gradient. All runs reuse one compiled program, because the ion temperature
is a differentiable input.
