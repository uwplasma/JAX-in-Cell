# Performance

The whole time loop is one XLA program, so the cost per step is set by the particle
count and, weakly, by the grid.

```{figure} ../_static/figures/scaling.png
:width: 100%
:alt: Cost per particle per step against particle count, and cost per step against grid size

(a) Cost per particle per step for both integrators. It falls until the device has
enough work and is flat after that. (b) Cost per step against the number of cells at a
fixed particle count.
```

On {{ scaling_device }} ({{ scaling_platform }}, JAX {{ scaling_jax_version }}) the
explicit scheme costs {{ scaling_ns_per_particle_step }} ns per particle per step at
{{ scaling_particles_max }} particles, and the implicit scheme with eight Picard
iterations about {{ scaling_implicit_over_explicit }} times that. At
{{ scaling_particles_max }} particles and 1024 cells a step takes
{{ scaling_ms_per_step_1024_cells }} ms.

## Where the time goes

Each particle touches three cells, so a deposit is $3N$ scatter-adds and a gather $3N$
reads, both independent of the grid. A step does four deposits and two gathers. The
field update and the filter are a few operations per cell, which is why panel (b) is
nearly flat until the grid becomes comparable to the particle count.

The deposit is a scatter-add rather than a dense particle-by-cell weight matrix. The
dense form keeps everything a matrix product, which looks appealing on a GPU, but it
costs $O(N N_x)$ instead of $O(N)$; replacing it was worth about a factor of three at
64 cells and more as the grid grows.

## Compilation

The first call to `run` compiles, which takes a second or two. It is reused whenever
the static arguments are unchanged: `steps`, `store_every`, `store_particles`, the
particle and cell counts, the boundary types and every `Solver` switch. Physical
parameters are pytree leaves, so a scan over density, drift or temperature compiles
once:

```python
base = jax.jit(lambda s: s.run(1000, seed=0).E)
for density in densities:
    E = base(simulation.replace(species=(electrons.replace(density=density), ions)))
```

Watch out for the opposite: changing `steps` in a loop recompiles every iteration.

## Memory

The particle history is almost always the binding constraint, not speed:

```{math}
\text{bytes} = \frac{\text{steps}}{\text{store\_every}} \times N \times 3 \times 8 \times 2 .
```

Six thousand steps of sixty thousand particles is 17 GB. Use `store_every` to thin the
history, `store_particles=False` when only the fields are needed, or `state` to run in
chunks ({doc}`running`). A long field-only run costs almost nothing to store: the same
six thousand steps of a 128-cell grid is under a hundred megabytes.

## GPUs and TPUs

Nothing in the package is CPU-specific; install a JAX build for the accelerator and
the same program runs on it. The gain is largest where the particle count is large
enough to fill the device — panel (a) shows the cost per particle still falling at
{{ scaling_particles_max }} particles on a CPU, and an accelerator moves that knee
much further out. Double precision is enabled at import
(`jax_enable_x64`), which matters for the conservation properties and costs a factor
of two on hardware optimised for single precision; a run that does not need it can
override the flag before importing.

## Practical advice

* Measure with `block_until_ready()`; JAX is asynchronous and a timing without it
  measures dispatch, not work.
* Discard the first call from any timing.
* Prefer one long `run` to many short ones: the `lax.scan` has no per-step Python
  overhead, but each `run` call does have a dispatch cost.
* Reach for the implicit scheme when it lets you take a step more than
  {{ scaling_implicit_over_explicit }} times larger; otherwise the explicit one is
  cheaper for the same physics.
