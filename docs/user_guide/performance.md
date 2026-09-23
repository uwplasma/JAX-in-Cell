# Performance

The whole time loop is one XLA program, so the cost per step is set by the particle count
and, weakly, by the grid.

```{figure} ../_static/figures/scaling.png
:width: 100%
:alt: Cost per particle per step against particle count, and cost per step against grid size

(a) Cost per particle per step against the number of pseudo-particles, for both
integrators: it falls until the device has enough work and is flat after that. (b) Cost per
step against the number of cells, from 32 to 1024, at 40 000 particles.
```

Measured on {{ scaling_device }} ({{ scaling_platform }}, JAX {{ scaling_jax_version }}),
over {{ scaling_steps }} steps:

| | |
|---|---|
| explicit, at {{ scaling_particles_max }} particles | {{ scaling_ns_per_particle_step }} ns per particle per step |
| implicit, 8 Picard iterations | {{ scaling_ns_per_particle_step_implicit }} ns per particle per step |
| implicit over explicit | {{ scaling_implicit_over_explicit }}× |
| at {{ scaling_particles_max }} particles and 1024 cells | {{ scaling_ms_per_step_1024_cells }} ms per step |

## Where the time goes

Each particle touches three cells, so a deposit is $3N$ scatter-adds and a gather $3N$
reads, both independent of the grid. An explicit step does six deposits — the charge and
the two transverse currents for each half step — and one gather of $\mathbf E$ and
$\mathbf B$ together. The field update and the filter are a few operations per cell, which
is why panel (b) is nearly flat until the grid becomes comparable to the particle count.

* The deposit is a **scatter-add**, not a dense particle-by-cell weight matrix. The dense
  form keeps everything a matrix product, which looks appealing on a GPU, but costs
  $O(N N_x)$ instead of $O(N)$; replacing it was worth about a factor of three at 64 cells
  and more as the grid grows.
* The indices are always inside the grid, so the deposit and the gather could skip JAX's
  bounds handling with `mode="promise_in_bounds"`. Timed over whole steps on an idle RTX
  A4000 that changed nothing by more than 1 %, so they keep the default.

## Compilation

The first call to `run` compiles, and the program is reused while the static arguments are
unchanged ({doc}`running`), so a scan over density, drift or temperature compiles once:

```python
base = jax.jit(lambda s: s.run(1000, seed=0).E)
for density in densities:
    E = base(simulation.replace(species=(electrons.replace(density=density), ions)))
```

Watch out for the opposite: changing `steps` in a loop recompiles every iteration.

Memory, not speed, is usually the binding constraint; {doc}`running` gives the size of the
particle history and the options that bound it.

## GPUs and TPUs

Nothing in the package is CPU-specific; install a JAX build for the accelerator and the
same program runs on it. The gain is largest where the particle count is large enough to
fill the device — panel (a) of the figure above shows the cost per particle still falling
at {{ scaling_particles_max }} particles on a CPU, and an accelerator moves that knee much
further out.

```{figure} ../_static/figures/runtime_resolution.png
:width: 100%
:alt: Runtime of a two-stream run against particle count on a CPU and a GPU, and the growth rate against drift for three particle counts

(a) Wall-clock time of the quiet two-stream run of {doc}`../examples/two_stream`,
{{ runtime_steps }} steps on {{ two_stream_cells }} cells with half as many ions as
pseudo-electrons, against the number of pseudo-electrons: on the CPU of an
{{ runtime_cpu_device }} (JAX {{ runtime_cpu_jax_version }}) and on one
{{ runtime_gpu_device }} (JAX {{ runtime_gpu_jax_version }}), both in double precision,
compilation excluded, the best of five runs. (b) The growth rate of the seeded mode
against $kv_0/\omega_{pe}$ with {{ resolution_counts }} pseudo-electrons, fitted as in
{doc}`../examples/two_stream`, against the kinetic root.
```

| {{ runtime_counts_largest }} pseudo-electrons, {{ runtime_steps }} steps | |
|---|---|
| CPU, {{ runtime_cpu_device }} | {{ runtime_cpu_seconds_largest }} s |
| GPU, {{ runtime_gpu_device }} | {{ runtime_gpu_seconds_largest }} s |
| GPU over CPU | {{ runtime_gpu_speedup_largest }}× faster |

Both machines were shared when they were timed (load averages {{ runtime_cpu_load }} on
the laptop and {{ runtime_gpu_load }} on the host of the GPU), so the CPU curve is
noisier than an idle machine would give. Panel (b) is the price of fewer particles: the
growth rate is off the kinetic root by {{ resolution_mean_deviation_percent_1000 }} % on
average with 1000 pseudo-electrons, {{ resolution_mean_deviation_percent_4000 }} % with
4000 and {{ resolution_mean_deviation_percent_16000 }} % with 16000, and most at the
small drifts, where the rate is lowest. `docs/scripts/fig_runtime.py` draws the figure; run
on a GPU it records that device's timings, and on a CPU it times the CPU and reads them.

Measure before choosing single precision ({doc}`units`) for speed. On a CPU the two cost
about the same, and on the NVIDIA RTX A4000 we tested, with JAX 0.10.2, a single-precision
run was many times slower than a double-precision one.

## Apple silicon

Apple's plugin for its GPUs, `jax-metal`, is **not supported**. Its latest release, 0.1.1,
requires `jax==0.4.34` and has no double precision. On an M3 Max it:

* returns NaN from the second step of any run past about sixteen thousand particles with
  the history stored;
* never returns from larger ones;
* crashes the process on reverse-mode derivatives and on the loop the implicit integrator
  runs over its sub-steps;
* is little faster than the CPU where it does compute correctly.

Run on the CPU, or on a CUDA GPU.

## Practical advice

* Measure with `block_until_ready()`; JAX is asynchronous, and a timing without it
  measures dispatch, not work.
* Discard the first call from any timing.
* Prefer one long `run` to many short ones: the `lax.scan` has no per-step Python
  overhead, but each `run` call does have a dispatch cost.
* Reach for the implicit scheme when it lets you take a step more than
  {{ scaling_implicit_over_explicit }} times larger; otherwise the explicit one is cheaper
  for the same physics.
