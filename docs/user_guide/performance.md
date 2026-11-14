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
the two transverse currents for each half step, which an accelerator does as two scatters
of three columns — and one gather of $\mathbf E$ and $\mathbf B$ together. The field update and the filter are a few operations per cell, which
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

The host was shared when it was timed (load averages {{ runtime_cpu_load }} during the CPU
runs and {{ runtime_gpu_load }} during the GPU ones), so the CPU curve is slower and noisier
than an idle machine would give; the CPU runs were pinned to {{ runtime_cpu_cores }} cores. Panel (b) is the price of fewer particles: the
growth rate is off the kinetic root by {{ resolution_mean_deviation_percent_1000 }} % on
average with 1000 pseudo-electrons, {{ resolution_mean_deviation_percent_4000 }} % with
4000 and {{ resolution_mean_deviation_percent_16000 }} % with 16000, and most at the
small drifts, where the rate is lowest. `docs/scripts/fig_runtime.py` draws the figure; run
on a GPU it records that device's timings, and on a CPU it times the CPU and reads them.

## Devices and precision

The same two-stream run with {{ device_particles }} pseudo-electrons, {{ runtime_steps }} steps,
in both precisions on both devices (`docs/scripts/device_matrix.py`, warm, best of three):

| | seconds | growth rate / $\omega_{pe}$ | energy error |
|---|---|---|---|
| CPU, float64 | {{ device_cpu_f64_seconds }} | {{ device_cpu_f64_gamma }} | {{ device_cpu_f64_energy_error }} |
| CPU, float32 | {{ device_cpu_f32_seconds }} | {{ device_cpu_f32_gamma }} | {{ device_cpu_f32_energy_error }} |
| GPU, float64 | {{ device_gpu_f64_seconds }} | {{ device_gpu_f64_gamma }} | {{ device_gpu_f64_energy_error }} |
| GPU, float32 | {{ device_gpu_f32_seconds }} | {{ device_gpu_f32_gamma }} | {{ device_gpu_f32_energy_error }} |

The kinetic root is {{ device_theory_gamma }}. Single precision gives the same rate and energy
error here. On the CPU it was 1.7 times faster (a shared host, load {{ device_cpu_load }}); on
the {{ device_gpu_name }} it is about a hundred times slower. The deposit's
float32 scatter-add is the likely cause; it has not been profiled. Measure before choosing
single precision ({doc}`units`). On the RTX A4000 of the {ref}`profile record <profile-record>` below single precision is not slower: the explicit
forward run of 131072 electrons takes 60.9 ms in float32 against 107.9 ms in float64 before the deposit change, and 14.7 ms
against 50.3 ms after it.

The test suite runs in double precision on both devices. On the GPU a scatter-add sums in an
order that changes from call to call, so two identical runs agree to round-off rather than bit
for bit, and the tests that assert bit-identity (a run split by the progress meter or by a
restart against the run taken whole) hold on a CPU only.

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

## Reproducible local profiles

`docs/scripts/profile_pic.py` measures the actual compiled simulation, including
relativistic collision and radiation variants. It separates compilation, first
execution and synchronized warm executions (five by default); records compiler buffer estimates;
and optionally captures a JAX profiler trace. Results and HLO text go to
`PROFILE_DIR`, which must be outside the package checkout:

```bash
PROFILE_DIR=/tmp/jic-explicit python docs/scripts/profile_pic.py
PROFILE_CASE=implicit PROFILE_MODE=reverse PROFILE_CHECKPOINT=1 \
  PROFILE_DIR=/tmp/jic-implicit-grad python docs/scripts/profile_pic.py
PROFILE_CASE=collisions PROFILE_TRACE=1 PROFILE_DIR=/tmp/jic-collisions \
  python docs/scripts/profile_pic.py
```

Forward mode returns the complete output, including the work ledger and field
history, without particle histories. Reverse mode and JVP differentiate a scalar
final-field objective; JAX may discard diagnostics unused by that objective.

Use separate processes for the parent revision and this branch, with the same
`PROFILE_N`, `PROFILE_CELLS`, `PROFILE_STEPS`, precision and backend.
`PROFILE_REPEATS` sets the number of warm measurements, and
`PROFILE_EXPECT_BACKEND=cpu` or `gpu` rejects an unintended backend before timing.
Compiler temporary-buffer
estimates are not measured peak resident memory. `PROFILE_COMPILE_ONLY=1` permits
checking those estimates before attempting a gradient that may exceed device memory.
Keep reports and traces local; commit the script and a deliberately selected,
compressed documentation figure only when it communicates a useful result.

For optimization starts or parameter ensembles, `PROFILE_SHARDS=k`
distributes four simulations per device by default and checks the result against
the same batch on one device. Set the same `PROFILE_BATCH` for one and several
devices when measuring strong scaling; the batch must divide evenly across devices.
The profile uses common random numbers across its parameter cases. On a CPU, a small correctness rehearsal can use
`XLA_FLAGS=--xla_force_host_platform_device_count=2 PROFILE_SHARDS=2`; virtual CPU
devices do not establish accelerator scaling. This is ensemble sharding. A single
PIC domain needs explicit ownership and communication of particles, current sums
and collision-cell pairs before it can be distributed safely.

### Where custom kernels would help

Profile the complete forward and differentiated program before changing kernels.
Candidate accelerator work is fused field gathering and particle pushing,
collision sorting/pairing, and deposition with many particles competing for one
cell. Elementwise radiation arithmetic alone may already fuse efficiently under
XLA. A Pallas implementation earns its additional code only with measured benefit
on the target hardware, matching force/current conservation tests and validated
JVP/VJP rules. Keep the pure JAX implementation as the portable reference until
those tests pass. See the upstream [profiling guide](https://docs.jax.dev/en/latest/profiling.html)
and [Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html).

(profile-record)=
## Profile record

Measured on 2026-11-09 with `docs/scripts/profile_pic.py` and `docs/scripts/profile_workloads.py`,
JAX 0.12.0.1, Python 3.12. The GPU is one RTX A4000 (CUDA 13, driver 580) of a shared office host;
the CPU rows are one pinned core of its x86 CPU, or an Apple laptop core. Every row is a fresh
process; before/after rows alternate between a clean checkout of the parent
(`relativistic-radiative-control`) and of this branch, three rounds each. Tables give the median of the
round medians, and the smallest and largest in parentheses. Particle counts are electrons, with half as
many ions; 32 steps, double precision unless stated. Compiler temporary bytes are not peak memory.

### Where the time goes

Device traces of the parent, 131072 electrons, 256 cells, one forward run of 32 steps (kernel time is the
sum over the one GPU stream):

| case | kernel time | what it is made of |
|---|---|---|
| explicit | 114 ms | deposit scatter-adds 52 %; the rest is many small fused kernels |
| implicit | 1795 ms | scatter-adds 37 % (the deposit, and the transpose of the field gather); copies 10 % |
| radiation | 341 ms | two fused particle reductions, the Landau-Lifshitz arithmetic, 63 %; scatter-adds 12 % |
| collisions | 194 ms in a 466 ms span | 21 000 launches, so the device idles; a scatter is the largest kernel (13 %) |

The scatter-add into 64 to 256 cells, which hundreds of thousands of particles share, is the cost on a
GPU. The radiation arithmetic is compute-bound in double precision and was left alone; the collision loop
is launch-bound.

### What changed

* The charge and the two transverse currents of a half step share one stencil, so an accelerator
  deposits them in one scatter of three columns instead of three (a CPU keeps three, which measured faster there).
* Above 8192 particles an accelerator gives each particle one of 32 private copies of the grid and sums
  them, which removes most of the atomic contention. The gather reads from the same copies, so that its
  transpose, which is a scatter-add as well and which the implicit step takes every Picard iteration and
  reverse mode takes every step, is split in the same way. A CPU has no contention to remove and pays
  for the copies, so it keeps the single-copy code and its timings are unchanged.
* The contact-time refinement of a wall run with a prescribed drive runs for every particle in the box, so
  its `cond` only cost; it is called directly there. Under JAX 0.6.2, the last release for Python 3.10, the
  reverse pass of that `cond` returned NaN for the time-dependent wall-impact gradient of
  `tests/test_external_xyz.py`; the full suite except `test_api.py` passes under it now (549 passed, 1 skipped).

The numerical result does not change beyond the order of a sum: final-field energies agree to 1e-12
relative, AD agrees with JVP to 1e-12 and the derivative of the objective is the same to 1e-12
in every row below. In case names, `nN` is N electrons and `xN` is N cells.

Forward, GPU, median of three rounds (spread in parentheses), milliseconds:

| case | before ms | after ms | after/before |
|---|---|---|---|
| collisions-n2048-x64 | 45.0 (43.6-46.6) | 44.3 (43.7-46.1) | 0.98 |
| collisions-n2048-x256 | 46.1 (45.3-46.2) | 45.1 (44.7-50.6) | 0.98 |
| collisions-n16384-x64 | 62.1 (61.8-62.2) | 57.6 (56.8-58.5) | 0.93 |
| collisions-n16384-x256 | 60.3 (59.8-62.1) | 58.1 (57.7-63.1) | 0.96 |
| collisions-n131072-x64 | 226.0 (224.7-226.2) | 166.1 (166.0-166.3) | 0.73 |
| collisions-n131072-x256 | 198.5 (198.5-198.6) | 165.6 (165.5-165.7) | 0.83 |
| explicit-n2048-x64 | 5.4 (5.4-5.7) | 4.9 (4.9-5.0) | 0.91 |
| explicit-n2048-x256 | 5.5 (5.4-5.7) | 4.9 (4.9-5.2) | 0.90 |
| explicit-n16384-x64 | 21.6 (21.0-21.7) | 13.2 (13.1-13.5) | 0.61 |
| explicit-n16384-x256 | 17.7 (17.7-17.8) | 13.5 (13.3-13.5) | 0.76 |
| explicit-n131072-x64 | 107.9 (107.8-108.7) | 50.3 (50.1-50.3) | 0.47 |
| explicit-n131072-x256 | 82.0 (81.4-82.4) | 50.4 (50.2-50.5) | 0.61 |
| implicit-n2048-x64 | 56.5 (56.4-56.6) | 56.5 (56.4-56.6) | 1.00 |
| implicit-n2048-x256 | 57.7 (57.6-57.8) | 57.7 (57.4-58.3) | 1.00 |
| implicit-n16384-x64 | 173.2 (173.0-174.0) | 148.3 (148.3-148.3) | 0.86 |
| implicit-n16384-x256 | 146.7 (146.6-147.3) | 143.8 (142.9-143.9) | 0.98 |
| implicit-n131072-x64 | 1550 (1543-1552) | 916.3 (916.2-916.3) | 0.59 |
| implicit-n131072-x256 | 1226 (1226-1227) | 907.3 (907.2-907.4) | 0.74 |
| radiation-n2048-x64 | 8.5 (8.2-8.5) | 7.9 (7.6-7.9) | 0.93 |
| radiation-n2048-x256 | 8.2 (8.0-8.2) | 7.4 (7.3-7.9) | 0.91 |
| radiation-n16384-x64 | 51.7 (51.7-52.0) | 48.2 (46.6-48.8) | 0.93 |
| radiation-n16384-x256 | 48.6 (48.6-49.1) | 46.7 (46.3-46.7) | 0.96 |
| radiation-n131072-x64 | 375.3 (372.4-375.3) | 309.9 (309.8-310.0) | 0.83 |
| radiation-n131072-x256 | 342.3 (342.1-342.4) | 310.3 (310.3-310.4) | 0.91 |

Float32, same GPU, explicit and radiation, 64 cells (before/after):

| case | before ms | after ms | after/before |
|---|---|---|---|
| explicit-n16384-x64 | 11.3 (11.3-11.3) | 3.9 (3.8-3.9) | 0.34 |
| explicit-n131072-x64 | 60.9 (60.6-61.1) | 14.7 (14.6-14.8) | 0.24 |
| radiation-n16384-x64 | 12.8 (12.8-12.9) | 6.4 (6.3-6.6) | 0.50 |
| radiation-n131072-x64 | 82.6 (82.5-82.7) | 38.3 (37.9-38.7) | 0.46 |

Differentiated, GPU, 64 cells, 32 steps, no checkpointing, milliseconds:

| case | before ms | after ms | after/before |
|---|---|---|---|
| explicit-jvp-n2048-x64 | 7.8 (7.8-7.8) | 6.7 (6.7-6.7) | 0.86 |
| explicit-jvp-n16384-x64 | 26.6 (26.6-26.6) | 15.1 (15.1-15.1) | 0.57 |
| explicit-reverse-n2048-x64 | 11.2 (11.2-11.2) | 10.0 (10.0-10.0) | 0.89 |
| explicit-reverse-n16384-x64 | 32.9 (32.9-32.9) | 24.8 (24.8-24.8) | 0.75 |
| implicit-jvp-n2048-x64 | 104.8 (104.8-104.8) | 104.9 (104.9-104.9) | 1.00 |
| implicit-jvp-n16384-x64 | 315.4 (315.4-315.4) | 250.4 (250.4-250.4) | 0.79 |
| implicit-reverse-n2048-x64 | 264.1 (264.1-264.1) | 262.2 (262.2-262.2) | 0.99 |
| implicit-reverse-n16384-x64 | 820.2 (820.2-820.2) | 765.5 (765.5-765.5) | 0.93 |
| radiation-jvp-n2048-x64 | 12.8 (12.8-12.8) | 11.1 (11.1-11.1) | 0.87 |
| radiation-jvp-n16384-x64 | 49.6 (49.6-49.6) | 39.0 (39.0-39.0) | 0.79 |
| radiation-reverse-n2048-x64 | 22.2 (22.2-22.2) | 21.6 (21.6-21.6) | 0.97 |
| radiation-reverse-n16384-x64 | 67.7 (67.7-67.7) | 61.5 (61.5-61.5) | 0.91 |

On the two x86 CPUs the forward medians do not move (after/before 1.00 to 1.01 on the pinned office core
for all four cases at 2048 and 16384 electrons; and the three workload timings below, taken on the Apple core, are the same to within 1 %).

### Checkpointing and memory

Compiler temporary bytes of the reverse program on the GPU, 32 steps, 64 cells, double precision
(`checkpoint=False` against `True`):

| case | 2048 electrons | 16384 electrons |
|---|---|---|
| explicit | 72.5 MB against 8.9 MB | 584 MB against 69 MB |
| implicit | 1310 MB against 53 MB | 10 481 MB against 426 MB |
| radiation | 177 MB against 11 MB | 1421 MB against 84 MB |
| collisions | 250 MB against 13 MB | 1886 MB against 96 MB |

Without checkpointing the implicit reverse program of 131072 electrons asks for 78 GiB and does not fit
the 16 GB device; the explicit one needs 4.7 GB. These are compiler estimates, and the first
failure is a device-memory one on a host whose GPU is shared. Timings of these reverse runs on the shared
card varied by up to 2x between rounds, so no speed ratio between the two settings is claimed.

### Real workloads

| workload | warm median | compile | where |
|---|---|---|---|
| `beam_laser.py` `final_mean_gamma(a0)`, 16 particles, 8192 steps | 32 ms | 0.4 s | Apple CPU |
| `rf_source_control.py` `metrics(a)`, 512 steps | 1.8 ms | 0.3 s | Apple CPU |
| the same, four-coefficient `jacfwd` | 9.8 ms | 0.8 s | Apple CPU |
| maintained sheath, 6000 steps, 120 000 slots | 555 s (first call) | 3 s | office CPU core |
| maintained sheath, 6000 steps, 120 000 slots | 32.8 s before, 29.8 s after | 5 s | GPU |

The laser and RF objectives take milliseconds; the 33 s and 93 s of the examples are compilation of a
program per refinement step count and the validation around them, not stepping. The sheath is the one
workload that is long: on the CPU a 300-step trace has no dominant kernel (the pool's `top_k` sort is
11 %, the rest is spread over the dynamic-update-slice fusions and dots), and the GPU is 19 times faster.

### Not done, and why

* Collisions on the GPU are launch-bound: 21 000 kernels per 32 steps. Fusing the pairing needs a custom
  kernel; no Pallas kernel was written because the trace shows idle time, not a slow kernel.
* Writing the momentum changes of both collision partners as one scatter was slower on the CPU
  (635 ms against 503 ms at 16384 electrons) and was dropped.
* The implicit step is still 10 to 15 times an explicit one at equal particle count, which is the eight
  Picard iterations; the scatter fix took 41 % off it at 131072 electrons.
* The 78 GiB case and one checkpointed 131072-electron run could not be completed on a 16 GB card that
  another job was also using; they are failures of the device, not timings.
