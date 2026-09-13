# Roadmap

## Done

* Explicit leapfrog with the Boris pusher, relativistic or not.
* Implicit Crank-Nicolson integrator with a fixed-length Picard iteration, energy
  conserving to round-off and differentiable.
* Charge-conserving current deposit; the discrete Gauss law holds to round-off with no
  correction step.
* Ampere and Gauss field solvers.
* Any number of species, with cross-referenced temperatures.
* Periodic, reflective, absorbing and thermal walls, chosen separately for particles and
  fields, with a first-order Mur radiating condition on the fields. Absorbing walls are
  conductors, short-circuited to each other or floating opposite a symmetry plane.
* Walls that return part of each particle, as a fixed fraction or a law in the impact
  speed, with a coefficient of restitution per wall; with a thermal source wall a
  floating electrode holds the sheath drop of Hobbs and Wesson
  ({doc}`../examples/sheath`, {doc}`../examples/wall_reflection`).
* Compensated digital filter with arbitrary strides.
* Binary Coulomb collisions (Takizuka-Abe), verified against the Fokker-Planck rates.
* Quiet starts, and {func}`~jaxincell.quiet_start` for building custom conditions.
* Static external fields as arrays, differentiable like any other parameter.
* `store_every`, `store_particles` and restarts, so that long runs fit in memory.
* openPMD output.
* Differentiable end to end, with `vmap` over seeds for ensembles.

## Planned

* Precision as a choice in the examples. Every example script lets the user pick double
  or single precision, and the examples and the README set `JAX_ENABLE_X64` explicitly
  rather than relying on the default. The choice should carry what it costs: single
  precision is correct, but on the RTX A4000 tested it was many times slower than double
  precision, while on a CPU the two cost about the same ({doc}`../user_guide/performance`).
* A profile of `mode="promise_in_bounds"` for the scatter-add in `deposit`. The indices
  are always in range after `map_indices`, and on the RTX A4000 the option made the
  float64 scatter-add alone about a quarter faster. Adopt it only if an alternating A/B
  of whole steps, on the CPU and on the GPU, for both integrators and both precisions,
  shows it never slower and the fields identical to round-off, with a test that pins
  the equivalence.
* Particle sources and sinks, which need a pool of inactive particles because array
  shapes are static. An ionisation source is what a bounded-plasma run needs to reach a
  true steady state instead of slowly draining ({doc}`../examples/sheath`).
* A series RLC circuit between the two electrodes, so that a wall can be biased or left
  genuinely floating rather than short-circuited to its partner {cite}`verboncoeur1993`.
* Time-dependent external fields.
* Ionisation and recombination.
* Secondary electron emission with an energy-dependent yield {cite}`furman2002`, which,
  unlike reflection, creates electrons and needs the same pool of inactive particles.
* A two-dimensional version. `_core.py` would change thoroughly; the configuration
  objects, the time loop, the diagnostics and the differentiability would not.

## Contributing

Pick something from the list, or something not on it, and open an issue first so that
the design can be discussed before the code is written. {doc}`contributing` has the
practical details and {doc}`architecture` describes where things go.
