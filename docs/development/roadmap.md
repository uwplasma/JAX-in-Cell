# Roadmap

## Done

* Explicit leapfrog with the Boris pusher, relativistic or not.
* Implicit Crank-Nicolson integrator with a fixed-length Picard iteration, energy
  conserving to round-off and differentiable.
* Charge-conserving current deposit; the discrete Gauss law holds to round-off with no
  correction step.
* Ampere and Gauss field solvers.
* Any number of species, with cross-referenced temperatures.
* Periodic, reflective and absorbing walls, chosen separately for particles and fields,
  with a first-order Mur radiating condition on the fields. Absorbing walls are
  conductors short-circuited to each other, so a bounded plasma forms a sheath
  ({doc}`../examples/sheath`).
* Compensated digital filter with arbitrary strides.
* Binary Coulomb collisions (Takizuka-Abe), verified against the Fokker-Planck rates.
* Quiet starts, and {func}`~jaxincell.quiet_start` for building custom conditions.
* Static external fields as arrays, differentiable like any other parameter.
* `store_every`, `store_particles` and restarts, so that long runs fit in memory.
* openPMD output.
* Differentiable end to end, with `vmap` over seeds for ensembles.

## Planned

* Particle sources and sinks, which need a pool of inactive particles because array
  shapes are static. An ionisation source is what a bounded-plasma run needs to reach a
  true steady state instead of draining ({doc}`../examples/sheath`).
* A series RLC circuit between the two electrodes, so that a wall can be biased or left
  genuinely floating rather than short-circuited to its partner {cite}`verboncoeur1993`.
* Time-dependent external fields.
* Ionisation and recombination.
* Velocity-dependent boundary conditions (secondary emission, thermal re-injection).
* A two-dimensional version. `_core.py` would change thoroughly; the configuration
  objects, the time loop, the diagnostics and the differentiability would not.

## Contributing

Pick something from the list, or something not on it, and open an issue first so that
the design can be discussed before the code is written. {doc}`contributing` has the
practical details and {doc}`architecture` describes where things go.
