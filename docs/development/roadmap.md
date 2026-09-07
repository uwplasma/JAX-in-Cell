# Roadmap and branches

## Done

* Explicit electromagnetic and electrostatic field solvers.
* Relativistic Boris pusher.
* Implicit Crank-Nicolson integrator with Picard iteration.
* Multiple electron and ion populations with cross-referenced temperatures.
* Periodic, reflective and absorbing boundaries for particles and fields.
* Compensated digital filter.
* Differentiable inputs and runtime re-execution without recompilation.

## Planned

* Binary collisions, so that a plasma can relax to a Maxwellian.
* Particle sources and sinks (the `source_parameters` section is reserved for this).
* Time-dependent and user-defined external fields (the `*_function` parameters are
  reserved for this).
* Output of selected steps only, to bound the memory of long runs.
* Output in the openPMD standard.
* Mixed and velocity-dependent boundary conditions.
* A two-dimensional version.

## Open pull requests and branches

Work in progress lives on branches of the main repository and in open pull requests;
the following existed at the time of writing and are the place to look before
starting on the same topic: binary collisions (`collsion`), particle sources
(`ds/source_particles`), openPMD output (`ds/OpenPMD`), three-dimensional external
fields (`ds/3D_external_fields`), mixed boundary conditions (`rishi/mixed_BCs`),
snapshot output (pull request from `zhiping0913`), and improved charge conservation
in the explicit scheme (`rj/full_EM_2`).
