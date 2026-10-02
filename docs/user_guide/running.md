# Running simulations

## The `Simulation` object

{class}`jaxincell.Simulation` is constructed from a parameter dictionary or a path to a
TOML file. Construction validates the parameters, builds the grid, generates the
initial phase space and computes the initial electric field from Gauss's law. Nothing
is compiled yet.

```python
from jaxincell import Simulation

sim = Simulation(parameters)         # dictionary
sim = Simulation("input.toml")       # path, loaded with load_parameters
sim = Simulation()                   # built-in defaults
```

The cleaned parameter sections are attributes: `sim.domain_parameters`,
`sim.species_parameters`, `sim.solver_parameters`, `sim.external_field_parameters`,
`sim.source_parameters`. Species are keyed by their canonical labels
(`sim.species_parameters["electrons"]["_electrons0"]`). The initial state is also
available: `sim.grid`, `sim.dx`, `sim.dt`, `sim.box_size`, `sim.positions`,
`sim.velocities`, `sim.weights`, `sim.charges`, `sim.masses`, `sim.fields`.

## `run`

```python
output = sim.run()
output = sim.run(input_parameters)
```

`run` (an alias of `simulation`) executes all `total_steps` steps inside one
`jax.lax.scan` and returns the output dictionary described in {doc}`output`. The first
call on a given configuration compiles the program with XLA; the compile time is a few
seconds for the examples and grows with the code path (the implicit scheme, the
relativistic pusher and non-periodic boundaries each add branches). Subsequent calls
are fast. Wrap the call in `jax.block_until_ready` when timing it, because JAX returns
before the computation has finished.

The optional argument is a dictionary of differentiable parameters (see
{doc}`differentiation`). It changes the values used in this call without touching the
stored configuration and without recompiling:

```python
for drift in (4e7, 6e7, 8e7):
    output = sim.run({"electrons": {"electrons0": {"drift_speed_x": drift}}})
```

Species can be addressed by user label or canonical label, and a value given at the
type level applies to every population of that type:

```python
sim.run({"electrons": {"beam": {"drift_speed_x": 8e7}}})       # one population
sim.run({"electrons": {"drift_speed_x": 8e7}})                 # all electron populations
sim.run({"timestep_over_spatialstep_times_c": 0.5})            # a domain parameter
```

`sim.input_parameters` returns the differentiable parameters that were given under
`input_parameters` at construction, as a dictionary ready to be passed back to `run`
or to `jax.grad`.

## Changing parameters after construction

Assigning to a section attribute replaces that section, re-validates it and
re-initialises the state:

```python
sim.solver_parameters = {"time_evolution_algorithm": 1}
sim.domain_parameters = {**sim.domain_parameters, "total_steps": 2000}
```

The assignment takes the complete new section (missing keys revert to defaults), so
copy the existing one when only one key should change. Assigning to
`sim.input_parameters` re-routes a new set of differentiable inputs.

## When does JAX recompile?

The compiled program depends on everything that is not a differentiable input:
particle counts, grid size, number of steps, algorithm switches, boundary conditions,
filter passes and strides, seeds. `Simulation` hashes each section and passes the
hashes as static arguments, so any change of those values compiles a new program,
while changes of differentiable values reuse the existing one. The `Simulation`
object itself is also a static argument, so a second object compiles its own program
even when its parameters are identical; reuse one object when running many cases.

## Command line

```bash
jaxincell               # defaults
jaxincell input.toml
```

The entry point loads the file, runs the simulation, calls {func}`jaxincell.diagnostics`
and {func}`jaxincell.plot`. It has no further options; use a script for anything else.

## Reproducibility

Runs are deterministic for a given seed, parameter set, JAX version and device.
Results on a GPU differ from those on a CPU at the level of floating-point rounding,
which is amplified by the instabilities being simulated; growth rates and energies agree,
individual particle trajectories do not after many e-foldings.
