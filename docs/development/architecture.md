# Code structure

## Modules

The package is a flat set of modules under `jaxincell/`; the names start with an
underscore and `__init__.py` re-exports their public names.

| module | contents |
|---|---|
| `_simulation.py` | the `Simulation` class: parameter handling, state initialisation, the compiled `_simulation` method with the `lax.scan` time loop, output assembly |
| `_algorithms.py` | `Boris_step` and `CN_step`, one time step of each integrator |
| `_particles.py` | Boris pushers (non-relativistic, relativistic), field interpolation |
| `_sources.py` | charge and current deposition, spline weights |
| `_fields.py` | curl operators, half-step field updates, electrostatic solvers |
| `_boundary_conditions.py` | particle boundary maps, field ghost cells |
| `_filters.py` | compensated binomial filter |
| `_state_initialization.py` | grid, particle sampling, weights, initial fields, the `print_info` summary |
| `_routing.py` | routing of differentiable inputs between the user tree and the sections |
| `_parameters/` | one module per parameter section with defaults, validation and hashing; `_species_definitions.py` and `_species_parameters.py` handle populations and cross references |
| `_diagnostics.py` | post-processing |
| `_plot.py` | the overview animation |
| `_constants.py` | physical constants |
| `__main__.py` | the `jaxincell` command |

## Data flow

1. `Simulation.__init__` copies the user tree, pulls differentiable values under
   `input_parameters` to their sections, overlays defaults, validates each section
   and stores the cleaned sections as attributes. It then builds the domain state
   (`dx`, `dt`, `grid`, `box_size`), samples the particles, computes the weights and
   the initial fields, and hashes each section.
2. `Simulation.run` cleans the runtime inputs and calls the jitted `_simulation`
   method with the section hashes as static arguments. Inside, the sections are
   merged with the runtime inputs, the state is rebuilt from them (so that gradients
   flow from the inputs through the initial condition), and `lax.scan` runs the step
   function for `total_steps` iterations.
3. The step function is `Boris_step` or `CN_step`. Both take and return a `carry`
   tuple of arrays and emit the per-step output tuple that `scan` stacks along a new
   leading axis.
4. `assemble_output` merges the stacked histories with the parameter sections into
   the output dictionary. `diagnostics` and `plot` work on that dictionary outside
   JAX.

## Conventions

* All arrays are JAX arrays inside the compiled region; NumPy is used only in
  `diagnostics` and `plot`.
* Static configuration (integers, booleans, tuples, strings) must not change between
  calls of the same compiled program; it is hashed into the static arguments.
  Floating-point physics parameters are traced and can be differentiated.
* Particle loops are `jax.vmap` over the particle axis; reductions to the grid are
  sums over that axis or scatter-adds.
* Boundary conditions are integer codes evaluated with `jnp.where` and `jnp.select`
  so that they do not create separate compiled branches.
* Functions that depend on a static value use `functools.partial(jit, static_argnames=...)`.

## Adding a parameter

Add the key with its default and validation to the section module under
`_parameters/`, add it to the section's differentiable list if it is a
floating-point physics input, read it where it is used, and add a test. The
documentation tables in the user guide are maintained by hand; update the one for
the section.

## Adding a diagnostic

Anything computable from the output dictionary belongs in `_diagnostics.py` or in a
user script; anything that needs per-step access inside the loop has to be added to
the `step_data` tuple returned by the step functions and to the unpacking in
`_simulation`.
