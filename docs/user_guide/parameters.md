# Configuring a simulation

## The parameter tree

A simulation is described by a nested dictionary with five sections. Each section is
optional; a missing section or key takes the default value.

| section | contents | page |
|---|---|---|
| `domain_parameters` | box size, grid, time step, number of steps, boundary conditions | {doc}`domain` |
| `species_parameters` | one entry per electron or ion population | {doc}`species` |
| `solver_parameters` | integrator, field solver, filter, implicit-solver settings, seed | {doc}`solver` |
| `external_field_parameters` | static external electric and magnetic fields | {doc}`external_fields` |
| `source_parameters` | particle sources (accepted but not active on `main`) | {doc}`external_fields` |

In TOML the sections are tables; species are nested tables named
`species_parameters.<type>.<label>`:

```toml
[domain_parameters]
length = 0.01
number_grid_points = 70

[species_parameters.electrons.electrons0]
number_pseudoparticles = 3500

[species_parameters.ions.ions0]
number_pseudoparticles = 3500
```

In Python the same tree is a dictionary of dictionaries. `jaxincell.load_parameters`
converts a TOML file to that dictionary, so anything written for one form works for the
other.

## Validation

{class}`jaxincell.Simulation` copies the tree, overlays the defaults, and validates every
section with assertions when it is constructed. Errors are raised immediately, before
any compilation:

```python
>>> Simulation({"domain_parameters": {"total_steps": 0}})
AssertionError: Total number of time steps must be an integer.
```

Integer-valued parameters must be Python `int` (a TOML `500` is fine, `500.0` is not),
booleans must be `true`/`false`, and lists such as `filter_strides` are converted to
tuples. Floating-point physical parameters are converted to JAX arrays so that they can
participate in automatic differentiation.

## Species labels

Each population is identified by its type (`electrons` or `ions`) and a label that you
choose (`electrons0`, `beam`, `protons`, ...). Internally the code assigns canonical
labels `_electrons0`, `_electrons1`, ... and `_ions0`, `_ions1`, ... in the order in
which the populations appear, and keeps your label under `user_label`. Both labels can be
used when passing runtime inputs, see {doc}`running`. Cross references between species
(a string value such as `"_electrons0"`) must use the canonical label.

If `species_parameters.electrons` is omitted entirely, a single population with the
built-in defaults is created; the same holds for ions. The first electron population
and the first ion population use a different set of defaults from any additional
populations (the first ones set up a two-stream instability, the others are cold and
at rest), see {doc}`species`.

## Differentiable inputs

A subset of the parameters can be changed at run time without recompiling and can be
differentiated with respect to. Passing them under an `input_parameters` key of the
tree, or directly to `Simulation.run`, routes them to the right section:

```python
parameters = {
    "domain_parameters": {...},
    "input_parameters": {"electrons": {"electrons0": {"drift_speed_x": 6e7}}},
}
sim = Simulation(parameters)
output = sim.run()                                   # uses drift_speed_x = 6e7
output = sim.run({"electrons": {"electrons0": {"drift_speed_x": 7e7}}})   # same compiled program
```

Which parameters are differentiable is listed in each section's table and in
{doc}`differentiation`. Passing a non-differentiable parameter to `run` raises a
`ValueError` that names the offending key.

## Precedence

When a parameter appears in more than one place the order is, from lowest to highest
priority: built-in default, value in its section, value under `input_parameters` at
construction, value passed to `run`. Setting a section attribute of the `Simulation`
object (for example `sim.solver_parameters = {...}`) replaces that section and
re-initialises the state, see {doc}`running`.
