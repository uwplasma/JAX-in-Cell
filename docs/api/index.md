# API reference

Everything importable from `jaxincell` is listed here. The top-level names are the
public interface; the modules they come from start with an underscore and are not part
of the documented API, although the numerical kernels are documented because the
{doc}`../numerics/index` pages refer to them.

```{toctree}
:maxdepth: 1

configuration
simulation
diagnostics
kernels
constants
```

## Overview

| name | purpose |
|---|---|
| {class}`jaxincell.Domain` | box, grid, time step and walls |
| {class}`jaxincell.Species` | one population of pseudo-particles |
| {class}`jaxincell.Solver` | integrator, field solver and filter |
| {class}`jaxincell.Collisions` | binary Coulomb collision model |
| {class}`jaxincell.Simulation` | the whole problem; `.run()` executes it |
| {class}`jaxincell.Output` | the result of a run |
| {func}`jaxincell.quiet_start` | quiet-start positions and velocities as plain arrays |
| {func}`jaxincell.load_toml` | build a simulation from a TOML file |
| {func}`jaxincell.diagnostics` | energies, momentum, Gauss residual, temperatures |
| {func}`jaxincell.energies`, {func}`jaxincell.gauss_residual`, {func}`jaxincell.temperatures`, {func}`jaxincell.dominant_frequency` | the individual diagnostics |
| {func}`jaxincell.plot` | animated overview figure, optional MP4 |
| {func}`jaxincell.openpmd.write_openpmd` | export to the openPMD standard |
| `jaxincell.epsilon_0`, `jaxincell.speed_of_light`, … | physical constants |

## Pytree structure

Every configuration object and the {class}`~jaxincell.Output` are frozen dataclasses
registered as JAX pytrees. Physical quantities are leaves; structural settings are
static and part of the treedef. That is what lets `jax.grad` and `jax.vmap` apply to
functions of a `Simulation` directly, and what decides whether changing a value
recompiles the program.

```python
import jax
jax.tree_util.tree_structure(simulation)   # the static part
jax.tree_util.tree_leaves(simulation)      # the differentiable part
```
