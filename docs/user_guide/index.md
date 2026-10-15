# User guide

Four objects describe a simulation and one method runs it.

```{code-block} python
:caption: the whole interface, in one block

from jaxincell import Domain, Species, Solver, Collisions, Simulation, diagnostics, plot

domain    = Domain(length=0.01, cells=64, dt_over_dx_c=1.0)      # the box and the grid
electrons = Species.electrons(n=10000, density=1e17, vth=(1e6, 0, 0))
ions      = Species.ions(n=10000, density=1e17, electrons=electrons)
solver    = Solver(algorithm="explicit", filter_passes=2)        # the numerics

simulation = Simulation(domain, [electrons, ions], solver)
output     = simulation.run(1000, seed=0)
print(diagnostics(output)["total"][-1])
plot(output)
```

Every one of them is a frozen dataclass and a JAX pytree: physical quantities are
leaves, so they can be changed without recompiling and differentiated with respect to;
structural choices (particle counts, cell counts, boundary types, algorithm names) are
static and become part of the compiled program.

## Moving from the parameter-dictionary API

This research API uses SI inputs and separate configuration objects; old dictionaries
and TOML sections need an explicit translation.

| previous input or operation | research equivalent |
|---|---|
| `Simulation(parameters)` | `Simulation(Domain(...), [Species(...), ...], Solver(...))` |
| `total_steps`, then `sim.run()` | `sim.run(steps)` |
| `number_grid_points` | `Domain(cells=...)`; at least four cells |
| `vth_over_c_x/y/z` | `Species(vth=(vx, vy, vz))` in m/s; multiply the old ratios by $c$ |
| particle codes `0`, `1`, `2` | `"periodic"`, `"reflective"`, `"absorbing"` |
| partial-return codes `3`, `4` | absorbing particle walls with constant or callable `Species.reflection` |
| numeric time algorithm `0`, `1` | `Solver(algorithm="explicit" or "implicit")` |
| solver `snapshot_steps` | `sim.run(steps, snapshot_steps=...)`; zero selects the first completed step |
| mutable parameter setters | `dataclasses.replace` on the frozen objects |

Choose the field model and solver explicitly using {doc}`solver`; the numeric field
solver options are not interchangeable with the new model choices. Source rates,
wall histories and restart formats also have explicit contracts ({doc}`sources`, {doc}`output`).

```{toctree}
:maxdepth: 1

domain
species
solver
collisions
sources
external_fields
running
output
plotting
differentiation
performance
units
```
