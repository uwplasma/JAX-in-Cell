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

```{toctree}
:maxdepth: 1

domain
species
solver
collisions
external_fields
running
output
plotting
differentiation
performance
units
```
