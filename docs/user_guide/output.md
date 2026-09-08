# The output

{meth}`~jaxincell.Simulation.run` returns an {class}`~jaxincell.Output`, a frozen
dataclass and a pytree. Histories have the stored step as their first axis.

| field | shape | meaning |
|---|---|---|
| `t` | `(S,)` | time of each stored state, s |
| `x`, `v` | `(S, N, 3)` | particle positions and velocities, or `None` |
| `E`, `B`, `J` | `(S, cells, 3)` | fields and current density |
| `rho` | `(S, cells)` | charge density at the cell centres |
| `grid` | `(cells,)` | cell centres |
| `dx`, `dt`, `length` | scalars | grid spacing, time step, box length |
| `charge`, `mass`, `weight` | `(N,)` | per pseudo-particle; `charge` is zero for absorbed particles |
| `species` | `(N,)` | index of the species each particle belongs to |
| `names`, `counts` | tuples | species names and particle counts |
| `state` | pytree | the final loop state, for a restart |

`E` and `J` are at the cell faces, `B` and `rho` at the centres
({doc}`../numerics/discretization`).

## Per species

```python
x_e, v_e = output.particles("electrons")     # (S, n_e, 3) each
```

or, by hand, using the `species` index:

```python
import numpy as np
electrons = np.asarray(output.species) == 0
```

## Diagnostics

```python
from jaxincell import diagnostics

d = diagnostics(output)
d["total"]            # total energy at every stored step
d["gauss_residual"]   # relative violation of the discrete Gauss law
d["temperatures"]     # per species, per component, in eV
```

The full list and what each one means is in {doc}`../numerics/diagnostics`. The
individual functions — {func}`~jaxincell.energies`,
{func}`~jaxincell.gauss_residual`, {func}`~jaxincell.temperatures`,
{func}`~jaxincell.dominant_frequency` — are exported too, so only what is needed has
to be computed.

## Saving

The arrays are ordinary JAX arrays, so anything that takes NumPy works:

```python
import numpy as np
np.savez_compressed("run.npz", t=output.t, E=output.E, rho=output.rho)
```

For an interchange format the package can write openPMD {cite}`openpmd`, the
community standard for particle-in-cell output, which the visualisation tools of the
field read directly:

```python
from jaxincell.openpmd import write_openpmd

write_openpmd(output, "run.h5")       # needs `pip install jaxincell[openpmd]`
```

One iteration per stored step, meshes for `E`, `B`, `J` and `rho` with the right
staggering recorded in the file, and one particle species per `Output.names` carrying
position, momentum, weighting, charge and mass.

## Reading a run back

An `Output` is a pytree, so `jax.tree_util` flattens and rebuilds it, and the
diagnostics work on a reconstructed one as long as the fields they need are present.
The simplest durable choice is to save the arrays and recompute the diagnostics on
load; they are cheap compared with the run that produced them.
