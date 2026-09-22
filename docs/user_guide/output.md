# The output

{meth}`~jaxincell.Simulation.run` returns an {class}`~jaxincell.Output`, a frozen dataclass
and a pytree. Histories have the stored step as their first axis.

## Fields

| field | shape | meaning |
|---|---|---|
| `t` | `(S,)` | time of each stored state, s |
| `x`, `v` | `(S, N, 3)` | particle positions and velocities, or `None` |
| `E`, `B`, `J` | `(S, cells, 3)` | fields and current density |
| `rho` | `(S, cells)` | charge density at the cell centres |
| `grid` | `(cells,)` | cell centres |
| `dx`, `dt`, `length` | scalars | grid spacing, time step, box length |
| `charge`, `mass` | `(N,)` | of one physical particle, for each pseudo-particle |
| `weight` | `(S, N)` | physical particles per pseudo-particle and unit area; zero once a wall has collected it; `None` with `x` |
| `species` | `(N,)` | index of the species each particle belongs to |
| `names`, `counts` | tuples | species names and particle counts |
| `field_bc` | tuple | wall codes the fields were solved with, which the Gauss diagnostic needs |
| `state` | pytree | the final loop state, for a restart |

`E` and `J` are at the cell faces, `B` and `rho` at the centres
({doc}`../numerics/discretization`).

## Per species

```python
x_e, v_e = output.particles("electrons")     # (S, n_e, 3) each
```

Or by hand, using the `species` index:

```python
import numpy as np
electrons = np.asarray(output.species) == 0
```

## Diagnostics

```python
from jaxincell import diagnostics

d = diagnostics(output)
d["total"]            # total energy at every stored step
d["energy_error"]     # |W(t) - W(0)| / W(0)
d["momentum_error"]   # |P(t) - P(0)| / sum_p |p_p(0)|
d["gauss_residual"]   # violation of the discrete Gauss law, relative to e n / eps0
d["potential"]        # electrostatic potential at the faces, zero at the left wall
d["temperatures"]     # per species, per component, in eV
```

* `d["dominant_frequency"]` needs at least two stored steps and is NaN for a run that
  stored one.
* The full list is in {doc}`../numerics/diagnostics`.
* {func}`~jaxincell.energies`, {func}`~jaxincell.gauss_residual`,
  {func}`~jaxincell.temperatures` and {func}`~jaxincell.dominant_frequency` are exported
  too, so only what is needed has to be computed.

## Saving

The arrays are ordinary JAX arrays, so anything that takes NumPy works:

```python
import numpy as np
np.savez_compressed("run.npz", t=output.t, E=output.E, rho=output.rho)
```

For interchange the package writes openPMD {cite}`openpmd`, the community standard for
particle-in-cell output, which the visualisation tools of the field read directly:

```python
from jaxincell.openpmd import write_openpmd

write_openpmd(output, "run.h5")       # needs `pip install jaxincell[openpmd]`
```

* One iteration per stored step.
* Meshes for `E`, `B`, `J` and `rho`, with the right staggering recorded in the file.
* One particle species per `Output.names`, carrying position, momentum and weighting per
  particle, and charge, mass and a zero `positionOffset` as constant records.
* The momentum is the one the pusher advances: $\gamma m\mathbf v$ for a relativistic run,
  $m\mathbf v$ otherwise.

openPMD's `weighting` counts physical particles, while `Output.weight` counts them per unit
area of the $y$-$z$ plane ({doc}`units`). The export multiplies by the transverse area the
run stands for, `area` in m², recorded on the `weighting` record as `transverseArea`:

```python
write_openpmd(output, "run.h5", area=domain.length_y * domain.length_z)
```

The default, 1 m², writes the per-unit-area weights unchanged. The meshes are volume
densities and do not depend on it.

## Reading a run back

An `Output` is a pytree, so `jax.tree_util` flattens and rebuilds it, and the diagnostics
work on a reconstructed one as long as the fields they need are present. The simplest
durable choice is to save the arrays and recompute the diagnostics on load: they are cheap
next to the run that produced them.
