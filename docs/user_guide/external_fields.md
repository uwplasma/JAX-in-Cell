# External fields

`Simulation(..., external_E=..., external_B=...)` adds static fields that are gathered
at the particles along with the self-consistent ones and never evolve.

```python
import numpy as np
from jaxincell import Simulation

B = np.zeros((domain.cells, 3))
B[:, 0] = 0.05                                 # 50 mT along x, uniform
simulation = Simulation(domain, [electrons, ions], solver, external_B=B)
```

Both are arrays of shape `(cells, 3)` or `None`. They sit on the same grids as the
self-consistent fields: `external_E` on the cell faces, `external_B` on the cell
centres ({doc}`../numerics/discretization`).

## Why they are arrays

An amplitude and a wavenumber would cover a sinusoid and nothing else. An array covers
a sinusoid, a mirror field, a measured profile, a gradient, a localised pulse:

```python
x = np.asarray(domain.grid)
B = np.zeros((domain.cells, 3))
B[:, 0] = B0 * (1 + 0.3 * np.cos(2 * np.pi * x / domain.length))   # a magnetic mirror
```

They are pytree leaves, so they can be differentiated with respect to — optimising a
coil profile against a confinement diagnostic needs nothing beyond `jax.grad`.

## $B_x$ is the interesting one

In one dimension $\nabla\times\mathbf B$ has no $x$ component, so $B_x$ cannot evolve:
it is exactly the field the code cannot generate itself and therefore the one worth
imposing. A uniform $B_x$ magnetises the plasma, gives the particles a gyration in the
$y$-$z$ plane at $\Omega_c = qB_x/m$, and opens up the magnetised wave physics —
upper-hybrid oscillations, Bernstein modes, cyclotron damping.

Resolve the gyration: the Boris rotation needs $\Omega_c\Delta t \lesssim 0.3$ for a
few per cent accuracy, and $\Omega_c\Delta t < 2$ to stay stable at all.

```python
from jaxincell import elementary_charge, mass_electron
print(elementary_charge * 0.05 / mass_electron * domain.dt)   # Omega_c dt
```

## Time-dependent fields

There is no hook for one. A field that has to vary in time is a physical field, and the
honest way to put it in is as a current or a charge — a driven antenna as a species, an
imposed wave as an initial condition on $\mathbf E$ and $\mathbf B$ through the
restart state ({doc}`running`).
