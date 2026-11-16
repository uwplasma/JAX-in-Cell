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

## Fields that vary in y and z

Particles carry all three coordinates, and $y$ and $z$ are periodic over the domain's
`length_y` and `length_z`. An external field of shape `(cells, ny, nz, 3)` is read on the
centres of an $(x, y, z)$ grid: the domain's cells along $x$, and `ny` and `nz` cells
spanning `length_y` and `length_z`. It is gathered at each particle's $x$, $y$ and $z$ with
the same quadratic spline as the $x$ gather, applied along each axis in turn, so a grid with
`ny = nz = 1` is the flat field to round-off. E and B on such a grid both sit on the
centres. The two forms can be mixed, one field flat and the other on a grid.

```python
y = (np.arange(ny) + 0.5) * domain.length_y / ny - domain.length_y / 2
B = np.zeros((domain.cells, ny, 1, 3))
B[..., 0] = B0 * (1 + y / L)[None, :, None]           # a gradient across the box: grad-B drift
simulation = Simulation(domain, [electrons], solver, external_B=B)
```

The self-consistent fields still depend on $x$ alone, so this is for imposed structure: drifts,
mirrors, the gradients of a device, test particles. A flat field costs what it did; a field on a
grid gathers 27 points per particle instead of 3. {func}`~jaxincell.magnetic_moment` gives
$\mu = p_\perp^2/(2mB)$ relative to the external field, and
{meth}`~jaxincell.Simulation.external_fields_at` the external fields anywhere.
{doc}`../examples/external_fields_3d` checks the grad-$B$ drift and a mirror bounce against
guiding-centre theory. The `[external]` table of an input file stays uniform.

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

In one dimension $\nabla\times\mathbf B$ has no $x$ component, so $B_x$ cannot evolve: it
is exactly the field the code cannot generate itself, and therefore the one worth imposing.

* A uniform $B_x$ magnetises the plasma and gives the particles a gyration in the $y$-$z$
  plane at $\Omega_c = qB_x/m$, opening up upper-hybrid oscillations, Bernstein modes and
  cyclotron damping.
* Resolve that gyration: the Boris rotation needs $\Omega_c\Delta t \lesssim 0.3$ for a few
  per cent accuracy, and $\Omega_c\Delta t < 2$ to stay stable at all.
* {doc}`../examples/sheath_magnetized` is a worked case with the field oblique to a wall.

```python
from jaxincell import elementary_charge, mass_electron
print(elementary_charge * 0.05 / mass_electron * domain.dt)   # Omega_c dt
```

## Time-dependent fields

There is no hook for one. A field that has to vary in time is a physical field, and the
honest way to put it in is as a current or a charge — a driven antenna as a species, an
imposed wave as an initial condition on $\mathbf E$ and $\mathbf B$ through the
restart state ({doc}`running`).
