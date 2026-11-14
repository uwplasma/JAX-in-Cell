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
* Resolve that gyration: in a uniform magnetic field the Boris angle is
  $2\arctan(\Omega_c\Delta t/2)$, so it stays bounded at large steps while its phase
  becomes inaccurate. Use the relativistic frequency $|q|B/(m\gamma)$ where needed
  and check the orbit phase under timestep refinement.
* {doc}`../examples/sheath_magnetized` is a worked case with the field oblique to a wall.

```python
from jaxincell import elementary_charge, mass_electron
print(elementary_charge * 0.05 / mass_electron * domain.dt)   # Omega_c dt
```

## Time-dependent fields

`prescribed_fields(x, t, parameters)` returns electric and magnetic arrays broadcastable
to `(N, 3)` in SI units at the supplied particle positions. It is a pure JAX function;
`field_parameters` is a dynamic pytree of its amplitudes, phases or coefficients.
The returned fields add to the static arrays and the self-consistent fields.
Evaluate each position independently of the other particles in the input batch:
source and boundary events may evaluate the callback on a single particle.

```python
import jax.numpy as jnp
from jaxincell import speed_of_light as c

def wave(x, t, parameters):
    amplitude, omega, phase = parameters
    Ey = amplitude * jnp.sin(omega * (t - x[:, 0] / c) + phase)
    E = jnp.zeros_like(x).at[:, 1].set(Ey)
    B = jnp.zeros_like(x).at[:, 2].set(Ey / c)
    return E, B

simulation = Simulation(domain, [electrons, ions], solver,
                        prescribed_fields=wave, field_parameters=(1e4, 1e10, 0.))
```

This example is a vacuum plane wave traveling along $+x$. An imposed plasma RF
mode needs the appropriate dispersion and polarization; the callback does not
enforce Maxwell's equations or solve the antenna or the background dielectric response. For a periodic domain,
use a compatible spatial period. Time is absolute simulation time in seconds,
carried in `State.time`, and the integrators evaluate it at their force stages.
For identical continuation, reconstruct the same simulation, including its
timestep, fields and radiation settings: archives preserve time and state but
do not serialize or validate those settings. Changing the timestep also changes
the periodic explicit half-position convention and requires a phase conversion.

Define the callable once; changing numeric parameters then does not create a new
program. Differentiating the parameters differentiates the imposed waveform and
its effect on particles. The work supplied by all imposed electric fields is
recorded per species in `output.work.prescribed_work`. That measures work delivered
to the simulated particles; it is not wall-plug RF power or incident antenna power.
See {doc}`../numerics/radiation` for the radiation and mechanical-work budgets.
