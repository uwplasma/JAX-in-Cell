# The box, the grid and the walls

{class}`~jaxincell.Domain` holds everything geometric.

```python
from jaxincell import Domain

domain = Domain(length=0.01, cells=64, dt_over_dx_c=1.0,
                particle_bc="periodic", field_bc="periodic")
```

| argument | meaning | default |
|---|---|---|
| `length` | box length $L$ in metres; the box is $[-L/2, L/2]$ | `1e-2` |
| `cells` | number of cells $N_x$ (static) | `64` |
| `dt_over_dx_c` | the ratio $c\Delta t/\Delta x$ | `1.0` |
| `time_step` | $\Delta t$ in seconds, instead of `dt_over_dx_c` | `None` |
| `particle_bc` | wall type for particles, one name or a `(left, right)` pair (static) | `"periodic"` |
| `field_bc` | wall type for fields (static) | `"periodic"` |
| `restitution` | the normal velocity of whatever a wall sends back is multiplied by `-restitution`; one number or a `(left, right)` pair | `1.0` |
| `length_y`, `length_z` | periods of the two ignorable coordinates | `1e-2` |

Derived quantities are properties, so they follow the arguments:

```python
domain.dx       # length / cells
domain.dt       # the step in seconds, however it was given
domain.courant  # c dt / dx, however it was given
domain.grid     # cell centres, shape (cells,)
```

## Setting the time step

Give the step one way or the other, not both: `dt_over_dx_c` fixes $\Delta t$ through the
grid rather than in seconds, so refining the mesh refines the step with it, and
`time_step` says the seconds. `Domain.dt` and `Domain.courant` are the two readings of
whichever was given.

Which is the natural input depends on the physics, and so does the value:

* **Electromagnetic problems** need $c\Delta t/\Delta x \le 1$; at exactly one the
  vacuum wave propagates without error.
* **Electrostatic problems** never excite the transverse fields, so the light-wave limit
  does not apply, the step is set by the plasma frequency or the gyro-frequency instead,
  and `time_step=0.1 / omega_pe` says that where a Courant number would have to be worked
  out from the grid. Values of `dt_over_dx_c` well above one are normal there — the two-stream runs in
  {doc}`../numerics/verification` use {{ energy_courant }}. What binds instead is
  $\omega_p\Delta t \lesssim 0.2$.

```python
print(f"omega_pe dt = {float(simulation.plasma_frequency() * domain.dt):.3f}")
print(f"dx / lambda_D = {float(domain.dx / simulation.debye_length()):.2f}")
```

{doc}`../numerics/stability` collects all four resolution conditions.

:::{warning}
Running above the Courant limit with transverse particle motion — an isotropic
temperature, a magnetic field, collisions — makes the explicit field solver diverge
within a few steps. {class}`~jaxincell.Simulation` warns when it sees that
combination; take the warning seriously, or switch to `algorithm="implicit"`.
:::

## Walls

`"periodic"`, `"reflective"`, `"absorbing"` and, for particles only, `"thermal"`, either
as one name for both ends or as a pair:

```python
Domain(particle_bc=("thermal", "absorbing"), field_bc=("reflective", "absorbing"))
```

A periodic wall needs a periodic partner, which is checked at construction; this and
every other invalid choice (an unknown wall name, fewer than four cells, a thermal field
wall, a restitution outside $[0, 1]$) raises `ValueError`. The
particle and field walls are set separately, which is occasionally useful (particles
reflected while radiation leaves) but usually they should match; a thermal particle wall
takes a reflective field wall.

Two absorbing walls are treated as conductors that keep the charge they collect,
short-circuited to one another, so they stay at the same potential and the plasma is
free to float above them. One absorbing wall opposite a reflective or thermal one is a
floating electrode on its own. {doc}`../numerics/boundaries` explains why the
alternative, holding one wall's field at zero, piles all the collected charge onto the
other.

What each one does to particles, to the fields and to the charge budget is described in
{doc}`../numerics/boundaries`. In short: periodic recirculates; reflective mirrors the
position and reverses the normal velocity; absorbing collects the particle, all of it
unless its species returns a fraction (`Species.reflection`), and parks what it keeps
outside the grid, with a first-order Mur radiating condition on the fields so that
outgoing waves leave without reflection; thermal mirrors the position and redraws the
velocity from the half-Maxwellian of the species, standing for the plasma beyond the
box. That last pair, a thermal wall facing a floating conductor, is what
{doc}`../examples/sheath_reflection` uses.

## The ignorable coordinates

Particles carry $y$ and $z$ positions, wrapped periodically with periods `length_y` and
`length_z`. Nothing depends on them — the fields are functions of $x$ alone — so they
matter only for a plot or for a diagnostic that wants them. Leave them at the default
unless there is a reason not to.
