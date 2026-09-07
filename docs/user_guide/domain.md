# Domain parameters

The `domain_parameters` section defines the simulation box, the grid, the time step
and the boundary conditions.

| parameter | default | type | differentiable | meaning |
|---|---|---|---|---|
| `length` | `1e-2` | float | yes | Box length $L$ in metres along $x$. The box spans $[-L/2, L/2]$. |
| `length_y`, `length_z` | `0` | float | yes | Periodic extent in $y$ and $z$; `0` means "same as `length`". Only used to wrap the $y$ and $z$ coordinates of particles. |
| `number_grid_points` | `50` | int | no | Number of cells $N_x$ along $x$. |
| `number_grid_points_y`, `number_grid_points_z` | `0` | int | no | Accepted for future use; `0` is replaced by `3`. No field is defined on a $y$ or $z$ grid. |
| `total_steps` | `350` | int | no | Number of time steps. Every step is stored in the output. |
| `timestep_over_spatialstep_times_c` | `1.0` | float | yes | $c\,\Delta t/\Delta x$. |
| `particle_BC_left`, `particle_BC_right` | `0` | int | no | Particle boundary condition at $x=-L/2$ and $x=+L/2$: `0` periodic, `1` reflective, `2` absorbing. |
| `field_BC_left`, `field_BC_right` | `0` | int | no | Field boundary condition: `0` periodic, `1` reflective, `2` absorbing. |

## Derived quantities

```{math}
\Delta x = \frac{L}{N_x}, \qquad
x_i = -\frac{L}{2} + \left(i + \tfrac12\right)\Delta x, \quad i = 0, \dots, N_x - 1, \qquad
\Delta t = \texttt{timestep\_over\_spatialstep\_times\_c}\;\frac{\Delta x}{c}.
```

`grid` in the output holds the cell centres $x_i$. Electric field and current density
are stored at the cell faces $x_{i+1/2}$, the magnetic field at the cell centres; the
output arrays have one value per cell for every quantity and the
{doc}`../numerics/discretization` page explains which location each one refers to.

## Choosing the resolution

The grid spacing should resolve the electron Debye length. With the quadratic spline
shape function and the digital filter switched on, $\Delta x \lesssim 2\lambda_D$ is
safe; the finite-grid instability appears for coarser grids. The spacing is not set
directly: `grid_points_per_Debye_length` in the species section fixes
$\lambda_D/\Delta x$, and the density follows from it (see {doc}`species`).

The time step has three constraints, discussed in {doc}`../numerics/stability`:

* plasma oscillations: $\omega_{pe}\Delta t \lesssim 0.2$ for accuracy (the leapfrog
  limit is $\omega_{pe}\Delta t < 2$);
* particle motion: a pseudo-particle should not cross more than one cell per step,
  $v_{\max}\Delta t < \Delta x$, because the charge-conserving current deposit sweeps a
  window of six cells around each particle;
* light waves, explicit scheme only: $c\,\Delta t/\Delta x \le 1$ whenever a transverse
  field component can be excited. Purely electrostatic runs with velocities only along
  $x$ do not excite transverse fields and may use a larger value, as the examples do.

The implicit Crank-Nicolson scheme removes the light-wave constraint but not the other
two.

## Boundary conditions

The codes apply to the $x$ boundaries only; $y$ and $z$ are always periodic with
periods `length_y` and `length_z`. Particle and field conditions are independent, but
the physically consistent combinations are the diagonal ones: periodic with periodic,
reflective with reflective, absorbing with absorbing. {doc}`boundaries` describes what
each condition does to particles and fields, and {doc}`../numerics/boundaries` gives the
ghost-cell formulas.
