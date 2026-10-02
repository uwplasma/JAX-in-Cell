# Solver parameters

The `solver_parameters` section selects the time integrator and the field solver and
sets the digital filter, the implicit-solver controls and the random seed.

| parameter | default | differentiable | meaning |
|---|---|---|---|
| `time_evolution_algorithm` | `0` | no | `0` explicit leapfrog with the Boris pusher; `1` implicit Crank-Nicolson with Picard iteration. |
| `field_solver` | `0` | no | `0` electromagnetic: $E_x$ follows Ampere's law; `1` electrostatic: $E_x$ is recomputed from Gauss's law by FFT every step. Only `0` and `1` are accepted. |
| `relativistic` | `false` | no | Use the relativistic Boris pusher (explicit scheme only). |
| `filter_passes` | `5` | no | Number of passes of the compensated binomial filter applied to $\rho$ and $\mathbf J$ (explicit scheme only). `0` disables it; `1` is a no-op, see below. |
| `filter_alpha` | `0.5` | yes | Weight of the centre point in each binomial pass, $0 < \alpha < 1$. |
| `filter_strides` | `(1, 2, 4)` | no | Cell offsets of the three-point stencil; the filter is applied once per stride. |
| `max_number_of_Picard_iterations_implicit_CN` | `20` | no | Iteration cap of the implicit solver. |
| `tolerance_Picard_iterations_implicit_CN` | `1e-6` | no | Relative change of $\mathbf E$ between iterates below which the implicit solver stops. |
| `number_of_particle_substeps_implicit_CN` | `2` | no | Particle sub-steps per field step in the implicit solver. |
| `seed` | `1701` | no | Seed of the random number generators, see {doc}`species`. |
| `print_info` | `true` | no | Print the derived plasma parameters at the start of the run. |

## Choosing the integrator

The explicit scheme (`0`) is the default. It is second order in time, uses the
charge-conserving current deposit, supports all boundary conditions, the relativistic
pusher and the digital filter, and costs one particle push and two half field updates
per step. Its total energy drifts slowly, typically by $10^{-3}$ relative over a few
hundred plasma periods in the examples.

The implicit scheme (`1`) solves the field and particle equations together with a
time-centred discretisation and conserves total energy to round-off. It has no
light-wave Courant limit, so it is the natural choice for large time steps in
electromagnetic problems. Each step costs up to
`max_number_of_Picard_iterations_implicit_CN` particle pushes times the number of
sub-steps. Its deposition and interpolation are written for periodic boundaries only,
and it ignores the digital filter and the `relativistic` switch. Details in
{doc}`../numerics/implicit`.

## Choosing the field solver

With `field_solver = 0` the longitudinal field $E_x$ is advanced with Ampere's law from
the deposited current, and Gauss's law is satisfied because the current deposit
satisfies the discrete continuity equation. With `field_solver = 1` the code
additionally overwrites $E_x$ at the end of every step with the solution of Gauss's law
from the deposited charge density, computed by FFT. This is the electrostatic mode: it
is exact for periodic boundaries, removes any accumulated error in $\nabla\cdot\mathbf E$
and costs one FFT per step. The transverse components $E_y$, $E_z$ and the magnetic
field are advanced in the same way in both modes. See {doc}`../numerics/field_solvers`.

## The digital filter

The charge and current densities can be smoothed before they enter the field
equations. One pass with stride $s$ replaces $f_j$ by
$\alpha f_j + \tfrac{1-\alpha}{2}(f_{j-s} + f_{j+s})$. A setting of $p$ passes applies
$p - 1$ such passes followed by one compensation pass with
$\alpha_c = p - \alpha(p-1)$, for each stride in `filter_strides` in turn. The
compensation pass flattens the response at long wavelengths so that the resolved
physics is not damped. Consequences of the formula:

* `filter_passes = 1` applies only the compensation pass with $\alpha_c = 1$, which
  is the identity. The smallest setting that filters is `2`.
* With the defaults (five passes, strides 1, 2 and 4) the response drops to zero for
  $k\Delta x \gtrsim 0.15\pi$, that is for wavelengths shorter than about thirteen
  cells. Modes you want to study must be longer than that; the bump-on-tail example
  switches the filter off for this reason. The response curves are shown in
  {doc}`../numerics/filtering`.
* The number of regular passes is capped at 16 inside the compiled code.

The filter is applied with the field boundary conditions: periodic wrap, clamped
values (reflective) or zeros outside the box (absorbing).

## Implicit-solver controls

The Picard iteration stops when
$\max|\mathbf E^{(k+1)} - \mathbf E^{(k)}| / \max|\mathbf E^{(k+1)}|$ falls below the
tolerance or when the iteration cap is reached; there is no warning in the second
case. With the default tolerance the examples converge in a handful of iterations.
Particle sub-stepping divides each field step into `number_of_particle_substeps_implicit_CN`
pushes with the time-centred fields, which keeps the particle orbits accurate when the
field time step is large.
