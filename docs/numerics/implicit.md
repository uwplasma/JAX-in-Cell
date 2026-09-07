# Implicit Crank-Nicolson scheme

`time_evolution_algorithm = 1` selects a time-centred, implicit discretisation of the
particle and field equations, solved at every step by Picard (fixed-point) iteration.
The scheme is modelled on the energy-conserving implicit particle-in-cell method of
Chen, Chacón and Barnes {cite}`chen2011,chen2014` and Markidis and Lapenta
{cite}`markidis2011`: fields and particles are advanced with mid-point averages, the
current is orbit-averaged over particle sub-steps, and the same quadratic spline is
used for deposit and gather. Its distinguishing property is that the total energy is
conserved to round-off once the iteration has converged, independently of the time
step.

## Discrete equations

Given $\mathbf E^n$, $\mathbf B^n$, $x^n$, $\mathbf v^n$, the scheme seeks
$\mathbf E^{n+1}$, $\mathbf B^{n+1}$, $x^{n+1}$, $\mathbf v^{n+1}$ that satisfy

```{math}
\mathbf E^{n+1/2} = \tfrac12\left(\mathbf E^n + \mathbf E^{n+1}\right), \qquad
\mathbf B^{n+1} = \mathbf B^n - \Delta t\,\nabla\times\mathbf E^{n+1/2}, \qquad
\mathbf B^{n+1/2} = \tfrac12\left(\mathbf B^n + \mathbf B^{n+1}\right),
```

```{math}
\mathbf E^{n+1} = \mathbf E^n + \Delta t\left(c^2\nabla\times\mathbf B^{n+1/2} - \frac{\bar{\mathbf J} - \langle\bar{\mathbf J}\rangle}{\epsilon_0}\right),
```

where $\bar{\mathbf J}$ is the current averaged over the particle orbits during the
step, and $\langle\cdot\rangle$ is the spatial mean. Subtracting the mean current
removes the $k = 0$ component, which in a periodic box is not constrained by Gauss's
law and would otherwise accumulate a uniform electric field from any net drift.

The particles are advanced over $N_{sub}$ sub-steps of length $\Delta\tau = \Delta t/N_{sub}$
in the time-centred fields. For sub-step $\nu$, with $\mathbf E^{n+1/2}$ and
$\mathbf B^{n+1/2}$ gathered at the staggered position $x^{\nu+1/2}$ of the previous
iterate,

```{math}
\mathbf v^{\nu+1} = \text{Boris}\left(\mathbf v^{\nu};\, \mathbf E^{n+1/2}(x^{\nu+1/2}),\, \mathbf B^{n+1/2}(x^{\nu+1/2}),\, \Delta\tau\right), \qquad
\bar{\mathbf v}^{\nu} = \tfrac12\left(\mathbf v^{\nu} + \mathbf v^{\nu+1}\right),
```
```{math}
x^{\nu+1} = x^{\nu} + \bar v_x^{\nu}\,\Delta\tau, \qquad
x^{\nu+1/2} = x^{\nu+1} - \tfrac12\bar v_x^{\nu}\,\Delta\tau,
```

and the orbit-averaged current is accumulated with the quadratic spline at the
staggered positions,

```{math}
\bar{\mathbf J}_i = \frac{1}{\Delta t}\sum_{\nu=0}^{N_{sub}-1}\Delta\tau\sum_p q_p\,\bar{\mathbf v}_p^{\nu}\,S_2\!\left(\frac{x_i - x_p^{\nu+1/2}}{\Delta x}\right).
```

The Boris rotation with the mid-point fields is the Crank-Nicolson discretisation of
the Lorentz force, and the use of $\bar{\mathbf v}$ both for the position update and
for the current makes the work done by the field on the particles equal, at the
discrete level, to the change of the field energy. Boundary conditions are applied to
the positions after every sub-step.

## Picard iteration

The unknown $\mathbf E^{n+1}$ appears on both sides (through the fields that push the
particles and produce $\bar{\mathbf J}$). Starting from the guess
$\mathbf E^{n+1,(0)} = \mathbf E^n$, iteration $k$ evaluates the right-hand sides above
with $\mathbf E^{n+1,(k)}$ and produces $\mathbf E^{n+1,(k+1)}$. The iteration stops when

```{math}
\frac{\max\left|\mathbf E^{n+1,(k+1)} - \mathbf E^{n+1,(k)}\right|}{\max\left|\mathbf E^{n+1,(k+1)}\right| + 10^{-12}} < \texttt{tolerance\_Picard\_iterations\_implicit\_CN}
```

or after `max_number_of_Picard_iterations_implicit_CN` iterations. The particle
sub-step positions of the previous iterate are carried along so that each iteration
starts the sub-stepping from the converged orbit of the last one. The loop is a
`lax.while_loop` with a data-dependent trip count.

Picard iteration converges when the mapping is a contraction, which in practice
requires $\omega_{pe}\Delta t$ and $\Omega_c\Delta t$ to be at most of order one; it is
not a Newton method and does not benefit from a preconditioner. With the default
tolerance of $10^{-6}$ the examples converge in a few iterations.

## Properties

Energy conservation
: For the two-stream configuration of `examples/input.toml` the relative change of
  the total energy stays at {{ energy_error_max_implicit }} over the run, against
  {{ energy_error_max_explicit }} for the explicit scheme with the same time step,
  see {doc}`verification`.

Time step
: There is no light-wave Courant condition; the fields are unconditionally stable for
  the source-free Maxwell equations. The step is still limited by the accuracy of the
  particle orbits, which the sub-stepping helps with, and by the convergence of the
  fixed-point iteration.

Cost
: Each Picard iteration performs $N_{sub}$ Boris pushes and current deposits over all
  particles plus two curl evaluations, so a step costs several times an explicit step.

## Limitations of the current implementation

* Deposit and gather use periodic index wrapping regardless of the field boundary
  codes; the particle boundary codes are still honoured. Use periodic boundaries.
* The digital filter is not applied; `filter_passes` is ignored.
* The particle push is the non-relativistic Boris rotation; `relativistic` is ignored.
* The charge density stored in the output is deposited from $x^{n+1}$ without
  filtering.
* Reverse-mode automatic differentiation (`jax.grad`) is not available through the
  while loop; forward mode (`jax.jvp`, `jax.jacfwd`) is, see
  {doc}`../user_guide/differentiation`.
