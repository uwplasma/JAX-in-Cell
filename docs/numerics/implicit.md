# Implicit scheme

`Solver(algorithm="implicit")` selects a Crank-Nicolson scheme that keeps the discrete
Gauss law to round-off and, when the particle-field iteration converges in a periodic box,
the total energy: the energy-conserving scheme of Chen,
Chacón and Barnes {cite}`chen2011,chen2014`, with the longitudinal current and force of the
discrete gradient of Kormann and Sonnendrücker {cite}`kormann2021`. The fixed-point iteration
is a `lax.scan` of a fixed length, so that the whole loop stays differentiable.

## Why bother

The explicit leapfrog is fast and its energy error is bounded, but it is not zero, and
it grows with $\omega_{pe}\Delta t$. Two situations make that a problem: long runs
where a slow energy drift competes with the physics being studied, and stiff problems
where the explicit stability limits force a step far below the timescale of interest.
The linear periodic vacuum update is stable at any time step. A plasma still requires
a converged nonlinear solve, and a large step can lose phase and orbit accuracy even
when its energy is conserved.

## The discrete equations

The fields are centred at $t^{n+1/2}$. With
$\mathbf E^{n+1/2} = \tfrac12(\mathbf E^n + \mathbf E^{n+1})$ and likewise for $\mathbf B$,

```{math}
\frac{\mathbf E^{n+1} - \mathbf E^n}{\Delta t} = c^2\nabla\times\mathbf B^{n+1/2} - \frac{\mathbf J^{n+1/2}}{\epsilon_0}, \qquad
\frac{\mathbf B^{n+1} - \mathbf B^n}{\Delta t} = -\nabla\times\mathbf E^{n+1/2},
```

and $\mathbf J^{n+1/2}$ is the mean of the currents of `substeps` particle sub-steps of
$\Delta\tau = \Delta t/N_\nu$. In each, a particle moves on the straight line from $x_p^\nu$ to
$x_p^{\nu+1} = x_p^\nu + \Delta\tau\,\bar v_{x,p}$ and is pushed by the Boris step,

```{math}
:label: implicit-push
\frac{\mathbf u_p^{\nu+1} - \mathbf u_p^\nu}{\Delta\tau} = \frac{q_p}{m_p}\left(\mathbf E_p + \bar{\mathbf v}_p\times\mathbf B(x_p^{\nu+1/2})\right), \qquad
\bar{\mathbf v}_p = \frac{\mathbf u_p^\nu + \mathbf u_p^{\nu+1}}{\gamma_p^\nu + \gamma_p^{\nu+1}},
```

with $\mathbf u = \gamma\mathbf v$ and $\gamma = 1$ in a Newtonian run, where $\bar{\mathbf v}$ is the
mean of the two velocities. The Boris step changes $|\mathbf u|^2$ by exactly
$(q/m)\Delta\tau\,\mathbf E_p\cdot(\mathbf u^\nu + \mathbf u^{\nu+1})$, so the kinetic energy,
$\tfrac12 m|\mathbf v|^2$ or $(\gamma - 1)mc^2$, changes by $q\,\mathbf E_p\cdot\bar{\mathbf v}\,\Delta\tau$
to round-off, and the magnetic field does no work. What remains is to choose $\mathbf E_p$
and $\mathbf J$ so that this work is what the current takes from the field, which is where
the transverse and the longitudinal components part ways.

**Transverse.** $E_y$, $E_z$ and $\mathbf B$ are gathered at the mid-point
$x_p^{\nu+1/2}$, from the centres with the deposit's spline ({doc}`boundaries`), and
$J_y$, $J_z$ are the transpose of that gather applied to $q_p\bar v_{y,p}$, $q_p\bar v_{z,p}$.
The code obtains it from `jax.vjp` of the gather, the transpose by construction whatever
the walls.

**Longitudinal current.** $J_x$ is the continuity current of the deposits at the two ends
of the sub-step, with the weights the walls leave,

```{math}
:label: implicit-continuity
\frac{\rho_i(x^{\nu+1}) - \rho_i(x^\nu)}{\Delta\tau} + \frac{J^\nu_{x,i+1/2} - J^\nu_{x,i-1/2}}{\Delta x} = 0,
```

the current the explicit scheme takes ({doc}`deposition`), closed at the walls in the same
way, for a particle that crosses any number of cells, a wall or a periodic boundary.

**Longitudinal force.** {eq}`implicit-continuity` makes $J_x^\nu = \mathcal T(\rho^{\nu+1} - \rho^\nu)
+ \langle J\rangle$ with $\mathcal T$ linear, and $\rho = \mathcal D(x, qw)$ is linear in the charges.
The work the current takes from the field is then

```{math}
\Delta\tau\,\Delta x\sum_i E_{x,i+1/2} J^\nu_{x,i+1/2}
= \sum_p q_p w_p\left[\Phi(x_p^{\nu+1}) - \Phi(x_p^\nu)\right] + \langle E_x\rangle\sum_p q_p w_p \Delta\tau\,\bar v_{x,p},
\qquad \Phi = \Delta x\,\mathcal D^{\mathsf T}\,\mathcal T^{\mathsf T} E_x,
```

where the transposes turn $E_x$ into a potential $\Phi$ at the particles, interpolated with
the spline and the walls of the deposit, and the last term is the work of the mean current
of a periodic box. The force that does exactly this work is the discrete gradient

```{math}
:label: discrete-gradient
E_{x,p} = \frac{\Phi(x_p^{\nu+1}) - \Phi(x_p^\nu)}{\Delta\tau\,\bar v_{x,p}} + \langle E_x\rangle ,
```

the one-dimensional form of the line integral of Kormann and Sonnendrücker
{cite}`kormann2021`: $\Phi$ integrates the $S_1$ face interpolant for quadratic weights
or $S_4$ for quintic weights, and its difference is the orbit integral in closed form. The displacement in the
denominator is the unwrapped one, and $\Phi$ is read where the deposit puts the particle,
mirrored or wrapped. `shape_order=5` selects the same quintic charge weights for the
endpoint deposit and its transposed potential, with periodic particle and field boundaries.
Its potential slope is the quartic face-field interpolant; `shape_order=2` retains the
linear interpolant. Transverse gathering and its current use the selected spline too.
This is an optional particle shape within the existing implicit algorithm, not the complete
SHARP field/interpolation method. Continuity and the work identity retain their algebraic
form; quintic smoothing does not imply exact continuous momentum or converged physical phase.
The figures and measured comparisons below retain the default quadratic weighting.

When the displacement is below $\sqrt\epsilon$ of a cell, with
$\epsilon$ the machine epsilon, the quotient has lost half its digits and $E_{x,p}$ is the
slope $\Phi'(x_p^{\nu+1/2})$ instead. For quadratic weights it equals the quotient while
both ends lie in one spline piece; for quintic weights it differs by
$O((\Delta x_p/\Delta x)^2)$ even within a piece. At this threshold that local truncation
is on the machine-epsilon scale. Knot crossings, cancellation and finite Picard closure
still need separate checks. The code writes none of these operators out: $\mathcal T^{\mathsf T}$
and $\mathcal D^{\mathsf T}$ are `jax.linear_transpose` of `current_from_continuity` and of
`deposit`, and the slope is `jax.jvp` of $\Phi$.

### Optional short-orbit precision

`Solver(algorithm="implicit", orbit_force="integral")` evaluates the same longitudinal
force by a polynomial integral for periodic particle sub-steps with
$|\ell|=|\Delta\tau\bar v_x|\leq\Delta x$. The default `orbit_force="secant"`, all
nonperiodic runs and longer sub-steps retain the potential calculation above.
The integral uses the local starting cell and the nominal unwrapped displacement:

```{math}
\bar E_p=\int_0^1 F(x_p+s\ell)\,ds.
```

Here $F$ is the linear ($S_1$) or quartic ($S_4$) face interpolant, including its mean.
The orbit is split at spline knots. A linear piece averages to its midpoint value;
a quartic piece of signed length $d$ averages to
$F(x_m)+d^2 F''(x_m)/24+d^4 F^{(4)}(x_m)/1920$.
This avoids subtracting nearby potentials or first rounding a global endpoint.
At zero displacement its derivative is $F'(x_p)/2$ wherever $F$ is differentiable;
a quadratic particle shape has a one-sided limit at a face-field knot.

Charge deposition, current, transverse force and Picard iteration are unchanged.
Rounded endpoint deposition and finite iteration still require independent work,
continuity and phase checks. This improves force precision; it does not restore
continuous momentum conservation or establish long-time kinetic accuracy.
Write `save_state(path, state, simulation=sim)` to retain the selected force and shape:
these integral-force archives use format 3, which secant-only readers reject.
A bare `State` carries no solver metadata; saving it without `simulation` writes
format 1 and cannot identify the selected force. Default archives retain their existing formats.

## Why both laws hold

**Charge.** Summed over the sub-steps, {eq}`implicit-continuity` telescopes, and Ampere's law
advances $E_x$ by $-\mathcal T(\rho^{n+1} - \rho^n)/\epsilon_0$, the change of the Gauss
solve with the same wall closures. The initial field solves the Gauss law, so every later
one does, to round-off and at every wall, whether or not the Picard iteration has
converged, since the field and the density a step returns come from the same orbit.

**Energy.** Take the discrete field energy $W_F = \tfrac12\sum_i(\epsilon_0|\mathbf E_i|^2 + |\mathbf B_i|^2/\mu_0)\Delta x$.
Dotting the field equations with $\mathbf E^{n+1/2}$ and $\mathbf B^{n+1/2}$ and summing
over the grid,

```{math}
\frac{W_F^{n+1} - W_F^n}{\Delta t} = -\Delta x\sum_i \mathbf E_i^{n+1/2}\cdot\mathbf J_i^{n+1/2},
```

in a periodic box, where the staggered differences are exact adjoints. Nonperiodic
ghosts also contribute boundary work; their stored field energy need not be constant
({doc}`field_solvers`). The particles gain $\sum_p q_p w_p\mathbf E_p\cdot\bar{\mathbf v}_p\Delta\tau$ per
sub-step, from {eq}`implicit-push`: transversely that is the transpose of the gather, and
longitudinally {eq}`discrete-gradient`, so the two cancel, once the orbit the force was
computed on is the orbit the particles move on — which is what the Picard iteration
converges to. A wall that collects, re-emits or slows particles changes the energy
physically, and in those runs only the Gauss law is left to check.

**Momentum** is not conserved. The discrete gradient, unlike the centred gather of the
explicit scheme, does not make the force between two particles antisymmetric, as in Chen,
Chacón and Barnes {cite}`chen2011`: over the two-stream run the momentum error reaches
{{ momentum_error_implicit }}, against {{ momentum_error_relative }} for the explicit scheme.

## The electrostatic model

`Solver(algorithm="implicit", model="electrostatic")` keeps everything above for $E_x$ and
drops the rest: $E_y$, $E_z$ and $\mathbf B$ are not evolved (external fields still act), and
$E_x$ is advanced by Ampere's law with the same continuity current,

```{math}
:label: implicit-electrostatic
E_{x,i+1/2}^{n+1} = E_{x,i+1/2}^{n} - \frac{\Delta t}{\epsilon_0}\left(J^{n+1/2}_{x,i+1/2} - \langle J_x^{n+1/2}\rangle\right),
```

with the mean $\langle J_x\rangle$ subtracted in a periodic box only. $E_x$ is **not** re-solved
from $\rho^{n+1}$ afterwards: the Gauss law already holds by the telescoping argument above,
and a projection would replace the field whose work the discrete gradient balances.
Subtracting the mean current is the convention of the explicit electrostatic model, where the
Gauss solve returns $\langle E_x\rangle = 0$: the initial field has zero mean, so every later one
does, and the uniform plasma oscillation a net current would drive is absent. The energy
balance is unchanged, since $\sum_i E_{x,i}\langle J_x\rangle = N\langle E_x\rangle\langle J_x\rangle = 0$
and the mean-current term of {eq}`discrete-gradient` vanishes with $\langle E_x\rangle$. Between
walls the current is the wall-closed one and nothing is subtracted.

On the two-stream run of the tests (2000 electrons and ions, 64 cells, 150 steps at
$c\Delta t/\Delta x = 4.5$) the energy error is $2.8\times10^{-11}$ after 4 Picard iterations and
$2.3\times10^{-16}$ after 8 and 12, the Gauss residual $\le 2\times10^{-15}$ and
$|\langle E_x\rangle|/\max|E_x| \le 6\times10^{-17}$ at every step.

## Solving the system

For a prescribed current the Maxwell equations are linear. Eliminating midpoint
$\mathbf B$ gives a transverse Helmholtz system. In a periodic box,

```{math}
\left(I+\frac{c^2\Delta t^2}{4}\,\operatorname{curl}_B\operatorname{curl}_E\right)\mathbf E^{n+1/2}
=\mathbf E^n+\frac{\Delta t}{2}\left(c^2\operatorname{curl}_B\mathbf B^n-\mathbf J/\epsilon_0\right).
```

Its Fourier denominator is $1+C^2\sin^2(k\Delta x/2)$, with $C=c\Delta t/\Delta x$;
the code inverts it directly using the discrete curl symbol, not the continuum $ik$
{cite}`kormann2021`. Reflective, absorbing and mixed walls give a tridiagonal system.
The radiating ghost depends on midpoint $\mathbf B$, which is solved together with
$\mathbf E$. Thus vacuum propagation requires no particle iteration to converge.

The particle-field coupling remains nonlinear. Starting from the old fields and
every particle streaming freely at its present velocity, each Picard iteration:

1. forms midpoint fields from the previous iterate;
2. sub-step the particles in those fields, with $\mathbf E_p$ from the orbit of the previous
   iteration, accumulating the current of the new orbit;
3. solves the linear Maxwell system for that current;

repeated `picard_iterations` times. The state the step returns is the last iteration's
particles, field and density, so that the Gauss law holds exactly. The particles are
advanced in `substeps` sub-steps per field step, which resolves orbits that turn inside
one field step without refining the field grid.

:::{important}
The end and the mean velocity of every sub-step are carried from one Picard iteration to
the next, rather than recomputed from the start of the step. The force of the next iteration
is computed on that orbit, so at convergence the force and the current share it, which is
the condition for the energy to be exact.
:::

A filter keeps both laws only if it acts symmetrically on the current and on the gathered
field {cite}`chen2011`; that pair is not implemented, so the implicit scheme refuses
`filter_passes`, and `field_solver="gauss"`, which would overwrite the $E_x$ the energy
balance needs; `model="electrostatic"` does not, as above.

## Convergence

The Gauss law holds at every iteration count. For the verification run, the energy
error falls geometrically with the count until it reaches round-off:

| Picard iterations | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| energy error | {{ energy_error_max_implicit_1 }} | {{ energy_error_max_implicit_2 }} | {{ energy_error_max_implicit_4 }} | {{ energy_error_max_implicit_8 }} |

```{figure} ../_static/figures/conservation.png
:width: 100%
:alt: Energy, momentum and Gauss-law errors of the explicit scheme and of the implicit scheme at 1, 2, 4 and 8 Picard iterations

(a) Total energy error against time. The implicit curves are one Picard iteration
apart; at eight the error is at the round-off of double precision.
(b) Momentum error. (c) The Gauss residual, at round-off for both schemes, in a periodic
box and between absorbing walls ({doc}`../examples/conservation`).
```

The default is eight; it is not a convergence guarantee. Check fields, particle
observables and gradients against a larger iteration count and a smaller time step.
Stiff particle coupling can make Picard diverge, in which case more iterations do
not repair the step. Substeps resolve particle orbits but do not certify convergence
of the field-particle solve.

Periodic vacuum modes have unit amplification modulus and phase advance
$2\arctan[C\sin(k\Delta x/2)]$, independently of `picard_iterations`. This is the
Crank-Nicolson solution of the discrete Maxwell equations, not exact continuum
propagation. Time-step refinement recovers the semi-discrete wave at second order.
A fixed iteration count, rather than a tolerance and a `while` loop, is a deliberate
choice: `lax.while_loop` has no reverse-mode derivative, so a tolerance-based solver
would not have that derivative directly. With a fixed count, `jax.grad` differentiates
the executed finite solve; it represents the converged scheme only after iteration
and time-step checks. See {doc}`../user_guide/differentiation`.

## When to use which

| | explicit | implicit |
|---|---|---|
| cost per step | 1 | depends on grid, substeps and iteration count |
| energy error | {{ energy_error_max_explicit }}, bounded | {{ energy_error_max_implicit_8 }} |
| momentum error, periodic | {{ momentum_error_relative }} | {{ momentum_error_implicit }} |
| Gauss law | round-off | round-off |
| stability | $\omega_{pe}\Delta t \lesssim 2$, $c\Delta t < \Delta x$ for general light waves | periodic vacuum: any step; plasma: converged solve required |
| physical grid resolution | requires independent refinement | requires independent refinement |
| reverse-mode gradients | yes | yes |

Start explicit. Move to implicit when the energy budget matters, when the step you
want breaks an explicit stability limit, or when the Debye length is too small to
resolve.
