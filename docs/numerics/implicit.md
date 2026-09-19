# Implicit scheme

`Solver(algorithm="implicit")` selects a Crank-Nicolson scheme that conserves the discrete
total energy and the discrete Gauss law to round-off: the energy-conserving scheme of Chen,
Chacón and Barnes {cite}`chen2011,chen2014`, with the longitudinal current and force of the
discrete gradient of Kormann and Sonnendrücker {cite}`kormann2021`. The fixed-point iteration
is a `lax.scan` of a fixed length, so that the whole loop stays differentiable.

## Why bother

The explicit leapfrog is fast and its energy error is bounded, but it is not zero, and
it grows with $\omega_{pe}\Delta t$. Two situations make that a problem: long runs
where a slow energy drift competes with the physics being studied, and stiff problems
where the explicit stability limits force a step far below the timescale of interest.
The Crank-Nicolson scheme is unconditionally stable and conserves the discrete total
energy exactly, at the price of solving a nonlinear system every step.

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
$2(q/m)\Delta\tau\,\mathbf E_p\cdot(\mathbf u^\nu + \mathbf u^{\nu+1})$, so the kinetic energy,
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
{cite}`kormann2021`: $\Phi$ is the integral of the $S_1$ interpolant of $E_x$, and its
difference is their integral along the orbit, in closed form. The displacement in the
denominator is the unwrapped one, and $\Phi$ is read where the deposit puts the particle,
mirrored or wrapped. When the displacement is below $\sqrt\epsilon$ of a cell, with
$\epsilon$ the machine epsilon, the quotient has lost half its digits and $E_{x,p}$ is the
slope $\Phi'(x_p^{\nu+1/2})$ instead, which equals the quotient while both ends lie in
one piece of the spline and otherwise differs from it by the square of the displacement
in cells, the round-off. The code writes none of these operators out: $\mathcal T^{\mathsf T}$
and $\mathcal D^{\mathsf T}$ are `jax.linear_transpose` of `current_from_continuity` and of
`deposit`, and the slope is `jax.jvp` of $\Phi$.

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

the curl terms cancelling because the staggered difference operators are exact adjoints of
one another. The particles gain $\sum_p q_p w_p\mathbf E_p\cdot\bar{\mathbf v}_p\Delta\tau$ per
sub-step, from {eq}`implicit-push`: transversely that is the transpose of the gather, and
longitudinally {eq}`discrete-gradient`, so the two cancel, once the orbit the force was
computed on is the orbit the particles move on — which is what the Picard iteration
converges to. A wall that collects, re-emits or slows particles changes the energy
physically, and in those runs only the Gauss law is left to check.

**Momentum** is not conserved. The discrete gradient, unlike the centred gather of the
explicit scheme, does not make the force between two particles antisymmetric, as in Chen,
Chacón and Barnes {cite}`chen2011`: over the two-stream run the momentum error reaches
{{ momentum_error_implicit }}, against {{ momentum_error_relative }} for the explicit scheme.

## Solving the system

The equations are nonlinear because the orbit depends on the field and the field on
the orbit. The code uses Picard iteration: starting from $\mathbf E^{n+1} = \mathbf E^n$ and
from every particle streaming freely at its present velocity,

1. form $\mathbf E^{n+1/2}$ and, from Faraday, $\mathbf B^{n+1/2}$;
2. sub-step the particles in those fields, with $\mathbf E_p$ from the orbit of the previous
   iteration, accumulating the current of the new orbit;
3. update $\mathbf E^{n+1}$ from Ampere;

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
balance needs.

## Convergence

The Gauss law holds at every iteration count. The energy error falls geometrically with
the count until it reaches round-off:

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

The default is eight, which reaches round-off for the problems in {doc}`verification`
while costing about {{ scaling_implicit_over_explicit }} times an explicit step.
A fixed iteration count, rather than a tolerance and a `while` loop, is a deliberate
choice: `lax.while_loop` has no reverse-mode derivative, so a tolerance-based solver
would not be differentiable. With a fixed count the whole scheme is, and
`jax.grad` runs through it; see {doc}`../user_guide/differentiation`.

## When to use which

| | explicit | implicit |
|---|---|---|
| cost per step | 1 | about {{ scaling_implicit_over_explicit }} |
| energy error | {{ energy_error_max_explicit }}, bounded | {{ energy_error_max_implicit_8 }} |
| momentum error, periodic | {{ momentum_error_relative }} | {{ momentum_error_implicit }} |
| Gauss law | round-off | round-off |
| stability | $\omega_{pe}\Delta t \lesssim 2$, $c\Delta t \le \Delta x$ for light waves | unconditional |
| grid resolution | $\Delta x \lesssim \lambda_D$ | can exceed $\lambda_D$ |
| reverse-mode gradients | yes | yes |

Start explicit. Move to implicit when the energy budget matters, when the step you
want breaks an explicit stability limit, or when the Debye length is too small to
resolve.
