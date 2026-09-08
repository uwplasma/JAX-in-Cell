# Implicit scheme

`Solver(algorithm="implicit")` selects the energy-conserving Crank-Nicolson scheme of
Chen, Chacon and Barnes {cite}`chen2011`, with the fixed-point iteration written as a
`lax.scan` of a fixed length so that the whole loop stays differentiable.

## Why bother

The explicit leapfrog is fast and its energy error is bounded, but it is not zero, and
it grows with $\omega_{pe}\Delta t$. Two situations make that a problem: long runs
where a slow energy drift competes with the physics being studied, and stiff problems
where the explicit stability limits force a step far below the timescale of interest.
The Crank-Nicolson scheme is unconditionally stable and conserves the discrete total
energy exactly, at the price of solving a nonlinear system every step.

## The discrete equations

Everything is centred at $t^{n+1/2}$. With
$\mathbf E^{n+1/2} = \tfrac12(\mathbf E^n + \mathbf E^{n+1})$ and likewise for
$\mathbf B$ and for the particle quantities,

```{math}
\frac{\mathbf E^{n+1} - \mathbf E^n}{\Delta t} = c^2\nabla\times\mathbf B^{n+1/2} - \frac{\mathbf J^{n+1/2}}{\epsilon_0}, \qquad
\frac{\mathbf B^{n+1} - \mathbf B^n}{\Delta t} = -\nabla\times\mathbf E^{n+1/2},
```
```{math}
\frac{x_p^{n+1} - x_p^n}{\Delta t} = v_{x,p}^{n+1/2}, \qquad
\frac{\mathbf v_p^{n+1} - \mathbf v_p^n}{\Delta t} = \frac{q_p}{m_p}\left(\mathbf E(x_p^{n+1/2}) + \mathbf v_p^{n+1/2}\times\mathbf B(x_p^{n+1/2})\right),
```

with the current deposited from the same mid-point orbit that the fields are gathered
at,

```{math}
:label: orbit-current
\mathbf J^{n+1/2}_{i+1/2} = \sum_p q_p \mathbf v_p^{n+1/2}\, S_2\!\left(\frac{x_p^{n+1/2} - x_{i+1/2}}{\Delta x}\right).
```

## Why this conserves energy

Take the discrete field energy $W_F = \tfrac12\sum_i(\epsilon_0|\mathbf E_i|^2 + |\mathbf B_i|^2/\mu_0)\Delta x$
and the particle energy $W_P = \tfrac12\sum_p m_p|\mathbf v_p|^2$. Dotting the field
equations with $\mathbf E^{n+1/2}$ and $\mathbf B^{n+1/2}$ and summing over the grid,

```{math}
\frac{W_F^{n+1} - W_F^n}{\Delta t} = -\Delta x\sum_i \mathbf E_i^{n+1/2}\cdot\mathbf J_i^{n+1/2},
```

the curl terms cancelling because the staggered difference operators are exact
adjoints of one another. Dotting the velocity equation with $\mathbf v_p^{n+1/2}$ kills
the magnetic term ($\mathbf v\times\mathbf B \perp \mathbf v$) and gives

```{math}
\frac{W_P^{n+1} - W_P^n}{\Delta t} = \sum_p q_p\,\mathbf v_p^{n+1/2}\cdot\mathbf E(x_p^{n+1/2}).
```

The two right-hand sides cancel **provided the same shape function, at the same
mid-point position, is used to gather $\mathbf E$ and to deposit $\mathbf J$** — which
is exactly {eq}`orbit-current`. That condition is the whole content of the scheme, and
it is what makes the energy error vanish rather than merely stay bounded.

## Solving the system

The equations are nonlinear because the orbit depends on the field and the field on
the orbit. The code uses Picard iteration: starting from $\mathbf E^{n+1} = \mathbf E^n$,

1. form $\mathbf E^{n+1/2}$ and, from Faraday, $\mathbf B^{n+1/2}$;
2. sub-step the particles in those fields, accumulating {eq}`orbit-current`;
3. update $\mathbf E^{n+1}$ from Ampere;

repeated `picard_iterations` times, then once more to produce the state. The particles
are advanced in `substeps` sub-steps per field step, which resolves orbits that turn
inside one field step without refining the field grid.

:::{important}
The mid-point positions of every sub-step are carried from one Picard iteration to the
next, rather than recomputed from the start of the step. Recomputing them leaves the
gather and the deposit on slightly different orbits, and the energy error then sticks
at $4\times10^{-4}$ however many iterations are taken. Carrying them makes the
iteration converge properly.
:::

## Convergence

The energy error falls geometrically with the iteration count until it reaches
round-off:

| Picard iterations | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| energy error | {{ energy_error_max_implicit_1 }} | {{ energy_error_max_implicit_2 }} | {{ energy_error_max_implicit_4 }} | {{ energy_error_max_implicit_8 }} |

```{figure} ../_static/figures/conservation.png
:width: 100%
:alt: Energy error of the explicit scheme and of the implicit scheme at 1, 2, 4 and 8 Picard iterations

(a) Total energy error against time. The implicit curves are one Picard iteration
apart; at eight the error is at the round-off of double precision.
(b) The Gauss residual, at round-off for both schemes.
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
| stability | $\omega_{pe}\Delta t \lesssim 2$, $c\Delta t \le \Delta x$ for light waves | unconditional |
| grid resolution | $\Delta x \lesssim \lambda_D$ | can exceed $\lambda_D$ |
| reverse-mode gradients | yes | yes |

Start explicit. Move to implicit when the energy budget matters, when the step you
want breaks an explicit stability limit, or when the Debye length is too small to
resolve.
