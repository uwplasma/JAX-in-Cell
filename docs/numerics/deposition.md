# Deposition and charge conservation

The sources that Maxwell's equations need are the charge density $\rho$ and the
current density $\mathbf J$. Depositing them naively breaks Gauss's law; this page
derives the deposit the code actually uses and shows why the law then holds to
round-off.

## Why the obvious current is not good enough

The particle representation of {doc}`equations` suggests

```{math}
:label: naive-current
J_{x,i} = \sum_p q_p v_{x,p}\, S_2\!\left(\frac{x_p - x_i}{\Delta x}\right).
```

Take the divergence of Ampere's law and subtract the time derivative of Gauss's law:

```{math}
\frac{\partial}{\partial t}\left(\nabla\!\cdot\!\mathbf E - \frac{\rho}{\epsilon_0}\right)
= -\frac{1}{\epsilon_0}\left(\nabla\!\cdot\!\mathbf J + \frac{\partial\rho}{\partial t}\right).
```

In the continuum the right-hand side vanishes identically. On the grid it does not:
{eq}`naive-current` evaluates the shape function at $x_p$, while $\rho$ changes because
the *whole cloud* moved, and the two differ at second order in $v\Delta t/\Delta x$.
The error accumulates, so $\nabla\!\cdot\!\mathbf E - \rho/\epsilon_0$ drifts linearly
in time and the longitudinal field acquires an unphysical offset. Codes that deposit
{eq}`naive-current` therefore need a divergence-cleaning or Poisson-correction step
every so often.

## The current that conserves charge exactly

Instead of asking what $\mathbf J$ *is*, ask what it must be for the discrete
continuity equation

```{math}
:label: discrete-continuity
\frac{\rho_i^{n+1/2} - \rho_i^{n-1/2}}{\Delta t}
+ \frac{J_{x,i+1/2} - J_{x,i-1/2}}{\Delta x} = 0
```

to hold cell by cell. This is the one-dimensional case of the Villasenor-Buneman
{cite}`villasenor1992` and Esirkepov {cite}`esirkepov2001` deposits, where it can be
solved in closed form. Summing {eq}`discrete-continuity` from the left wall gives

```{math}
:label: cumsum-current
J_{x,i+1/2} = J_{x,-1/2} - \frac{\Delta x}{\Delta t}\sum_{j\le i}\left(\rho_j^{n+1/2} - \rho_j^{n-1/2}\right),
```

a single cumulative sum over the grid. In the code this is
{func}`~jaxincell._core.current_from_continuity`, three lines long: deposit $\rho$ at
the two half-step positions, difference, `cumsum`.

The integration constant $J_{x,-1/2}$ is the current through the left wall:

* **Periodic walls.** There is no wall, so the constant is fixed instead by requiring
  that the mean of $J_x$ over the box equal the mean current carried by the particles,
  $\langle J_x\rangle = L^{-1}\sum_p q_p v_{x,p}$. Without this the uniform part of the
  current, which a cumulative sum cannot see, would be lost and a net beam would not
  drive the field it should.
* **Reflective or absorbing walls.** Nothing crosses the wall, so $J_{x,-1/2} = 0$ and
  the cumulative sum is taken as it stands.

Because the shape functions are compact, $\rho^{n+1/2} - \rho^{n-1/2}$ has zero sum
over the grid in a periodic box, so the cumulative sum returns to its starting value
and no discontinuity appears at the wrap point.

The transverse currents carry no divergence in one dimension, so they are deposited
directly as {eq}`naive-current` with $v_y$ and $v_z$, at the cell centres, then
averaged onto the faces where $E_y$ and $E_z$ live.

## What this buys

With {eq}`discrete-continuity` satisfied and the initial field taken from Gauss's law,
advancing $E_x$ with Ampere's law preserves

```{math}
\frac{E_{x,i+1/2} - E_{x,i-1/2}}{\Delta x} = \frac{\rho_i}{\epsilon_0}
```

for all time, up to floating-point round-off. Measured over the two-stream run of
{doc}`verification`, the largest relative residual is
{{ gauss_residual_max_explicit }}, and the charge on the grid matches the charge on
the particles to {{ charge_error_relative }} of the total. No correction step, no
Poisson solve inside the loop.

```{figure} ../_static/figures/conservation.png
:width: 100%
:alt: Energy error and Gauss-law residual for the explicit and implicit schemes

(a) Total energy error. (b) The Gauss-law residual stays at round-off for both
schemes because both deposit the charge-conserving current.
```

## Momentum conservation

Using the same $S_2$ for the deposit and the gather makes the force between two
pseudo-particles antisymmetric, so the interpolation contributes nothing to the total
momentum {cite}`birdsall1991`. What is left is the residual of the staggered field
solve, which is small but not zero: over the same run the total particle momentum
drifts by {{ momentum_error_relative }} of $\sum_p m_p|v_{x,p}|$.

A code that gathers with a different shape than it deposits, or that interpolates the
field to the particle from a different grid than the one the charge was written to,
loses this property and develops a self-force: a single particle in an empty periodic
box accelerates itself. The test suite checks the momentum budget directly.

## Cost

Each particle touches three cells, so one deposit is $3N$ scatter-adds and one gather
is $3N$ reads, independent of $N_x$. A step performs three charge deposits and four
transverse-current deposits, and two gathers, one for $\mathbf E$ and one for
$\mathbf B$. Three charge deposits and not four: the density at $t^{n+1/2}$ that the
second half step starts from is the one the first half step ended on, and reusing it
is not only cheaper but necessary, for the reason given under
{doc}`boundaries`.
Measured on one CPU core, the explicit scheme costs
{{ scaling_ns_per_particle_step }} ns per particle per step at
{{ scaling_particles_max }} particles; see {doc}`../user_guide/performance`.
