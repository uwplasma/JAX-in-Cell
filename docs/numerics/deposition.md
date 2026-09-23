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
* **Walls.** The constant follows the same closure as the Gauss solve of
  {doc}`field_solvers`, applied to $-\partial_t\rho$ in place of $\rho/\epsilon_0$: no
  current through a reflective wall, the sum run from that wall, and between two
  absorbing walls no change in the potential drop, the rest being the current in the
  external circuit ({doc}`boundaries`). Sharing the closure is what keeps the field
  Ampere's law advances on the one the initial Gauss solve chose.

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
:alt: Energy, momentum and Gauss-law errors of the explicit and implicit schemes

(a) Total energy error. (b) Momentum error. (c) The Gauss-law residual, at round-off for
both schemes in a periodic box and between absorbing walls: the implicit scheme takes the
same charge-conserving current, from the two ends of every sub-step ({doc}`implicit`).
```

## Momentum conservation

The field is gathered with the same $S_2$ the charge was deposited with, and from the same
grid: $E_x$ is first averaged from the faces to the centres,
$E_i = \tfrac12(E_{i-1/2} + E_{i+1/2}) = -(\phi_{i+1} - \phi_{i-1})/2\Delta x$, a centred
difference. The gather is then the transpose of the deposit composed with an antisymmetric
operator, so the force between two pseudo-particles is antisymmetric and a particle exerts
none on itself {cite}`birdsall1991`: in a periodic electrostatic run the total momentum is
conserved to round-off. Over the same run it drifts by {{ momentum_error_relative }} of
$\sum_p m_p|v_{x,p}|$.

The price is paid in energy. An explicit particle-in-cell scheme can conserve momentum or,
to the order of its time step, energy, but not both: the centred gather does the first, a
gather consistent with the staggered current the second {cite}`birdsall1991`. The
two-stream run of {doc}`explicit` changed its total energy by $4\times10^{-5}$ when the
field was gathered from the faces, and changes it by {{ energy_error_max_explicit }} now,
still bounded over the run. A run that needs the energy exact should use the
{doc}`implicit` scheme, which conserves it and the Gauss law to round-off at any wall and
pays with the momentum, {{ momentum_error_implicit }} over the same run.

A code that gathers with a different shape than it deposits, or that interpolates the
field to the particle from a different grid than the one the charge was written to,
loses this property and develops a self-force: a single particle in an empty periodic
box accelerates itself. Gathering $E_x$ directly from the faces with $S_2$ is such a
case, and this code did it until the self-force was measured: up to 8 % of the particle's
own field, depending on where in its cell it sat, and a momentum drift of $2\times10^{-5}$
over the run above. The test suite checks the self-force, the image forces at every wall
({doc}`boundaries`) and the momentum budget directly.

## Cost

Each particle touches three cells, so one deposit is $3N$ scatter-adds and one gather
is $3N$ reads, independent of $N_x$. A step performs two charge deposits and four
transverse-current deposits, and two gathers, one for $\mathbf E$ and one for
$\mathbf B$. Two charge deposits and not four: the density at $t^n$ is the one the
previous step ended on, carried in the state, and the density at $t^{n+1/2}$ that the
second half step starts from is the one the first half step ended on. Reusing the first
only saves a deposit; reusing the second is necessary, for the reason given under
{doc}`boundaries`.
Measured on one CPU core, the explicit scheme costs
{{ scaling_ns_per_particle_step }} ns per particle per step at
{{ scaling_particles_max }} particles; see {doc}`../user_guide/performance`.
