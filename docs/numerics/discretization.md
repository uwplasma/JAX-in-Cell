# Discretisation

## Grid

The box $[-L/2, L/2]$ is divided into $N_x$ cells of size $\Delta x = L/N_x$ with
centres $x_i = -L/2 + (i + \tfrac12)\Delta x$. Quantities live on two interleaved sets
of points, following the Yee arrangement {cite}`yee1966`:

* cell centres $x_i$: the charge density $\rho$ and the magnetic field $\mathbf B$;
* cell faces $x_{i+1/2}$: the electric field $\mathbf E$ and the whole current
  density $\mathbf J$.

The transverse currents are deposited at the centres, where the particle shape
function is defined, and averaged onto the faces, $J_{i+1/2} = (J_i + J_{i+1})/2$,
so that they sit where $E_y$ and $E_z$ do. Depositing them at the centres and
adding them to a face-centred field would misplace them by half a cell.

```{figure} ../_static/figures/staggered_grid.png
:width: 100%
:alt: Staggered grid with cell centres and cell faces

Staggered grid. All output arrays have $N_x$ entries; entry $i$ of a face-centred
quantity refers to $x_{i+1/2}$.
```

With this arrangement the two curls become centred differences of second order:

```{math}
(\nabla\times\mathbf E)_i = \left(0,\; -\frac{E_{z,i+1/2} - E_{z,i-1/2}}{\Delta x},\; \frac{E_{y,i+1/2} - E_{y,i-1/2}}{\Delta x}\right), \qquad
(\nabla\times\mathbf B)_{i+1/2} = \left(0,\; -\frac{B_{z,i+1} - B_{z,i}}{\Delta x},\; \frac{B_{y,i+1} - B_{y,i}}{\Delta x}\right).
```

The values needed beyond the first and last cell come from ghost cells whose content
depends on the boundary condition, see {doc}`boundaries`.

## Time levels

The explicit scheme is a leapfrog: velocities are known at integer times $t^n = n\Delta t$
and positions at half-integer times $t^{n+1/2}$. The fields are advanced in two half
steps so that they are available at $t^{n+1/2}$ when the particles are pushed.

```{figure} ../_static/figures/time_staggering.png
:width: 100%
:alt: Time levels of positions, velocities, fields and currents in the explicit scheme

Time levels of the explicit scheme. The current $\mathbf J^n$ is computed from the
motion between $x^{n-1/2}$ and $x^{n+1/2}$ and is therefore centred at $t^n$.
```

The implicit scheme keeps positions, velocities and fields all at integer times and
uses their averages at $t^{n+1/2}$ inside the iteration, see {doc}`implicit`.

## Shape function

Charge assignment and field interpolation use the quadratic B-spline (the
"triangular-shaped cloud" of Hockney and Eastwood {cite}`hockney1988`), which spreads
each pseudo-particle over three cells:

```{math}
S_2(\xi) = \frac{1}{\Delta x}\begin{cases}
\dfrac34 - \xi^2, & |\xi| \le \dfrac12,\\[6pt]
\dfrac12\left(\dfrac32 - |\xi|\right)^2, & \dfrac12 < |\xi| \le \dfrac32,\\[6pt]
0, & \text{otherwise},
\end{cases}
\qquad \xi = \frac{x - x_i}{\Delta x}.
```

```{figure} ../_static/figures/shape_functions.png
:width: 100%
:alt: Shape functions of order 0, 1 and 2, and the S2 weights for one particle

(a) The nearest-grid-point, linear and quadratic shape functions. (b) The three
weights that $S_2$ assigns to a particle at $x_i + 0.3\,\Delta x$; they sum to one for
any position.
```

The weights are continuous together with their first derivative as the particle
crosses a cell boundary, which makes the self-force and the grid noise smaller than
with linear weighting, at the price of three grid accesses per particle instead of
two. The same $S_2$ is used to gather the fields at the particle:

```{math}
\mathbf E(x_p) = \Delta x\sum_i \mathbf E_{i+1/2}\,S_2\!\left(\frac{x_p - x_{i+1/2}}{\Delta x}\right), \qquad
\mathbf B(x_p) = \Delta x\sum_i \mathbf B_i\,S_2\!\left(\frac{x_p - x_i}{\Delta x}\right),
```

each with its own set of three points. Using one and the same kernel for deposit and
gather makes the interaction between two pseudo-particles antisymmetric, so that the
grid force conserves momentum {cite}`birdsall1991`.

## Charge deposit

The charge density on a given set of points ($x_i$ or $x_{i+1/2}$) is

```{math}
\rho_i = \sum_p q_p\,S_2\!\left(\frac{x_p - x_i}{\Delta x}\right),
```

with $q_p = q_s w_s$ the charge of the pseudo-particle. The part of a cloud that falls
outside the box is treated by the boundary condition: wrapped to the other end for
periodic walls, folded back onto the boundary cell for reflective walls, dropped for
absorbing walls ({doc}`boundaries`).

The deposit is a scatter-add: each particle writes into the three cells it touches,
through `jnp.zeros(n).at[idx].add(...)`, so the cost is $O(N)$ and independent of the
number of cells. Writing it instead as a dense weight per particle and cell, which is
the obvious way to keep everything a matrix product, costs $O(N N_x)$ and is what an
earlier version of the code did; the scatter is about three times faster at
$N_x = 64$ and the gap widens with the grid.

## Time step

The time step is set by the ratio $c\,\Delta t/\Delta x$, the argument
`dt_over_dx_c` of {class}`~jaxincell.Domain`, so that $\Delta t$ follows the grid when
the resolution is changed. Its constraints are collected in {doc}`stability`.
