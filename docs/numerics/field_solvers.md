# Field solvers

## Finite-difference time domain

Both integrators advance the transverse fields with the curl operators of the
staggered grid,

```{math}
(\nabla\times\mathbf E)_i = \frac{1}{\Delta x}\left(0,\; -(E_{z,i+1/2} - E_{z,i-1/2}),\; E_{y,i+1/2} - E_{y,i-1/2}\right), \qquad
(\nabla\times\mathbf B)_{i+1/2} = \frac{1}{\Delta x}\left(0,\; -(B_{z,i+1} - B_{z,i}),\; B_{y,i+1} - B_{y,i}\right),
```

inserted into Faraday's and Ampere's laws. Each operator is a two-point difference
that is centred on the point where the result lives, which is the Yee scheme
{cite}`yee1966` in one dimension. The functions {func}`jaxincell.curlE` and
{func}`jaxincell.curlB` implement them with ghost cells at the two ends, and
{func}`jaxincell.field_update1` and {func}`jaxincell.field_update2` are the two
half-step compositions used by the explicit scheme (E then B, and B then E).

In the explicit scheme the update is stable for source-free light waves when
$c\,\Delta t \le \Delta x$. The longitudinal field $E_x$ is not part of any curl: it
obeys $\partial_t E_x = -J_x/\epsilon_0$ exactly on the grid and therefore has no
Courant limit of its own.

## Gauss's law

With `field_solver = 1`, the explicit scheme replaces $E_x$ at the end of every step
by the solution of $\partial_x E_x = \rho/\epsilon_0$ for the charge density deposited
on the cell faces. The solution uses the fast Fourier transform,

```{math}
\hat E_x(k) = -\frac{i\,\hat\rho(k)}{\epsilon_0 k}, \qquad \hat E_x(0) = 0,
```

which is exact for the discrete Fourier modes of a periodic box and removes the mean
field ({func}`jaxincell.E_from_Gauss_1D_FFT`). The equivalent route through the
potential, $\hat\phi = \hat\rho/(\epsilon_0 k^2)$ followed by $\hat E_x = -ik\hat\phi$,
is provided as {func}`jaxincell.E_from_Poisson_1D_FFT` and gives the same field. Both
assume periodicity; with reflective or absorbing walls the transform sees a
discontinuity at the box ends.

The direct finite-difference solution $E_{i+1/2} = E_{i-1/2} + \Delta x\,\rho_i/\epsilon_0$
with $E_{-1/2} = 0$, implemented in {func}`jaxincell.E_from_Gauss_1D_Cartesian` as a
bidiagonal solve, is used once to compute the initial electric field from the initial
charge density. It does not assume periodicity: the field vanishes at the left wall
and, for a neutral box, returns to zero at the right wall.

## Which mode to use

In the electromagnetic mode (`field_solver = 0`) Gauss's law is maintained by the
charge-conserving current deposit and never solved after the first step. This is the
mode used by all the examples and verified in {doc}`verification`. The electrostatic
mode (`field_solver = 1`) recomputes $E_x$ from $\rho$ every step and so does not
depend on the continuity property of the deposit, at the cost of one pair of FFTs
per step.

```{note}
In the current release the two modes do not give the same growth rate for the
two-stream test of {doc}`verification`: with `field_solver = 1` the measured rate is
about half the kinetic value, whereas `field_solver = 0` agrees with theory. The
charge density that feeds the FFT solver is deposited on the cell faces with the
wall treatment written for cell centres, which does not wrap the whole charge cloud
of particles within half a cell of the left wall. Until this is resolved, use
`field_solver = 0` for quantitative work.
```

## External fields

Static external arrays for $\mathbf E$ and $\mathbf B$ are added to the
self-consistent fields before the gather and take no part in the field update; see
{doc}`../user_guide/external_fields`.
