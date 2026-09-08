# Field solvers

`Solver(field_solver=...)` chooses how the longitudinal field $E_x$ is obtained.
The transverse fields are always advanced by the Yee update of {doc}`explicit`.

## `"ampere"` (default)

$E_x$ is advanced in time by Ampere's law,

```{math}
\frac{\partial E_x}{\partial t} = -\frac{J_x}{\epsilon_0},
```

with the charge-conserving current of {doc}`deposition`. Because that current
satisfies the discrete continuity equation exactly, the discrete Gauss law is
preserved for all time once it holds initially, and the initial field is built from
Gauss's law at $t=0$. No elliptic solve appears inside the time loop, and the
residual stays at round-off ({{ gauss_residual_max_explicit }} over the two-stream
run). This is the recommended setting and the only one that is exactly local, hence
the only one that parallelises without a global reduction.

## `"gauss"`

$E_x$ is recomputed from the charge density at the end of every step by solving the
discrete Gauss law

```{math}
:label: discrete-gauss
\frac{E_{x,i+1/2} - E_{x,i-1/2}}{\Delta x} = \frac{\rho_i}{\epsilon_0}.
```

This throws away whatever $E_x$ the time integration produced, which makes the scheme
insensitive to an accumulated error in $J_x$ but also to any physics that lives in the
longitudinal field between deposits. It is useful as a cross-check, and for problems
started from a charge distribution rather than from a field.

### Solving it

**Periodic walls.** {eq}`discrete-gauss` is diagonal in Fourier space. The forward
difference has the exact symbol

```{math}
\widehat{D}(k) = \frac{1 - e^{-ik\Delta x}}{\Delta x},
```

so $\hat E_x(k) = \hat\rho(k)/(\epsilon_0\widehat D(k))$, with the $k=0$ mode set to
zero because a periodic box cannot support a uniform field. It matters that the
**finite-difference** symbol is used and not the continuum $ik$: the two differ by
$\mathcal{O}((k\Delta x)^2)$, and using $ik$ leaves a residual in
{eq}`discrete-gauss` as large as {{ two_stream_dx_over_debye }} times the field
itself at the resolutions typical of these runs, so Gauss's law is not actually
satisfied by the field the solver returns.

**Reflective or absorbing walls.** The field is integrated from the left wall, where
$E_{x,-1/2}=0$, by a cumulative sum: $E_{x,i+1/2} = (\Delta x/\epsilon_0)\sum_{j\le i}\rho_j$.

## Boundary values of the curls

Both curls need one value beyond the grid. `_left_ghost_E` and `_right_ghost_B` supply
it according to the wall type ({doc}`boundaries`): the opposite end for a periodic
wall, a copy of the boundary value for a reflective wall, and for an absorbing wall the
first-order Mur condition, which sets the ghost so that an outgoing plane wave leaves
without reflection:

```{math}
E_{y,-1/2} = -2cB_{z,0} - E_{y,0}, \qquad E_{z,-1/2} = 2cB_{y,0} - E_{z,0},
```

and the mirror image on the right. First order means a wave arriving at normal
incidence is absorbed to the accuracy of the discretisation; in one dimension there is
no other angle of incidence, so this is as good as an open boundary gets here.

## Choosing

Use `"ampere"`. Reach for `"gauss"` only to check a result, or when the initial
condition is specified as a charge density and the field must follow from it exactly
at every step. The two agree to within the discretisation error on every problem in
{doc}`verification`.
