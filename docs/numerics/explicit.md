# Explicit scheme

`time_evolution_algorithm = 0` selects a leapfrog integrator with the Boris pusher for
the particles, a symmetric split-step update for the fields, and a current deposit
that satisfies the discrete continuity equation. One step advances the state from
$t^n$ to $t^{n+1}$ as follows.

## The state

The loop carries the fields $\mathbf E^n$, $\mathbf B^n$, the positions
$x^{n-1/2}$, $x^{n}$, $x^{n+1/2}$, the velocities $\mathbf v^n$, and the charges,
masses and charge-to-mass ratios of all pseudo-particles (the last three change only
when a particle is absorbed). Before the loop starts, the initial positions $x^0$ are
displaced by $\pm\tfrac12\Delta t\,\mathbf v^0$ to create $x^{\pm1/2}$.

## One step

1. **Current at $t^n$.** $J_x^n$ is computed from the motion $x^{n-1/2}\to x^{n+1/2}$
   (see below); $J_y^n$ and $J_z^n$ are deposited as $q_p v_{y,p}^n S_2$ and
   $q_p v_{z,p}^n S_2$ at $x^n$. The digital filter is applied.
2. **First half field update.** Ampere then Faraday, each over $\Delta t/2$:
   ```{math}
   \mathbf E^{n+1/2} = \mathbf E^n + \frac{\Delta t}{2}\left(c^2\nabla\times\mathbf B^n - \frac{\mathbf J^n}{\epsilon_0}\right), \qquad
   \mathbf B^{n+1/2} = \mathbf B^n - \frac{\Delta t}{2}\nabla\times\mathbf E^{n+1/2}.
   ```
3. **Gather.** The external fields, if any, are added, and $\mathbf E^{n+1/2}$ and
   $\mathbf B^{n+1/2}$ are interpolated to $x^{n+1/2}$ with $S_2$.
4. **Push.** The Boris rotation advances $\mathbf v^n\to\mathbf v^{n+1}$ and the
   position $x^{n+3/2} = x^{n+1/2} + \Delta t\,v_x^{n+1}$ (the $y$ and $z$ coordinates
   are advanced too).
5. **Boundary conditions** are applied to $x^{n+3/2}$ and $\mathbf v^{n+1}$, and the
   integer-time position $x^{n+1} = x^{n+3/2} - \tfrac12\Delta t\, v_x^{n+1}$ is formed
   and wrapped as well.
6. **Current at $t^{n+1}$** from the motion $x^{n+1/2}\to x^{n+3/2}$ and the transverse
   velocities at $x^{n+1}$.
7. **Second half field update.** Faraday then Ampere:
   ```{math}
   \mathbf B^{n+1} = \mathbf B^{n+1/2} - \frac{\Delta t}{2}\nabla\times\mathbf E^{n+1/2}, \qquad
   \mathbf E^{n+1} = \mathbf E^{n+1/2} + \frac{\Delta t}{2}\left(c^2\nabla\times\mathbf B^{n+1} - \frac{\mathbf J^{n+1}}{\epsilon_0}\right).
   ```
8. **Electrostatic correction** (only with `field_solver = 1`): $E_x^{n+1}$ is replaced
   by the solution of Gauss's law from the charge density deposited at the cell faces
   from $x^{n+1}$.
9. The charge density at the cell centres is deposited from $x^{n+1}$ for the output,
   and the step returns $(x^{n+1}, \mathbf v^{n+1}, \mathbf E^{n+1}, \mathbf B^{n+1},
   \mathbf J^{n+1}, \rho^{n+1})$.

Combining steps 2 and 7, the magnetic field is advanced over the full step with the
mid-point electric field, $\mathbf B^{n+1} = \mathbf B^n - \Delta t\,\nabla\times\mathbf E^{n+1/2}$,
and the electric field with the trapezoidal average of the curl of $\mathbf B$ and of
the current. The composition E-B-B-E is symmetric, hence second-order accurate for the
source-free Maxwell equations, and it makes the fields seen by the particles centred
at $t^{n+1/2}$, which is what the leapfrog needs to be second order.

## The Boris pusher

For the non-relativistic case {cite}`boris1970`, with $\mathbf E$ and $\mathbf B$
the fields at the particle and $\Delta t$ the step,

```{math}
\mathbf v^- = \mathbf v^n + \frac{q}{m}\frac{\Delta t}{2}\mathbf E, \qquad
\mathbf b = \frac{q}{m}\frac{\Delta t}{2}\mathbf B, \qquad
\mathbf v' = \mathbf v^- + \mathbf v^-\times\mathbf b,
```
```{math}
\mathbf v^+ = \frac{\mathbf v' + \mathbf v'\times\mathbf b + (\mathbf v'\cdot\mathbf b)\,\mathbf b}{1 + |\mathbf b|^2}, \qquad
\mathbf v^{n+1} = \mathbf v^+ + \frac{q}{m}\frac{\Delta t}{2}\mathbf E .
```

The middle step is an exact rotation of $\mathbf v^-$ about $\mathbf B$ by the angle
$2\arctan|\mathbf b|$, which equals $\Omega_c\Delta t$ to second order. It conserves
$|\mathbf v^-|$ exactly, so the magnetic field does no work at the discrete level. The
scheme is time reversible and, although not symplectic, preserves phase-space volume,
which is the reason for its long-time accuracy {cite}`qin2013`.

With `relativistic = true` the same sequence acts on the momentum
$\mathbf p = \gamma m\mathbf v$:

```{math}
\mathbf p^- = \mathbf p^n + \frac{q\Delta t}{2}\mathbf E, \qquad
\gamma^- = \sqrt{1 + \frac{|\mathbf p^-|^2}{m^2c^2}}, \qquad
\mathbf t = \frac{q\Delta t}{2 m\gamma^-}\mathbf B,
```
```{math}
\mathbf p^+ = \frac{(1 - |\mathbf t|^2)\,\mathbf p^- + 2(\mathbf p^-\cdot\mathbf t)\,\mathbf t + 2\,\mathbf p^-\times\mathbf t}{1 + |\mathbf t|^2}, \qquad
\mathbf p^{n+1} = \mathbf p^+ + \frac{q\Delta t}{2}\mathbf E,
```

after which $\gamma^{n+1}$ and $\mathbf v^{n+1} = \mathbf p^{n+1}/(\gamma^{n+1}m)$ follow.
The initial $\gamma^n$ is computed from $\mathbf v^n$, which is why velocities are
clipped to $0.99c$ at initialisation.

## Charge-conserving current deposit

The longitudinal current is not deposited as $q v_x S_2$. Instead, it is derived from
the change of the particle's charge cloud between the two half-step positions, so that
the discrete continuity equation holds exactly on the grid
{cite}`villasenor1992,esirkepov2001`:

```{math}
\frac{\rho_i^{n+1/2} - \rho_i^{n-1/2}}{\Delta t} + \frac{J_{x,i+1/2}^{n} - J_{x,i-1/2}^{n}}{\Delta x} = 0 .
```

For each particle the code forms $\Delta\rho_i = q_p[S_2(x_i - x^{n+1/2}) - S_2(x_i - x^{n-1/2})]/\Delta t$
on the cell centres and integrates it from the left,

```{math}
J_{x,i+1/2} = -\Delta x\sum_{j\le i}\Delta\rho_j ,
```

over a window of six cells starting three cells to the left of the cell that contains
$x^{n-1/2}$. The window is enough because the cloud spans three cells and moves by less
than one cell per step; the sum of $\Delta\rho$ over the window vanishes, so the
current returns to zero on both sides. Contributions of all particles are summed and
filtered.

Because the continuity equation is satisfied, advancing $E_x$ with Ampere's law
preserves Gauss's law to round-off: $\partial_x E_x - \rho/\epsilon_0$ stays at its
initial value, and the initial field is computed from Gauss's law. No correction step
is needed in the electromagnetic mode; the electrostatic mode `field_solver = 1`
recomputes $E_x$ from $\rho$ anyway, which also removes any error that the boundary
treatment introduces.

The transverse currents are deposited directly, $J_{y,i} = \sum_p q_p v_{y,p}^n S_2(x_i - x_p^n)$,
at the integer-time position.

## Time step and cost

The constraints on $\Delta t$ (plasma frequency, cell crossing, light waves) are
discussed in {doc}`stability`. Per step, the scheme evaluates $S_2$ at three positions
per particle for the currents and charge, gathers two fields, and performs one Boris
rotation; the field update and the filter cost a few operations per cell. The deposits
are written as dense operations over the whole grid for every particle (a `vmap`
followed by a sum), so their cost scales with the product of the particle and cell
counts; see {doc}`../user_guide/performance`. Everything is expressed as `vmap` over
particles and `scan` over steps, so the whole loop is one XLA program.
