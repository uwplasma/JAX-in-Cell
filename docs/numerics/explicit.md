# Explicit scheme

`Solver(algorithm="explicit")`, the default, is a leapfrog with the Boris pusher for
the particles, a symmetric split-step update for the fields, and the
charge-conserving current of {doc}`deposition`.

## The state

For periodic particles the loop carries the fields $\mathbf E^n$, $\mathbf B^n$, the half-step positions
$x^{n+1/2}$, the velocities $\mathbf v^n$ ($\gamma\mathbf v^n$ in a relativistic run, see
below), the pseudo-particle charges and charge-to-mass ratios (which change only when a
particle is absorbed), the charge density $\rho^n$ at the integer-time positions, and the
random key. Before the loop starts the initial positions are displaced by
$\tfrac12\Delta t\,\mathbf v^0$ to create $x^{1/2}$, and $\mathbf E^0$ is taken from
Gauss's law so that the constraint holds from the first step.

With particle walls the state instead carries $x^n$, alongside $\mathbf v^n$.
The first physical half drift reaches $x^{n+1/2}$ and applies any wall event there;
after the full Boris kick the second half drift reaches $x^{n+1}$ and applies its
events. Collisions then act at this physical endpoint. A wall hit after $t^{n+1}$
is handled on the next step: initialization consumes neither a future impact nor
a thermal redraw. A segment resolves at most one impact per particle; another contact
during its returned flight makes the result invalid and requires a shorter step.

The wall ledger uses momenta transported to the impact time in the gathered field,
and the returned momentum is transported back to its carried time. Ballistic return
flights, including thermal and relativistic ones, use the returned velocity rather
than mirroring a distance at an assumed speed. A held Newtonian electric field gives
a quadratic flight: its first outward contact is used, and a particle turning before
the wall is not reflected. Magnetic and relativistic contacts use a bounded Newton
solve of the partial-Boris mean-velocity flight; unresolved roots invalidate the
result. These flights approximate the physical orbit in varying or magnetic fields.
Refine the timestep, including the gyro angle, for impact observables and derivatives;
an event/no-event transition at grazing contact has no smooth physical derivative.
The periodic path below retains its half-position carry.

## One step

Write $x^{n} = x^{n+1/2} - \tfrac12\Delta t\,\mathbf v^n$ for the integer-time
position reconstructed from the state.

1. **Sources over the first half step.** $\rho^n$ is the density the previous step ended
   on, carried in the state, $\rho^{n+1/2}$ is deposited, and {eq}`cumsum-current` gives $J_x$ for the motion $x^n\to x^{n+1/2}$; the transverse
   currents are deposited at $x^{n+1/2}$. The digital filter of {doc}`filtering` is
   applied to every source.
2. **First half field update**, Ampere then Faraday over $\Delta t/2$:
   ```{math}
   \mathbf E \mathrel{+}= \frac{\Delta t}{2}\left(c^2\nabla\times\mathbf B - \frac{\mathbf J}{\epsilon_0}\right), \qquad
   \mathbf B \mathrel{-}= \frac{\Delta t}{2}\nabla\times\mathbf E .
   ```
3. **Gather and push.** $\mathbf E$ is averaged from the faces to the cell centres,
   where $\mathbf B$ and the charge live, and the fields, plus `external_E` and
   `external_B` if given, are interpolated to $x^{n+1/2}$ with the shape of the deposit,
   with image values beyond the walls ({doc}`boundaries`); the Boris
   rotation advances the momentum. The prescribed callable is evaluated at the
   midpoint time. With radiation, two RR half-kicks surround the Lorentz kick
   ({doc}`radiation`). Call the resulting velocity $\mathbf v_*$.
4. **Reach the integer position and collide.** First set
   $x^{n+1}=x^{n+1/2}+\tfrac12\Delta t\,\mathbf v_*$, with periodic wrapping.
   A {class}`~jaxincell.Collisions` model acts there at fixed position and produces
   $\mathbf v^{n+1}$ ({doc}`collisions`).
5. **Prepare the next half-position.**
   $x^{n+3/2}=x^{n+1}+\tfrac12\Delta t\,\mathbf v^{n+1}$.
   Without collisions these two half drifts are one full drift. The wall path
   instead carries the physical integer position and resolves its events as above.
6. **Sources over the second half step**, from $x^{n+1/2}$ to $x^{n+1}$, starting from
   the density step 1 already computed at $x^{n+1/2}$ rather than depositing it again.
   That matters at an absorbing wall: step 5 has just zeroed the charge of the
   particles that hit it, so a second deposit at $x^{n+1/2}$ would return a different
   density from the one the first half step ended on, and the charge would vanish
   between the two halves with no current to account for it.
7. **Second half field update**, Faraday then Ampere.
8. **Electrostatic correction**, only with `field_solver="gauss"`: $E_x$ is replaced by
   the solution of the discrete Gauss law from $\rho^{n+1}$ ({doc}`field_solvers`).

Composing steps 2 and 7 advances $\mathbf B$ over the whole step with the mid-point
electric field and $\mathbf E$ with the trapezoidal average of $\nabla\times\mathbf B$
and of the current. The composition E-B-B-E is symmetric, hence second-order accurate,
and it puts the fields the particles see at $t^{n+1/2}$, which is what the leapfrog
needs to be second order as well.

Splitting the step in two halves is not cosmetic. A single deposit over the whole step
would centre the current at $t^{n+1/2}$ while the field update needs it at $t^n$ and
$t^{n+1}$; the mismatch shows up as a drift of the Gauss residual by one part in
$10^{3}$ per half step.

## The Boris pusher

For the non-relativistic case {cite}`boris1970`, with $\mathbf E$ and $\mathbf B$ the
fields at the particle,

```{math}
\mathbf v^- = \mathbf v^n + \frac{q}{m}\frac{\Delta t}{2}\mathbf E, \qquad
\mathbf b = \frac{q}{m}\frac{\Delta t}{2}\mathbf B, \qquad
\mathbf v' = \mathbf v^- + \mathbf v^-\times\mathbf b,
```
```{math}
\mathbf v^+ = \mathbf v^- + \frac{2\,\mathbf v'\times\mathbf b}{1 + |\mathbf b|^2}, \qquad
\mathbf v^{n+1} = \mathbf v^+ + \frac{q}{m}\frac{\Delta t}{2}\mathbf E .
```

The middle step is an exact rotation of $\mathbf v^-$ about $\mathbf B$ through
$2\arctan|\mathbf b|$, which equals $\Omega_c\Delta t$ to second order. It conserves
$|\mathbf v^-|$ to round-off, so the magnetic field does no work at the discrete
level. The scheme is time reversible and, although not symplectic, preserves
phase-space volume, which is why its energy error stays bounded over long runs rather
than growing secularly {cite}`qin2013`.

With `Solver(relativistic=True)` the same three sub-steps act on the momentum
$\mathbf p = \gamma m\mathbf v$:

```{math}
\mathbf p^- = \mathbf p^n + \frac{q\Delta t}{2}\mathbf E, \qquad
\gamma^- = \sqrt{1 + \frac{|\mathbf p^-|^2}{m^2c^2}}, \qquad
\mathbf t = \frac{q\Delta t}{2 m\gamma^-}\mathbf B,
```

after which $\mathbf p^+$ follows the same rotation formula. Evaluating $\gamma^-$ from
$\mathbf p^-$ rather than from $\mathbf v^n$ is what makes the rotation angle correct for
a relativistic particle. A relativistic run carries $\mathbf u = \mathbf p/m = \gamma\mathbf v$
in its state instead of the velocity, so that $\gamma = \sqrt{1 + u^2/c^2}$ stays exact
at any energy: converting to $\mathbf v$ and back every step loses a fraction
$\gamma^2\epsilon$ of $\gamma$ each time, which in single precision took a particle started
at $\gamma = 1000$ to 1423 in a thousand steps with no field. The velocity is formed from
$\mathbf u$ only where it is needed, for the position update, the velocity output
and the walls. Relativistic collisions act on $\mathbf u$ directly, with no velocity
roundtrip. The initial $\mathbf u$ is supplied as `Species.u` or comes from the sampled velocities, of which only
one with $v^2/c^2 > 1 - 10^{-5}$ is changed, brought back to that speed along its own
direction ({doc}`initialization`).

Pairwise collision conservation does not imply exact conservation of the coupled
electromagnetic PIC step. Its second transverse-current deposit uses the
post-collision velocity, while the preceding half drift used $\mathbf v_*$.
Measure field/particle energy exchange under timestep refinement, separately from
the collision-stage `collision_work` and the exact discrete charge continuity.

## Accuracy

Over the growth and saturation of the two-stream instability of {doc}`verification`,
at $\omega_{pe}\Delta t = $ {{ energy_omega_pe_dt }} and $c\Delta t/\Delta x = $
{{ energy_courant }}, the total energy changes by
{{ energy_error_max_explicit }} and does not drift, while the momentum is conserved to
round-off, the trade the gather of {doc}`deposition` makes; the Gauss residual stays at
{{ gauss_residual_max_explicit }}. When the energy error itself has to be small, use
the {doc}`implicit` scheme, which drives it to round-off at a cost of about
{{ scaling_implicit_over_explicit }} times more work per step.
