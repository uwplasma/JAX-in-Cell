# Explicit scheme

`Solver(algorithm="explicit")`, the default, is a leapfrog with the Boris pusher for
the particles, a symmetric split-step update for the fields, and the
charge-conserving current of {doc}`deposition`.

## The state

The loop carries the fields $\mathbf E^n$, $\mathbf B^n$, the half-step positions
$x^{n+1/2}$, the velocities $\mathbf v^n$, the pseudo-particle charges and
charge-to-mass ratios (which change only when a particle is absorbed), and the random
key. Before the loop starts the initial positions are displaced by
$\tfrac12\Delta t\,\mathbf v^0$ to create $x^{1/2}$, and $\mathbf E^0$ is taken from
Gauss's law so that the constraint holds from the first step.

## One step

Write $x^{n} = x^{n+1/2} - \tfrac12\Delta t\,\mathbf v^n$ for the integer-time
position reconstructed from the state.

1. **Sources over the first half step.** $\rho^n$ and $\rho^{n+1/2}$ are deposited and
   {eq}`cumsum-current` gives $J_x$ for the motion $x^n\to x^{n+1/2}$; the transverse
   currents are deposited at $x^{n+1/2}$. The digital filter of {doc}`filtering` is
   applied to every source.
2. **First half field update**, Ampere then Faraday over $\Delta t/2$:
   ```{math}
   \mathbf E \mathrel{+}= \frac{\Delta t}{2}\left(c^2\nabla\times\mathbf B - \frac{\mathbf J}{\epsilon_0}\right), \qquad
   \mathbf B \mathrel{-}= \frac{\Delta t}{2}\nabla\times\mathbf E .
   ```
3. **Gather and push.** The fields, plus `external_E` and `external_B` if given, are
   interpolated to $x^{n+1/2}$ with the same $S_2$ used for the deposit, and the Boris
   rotation advances $\mathbf v^n\to\mathbf v^{n+1}$.
4. **Collisions**, if a {class}`~jaxincell.Collisions` model is set ({doc}`collisions`).
5. **Move.** $x^{n+3/2} = x^{n+1/2} + \Delta t\,\mathbf v^{n+1}$, then the boundary
   conditions of {doc}`boundaries` are applied.
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

after which $\mathbf p^+$ follows the same rotation formula and
$\mathbf v^{n+1} = \mathbf p^{n+1}/(\gamma^{n+1}m)$. Evaluating $\gamma^-$ from
$\mathbf p^-$ rather than from $\mathbf v^n$ is what makes the rotation angle correct
for a relativistic particle; the initial $\gamma$ does come from $\mathbf v$, which is
why velocities are clipped to $0.99c$ at initialisation.

## Accuracy

Over the growth and saturation of the two-stream instability of {doc}`verification`,
at $\omega_{pe}\Delta t = $ {{ energy_omega_pe_dt }} and $c\Delta t/\Delta x = $
{{ energy_courant }}, the total energy changes by
{{ energy_error_max_explicit }} and does not drift; the Gauss residual stays at
{{ gauss_residual_max_explicit }}. When the energy error itself has to be small, use
the {doc}`implicit` scheme, which drives it to round-off at a cost of about
{{ scaling_implicit_over_explicit }} times more work per step.
