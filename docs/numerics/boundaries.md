# Boundary conditions

`Domain(particle_bc=..., field_bc=...)` set the walls, either as one name applied to
both ends or as a `(left, right)` pair. The three kinds are `"periodic"`,
`"reflective"` and `"absorbing"`. A periodic wall needs a periodic partner; the other
two can be mixed.

```{figure} ../_static/figures/boundaries.png
:width: 100%
:alt: Electron phase space after 400 steps for periodic, reflective and absorbing walls

The same plasma, drifting to the right, after 400 steps with each wall type. Periodic
walls recirculate it, reflective walls turn it around, absorbing walls remove
{{ boundary_kept_percent_absorbing }} per cent of the electrons within the run.
```

## Particles

**Periodic.** A particle leaving one end re-enters at the other,
$x \to ((x + L/2) \bmod L) - L/2$. Nothing else changes.

**Reflective.** The position is mirrored about the wall, $x \to \pm L - x$, and the
normal velocity is multiplied by `-restitution`. At the default `restitution=1.0` this
is a specular bounce, and the total energy is conserved:
{{ boundary_energy_error_reflective }} over the run above. Values below one model a
lossy wall and remove energy on purpose.

**Absorbing.** The particle keeps its position outside the grid but its charge, its
charge-to-mass ratio and its velocity are set to zero, so it deposits nothing, feels
nothing and never returns. The arrays keep their shape, which is what lets the whole
loop stay a single compiled program with static shapes; the cost is that absorbed
particles still occupy memory. `Output.charge` is zero for them, which is how the
diagnostics and the plots tell them apart.

The transverse coordinates $y$ and $z$ are always periodic with periods `length_y` and
`length_z`. They do not affect the fields and exist only so that particle positions
stay bounded.

## Fields

The field walls enter in three places: the ghost values the two curls need, the way
the shape function is folded near the wall, and the integration constant of the
current.

**Periodic.** Ghost values wrap; the part of a particle's cloud that sticks out of one
end is deposited at the other.

**Reflective.** The ghost value repeats the boundary cell, a zero-gradient
extrapolation, and the part of the cloud outside the box is folded back onto the
boundary cell. Charge is conserved exactly.

**Absorbing.** The part of the cloud outside the box is dropped, so charge leaves the
system, which is what an open boundary means. The field ghosts use the first-order Mur
radiating condition of {doc}`field_solvers`, so an outgoing electromagnetic wave leaves
without reflection.

## A wall that absorbs charge has to keep it

The longitudinal field is fixed by the charge density only up to a constant, and that
constant is the boundary condition. Which one is right depends on what the wall is.

A **reflective** wall is a symmetry plane: nothing crosses it, so $E = 0$ there and the
field is integrated from that wall outwards. An **absorbing** wall is not a symmetry
plane. It is a conductor, and the particles it absorbs are neutralised on its surface,
leaving a surface charge that sets the field at the wall. Imposing $E = 0$ at one
absorbing wall and letting the other float is the same as insisting that all the charge
collected at both walls sits on one of them; the field there then ramps without bound as
the plasma drains, which is an artefact and not a sheath.

The standard closure for a bounded plasma is to treat the walls as electrodes carrying
surface charge, connected by an external circuit {cite}`lawson1989,verboncoeur1993`.
JAX-in-Cell takes the simplest member of that family: two absorbing walls are conductors
short-circuited to each other, so they stay at the same potential and

```{math}
\int_0^L E_x\,dx = \phi(0) - \phi(L) = 0 .
```

That is one line in each of the two places the constant is chosen — the mean is
subtracted from $E_x$ in the Gauss solve, and from $J_x$ in the continuity current, the
difference being the current the external circuit carries. The plasma is then free to
float to whatever potential balances the two fluxes, which is what a sheath is. A
biased or floating electrode with a series RLC circuit is the same construction with a
different equation for the constant {cite}`verboncoeur1993`; it is not implemented.

One consequence for the diagnostics: with a wall, the field beyond it is a degree of
freedom the output does not carry, so {func}`~jaxincell.gauss_residual` checks the
discrete Gauss law on the cells that do not need it. See {doc}`../examples/sheath` for
what the closure produces.

## Charge accounting at an absorbing wall

Charge leaves an absorbing box by two routes: with the particles that hit the wall, and
through the tail of the shape function of a particle sitting within $\tfrac32\Delta x$
of it. The second is a real property of an open boundary rather than a bug, but it is
worth knowing about, because the *net* charge in a quasi-neutral plasma is a small
difference of large numbers and a fractional loss of $10^{-4}$ of the gross charge can
be tens of per cent of the net. The test suite checks the gross budget: the charge
deposited on the grid matches the charge still carried by the particles to better than
$10^{-3}$ of the total.

For a periodic box no charge is lost at all: the two agree to
{{ charge_error_relative }}, which is round-off.

What does hold at every wall is the discrete Gauss law, to round-off, because the
current is derived from the same density the field is checked against
({doc}`deposition`). Three things have to line up for that, and the test suite checks
all three at every wall type, with and without filtering:

* the density at $t^{n+1/2}$ has to be shared between the two half steps, or the
  charge an absorbing wall removes disappears between them uncounted;
* the initial field has to be built from the density the loop starts from. The
  leapfrog carries $x^{n+1/2}$ and reconstructs $x^n$ as
  $\mathrm{wrap}(x^{n+1/2} - \tfrac12\Delta t\,\mathbf v)$, which at a reflecting
  wall is not where the particles were placed;
* the residual has to be measured with the same $E_{-1/2}$ the solver used: zero at a
  wall, the far end of the box only when the wall is periodic.

## Choosing

Periodic walls are the right default for studying a wave or an instability, because
they impose exactly the discrete Fourier modes the linear theory is written in.
Reflective walls model a mirror or a symmetry plane and keep the particle number
fixed. Absorbing walls model an open system: a sheath, a beam entering a vacuum, a
pulse leaving the box. Note that the plasma in an absorbing box is not in equilibrium
and will steadily lose particles and energy, {{ boundary_energy_error_absorbing }} of
it over the run in the figure above.
