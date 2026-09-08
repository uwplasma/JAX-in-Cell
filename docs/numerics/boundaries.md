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

## Choosing

Periodic walls are the right default for studying a wave or an instability, because
they impose exactly the discrete Fourier modes the linear theory is written in.
Reflective walls model a mirror or a symmetry plane and keep the particle number
fixed. Absorbing walls model an open system: a sheath, a beam entering a vacuum, a
pulse leaving the box. Note that the plasma in an absorbing box is not in equilibrium
and will steadily lose particles and energy, {{ boundary_energy_error_absorbing }} of
it over the run in the figure above.
