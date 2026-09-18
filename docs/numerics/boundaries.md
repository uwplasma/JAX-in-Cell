# Boundary conditions

`Domain(particle_bc=..., field_bc=...)` set the walls, either as one name applied to
both ends or as a `(left, right)` pair. The kinds are `"periodic"`, `"reflective"`,
`"absorbing"`, for particles only `"thermal"`, and for the field only `"open"`. A
periodic wall needs a periodic partner; the others can be mixed.

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
normal velocity is multiplied by `-restitution`, which may differ between the two
walls. At the default `restitution=1.0` this is a specular bounce, and the total energy
is conserved: {{ boundary_energy_error_reflective }} over the run above. Values below
one model a lossy wall and remove energy on purpose.

**Absorbing.** The particle keeps its position outside the grid but its weight, its
charge-to-mass ratio and its velocity are set to zero, so it deposits nothing, feels
nothing and never returns. The arrays keep their shape, which is what lets the whole
loop stay a single compiled program with static shapes; the cost is that absorbed
particles still occupy memory. `Output.weight` is zero for them, which is how the
diagnostics and the plots tell them apart. A species can have part of each particle
sent back instead, as {ref}`partial-reflection` describes.

**Thermal.** The position is mirrored as at a reflective wall, but the velocity is
drawn afresh, as if the particle came from a Maxwellian reservoir behind the wall at the
temperature of its species. The normal component follows the flux distribution
$(v/\sigma^2)\,e^{-v^2/2\sigma^2}$, sampled as $\sigma\sqrt{-2\ln U}$ with $U$ uniform,
and the tangential ones the Maxwellian, with $\sigma = v_{th}/\sqrt2$. This is the
source boundary of bounded-plasma simulations {cite}`schwager1990` and the `thermal`
boundary of EPOCH and PIConGPU. Whatever the rest of the box has done to the
distribution, what leaves the wall is Maxwellian; particles are conserved, energy is not,
since the wall is a heat bath. The fields need a wall of their own there, normally
`"reflective"`.

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

## What a particle feels next to a wall

The field is gathered to a particle with the spline its charge was deposited with, from
the grid it was deposited on: $\mathbf E$ is first averaged from the faces to the centres,
where $\mathbf B$ and $\rho$ live, and the unstored left wall face enters that average with
the value the walls give it ({func}`~jaxincell._core.wall_faces_E`). Within a cell of a wall
the spline of a particle reaches the centre beyond it, and the value there is the wall's:

* **Periodic.** The far end of the box.
* **Reflective.** The mirror image of the first centre, with the sign each component takes
  under the reflection: $E_x$, $B_y$ and $B_z$ change sign, $E_y$, $E_z$ and $B_x$ keep it.
  This is the method of images. The box then gathers exactly as the periodic box twice as
  long, holding every particle and its mirror image, would, and a particle feels its image
  and not itself.
* **Absorbing.** Zero. There is nothing beyond a conductor, and the deposit drops the same
  part of the cloud, so a particle that reaches past the wall is missing there both as a
  source and as a receiver of the field.

External fields are prescribed rather than solved for, and simply continue beyond a wall.

For a sheet of charge $\sigma$ at a distance $a \ge \Delta x$ from the left wall of a box
of length $L$, this gives the one-dimensional image forces exactly, $F = \sigma E$ with

| walls | $E$ at the sheet |
|---|---|
| periodic | $0$ |
| reflective, reflective | $(\sigma/\epsilon_0)(\tfrac12 - a/L)$, pushed from the nearer plane |
| reflective, absorbing | $+\sigma/2\epsilon_0$, towards the conductor, whatever $a$ |
| absorbing, reflective | $-\sigma/2\epsilon_0$ |
| absorbing, absorbing | $(\sigma/\epsilon_0)(a/L - \tfrac12)$, pulled to the nearer conductor |

and the test suite checks each to $10^{-10}$, the mirror image of each at every distance,
including inside the last cell, and the reflective box against the doubled periodic one.

Gathering from the centres is what makes the gather the transpose of the deposit
{cite}`birdsall1991`. Gathering $E_x$ straight from the faces with the same spline, as the
code once did, is not: a lone particle in a periodic box pushed itself with up to 8 % of
its own field, and near a wall the unstored left face read as zero or as its neighbour
while the stored right face read as itself. The implicit scheme gathers the transverse
fields this way and takes their current as its transpose; its longitudinal field and current
are the discrete gradient and the continuity current of {doc}`implicit`, which keep both the
energy and the charge whatever the walls. What stays one-sided is the transverse Yee update at the walls ({doc}`field_solvers`):
the right wall face is stored and advanced, the left one is only a ghost value, so an
electromagnetic wave meets the two walls of a reflective or absorbing box slightly
differently.

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

On the grid the potential lives at the centres, with the charge, and each conductor sits
half a cell beyond the last centre, so the integral is the trapezoidal sum over the
$N_x+1$ faces from wall to wall, the two wall faces with half weight. One function
imposes it in both places the constant is chosen, the Gauss solve for $E_x$ and the
continuity current for $J_x$, the current the closure removes being the one the
external circuit carries. The plasma is then free to float to whatever potential
balances the two fluxes, which is what a sheath is. A biased or floating electrode with
a series RLC circuit is the same construction with a different equation for the
constant {cite}`verboncoeur1993`; it is not implemented.

A single absorbing wall facing a reflective or thermal one, on either side, is a
floating electrode on its own: the symmetry plane fixes $E = 0$ at its end, the current
through it is zero, the field is integrated from it, and the electrode sits at whatever
potential the charge it has collected gives it. That is the setup of
{doc}`../examples/sheath_reflection`.

An **open** wall is the plane a {class}`~jaxincell.Source` supplies through, and it is
neither. A reservoir drives a current across it, so $E$ there is not zero and nothing at
that end fixes the constant. The collector opposite does instead: it holds the charge it
has collected, and a pillbox across its surface with $E = 0$ inside the conductor gives

```{math}
E_x(L^-) = -\sigma_w/\epsilon_0,
\qquad \sigma_w = \sum_s q_s\,W_{s,\rm collected} + \sigma_{\rm overlap},
```

with $W$ the weight on the wall ledger. The second term is the part of the live clouds
that reaches past the wall: a cloud is one and a half cells wide, so it crosses before
its centre does, and the deposit drops whatever lies outside the grid. That part is
neither in the volume nor irreversibly collected — the particle may still turn round —
but it is charge the wall already sees, and leaving it out of both makes a particle's
total swing between a half and one and a half of itself as it crosses, and the field
inside jump by half a particle at the moment the centre passes. With it, the deposited
and the exterior fractions sum to one at every sub-cell offset and the field does not
notice the crossing at all.

The field is integrated back from there and the potential is measured from the source
plane, which is the gauge. With a symmetry plane opposite and nothing crossing it,
closing on the electrode's charge and closing on $E = 0$ are the same problem; with a
source they are not, and only the electrode closure is right.
`field_bc=("open", "absorbing")` is the only pairing the open condition has, because it
needs a collector to close on, and for the same reason the implicit scheme, which carries
no surface charge, refuses it.

The same electrode closes the continuity current. Ampère's law makes the total current
uniform across a one-dimensional box, so $J_x + \epsilon_0\,\partial_t E_x$ is the current
in the external circuit, and a floating collector is connected to nothing:

```{math}
J_x(L^-) = \dot\sigma_w = -\epsilon_0\,\partial_t E_x(L^-).
```

Taken as a difference of $\sigma_w$ over the same interval the density change spans, that
is exact for the discrete continuity relation, and `Output.J` is then an absolute
conduction current whose total with the displacement current vanishes at every face to
round-off. Anchored at zero instead, it was an internal transport measured from the
source plane, and the residual was six tenths of its own size.

Two reflective walls are two symmetry planes, and the box
between them is half of a periodic box twice as long, holding the charge and its mirror
image; as in a periodic box the charge must then be neutral, the mean is removed, and
$E = 0$ at both walls. Each of these rules is its own mirror image, so a plasma and its
reflection, with the walls swapped, give reflected fields to round-off.

One consequence for the diagnostics: with a wall, the field beyond it is a degree of
freedom the output does not carry, so {func}`~jaxincell.gauss_residual` checks the
discrete Gauss law on the cells that do not need it. See {doc}`../examples/sheath_reflection` for
what the closure produces.

(partial-reflection)=
## Walls that send part of a particle back

A real surface does not collect every electron that reaches it. Some are reflected,
and slow electrons more readily than fast ones {cite}`cimino2004,furman2002`. Each
species therefore carries a `reflection` law $R$, the fraction of a particle an
absorbing wall returns: a number, a function of the normal impact speed $|v_x|$, or a
`(left, right)` pair of either.

```python
Species.electrons(..., reflection=0.3)                                        # 30 % of every electron
Species.electrons(..., reflection=lambda s: jnp.exp(-s ** 2 / (2 * u ** 2)))   # the slow ones
```

A particle of weight $w$ that crosses the wall leaves $(1-R)\,w$ on the conductor and
returns with $R\,w$, mirrored like a reflective bounce and with its normal velocity
multiplied by `-restitution`. The split is deterministic. WarpX offers the same law, a
per-species function of the normal velocity at an absorbing boundary, but draws a
random number and returns or keeps each macro-particle whole; splitting the weight
gives the mean of that process without its sampling noise, and keeps the result a
smooth function of $R$ that `jax.grad` can differentiate. The collected part becomes
surface charge through the same continuity current as an ordinary absorption, so the
discrete Gauss law still holds to round-off ({doc}`verification`).

### The wall samples the flux

Which particles reach a wall in a short time is biased towards the fast: from a uniform
plasma with velocity distribution $f$, those arriving with normal speed near $v$ are in
proportion to $v f(v)$. A law $R(v)$ therefore returns, from a Maxwellian of variance
$\sigma^2 = T/m$, its flux average

```{math}
R_{\rm eff} = \frac{\int_0^\infty R(v)\, v\, e^{-v^2/2\sigma^2}\,dv}{\int_0^\infty v\, e^{-v^2/2\sigma^2}\,dv}
            = \frac{1}{\sigma^2}\int_0^\infty R(v)\, v\, e^{-v^2/2\sigma^2}\,dv ,
```

not its average over the distribution. For a Gaussian law $R = e^{-v^2/2u^2}$ the
integral is elementary, $R_{\rm eff} = u^2/(u^2+\sigma^2)$, where the distribution
average would be $u/\sqrt{u^2+\sigma^2}$: one half against
{{ reflection_distribution_average_sigma }} at $u = \sigma$. The simulation returns
{{ reflection_returned_sigma }} ({doc}`../examples/wall_reflection`). Restitution then
takes a factor $e^2$ out of the energy that comes back.

This is why a law should be written with a fixed velocity scale. Normalising the speed by
the fastest particle in the run would give $R_{\rm eff} \approx 1 - \sqrt{\pi/2}\,
\sigma/v_{\max}$, and since the largest of $N$ Maxwellian samples grows like
$\sigma\sqrt{2\ln N}$, the wall would reflect more the more particles the run used: a
property of the sampling, not of the surface.

### What reflection does to a sheath

A floating wall collects the ions at the Bohm flux $n_s c_s$ and the electrons at the
one-way flux $\tfrac14 n_s \bar v_e\,e^{-e\Delta\phi/T_e}$ that clears the barrier. If it
returns the fraction $R_{\rm eff}$ of those electrons, only $1-R_{\rm eff}$ of the flux
counts, and the balance gives {cite}`hobbs1967`

```{math}
\frac{e\Delta\phi}{T_e} = \ln\!\left[(1-R_{\rm eff})\sqrt{\frac{m_i}{2\pi m_e}}\right]
= \frac12\ln\frac{m_i}{2\pi m_e} + \ln(1-R_{\rm eff}) .
```

For a velocity-dependent law the average is taken at the wall, over the electrons that
cleared the barrier. In a collisionless sheath those still form a half-Maxwellian at the
plasma temperature, because energy conservation slides the part of the distribution
above the barrier down onto a whole half-Maxwellian, so the flux average above is
exactly the one that counts. Hobbs and Wesson derived the formula for secondary
emission, where the emitted electrons leave cold and pile up in front of the wall,
which limits the coefficient. Reflected electrons leave as fast as they came, so that
limit does not arise, and the drop simply vanishes as $R_{\rm eff}$ approaches
$1-\sqrt{2\pi m_e/m_i}$. Restitution does not enter: however slowly the electrons
leave, the sheath field returns them to the plasma, and the number the wall keeps is
unchanged. {doc}`../examples/sheath_reflection` measures the drop for a wall that keeps everything
and for two laws with the same $R_{\rm eff}$.

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
it over the run in the figure above, and with them the fast tail of its distribution.
A thermal wall at the other end stands for the plasma beyond the box and keeps that
tail filled, which is what a comparison with sheath theory needs.
