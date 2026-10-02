# Coulomb collisions

Passing a {class}`~jaxincell.Collisions` object to a
{class}`~jaxincell.Simulation` adds binary Coulomb collisions to the particle
push ([where in the step](#where-in-the-step)), by the Monte Carlo scheme of Takizuka and Abe {cite}`takizuka1977`.

```python
from jaxincell import Collisions, Simulation

simulation = Simulation(domain, [electrons, ions],
                        collisions=Collisions(pairs=(("electrons", "ions"),
                                                     ("electrons", "electrons"))))
```

`pairs` names the species combinations to collide, including a species with itself;
`None` collides every combination. `coulomb_log` fixes $\ln\Lambda$; left at `None`
it is taken from the NRL formulary {cite}`nrl2019` for the electrons (see
[below](#the-coulomb-logarithm)).

## The scheme

Inside each cell the particles are paired at random. Each pair is scattered by
rotating its *relative* velocity $\mathbf u = \mathbf v_a - \mathbf v_b$ through a
random angle $\Theta$ about a random azimuth, and sharing the change in inverse
proportion to the masses:

```{math}
\mathbf v_a \to \mathbf v_a + \frac{m_b}{m_a + m_b}\Delta\mathbf u, \qquad
\mathbf v_b \to \mathbf v_b - \frac{m_a}{m_a + m_b}\Delta\mathbf u .
```

Because $|\mathbf u|$ is unchanged, a pair conserves momentum and kinetic energy
**exactly**, whatever the time step and however large the angle. That is the reason
for scattering pairs rather than drawing a random kick per particle.

The angle is drawn through $\delta = \tan(\Theta/2)$, with $\delta$ Gaussian of zero
mean and variance

```{math}
:label: ta-variance
\langle\delta^2\rangle = \frac{q_a^2 q_b^2\, n\, \ln\Lambda}{8\pi\epsilon_0^2 m_{ab}^2 u^3}\,\Delta t,
\qquad m_{ab} = \frac{m_a m_b}{m_a + m_b},
```

and $\sin\Theta = 2\delta/(1+\delta^2)$, $1-\cos\Theta = 2\delta^2/(1+\delta^2)$.

### Where the variance comes from

{eq}`ta-variance` has no free parameter: it is fixed by matching the perpendicular
velocity diffusion of the Fokker-Planck operator. For a test particle of mass $m_a$
moving much faster than the background it scatters off,

```{math}
\frac{d\langle\Delta v_\perp^2\rangle}{dt} = \nu_\perp v^2, \qquad
\nu_\perp = 2\nu_0, \qquad
\nu_0 = \frac{q_a^2q_b^2 n_b\ln\Lambda}{4\pi\epsilon_0^2 m_a^2 v^3}
```

{cite}`trubnikov1965,nrl2019`. In the binary model the particle undergoes one
collision per step, $|\Delta\mathbf u_\perp| = u\sin\Theta$ and, for small angles,
$\langle\sin^2\Theta\rangle \simeq 4\langle\delta^2\rangle$, so that
$\langle\Delta v_{a\perp}^2\rangle = 4(m_{ab}/m_a)^2u^2\langle\delta^2\rangle$.
Equating the two rates with $u \simeq v$ and $n = n_b$ gives exactly
{eq}`ta-variance`. Everything below is about arranging the pairs so that every
particle does see one collision's worth of scattering off the whole density of its
partner species.

## Pairing

Each step draws a fresh random order of the particles inside every cell, with one
sort per species per pair of species (by cell, then by random bits). Partners are
then read off directly from each cell's first index.

### A species with itself

As in Takizuka and Abe {cite}`takizuka1977`: particles 1-2, 3-4, … of the cell
collide. When the cell holds an odd number, three of them collide 1-2, 2-3 and 3-1,
each with half the variance {eq}`ta-variance`; each of the three is in two
collisions, so it too receives one collision's worth of scattering. The three are
done one after the other, each starting from the velocities the previous one left,
so every collision conserves momentum and energy exactly. A particle alone in its
cell does not collide.

### Two species

The two randomly ordered lists are matched rank for rank, once per particle.
Every particle in the shorter list collides; a random fraction
$N_{\rm short}/N_{\rm long}$ of the longer list does. The sampled pair density
[below](#unequal-weights-and-the-density-that-enters) accounts for those left
unmatched, so each species scatters off the whole density of the other on average.

No particle receives two simultaneous kicks. With equal weights every cell
conserves momentum and kinetic energy exactly even when its species counts differ.
The earlier scheme cycled through the shorter list and added kicks computed from
the same starting velocities; their cross terms created an energy error.

### Unequal weights and the density that enters

Pseudo-particles of different species, or particles a wall has partly collected,
carry different weights $w$. Following Nanbu and Yonemura {cite}`nanbu1998`, the
change is applied to each partner with probability $w_{\rm other}/\max(w_a, w_b)$,
which conserves momentum and energy on average instead of per pair. The density in
{eq}`ta-variance` is then

```{math}
:label: pair-density
n = \frac{n_a\, n_b}{n_{ab}}, \qquad
n_{ab} = \frac{c}{\Delta x}\sum_{\rm pairs\ in\ cell} f\,\min(w_i, w_j),
```

the form of Pérez et al. {cite}`perez2012`, with $f$ the fraction of the variance a
collision carries (½ in a triplet, 1 otherwise), $c = 1$ between two species and
$c = 2$ within one, where $n_a = n_b$ is the density of the species.

The rule follows from asking that every particle receive, on average, one
collision's worth of scattering off the whole density of the other species. With
weights $w_a$, $w_b$ and counts $N_a \ge N_b$ in a cell of length $\Delta x$: a
particle of $a$ is selected with probability $N_b/N_a$ and accepts with probability
$w_b/w_{\max}$, so it needs $(N_b/N_a)n\,w_b/w_{\max} = n_b$; a particle of $b$
is always selected and needs $n\,w_a/w_{\max} = n_a$.
Both give $n = N_a w_{\max}/\Delta x$, which is {eq}`pair-density` with $N_b$ pairs.

For equal weights this is $\max(n_a, n_b)$ between species and $n_a$ within one.
The larger scattering variance compensates for sampling fewer pairs; keep its
collisional time step small and check [convergence](#time-step). Dropping unmatched
particles without changing the density would scatter both species too slowly.

## Where in the step

The explicit step carries the position at half-integer times and the velocity at integer
times. One step is a kick centred at $t^{n+1/2}$ followed by two half drifts,

```{math}
\mathbf u^{n+1} = \mathbf u^{n} + \Delta t\,\tfrac{q}{m}\mathbf F(x^{n+1/2}), \qquad
x^{n+1} = x^{n+1/2} + \tfrac12\Delta t\, v^{n+1}, \qquad
x^{n+3/2} = x^{n+1} + \tfrac12\Delta t\, v^{n+1},
```

with the two half drifts merged into one in the code and the Boris rotation inside the kick. The collision operator $C$ changes velocities at fixed positions and conserves the
kinetic energy of every pair. It is inserted where both of its arguments are defined at the
same time, between the two half drifts:

```{math}
:label: collision-split
\mathbf u^{n+1}_{\rm kicked} = K\,\mathbf u^n, \qquad
x^{n+1} = x^{n+1/2} + \tfrac12\Delta t\, v^{n+1}_{\rm kicked}, \qquad
\mathbf u^{n+1} = C(x^{n+1})\,\mathbf u^{n+1}_{\rm kicked}, \qquad
x^{n+3/2} = x^{n+1} + \tfrac12\Delta t\, v^{n+1}.
```

The particles are paired in the cells of $x^{n+1}$, and the energy $\sum_p w_p(\tfrac12 m_p v_p^2 + q_p\phi(x_p))$
at $t^{n+1}$ is unchanged by the scattering, because the positions are. Between collisions the
leapfrog keeps its own bounded, second-order energy error.

Scattering at $x^{n+1/2}$ right after the kick, as the code did before, moves the integer-time position $x^{n+1} = x^{n+3/2} - \tfrac12\Delta t\,v^{n+1}$ with
the scattered velocity, and every collision changes the energy by
$\tfrac12\Delta t\sum_p q_p\mathbf E(x_p)\cdot\Delta\mathbf v_p = \tfrac12\Delta t\,\Delta\mathbf p_a\cdot
(q_a\mathbf E_a/m_a - q_b\mathbf E_b/m_b)$, which vanishes only for equal charge-to-mass ratios
in equal fields. Splitting the kick instead, $K(\Delta t/2)\,C\,K(\Delta t/2)$, is not an option
for a magnetised run: two Boris rotations of $\Delta t/2$ turn by
$4\arctan(\Omega\Delta t/4)$, not the $2\arctan(\Omega\Delta t/2)$ of one, so the gyration
phase would change with the collision model. {eq}`collision-split` leaves the kick alone.

The test (`tests/test_collisions.py`) is an isolated oscillator: electrons and a species of
25 electron masses and the same charge in one harmonic well
$E_x = k(x - x_0)$, all inside one cell, so that every collision between the two is exact,
at $\nu/\omega \approx 0.1$ over sixteen periods. The largest relative energy error is

| $\omega\Delta t$ | 0.4 | 0.2 | 0.1 |
|---|---|---|---|
| collisionless | $1.07\times10^{-3}$ | $2.45\times10^{-4}$ | $4.54\times10^{-5}$ |
| collisions at $x^{n+1}$ | $1.01\times10^{-3}$ | $1.96\times10^{-4}$ | $3.86\times10^{-5}$ |
| collisions at $x^{n+1/2}$ (before) | 22.1 | 6.83 | 1.53 |

second order with {eq}`collision-split` and at the collisionless level; the earlier placement
multiplied the energy by 23 at $\omega\Delta t = 0.4$.

The equal-weight collision operator now conserves kinetic energy even with unequal
cell counts (`tests/test_collisions.py`, including count ratios of nine to one and
unequal physical masses). This removes the drift from simultaneous reused partners;
it does not remove errors from the particle-field coupling or time discretisation.
Unequal weights retain the statistical energy and momentum conservation of the
acceptance rule [above](#unequal-weights-and-the-density-that-enters).

The implicit scheme collides after its step, with $x^{n+1}$ and $\mathbf u^{n+1}$ both at
$t^{n+1}$, so equal-weight scattering there too leaves the conserved energy unchanged.
The splitting is of first order, as in the explicit scheme before this correction, but exact in energy.

## The Coulomb logarithm

Left at `None`, $\ln\Lambda$ is the NRL electron-ion expression {cite}`nrl2019`,

```{math}
\ln\Lambda = \begin{cases}
23 - \ln\!\left(n_e^{1/2} Z\, T_e^{-3/2}\right), & T_e < 10 Z^2\ \mathrm{eV},\\
24 - \ln\!\left(n_e^{1/2}\, T_e^{-1}\right), & \text{otherwise},
\end{cases}
```

with $n_e$ in $\mathrm{cm^{-3}}$ and $T_e$ in eV, evaluated once, at the start of the
run, with $Z = 1$. The electrons are the lightest negatively charged species; their
temperature is $T_e = m_e v_{th}^2/2$ from the largest of the three components of
`vth`. A run with no negatively charged species needs `coulomb_log` given.

The expression turns small and then negative in a cold, dense plasma, where the
weak-coupling picture behind it fails; a negative logarithm would make the variance
negative. The result is kept at or above 2, the floor of Lee and More
{cite}`lee1984` that particle codes commonly adopt, and a zero temperature or density
gives the floor rather than a division by zero.

## Verification

The fast-beam limit above fixes both rates with no adjustable constant, which makes it
the sharpest available test. A beam a hundred times faster than its background slows
at $\nu_s = (1 + m_a/m_b)\nu_0$ and spreads at $\nu_\perp = 2\nu_0$:

```{figure} ../_static/figures/collisions.png
:width: 100%
:alt: Beam slowing down and perpendicular diffusion against the Fokker-Planck rates

(a) Slowing down and (b) perpendicular diffusion of a fast beam, for equal masses and
for a background a hundred times heavier. Lines are the Fokker-Planck rates, points
are the operator.
```

Measured against theory, the ratios are {{ collisions_nu_slow_ratio_equal_mass }} and
{{ collisions_nu_perp_ratio_equal_mass }} for equal masses,
{{ collisions_nu_slow_ratio_heavy }} and {{ collisions_nu_perp_ratio_heavy }} for a
background a hundred times heavier: at most
{{ collisions_max_deviation_percent }} % away.

The test suite repeats this check, and pins the pairing itself
(`tests/test_collisions.py`): every particle collides with a partner in its own cell
at 50 000 cells and 100 000 particles; within a species every live particle takes
part in exactly one collision or two half collisions, and one step conserves momentum
and energy to round-off, at 2, 3, 4, 10 and 100 particles per cell; with unequal cell
counts and equal weights, each cell conserves energy and momentum; with unequal
numbers and weights each species scatters off the density of the other to within
the statistical error; gradients stay finite for identical velocities; and the
oscillator of [where in the step](#where-in-the-step) keeps its collisionless energy error.

## Time step

The scheme assumes $\langle\delta^2\rangle \ll 1$ for the pairs that matter. Because
{eq}`ta-variance` goes as $u^{-3}$, the slowest pairs always violate it; they are
scattered through a large angle, which is harmless because energy is still conserved,
but the rate they represent saturates. Keep $\nu\Delta t$ below about $10^{-2}$ for
the bulk of the distribution, where $\nu$ is the collision frequency at the thermal
speed, and check the answer against a smaller step. For round-off safety the
variance is capped at $10^{30}$, an angle within $10^{-15}$ of $\pi$, and the relative
speed is floored at $10^{-10}$ m/s.

Note that collisional and plasma timescales are far apart in a weakly coupled plasma:
$\nu/\omega_{pe} \sim \ln\Lambda/(n\lambda_D^3)$, which is $10^{-5}$ or smaller for
most laboratory plasmas. Resolving both in one run is expensive, and often the point
of a collisional study is a transport coefficient that can be measured on a small
patch of plasma rather than a full device.
