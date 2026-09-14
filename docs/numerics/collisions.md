# Coulomb collisions

Passing a {class}`~jaxincell.Collisions` object to a
{class}`~jaxincell.Simulation` adds binary Coulomb collisions after the particle
push, by the Monte Carlo scheme of Takizuka and Abe {cite}`takizuka1977`.

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

The longer of the two lists in the cell drives. Each of its particles collides once,
with the partner of the same rank in the other list, cycling through that list when
it is shorter (`rank % count`). A particle of the shorter list thus collides
$N_{\rm long}/N_{\rm short}$ times per step on average. Simply dropping the
unmatched particles, which is what a rank-for-rank match does, would leave two thirds
of them uncollided at a count ratio of three to one.

The several collisions of a particle of the shorter list are computed from the
velocities at the start of the step and added. Momentum stays exact; kinetic energy
is exact in cells holding as many particles of both species, and otherwise carries an
error of second order in the scattering angle. Exactness would need as many
sequential passes as the largest count ratio of any cell, which is not known when
the program is compiled.

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
particle of $a$ collides once and accepts with probability $w_b/w_{\max}$, so it needs
$n\,w_b/w_{\max} = n_b = N_b w_b/\Delta x$; a particle of $b$ collides $N_a/N_b$
times and accepts with $w_a/w_{\max}$, so it needs $(N_a/N_b)\,n\,w_a/w_{\max} = n_a$.
Both give $n = N_b w_{\max}/\Delta x$, which is {eq}`pair-density` with $N_a$ pairs.

For equal weights {eq}`pair-density` reduces to $\min(n_a, n_b)$ between species, as
in the original scheme, and to $n_a$ within one. Note what does the work in that
case: the acceptance is one, and a particle of the shorter list sees the larger
density through the number of its collisions. The plain $\min(n_a, n_b)$ is wrong as
soon as the shorter list also carries the smaller weight: at four times fewer and four
times lighter particles, it makes both species scatter four times too slowly.

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
and energy to round-off, at 2, 3, 4, 10 and 100 particles per cell; with unequal
numbers and weights each species scatters off the density of the other to within
the statistical error; and gradients stay finite for identical velocities.

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
