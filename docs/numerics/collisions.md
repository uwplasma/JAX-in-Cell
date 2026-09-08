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
it is taken from the NRL formulary {cite}`nrl2019` at the density and temperature of
the first species.

## The scheme

Inside each cell the particles of the two species are paired at random. Each pair is
scattered by rotating its *relative* velocity $\mathbf u = \mathbf v_a - \mathbf v_b$
through a random angle $\Theta$ about a random azimuth, and sharing the change in
inverse proportion to the masses:

```{math}
\mathbf v_a \to \mathbf v_a + \frac{m_b}{m_a + m_b}\Delta\mathbf u, \qquad
\mathbf v_b \to \mathbf v_b - \frac{m_a}{m_a + m_b}\Delta\mathbf u .
```

Because $|\mathbf u|$ is unchanged, every pair conserves momentum and kinetic energy
**exactly**, whatever the time step and however large the angle. That is the reason
for scattering pairs rather than drawing a random kick per particle.

The angle is drawn through $\delta = \tan(\Theta/2)$, with $\delta$ Gaussian of zero
mean and variance

```{math}
:label: ta-variance
\langle\delta^2\rangle = \frac{q_a^2 q_b^2\, n\, \ln\Lambda}{8\pi\epsilon_0^2 m_{ab}^2 u^3}\,\Delta t,
\qquad m_{ab} = \frac{m_a m_b}{m_a + m_b}.
```

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
Equating the two rates with $u \simeq v$ gives exactly {eq}`ta-variance`.

### Which density enters

$n$ in {eq}`ta-variance` is the density of field particles a given particle sees:

* **Two different species.** The sparser of the two, $n = \min(n_a, n_b)$, as in the
  original scheme.
* **A species colliding with itself.** The whole species density. The code splits the
  species in two halves and pairs them, so both halves have to be counted; using the
  density of one half alone would halve every collision rate.

### Unequal particle numbers and weights

A cell rarely holds the same number of pseudo-particles of both species. The longer
list drives the pairing, cell by cell, and the shorter one is cycled through
(`rank % count_other`), so that **every** particle of the longer list collides once.
Simply dropping the unmatched particles, which is what a rank-for-rank match does,
would leave two thirds of them uncollided at a count ratio of three to one.

Cycling makes a particle of the shorter list take part in several collisions per
step, which is correct as long as each is accepted with the right probability. The
change is applied to a partner with probability $w_{\rm other}/\max(w_a, w_b)$, the
weight correction of Nanbu and Yonemura {cite}`nanbu1998`. The two effects cancel: a
species represented by three times fewer pseudo-particles at the same density has
three times the weight, is used three times as often, and accepts one collision in
three.

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
{{ collisions_max_deviation_percent }} % away. The test suite repeats this check and
also verifies that momentum and energy are conserved to round-off.

## Time step

The scheme assumes $\langle\delta^2\rangle \ll 1$ for the pairs that matter. Because
{eq}`ta-variance` goes as $u^{-3}$, the slowest pairs always violate it; they are
scattered through a large angle, which is harmless because energy is still conserved
exactly, but the rate they represent saturates. Keep $\nu\Delta t$ below about
$10^{-2}$ for the bulk of the distribution, where $\nu$ is the collision frequency at
the thermal speed, and check the answer against a smaller step.

Note that collisional and plasma timescales are far apart in a weakly coupled plasma:
$\nu/\omega_{pe} \sim \ln\Lambda/(n\lambda_D^3)$, which is $10^{-5}$ or smaller for
most laboratory plasmas. Resolving both in one run is expensive, and often the point
of a collisional study is a transport coefficient that can be measured on a small
patch of plasma rather than a full device.
