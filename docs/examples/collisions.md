# Collisions

`examples/collisions.py`

The Takizuka-Abe collision operator checked against the Fokker-Planck relaxation
rates, in the limit where they are closed-form and contain no adjustable constant.

```{figure} ../_static/figures/collisions.png
:width: 100%
:alt: Beam slowing down and perpendicular diffusion against theory

(a) Slowing down. (b) Perpendicular diffusion. Lines are theory, points the operator.
```

## The test

A beam a hundred times faster than the background it scatters off slows at
$\nu_s = (1 + m_a/m_b)\nu_0$ and spreads in angle at $\nu_\perp = 2\nu_0$, with

```{math}
\nu_0 = \frac{q_a^2q_b^2 n_b\ln\Lambda}{4\pi\epsilon_0^2 m_a^2 v^3}
```

{cite}`trubnikov1965,nrl2019`. Matching $\nu_\perp$ is what fixes the variance of the
scattering angle in the first place ({doc}`../numerics/collisions`), so this is a real
check of the implementation rather than a tautology: the measured ratios are
{{ collisions_nu_slow_ratio_equal_mass }} and
{{ collisions_nu_perp_ratio_equal_mass }} for equal masses.

The example calls the operator directly rather than through a simulation, because
collisional and plasma timescales are five orders of magnitude apart in a weakly
coupled plasma and a run that resolved both would be enormous. That separation is
physics, not a limitation of the code.

## Using collisions in a simulation

```python
from jaxincell import Collisions
simulation = Simulation(domain, [electrons, ions], solver,
                        collisions=Collisions(pairs=(("electrons", "ions"),)))
```

Keep `dt_over_dx_c <= 1` when collisions are on: they scatter velocity into the
transverse directions, which excites the light-wave branch that an electrostatic run
was safely ignoring. See {doc}`../user_guide/collisions`.
