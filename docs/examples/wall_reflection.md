# Wall reflection

A real surface does not collect every electron that reaches it, and slow electrons return
more readily than fast ones {cite}`cimino2004,furman2002`. This example checks what a
partly reflecting wall hands back against two formulas that have no free parameter.

```{figure} ../_static/figures/wall_reflection.png
:width: 100%
:alt: Returned fraction of particles and energy against the width of the reflection law, and the impact speeds of what was returned and collected

(a) The fraction of the electrons reaching a wall that come back, for a Gaussian law of
width $u$, against its flux average (solid) and its average over the distribution
(dotted). (b) The fraction of the normal energy flux that comes back, with restitution
$e$ = {{ reflection_restitution }}. (c) At $u = \sigma$, the impact speeds of the part
returned and the part collected, against $R(v)\,v f(v)$ and $v f(v)$.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| returned particle fraction at $u = \sigma$ | {{ reflection_returned_sigma }} | $u^2/(u^2+\sigma^2) = 1/2$, the flux average | — |
| returned particle fraction, all five widths | panel (a) | flux average | {{ reflection_max_error }} |
| returned energy flux, all five widths | panel (b) | $e^2\left[u^2/(u^2+\sigma^2)\right]^2$ | {{ reflection_energy_max_error }} |
| average over the distribution instead | — | $u/\sqrt{u^2+\sigma^2}$ = {{ reflection_distribution_average_sigma }} at $u=\sigma$ | ruled out |

**The wall samples the flux.** Particles reaching a wall with normal speed near $v$ arrive
in proportion to $v f(v)$, because fast ones come from further away, so a law $R(v)$
returns its flux average

```{math}
R_{\rm eff} = \frac{1}{\sigma^2}\int_0^\infty R(v)\,v\,e^{-v^2/2\sigma^2}\,dv = \frac{u^2}{u^2+\sigma^2}
```

for a Maxwellian of spread $\sigma$ and $R = e^{-v^2/2u^2}$. Averaging over the
distribution instead gives {{ reflection_distribution_average_sigma }} rather than one
half at $u = \sigma$, which the measurement excludes.

**Restitution takes the energy.** What comes back has its normal velocity multiplied by
$-e$, and the energy flux weights the speeds once more by $v^2$, so the returned share is
$e^2\left[u^2/(u^2+\sigma^2)\right]^2$.

The flux average is the coefficient that sets the floating potential of a wall
{cite}`hobbs1967`; {doc}`sheath_reflection` measures that, and
{doc}`../numerics/boundaries` derives the wall model.

## The setup

| | |
|---|---|
| particles | {{ reflection_particles }} |
| reaching a wall | {{ reflection_hits }} |
| density | $10^6\ \mathrm{m^{-3}}$, too tenuous for any field to act |
| duration | a tenth of a thermal transit, so nothing arrives twice |
| restitution | {{ reflection_restitution }} |

A pseudo-particle's weight is multiplied by $R$ at the wall, so the returned fraction is
read straight from the weights, with no counting noise. The law is an ordinary function of
the speed:

```python
law = lambda speed: jnp.exp(-speed ** 2 / (2 * u ** 2))
electrons = Species.electrons(n=n, density=1e6, vth=(np.sqrt(2) * sigma, 0, 0), reflection=law)
```

## How to run

```bash
python examples/2_intermediate/wall_reflection.py
```

A few seconds; each width compiles its own program, since the law is part of it. The
figure and the numbers come from `docs/scripts/fig_wall_reflection.py`, which runs this
setup.

## Things to try

* A constant `reflection=0.5` returns one half at every width: a number has no speed to
  weight.
* A step, `lambda s: jnp.where(s < v_c, 1.0, 0.0)`, returns $1 - e^{-v_c^2/2\sigma^2}$.
* `reflection=(law, 0.0)` reflects at the left wall only; the right one collects
  everything.
