# Wall reflection

`examples/wall_reflection.py`

A real surface does not collect every electron that reaches it. Some are reflected, and
slow electrons more readily than fast ones {cite}`cimino2004,furman2002`. A species can
carry a reflection law, the fraction of a particle an absorbing wall returns, as a
number or as a function of the impact speed, and a wall's restitution scales the speed
of what comes back. This example checks what such a wall hands back against two
formulas that have no free parameter.

```{figure} ../_static/figures/wall_reflection.png
:width: 100%
:alt: Returned fraction of particles and energy against the width of the reflection law, and the impact speeds of what was returned and collected

(a) The fraction of the electrons reaching a wall that come back, for a Gaussian law of
width $u$, against its flux average (solid) and its average over the distribution
(dotted). (b) The fraction of the normal energy flux that comes back, with restitution
$e = $ {{ reflection_restitution }}. (c) At $u = \sigma$, the impact speeds of the part
returned and the part collected, against $R(v)\,v f(v)$ and $v f(v)$.
```

## What is being tested

**The wall samples the flux.** From a uniform plasma with velocity distribution $f$,
the particles that reach a wall in a short time with normal speed near $v$ are in
proportion to $v f(v)$: fast particles arrive from further away. A law $R(v)$ therefore
returns its flux average, which for a Maxwellian of spread $\sigma$ and the Gaussian law
$R = e^{-v^2/2u^2}$ is

```{math}
R_{\rm eff} = \frac{1}{\sigma^2}\int_0^\infty R(v)\,v\,e^{-v^2/2\sigma^2}\,dv = \frac{u^2}{u^2+\sigma^2} .
```

Averaging over the distribution instead would give $u/\sqrt{u^2+\sigma^2}$,
{{ reflection_distribution_average_sigma }} rather than one half at $u = \sigma$. The run
returns {{ reflection_returned_sigma }} there and follows the flux average to within
{{ reflection_max_error }} at all five widths.

**Restitution takes the energy.** What comes back has its normal velocity multiplied by
$-e$. The energy flux weights the speeds once more by $v^2$, so the returned share of it
is

```{math}
e^2\,\frac{\int_0^\infty R(v)\,v^3 e^{-v^2/2\sigma^2}\,dv}{\int_0^\infty v^3 e^{-v^2/2\sigma^2}\,dv}
= e^2\left(\frac{u^2}{u^2+\sigma^2}\right)^2 ,
```

which the run follows to within {{ reflection_energy_max_error }}.

The flux average is the coefficient that sets the floating potential of a wall
{cite}`hobbs1967`; {doc}`sheath` measures that.

## The setup

Electrons at $10^6\ \mathrm{m^{-3}}$, too tenuous for any field to act, fill the box as a
quiet-start Maxwellian and run into both walls for a tenth of a thermal transit, so that
nothing arrives twice. {{ reflection_hits }} of the {{ reflection_particles }} reach a
wall. A pseudo-particle's weight is multiplied by $R$ at the wall, so the returned
fraction is read straight from the weights, with no counting noise.

The law is an ordinary function of the speed, written with `jax.numpy`:

```python
law = lambda speed: jnp.exp(-speed ** 2 / (2 * u ** 2))
electrons = Species.electrons(n=n, density=1e6, vth=(np.sqrt(2) * sigma, 0, 0), reflection=law)
```

## Running it

```bash
python examples/wall_reflection.py
```

A few seconds; each width compiles its own program, since the law is part of it.

## Things to try

* A constant `reflection=0.5` returns one half at every width: a number has no speed
  to weight.
* A step, `lambda s: jnp.where(s < v_c, 1.0, 0.0)`, returns $1 - e^{-v_c^2/2\sigma^2}$.
* `reflection=(law, 0.0)` reflects at the left wall only; the right one collects
  everything.
