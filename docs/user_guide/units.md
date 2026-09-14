# Units and conventions

Everything is SI, unmodified. There is no normalisation layer, so a number that goes in
or comes out is in the unit its physics has.

| quantity | unit |
|---|---|
| length, position | m |
| time | s |
| velocity, thermal speed, drift | m/s |
| number density | m⁻³ |
| charge | C, except `Species.charge`, which is in units of $e$ |
| mass | kg |
| electric field | V/m |
| magnetic field | T |
| current density | A/m² |
| charge density | C/m³ |
| energy | J/m², an energy per unit area of the ignorable directions |
| temperature (diagnostic output) | eV |

The physical constants importable from `jaxincell` are the CODATA 2018 values
({doc}`../api`).

## Thermal speed

The one convention worth stating twice, because codes differ:

```{math}
f(v) \propto \exp\!\left(-\frac{(v-u)^2}{v_{th}^2}\right), \qquad
v_{th} = \sqrt{\frac{2k_BT}{m}}, \qquad
\lambda_D = \frac{v_{th}}{\sqrt{2}\,\omega_p} = \sqrt{\frac{\epsilon_0 k_B T}{n q^2}} .
```

`vth` is therefore $\sqrt2$ times the standard deviation of one velocity component.
From a temperature in electronvolts:

```python
import numpy as np
from jaxincell import elementary_charge, mass_electron

vth = np.sqrt(2 * T_ev * elementary_charge / mass_electron)
```

and back, which is what {func}`~jaxincell.temperatures` reports:

```python
T_ev = mass_electron * vth ** 2 / (2 * elementary_charge)
```

## Weights

A {class}`~jaxincell.Species` of `n` pseudo-particles at number density `density`
in a box of length `length` gives each pseudo-particle the weight

```{math}
w = \frac{n_{\rm phys}\,L}{N},
```

the number of physical particles it stands for, per unit area of the $y$-$z$ plane.
`Output.charge` and `Output.mass` are those of one physical particle, and
`Output.weight` is the weight of every pseudo-particle at every stored step: an
absorbing wall lowers it as it collects the particle, to zero when it keeps the particle
whole. The charge a pseudo-particle carries is `charge * weight`.

## Energies per unit area

The simulation is one-dimensional, so an energy is an energy per unit area of the
$y$-$z$ plane, in J/m². Ratios — the relative energy error, the fraction in the field
— are unaffected, and those are what the diagnostics are usually used for.

## Precision

`jax_enable_x64` is switched on when the package is imported, unless JAX's own switch
says otherwise: start Python with `JAX_ENABLE_X64=0` for single precision. Double
precision is what makes the Gauss residual sit at $10^{-12}$ rather than about
$10^{-3}$, and the implicit energy error at round-off rather than $10^{-7}$. The physics
does not change: every rate, frequency and sheath comparison in the test suite passes in
single precision too, and only the tests that check conservation to round-off fail.
See {doc}`performance` for what it costs on a GPU.

Every script in `examples/` sets `JAX_ENABLE_X64` at its top, before anything imports
JAX, so the precision of a run is written in the script, and
`JAX_ENABLE_X64=0 python examples/two_stream.py` switches it from the shell.

## Frequencies

Angular, in rad/s, throughout: `plasma_frequency()`, `dominant_frequency()` and every
$\omega$ in this documentation. Rates are per second; when a figure axis reads
$t\,\omega_{pe}$, the time has been multiplied by the angular plasma frequency.
