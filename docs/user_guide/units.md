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

## Constants

```python
from jaxincell import (epsilon_0, mu_0, speed_of_light, elementary_charge,
                       mass_electron, mass_proton, boltzmann_constant)
```

CODATA 2018 values, exactly as published.

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

the number of physical particles it stands for. `Output.charge` and `Output.mass` are
the *pseudo-particle* charge and mass, that is $q w$ and $m w$; divide by
`Output.weight` for the physical ones.

## Energies per unit area

The simulation is one-dimensional, so an energy is an energy per unit area of the
$y$-$z$ plane, in J/m². Ratios — the relative energy error, the fraction in the field
— are unaffected, and those are what the diagnostics are usually used for.

## Precision

`jax_enable_x64` is switched on when the package is imported. Double precision is what
makes the Gauss residual sit at $10^{-12}$ rather than $10^{-4}$ and the implicit
energy error reach round-off; single precision would give up both. Override the flag
before importing `jaxincell` if a run genuinely does not need them.

## Frequencies

Angular, in rad/s, throughout: `plasma_frequency()`, `dominant_frequency()` and every
$\omega$ in this documentation. Rates are per second; when a figure axis reads
$t\,\omega_{pe}$, the time has been multiplied by the angular plasma frequency.
