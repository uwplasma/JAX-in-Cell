# Species

A {class}`~jaxincell.Species` is one population of pseudo-particles. Two constructors
cover the common cases and the general one covers the rest.

```python
from jaxincell import Species, speed_of_light as c

electrons = Species.electrons(n=20000, density=1e17, vth=(0.05 * c, 0, 0))
ions      = Species.ions(n=20000, density=1e17, electrons=electrons)
```

`Species.ions` derives the ion thermal speed from the electrons,
$v_{th,i} = v_{th,e}\sqrt{(T_i/T_e)(m_e/m_i)}$, so that the two are in thermal
equilibrium unless `temperature_ratio` says otherwise. Pass `vth` explicitly to
override it, and `mass_ratio` for a species other than protons — `mass_ratio=1e9`
is the usual way to get an immobile neutralising background.

## Arguments

| argument | meaning | default |
|---|---|---|
| `name` | label used in the output and by `Collisions` (static) | — |
| `n` | number of pseudo-particles (static) | — |
| `charge` | charge in units of the elementary charge | — |
| `mass` | mass in kilograms | — |
| `density` | number density in m⁻³; the weight is `density * length / n` | — |
| `vth` | thermal speed per component, $\sqrt{2k_BT/m}$ | `(0, 0, 0)` |
| `drift` | drift velocity per component, m/s | `(0, 0, 0)` |
| `perturbation_amplitude` | amplitude $a$ of the displacement $x \to x + a\sin(2\pi m x/L)$ | `0.0` |
| `perturbation_mode` | mode number $m$ of that displacement | `0.0` |
| `plus_minus` | negate $v_x$ on every second particle: two counter-streaming beams (static) | `False` |
| `quiet` | quiet start (static) | `False` |
| `random_positions` | uniformly random rather than equally spaced positions (static) | `False` |
| `x`, `v` | arrays of shape `(n, 3)` replacing the generated phase space | `None` |

## Thermal speed and temperature

The convention throughout is

```{math}
f(v) \propto \exp\!\left(-\frac{(v-u)^2}{v_{th}^2}\right), \qquad
v_{th} = \sqrt{\frac{2k_BT}{m}}, \qquad
\lambda_D = \frac{v_{th}}{\sqrt2\,\omega_p},
```

so `vth` is $\sqrt2$ times the standard deviation of the velocity distribution.
Converting from a temperature in electronvolts:

```python
import numpy as np
from jaxincell import elementary_charge, mass_electron

vth = np.sqrt(2 * T_ev * elementary_charge / mass_electron)
```

## Seeding a mode

`perturbation_amplitude` is a **displacement** in metres, not a density. To first
order it produces $\delta n/n = -ak\cos(kx)$ with $k = 2\pi m/L$, so the dimensionless
seed usually quoted in the literature is $ak$:

```python
seed = 0.01                                    # a k, one per cent
Species.electrons(..., perturbation_mode=1,
                  perturbation_amplitude=seed * length / (2 * np.pi))
```

## Quiet starts

`quiet=True` places positions on an even lattice and velocities at the quantiles of the
Maxwellian, following a bit-reversed sequence. The noise floor drops by orders of
magnitude, which is what makes a growth rate measurable over more than a couple of
e-foldings. It is the right default for anything compared against linear theory, and
the wrong one when the noise itself is the point — see
{doc}`../numerics/initialization`.

## Custom phase space

```python
from jaxincell import quiet_start

x, v = quiet_start(n, length, vth=(vx, 0.0, vz))
v[:, 2] += 1e-3 * vz * np.sin(2 * np.pi * x[:, 0] / length)
electrons = Species.electrons(n=n, density=n_e, vth=(vx, 0.0, vz)).replace(x=x, v=v)
```

`x` and `v` replace the generated phase space entirely, so `vth`, `drift`,
`perturbation_*`, `plus_minus` and `quiet` are then ignored for the sampling — though
`vth` and `density` are still what the linear-theory helpers and the Debye-length
property read, so keep them consistent with the arrays.

## Changing a species

Every configuration object is frozen and has `.replace()`:

```python
hotter = electrons.replace(vth=(2 * electrons.vth[0], 0, 0))
```

Because the physical fields are pytree leaves, replacing them does not trigger a
recompilation, and `jax.grad` differentiates through them ({doc}`differentiation`).
