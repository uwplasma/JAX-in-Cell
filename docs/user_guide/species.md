# Species

A {class}`~jaxincell.Species` is one population of pseudo-particles. Two named
constructors cover the common cases; the general one covers the rest.

```python
from jaxincell import Species, speed_of_light as c

electrons = Species.electrons(n=20000, density=1e17, vth=(0.05 * c, 0, 0))
ions      = Species.ions(n=20000, density=1e17, electrons=electrons)
```

`Species.ions` derives $v_{th,i} = v_{th,e}\sqrt{(T_i/T_e)(m_e/m_i)}$ from the electrons,
so the two are in thermal equilibrium unless `temperature_ratio` says otherwise. Pass
`vth` to override it, and `mass_ratio` for a species other than protons; `mass_ratio=1e9`
gives an immobile neutralising background.

## Arguments

| argument | meaning | default |
|---|---|---|
| `name` | label used in the output and by `Collisions` (static) | — |
| `n` | number of pseudo-particles (static) | — |
| `charge` | charge in units of the elementary charge | — |
| `mass` | mass in kilograms | — |
| `density` | number density in m⁻³; the weight is `density * length / n` | — |
| `vth` | thermal speed per component, $\sqrt{2k_BT/m}$; a bare number is $x$ alone | `(0, 0, 0)` |
| `drift` | drift velocity per component, m/s | `(0, 0, 0)` |
| `perturbation_amplitude` | amplitude $a$ of the displacement $x \to x + a\sin(2\pi m x/L)$ | `0.0` |
| `perturbation_mode` | mode number $m$ of that displacement | `0.0` |
| `plus_minus` | negate $v_x$ on every second particle: two counter-streaming beams (static) | `False` |
| `sampling` | how the initial phase space is drawn: `"quiet"`, `"lattice"` or `"random"` (static) | `"lattice"` |
| `x`, `v` | arrays of shape `(n, 3)` replacing the generated phase space | `None` |
| `reflection` | fraction of each particle an absorbing wall sends back: a number, a function of the normal impact speed in m/s, or a `(left, right)` pair | `0.0` |

* `vth` and `drift` take three components as a tuple, list or array (NumPy or JAX) whose
  last axis holds them.
* A bare number is the **x component alone**, the direction the grid resolves, the other
  two zero: `vth=2e6` is hot along $x$ and cold across it; isotropic is
  `vth=(2e6, 2e6, 2e6)`.
* Invalid input — no particles, a phase-space array of the wrong shape, a coefficient
  outside $[0, 1]$, an unknown wall — raises `ValueError` on construction and on
  `replace`, so the checks survive `python -O`.
* Arrays may carry leading ensemble axes; a tracer is checked only for its shape.

## Thermal speed and temperature

```{math}
f(v) \propto \exp\!\left(-\frac{(v-u)^2}{v_{th}^2}\right), \qquad
v_{th} = \sqrt{\frac{2k_BT}{m}}, \qquad
\lambda_D = \frac{v_{th}}{\sqrt2\,\omega_p},
```

so `vth` is $\sqrt2$ times the standard deviation. From a temperature in electronvolts:

```python
import numpy as np
from jaxincell import elementary_charge, mass_electron

vth = np.sqrt(2 * T_ev * elementary_charge / mass_electron)
```

## Seeding a mode

`perturbation_amplitude` is a **displacement** in metres, not a density. To first order
it gives $\delta n/n = -ak\cos(kx)$ with $k = 2\pi m/L$, so the dimensionless seed quoted
in the literature is $ak$:

```python
seed = 0.01                                    # a k, one per cent
Species.electrons(..., perturbation_mode=1,
                  perturbation_amplitude=seed * length / (2 * np.pi))
```

## How the phase space is drawn

| `sampling` | positions | velocities |
|---|---|---|
| `"quiet"` | an even lattice | the quantiles of the Maxwellian, in a bit-reversed order |
| `"lattice"` (the default) | an even lattice | drawn at random |
| `"random"` | uniformly at random | drawn at random |

* `"quiet"` drops the noise floor by orders of magnitude, which is what makes a growth
  rate measurable over more than a couple of e-foldings: the right choice for anything
  compared against linear theory.
* It is the wrong one when the noise itself is the point — a survey of unseeded modes
  needs a floor to grow out of. See {doc}`../numerics/initialization`.

## Custom phase space

```python
from jaxincell import quiet_start

x, v = quiet_start(n, length, vth=(vx, 0.0, vz))
v[:, 2] += 1e-3 * vz * np.sin(2 * np.pi * x[:, 0] / length)
electrons = Species.electrons(n=n, density=n_e, vth=(vx, 0.0, vz)).replace(x=x, v=v)
```

* `x` and `v` replace the generated phase space entirely, so `vth`, `drift`,
  `perturbation_*`, `plus_minus` and `sampling` are ignored for the sampling.
* `vth` and `density` are still what the linear-theory helpers and the Debye-length
  property read, so keep them consistent with the arrays.

## Walls that send particles back

An absorbing wall collects everything unless the species says otherwise. `reflection` is
the fraction of each particle the wall returns, bounced as a reflective wall would; the
wall keeps the rest of the weight.

```python
import jax.numpy as jnp

Species.electrons(..., reflection=0.25)          # a quarter of every electron, at both walls
Species.electrons(..., reflection=(0.0, 0.5))    # only the right wall reflects
sigma = vth / np.sqrt(2)                         # slow electrons come back, fast ones stay
Species.electrons(..., reflection=lambda speed: jnp.exp(-speed ** 2 / (2 * sigma ** 2)))
```

* A number is a pytree leaf, traced and differentiable like any physical parameter, and
  so is the number in a mixed pair such as `(law, 0.3)`.
* A function is compiled into the program: write it with `jax.numpy`, and define it once,
  because a new function object is a new program.
* A wall sees the flux, not the distribution, so a velocity-dependent law returns its
  flux average from a Maxwellian: $u^2/(u^2+\sigma^2)$ for a Gaussian of width $u$ — one
  half for the law above, not the 0.71 an average over the distribution would suggest
  ({doc}`../numerics/boundaries`).

## Changing a species

Every configuration object is frozen and has `.replace()`:

```python
hotter = electrons.replace(vth=(2 * electrons.vth[0], 0, 0))
```

Physical fields are pytree leaves, so replacing them does not recompile, and `jax.grad`
differentiates through them ({doc}`differentiation`).

## Ensembles of species

JAX rebuilds a species from its leaves without rechecking them, so the tree functions
work on it directly. Stacking members gives an ensemble for `jax.vmap`; a template with
`None` in every leaf says which leaves carry the ensemble axis:

```python
import jax, jax.numpy as jnp

ensemble = electrons.replace(density=jnp.array([1e17, 2e17]), x=jnp.stack([x_a, x_b]))
axes = jax.tree.map(lambda _: None, electrons).replace(density=0, x=0)
fields = jax.vmap(lambda s: Simulation(domain, [s, ions]).run(500).E[-1, :, 0],
                  in_axes=(axes,))(ensemble)
```

* A species with `None` where it always holds a number, like `axes`, is a template:
  stored as given, so its integer axes stay integers.
* `jax.tree.map(lambda *m: jnp.stack(m), a, b)` stacks every leaf of two species, for a
  `vmap` with `in_axes=0`.
