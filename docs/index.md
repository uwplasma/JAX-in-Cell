---
html_theme.sidebar_secondary.remove: true
---

# JAX-in-Cell

```{image} _static/JAX-in-Cell_logo.png
:width: 420px
:align: center
:alt: JAX-in-Cell
:class: only-light
```

```{image} _static/JAX-in-Cell_logo_dark.png
:width: 420px
:align: center
:alt: JAX-in-Cell
:class: only-dark
```

<p class="lead" style="text-align:center; max-width: 46rem; margin: 1rem auto;">
A one-dimensional, three-velocity (1D3V) electromagnetic particle-in-cell code written in JAX.
It runs on CPUs, GPUs and TPUs, compiles the whole time loop with XLA, and is differentiable end to end.
</p>

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Install
:link: getting_started/quickstart
:link-type: doc
`pip install jaxincell`, from source, or with GPU support, and three short runs.
:::

:::{grid-item-card} First simulation
:link: getting_started/first_simulation
:link-type: doc
Run the two-stream instability, read the diagnostics, make a plot.
:::

:::{grid-item-card} Numerical methods
:link: numerics/index
:link-type: doc
The equations, the Yee grid, the Boris and Crank-Nicolson schemes, deposition, filtering, boundaries.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc
Every input parameter, the output dictionary, plotting, gradients, performance.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc
Landau damping, two-stream, bump-on-tail, Weibel, collisions and optimisation.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
`Domain`, `Species`, `Solver`, `Simulation` and the numerical kernels.
:::
::::

```{figure} _static/figures/two_stream.png
:width: 100%
:alt: Two-stream instability simulated with JAX-in-Cell

Two counter-streaming electron beams. (a) The seeded mode, with the measured growth
rate and the rate predicted by the kinetic dispersion relation, which agree to
{{ two_stream_gamma_deviation_percent }} per cent. (b) The electron phase space after
saturation. {doc}`numerics/verification` shows how every number in this documentation
is produced.
```

## What the code does

JAX-in-Cell advances charged pseudo-particles in one spatial dimension and three velocity
components under the Lorentz force, and advances the electric and magnetic fields on a
staggered grid with Maxwell's equations.

| | |
|---|---|
| deposition | quadratic spline; the current deposit satisfies the discrete continuity equation |
| filtering | optional compensated binomial filter, to remove short-wavelength noise |
| collisions | binary Coulomb, through the Takizuka-Abe operator |
| integrators | explicit leapfrog with the Boris pusher (non-relativistic or relativistic), and implicit Crank-Nicolson solved by Picard iteration |
| boundaries | periodic, reflective, absorbing or thermal, chosen separately for particles and fields |
| species | any number of electron and ion populations, each with its own density, drift, temperature anisotropy and seed |

* The implicit scheme conserves energy to round-off and has no light-wave time-step limit.
* An absorbing wall can return part of each particle by a law in its impact speed.
* Because the entire simulation is a pure JAX function, `jax.grad` differentiates it with
  respect to drift speeds, temperatures, perturbation amplitudes, external field profiles
  or the full initial phase space. See {doc}`user_guide/differentiation`, and
  {doc}`examples/optimize_two_stream` for an inverse problem whose answer is known from
  linear theory.

## What it is checked against

Every rate and frequency quoted in this documentation is measured against a closed-form or
linear kinetic result, not against another simulation.

| case | reference | agreement |
|---|---|---|
| Landau damping rate at $k\lambda_D = 0.5$ | kinetic root | {{ landau_gamma_deviation_percent }} % |
| Landau frequency at $k\lambda_D = 0.5$ | kinetic root | {{ landau_omega_deviation_percent }} % |
| two-stream growth rate, across the unstable range | kinetic root | {{ two_stream_scan_mean_deviation_percent }} % mean, {{ two_stream_scan_max_deviation_percent }} % worst |
| Weibel growth rate, {{ weibel_modes_compared }} of {{ weibel_modes_run }} seeded runs that grow cleanly | transverse kinetic root | {{ weibel_mean_deviation_percent }} % mean, {{ weibel_max_deviation_percent }} % worst |
| the four collisional relaxation rates | Fokker-Planck | {{ collisions_max_deviation_percent }} % at worst |

{doc}`numerics/verification` collects them; {doc}`examples/index` runs them.

## Quick look

```python
import numpy as np
from jaxincell import Domain, Simulation, Solver, Species, diagnostics, plot, speed_of_light as c

electrons = Species.electrons(n=10000, density=4.37e17, vth=(0.05 * c, 0, 0),
                              drift=(6e7, 0, 0), plus_minus=True,
                              perturbation_amplitude=5e-7, perturbation_mode=1)
ions = Species.ions(n=10000, density=4.37e17, electrons=electrons)

simulation = Simulation(Domain(length=0.01, cells=64, dt_over_dx_c=4.5),
                        [electrons, ions], Solver(filter_passes=2))

output = simulation.run(1000, seed=0)   # compiled with XLA on the first call
diagnostics(output)                     # energies, momentum, Gauss residual, temperatures
plot(output)                            # animated fields, distributions and phase space
```

Every physical parameter is a JAX pytree leaf, so changing one does not recompile and
`jax.grad` differentiates through the whole run:

```python
import jax, jax.numpy as jnp

gradient = jax.grad(lambda s: jnp.sum(s.run(200, seed=0).E ** 2))(simulation)
print(gradient.species[0].drift, gradient.domain.length)
```

```{toctree}
:hidden:
:maxdepth: 2

getting_started/quickstart
user_guide/index
numerics/index
examples/index
api
development
```
