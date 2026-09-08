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
:link: getting_started/installation
:link-type: doc
`pip install jaxincell`, from source, or with GPU support.
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
:link: api/index
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

JAX-in-Cell advances charged pseudo-particles in one spatial dimension and three
velocity components under the Lorentz force, and advances the electric and magnetic
fields on a staggered grid with Maxwell's equations. Charge and current are deposited
with a quadratic spline, the current deposit satisfies the discrete continuity
equation, and an optional compensated binomial filter removes short-wavelength noise. Binary
Coulomb collisions are available through the Takizuka-Abe operator.

Two time integrators are available: an explicit leapfrog scheme with the Boris pusher
(non-relativistic or relativistic) and an implicit Crank-Nicolson scheme solved by Picard
iteration, which conserves energy to round-off and has no light-wave time-step limit.
Boundaries can be periodic, reflective or absorbing, chosen separately for particles
and fields. Any number of electron and ion populations can be defined, each with its own
density, drift, temperature anisotropy and seed.

Because the entire simulation is a pure JAX function, it can be differentiated with
`jax.grad` with respect to physical inputs such as drift speeds, temperatures,
perturbation amplitudes, external field profiles or the full initial phase space. The
{doc}`user_guide/differentiation` page shows how, and
{doc}`examples/optimisation` solves an inverse problem whose answer is known from
linear theory.

Every rate and frequency quoted in this documentation is measured against the linear
kinetic dispersion relation, not against another simulation: Landau damping to
{{ landau_gamma_deviation_percent }} per cent, the two-stream growth rate to
{{ two_stream_scan_mean_deviation_percent }} per cent across the unstable range, the
Weibel rate to {{ weibel_mean_deviation_percent }} per cent, and the collision
operator to {{ collisions_max_deviation_percent }} per cent of the Fokker-Planck
rates. {doc}`numerics/verification` collects them.

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

getting_started/index
user_guide/index
numerics/index
examples/index
api/index
development/index
```
