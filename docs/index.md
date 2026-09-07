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
Landau damping, two-stream, bump-on-tail, Weibel, optimisation and inference.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
`Simulation`, `diagnostics`, `plot` and the numerical kernels.
:::
::::

```{figure} _static/figures/two_stream.png
:width: 100%
:alt: Two-stream instability simulated with JAX-in-Cell

Two counter-streaming electron beams, from `examples/input.toml`. (a) Electrostatic
energy with the growth rate measured in the linear phase and the rate predicted by
the kinetic dispersion relation. (b-d) Electron phase space before, during and after
the instability. See {doc}`numerics/verification` for how these numbers are obtained.
```

## What the code does

JAX-in-Cell advances charged pseudo-particles in one spatial dimension and three
velocity components under the Lorentz force, and advances the electric and magnetic
fields on a staggered grid with Maxwell's equations. Charge and current are deposited
with a quadratic spline, the current deposit satisfies the discrete continuity
equation, and an optional compensated binomial filter removes short-wavelength noise.

Two time integrators are available: an explicit leapfrog scheme with the Boris pusher
(non-relativistic or relativistic) and an implicit Crank-Nicolson scheme solved by Picard
iteration, which conserves energy to round-off and has no light-wave time-step limit.
Boundaries can be periodic, reflective or absorbing, chosen separately for particles
and fields. Any number of electron and ion populations can be defined, each with its own
density, drift, temperature anisotropy and seed.

Because the entire simulation is a pure JAX function, it can be differentiated with
`jax.grad` with respect to physical inputs such as drift speeds, temperatures,
perturbation amplitudes or the full initial phase space. The
{doc}`user_guide/differentiation` page shows how, and the
{doc}`examples/index` include a parameter scan, a gradient check against finite
differences, an optimisation and an inverse problem.

## Quick look

```python
from jaxincell import Simulation, diagnostics, plot

parameters = {
    "domain_parameters": {"length": 1e-2, "number_grid_points": 64, "total_steps": 1000,
                          "timestep_over_spatialstep_times_c": 1.0},
    "species_parameters": {
        "electrons": {"electrons0": {"number_pseudoparticles": 5000, "vth_over_c_x": 0.05,
                                     "drift_speed_x": 6e7, "velocity_plus_minus_x": True,
                                     "grid_points_per_Debye_length": 0.5,
                                     "perturbation_amplitude_x": 5e-7, "perturbation_wavenumber_x": 1}},
        "ions": {"ions0": {"number_pseudoparticles": 5000, "grid_points_per_Debye_length": 0.5,
                           "vth_over_c_x": "_electrons0"}},
    },
}

output = Simulation(parameters).run()   # compiled with XLA on first call
diagnostics(output)                     # energies, species split, dominant frequency
plot(output)                            # animated fields, distributions and phase space
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
