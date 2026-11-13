# Parameters and sampling

What a run is made of: physical inputs (a temperature and a density, in SI units),
numerical inputs (cells per Debye length, $\omega_{pe}\Delta t$, particles per cell), the
scales the code derives from them, and the three ways of loading the particles. The same
electron plasma is loaded three ways and the loading noise is measured against what random
sampling predicts.

```{figure} ../_static/figures/parameters_and_sampling.png
:width: 60%
:alt: Electric field energy of a thermal plasma loaded three ways

The electric energy of the same thermal plasma, loaded at random, on a lattice with random
velocities, and with the quiet start. No wave is seeded: all of it is loading noise.
```

## What is measured against what

| `sampling` | density spread per cell | temperature error | electric energy at the end (J/m$^2$) |
|---|---|---|---|
| `"random"` | {{ sampling_random_density }} | {{ sampling_random_temperature }} | {{ sampling_random_energy }} |
| `"lattice"` (default) | {{ sampling_lattice_density }} | {{ sampling_lattice_temperature }} | {{ sampling_lattice_energy }} |
| `"low_noise"` | {{ sampling_low_noise_density }} | {{ sampling_low_noise_temperature }} | {{ sampling_low_noise_energy }} |
| random sampling predicts | $1/\sqrt{N_{\rm cell}}$ = {{ sampling_predicted_density }} | $\pm\sqrt{2/N}$ = {{ sampling_predicted_temperature }} | — |

Random positions fill the cells as a Poisson process, so the density spread is the
$1/\sqrt{N_{\rm cell}}$ it predicts; a lattice removes it to the round-off of cell edges.
Random velocities miss the requested temperature by about $\sqrt{2/N}$; the quiet start's
quantiles miss it by ten times less. The field noise follows, over an order of magnitude
between each loading.

## The inputs and the scales

The script takes $T_e$ = 10 eV and $n$ = $10^{18}$ m$^{-3}$ and derives, in the code's
convention $v_{th} = \sqrt{2T/m}$:

| | |
|---|---|
| $v_{th}$ | {{ sampling_v_th }} m/s |
| $\omega_{pe}$ | {{ sampling_omega_pe }} rad/s |
| $\lambda_D = v_{th}/(\sqrt2\,\omega_{pe})$ | {{ sampling_debye_m }} m |

It then sets the box in Debye lengths, the time step as `Domain(time_step=...)` from
$\omega_{pe}\Delta t$, and checks that `Simulation.plasma_frequency()` and
`Simulation.debye_length()` report the same scales it computed by hand.

## How to run

```bash
python examples/1_basic/parameters_and_sampling.py
```

It writes `parameters_and_sampling/run.json` and its figure;
`docs/scripts/fig_parameters_and_sampling.py` runs it for this page.
