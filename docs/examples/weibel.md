# Weibel instability

A plasma hotter across the simulation axis than along it is unstable to purely growing
transverse magnetic modes {cite}`weibel1959`. It is the electromagnetic instability of the
set: it converts anisotropy in the distribution into magnetic field.

```{figure} ../_static/figures/weibel.png
:width: 100%
:alt: Weibel mode growth below and above the cutoff and the growth rate against wavenumber

(a) $|B_{y,k}(t)|$ for eight modes of one unseeded box, solid below the cutoff and dotted
above it: the modes below $k_c$ grow, those above it do not. (b) The growth rate of seven
seeded single-mode runs (filled circles) against the kinetic root (solid); open circles are
the runs whose fit did not reach $R^2 = 0.85$ and are excluded. The grey line marks $k_c$.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| growth rate, {{ weibel_modes_compared }} of {{ weibel_modes_run }} seeded runs | panel (b) | transverse kinetic root | {{ weibel_mean_deviation_percent }} % mean, {{ weibel_max_deviation_percent }} % worst |
| marginal wavenumber $k_c c/\omega_{pe}$ | gain splits at the cutoff | {{ weibel_kc_c_over_wpe }} $= \sqrt{T_z/T_x - 1}$ | — |
| gain below the cutoff | {{ weibel_gain_min_unstable }} at least | growth | — |
| gain above the cutoff | {{ weibel_gain_max_stable }} at most | no growth | — |
| fastest rate $\gamma/\omega_{pe}$ | panel (b) | {{ weibel_gamma_max_theory }} | — |
| energy drift | {{ weibel_energy_error }} | zero | — |

The marginal wavenumber is a sharp prediction that needs no fitting. Setting $\omega = 0$
in the transverse dispersion relation gives

```{math}
k_c c = \omega_{pe}\sqrt{\frac{T_z}{T_x} - 1},
```

so putting several wavelengths in one box makes every mode below $k_c$ grow and none above
it. The gain figures above are how far apart the two groups end up.

## The setup

| | |
|---|---|
| anisotropy $T_z/T_x$ | {{ weibel_anisotropy }} |
| cells | {{ weibel_cells }} |
| particles | {{ weibel_particles }} |
| steps | {{ weibel_steps }}, to $t\,\omega_{pe} = {{ weibel_t_end }}$ |
| `dt_over_dx_c` | {{ weibel_courant }} |
| seed amplitude, panel (b) | {{ weibel_seed_amplitude }} |

Two things this example needs:

* **A Courant number at or below one.** The instability lives in the transverse fields, so
  the explicit field solve is subject to the light-wave limit
  ({doc}`../numerics/stability`).
* **A bi-Maxwellian initial condition.** $T_z \ne T_x$ is `vth=(v, 0, v * sqrt(ratio))`,
  which `Species` supports directly. Seeding one mode coherently, as panel (b) does, needs
  `quiet_start` plus a transverse current — that is what {func}`~jaxincell.quiet_start` is
  for.

## How to run

```bash
python examples/2_intermediate/weibel.py
jaxincell inputs/weibel.toml
```

The figure and the numbers come from `docs/scripts/fig_weibel.py`: panel (a) is the
example's own unseeded box, panel (b) the seeded single-mode runs.

## Things to try

* Change the anisotropy and check that the cutoff moves as $\sqrt{T_z/T_x - 1}$.
* Let it run past saturation: the field feeds back on the particles, isotropising the
  distribution and shutting the instability off.
* Look at $B_y$ in real space rather than in $k$: the growing modes are current filaments,
  and their merging is what the late nonlinear stage is about.
