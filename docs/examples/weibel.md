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
(c) Every mode of one unseeded box of {{ weibel_wide_wavelengths }} marginal wavelengths,
fitted over one linear window, against the same kinetic root; open circles have $R^2 < 0.8$.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| growth rate, {{ weibel_modes_compared }} of {{ weibel_modes_run }} seeded runs | panel (b) | transverse kinetic root | {{ weibel_mean_deviation_percent }} % mean, {{ weibel_max_deviation_percent }} % worst |
| growth rate mode by mode, {{ weibel_wide_modes_compared }} of {{ weibel_wide_modes_unstable }} unstable modes of one unseeded box | panel (c) | transverse kinetic root | {{ weibel_wide_mean_deviation_percent }} % mean, {{ weibel_wide_max_deviation_percent }} % worst |
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

A gain is not a growth rate once a mode has saturated, and in a wide box the long
wavelengths saturate before the modes near the cutoff have grown. Panel (c) therefore fits
every mode over one window, from $t\,\omega_{pe} = 20$, when the noise has settled into the
growing root, to the time the total magnetic energy reaches 5 % of its maximum
($t\,\omega_{pe}$ = {{ weibel_wide_window }} here). A mode is compared when its fit reaches
$R^2 \ge 0.8$; close to the cutoff the growth is too slow to rise out of the noise inside the
window, and those modes are drawn open. The run then goes on to saturation, where the
filaments merge.

## The setup

| | |
|---|---|
| anisotropy $T_z/T_x$ | {{ weibel_anisotropy }} |
| cells | {{ weibel_cells }} |
| particles | {{ weibel_particles }} |
| steps | {{ weibel_steps }}, to $t\,\omega_{pe}$ = {{ weibel_t_end }} |
| `dt_over_dx_c` | {{ weibel_courant }} |
| seed amplitude, panel (b) | {{ weibel_seed_amplitude }} |
| wide box, panel (c) | {{ weibel_wide_wavelengths }} wavelengths, {{ weibel_wide_particles }} particles, {{ weibel_wide_steps }} steps |

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
python examples/2_intermediate/weibel.py            # both boxes, a few minutes on a GPU
python examples/2_intermediate/weibel.py --quick    # smoke preset: six wavelengths, fewer particles
jaxincell inputs/weibel.toml
```

The example writes `weibel/run.json` (settings, every fitted rate, provenance),
`weibel/modes.npz` and its figure. The figure above comes from `docs/scripts/fig_weibel.py`:
panel (a) is the example's four-wavelength box, panel (b) the seeded single-mode runs, and
panel (c) the example's own wide-box run, which the script runs and reads back. The kinetic
root is {func}`jaxincell.theory.weibel_rate`.

## Things to try

* Change the anisotropy and check that the cutoff moves as $\sqrt{T_z/T_x - 1}$.
* Let it run past saturation: the field feeds back on the particles, isotropising the
  distribution and shutting the instability off.
* Look at $B_y$ in real space rather than in $k$: the growing modes are current filaments,
  and their merging is what the late nonlinear stage is about.
