# Landau damping

A small-amplitude Langmuir wave decays without any collisions, because the electrons
travelling at the phase velocity absorb it {cite}`landau1946`. At
$k\lambda_D$ = {{ landau_k_lambda_D }} both the rate and the frequency are compared with
the least-damped root of the kinetic dispersion relation {cite}`canosa1972`.

```{figure} ../_static/figures/landau_damping.png
:width: 100%
:alt: Landau damping of the seeded mode and the measured dispersion relation

(a) $|E_k(t)|$ at $k\lambda_D$ = {{ landau_k_lambda_D }}, the maxima used (circles), the
fit through them (dashed) and the kinetic rate (dotted), above the noise floor (grey).
(b) The measured frequency (circles) against the kinetic root (solid) and Bohm-Gross
(dashed), over $k\lambda_D = 0.05$ to $0.5$.
```

## What is measured against what

| quantity | measured | kinetic root | deviation |
|---|---|---|---|
| damping rate $\gamma/\omega_{pe}$ | {{ landau_gamma_measured }} | {{ landau_gamma_theory }} | {{ landau_gamma_deviation_percent }} % |
| frequency $\omega/\omega_{pe}$ | {{ landau_omega_measured }} | {{ landau_omega_theory }} | {{ landau_omega_deviation_percent }} % |
| frequency over the $k$ scan, panel (b) | — | kinetic root | {{ landau_dispersion_max_deviation_percent }} % at worst |

Both parts come from the maxima of $|E_k(t)|$: their spacing gives the frequency, because
successive maxima of a modulus are half a period apart, and their envelope gives the rate.
{{ landau_peaks_used }} maxima are used.

## The setup

| | |
|---|---|
| cells | {{ landau_cells }} |
| particles | {{ landau_particles }} |
| $\omega_{pe}\Delta t$ | {{ landau_omega_pe_dt }} |
| seed | $ak$ = {{ landau_seed_ak }} |
| sampling | `sampling="quiet"` |

The quiet start is what makes the measurement possible: the wave has to be followed over
three e-foldings before it disappears into the discrete-particle noise, and a random start
puts that floor two e-foldings down. Placing the velocities at the quantiles of the
Maxwellian drops the floor by orders of magnitude — see {doc}`../numerics/initialization`.

## How to run

```bash
python examples/1_basic/landau_damping.py
jaxincell inputs/landau_damping.toml
```

The figure and the numbers come from `docs/scripts/fig_landau_damping.py`, which runs this
setup and adds the scan in $k$ of panel (b).

## Things to try

* Raise `k_lambda_d` towards 1: the damping becomes so strong that the wave never
  completes a period.
* Lower it towards 0.1: the rate becomes exponentially small, and the noise floor is
  reached before any damping is visible.
* Raise the seed to $ak \sim 0.1$: the wave traps the resonant electrons, damping stops,
  and the amplitude oscillates at the bounce frequency instead — the nonlinear regime of
  O'Neil {cite}`oneil1965`.
