# Landau damping

`examples/landau_damping.py`

A small-amplitude Langmuir wave decays without any collisions, because the electrons
travelling at the phase velocity absorb it {cite}`landau1946`.

```{figure} ../_static/figures/landau_damping.png
:width: 100%
:alt: Landau damping of the seeded mode and the measured dispersion relation

(a) The mode amplitude and the fit through its maxima. (b) The measured frequency
against $k\lambda_D$.
```

## Theory

At $k\lambda_D = 0.5$ the least damped root of

```{math}
1 + \frac{1}{(k\lambda_D)^2}\left[1 + \zeta Z(\zeta)\right] = 0, \qquad \zeta = \frac{\omega}{k v_{th}},
```

is $\omega/\omega_{pe} = 1.4157 - 0.1533\,i$ {cite}`canosa1972`. The script measures
both parts from the maxima of $|E_k(t)|$: their spacing gives the frequency, because
successive maxima of a modulus are half a period apart, and their envelope gives the
rate. It reaches {{ landau_gamma_measured }} and {{ landau_omega_measured }}.

## Why the quiet start matters

The wave has to be followed over three e-foldings before it disappears into the
discrete-particle noise. A random start puts that floor two e-foldings down, which is
not enough to fit anything. `quiet=True` places the velocities at the quantiles of the
Maxwellian and drops the floor by orders of magnitude; that plus
{{ landau_particles }} particles is what makes the measurement possible. See
{doc}`../numerics/initialization`.

## Things to try

* Raise `k_lambda_d` towards 1 and watch the damping become so strong that the wave
  never completes a period.
* Lower it towards 0.1 and watch the rate become exponentially small, until the noise
  floor is reached before any damping is visible.
* Raise the seed to $ak \sim 0.1$: the wave traps the resonant electrons, damping
  stops, and the amplitude oscillates at the bounce frequency instead — the nonlinear
  regime of O'Neil {cite}`oneil1965`.
