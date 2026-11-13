# Langmuir waves

The simplest plasma oscillation, measured across a range of wavenumbers. Six runs, one per
$k\lambda_D$ from 0.05 to 0.3, each measuring the frequency from the spacing of the maxima
of $|E_k(t)|$, which are half a period apart.

```{figure} ../_static/figures/landau_damping.png
:width: 100%
:alt: Landau damping of the seeded mode and the measured dispersion relation

Panel (b) is the comparison this example makes, carried out to $k\lambda_D = 0.5$: the
frequency measured in eight separate runs (circles) against the kinetic root (solid) and
the Bohm-Gross frequency (dashed), which part as $k\lambda_D$ grows. Panel (a) is the
damping measurement of {doc}`landau_damping`.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| frequency, $k\lambda_D = 0.05$ to $0.5$ | panel (b) above | kinetic root | {{ landau_dispersion_max_deviation_percent }} % at worst |
| frequency, the example's own six runs | printed per $k$ | kinetic root and Bohm-Gross | both printed |

The kinetic root is the least-damped solution of

```{math}
1 + \frac{1}{(k\lambda_D)^2}\left[1 + \zeta Z(\zeta)\right] = 0, \qquad \zeta = \frac{\omega}{k v_{th}},
```

{cite}`landau1946`, whose small-$k\lambda_D$ limit is the fluid result of Bohm and Gross
{cite}`bohm1949`, $\omega^2 = \omega_{pe}^2(1 + 3k^2\lambda_D^2)$.

## Kinetic, not fluid

The two theories part as $k\lambda_D$ grows, because the thermal correction stops being a
small expansion parameter. Roots from `docs/scripts/dispersion.py`:

| $k\lambda_D$ | kinetic root above the fluid frequency |
|---|---|
| 0.2 | 0.5 % |
| 0.3 | 2.9 % |
| 0.5 | 7.0 % |

The measurement follows the kinetic root to within a quarter of a per cent at every
wavenumber of the scan, where the fluid frequency would be 2.8 per cent low at 0.3. The
code is solving the Vlasov equation, not a fluid model with a pressure term.

## Two details that make the agreement visible

* **The grid shifts the frequency by a known amount.** Depositing and gathering with the
  quadratic spline, $S(k) = \mathrm{sinc}^3(k\Delta x/2)$, and the staggered Gauss law,
  which replaces $k$ by $K = (2/\Delta x)\sin(k\Delta x/2)$, make a cold plasma oscillate
  at $\omega_{pe}S(k)\sqrt{k/K}$ — 0.4 per cent low at 32 cells per wavelength
  ({doc}`../numerics/deposition`). The script compares with the theory times that factor.
* **The seed has to clear the noise.** $ak = 10^{-2}$ followed for 1600 steps keeps the
  wave far enough above the particle noise for the maxima to mark its period. With a tenth
  of it, the noise moves them by a per cent or more at $k\lambda_D \ge 0.25$.

## How to run

```bash
python examples/1_basic/langmuir_wave.py
jaxincell inputs/langmuir_wave.toml
```

The six per-$k$ numbers are printed by the example itself; no figure script records them.
