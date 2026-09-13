# Langmuir waves

`examples/langmuir_wave.py`

The simplest plasma oscillation, measured across a range of wavenumbers. A small density
perturbation of a Maxwellian oscillates at the real part of the least-damped root of the
kinetic dispersion relation {cite}`landau1946`,

```{math}
1 + \frac{1}{(k\lambda_D)^2}\left[1 + \zeta Z(\zeta)\right] = 0, \qquad \zeta = \frac{\omega}{k v_{th}},
```

whose limit at small $k\lambda_D$ is the fluid result of Bohm and Gross {cite}`bohm1949`,

```{math}
\omega^2 = \omega_{pe}^2\left(1 + 3k^2\lambda_D^2\right).
```

## Running it

```bash
python examples/langmuir_wave.py
```

Six runs, one per $k\lambda_D$ from 0.05 to 0.3, each measuring the frequency from the
spacing of the maxima of $|E_k(t)|$, which are half a period apart. The script prints
each against the kinetic root and the Bohm-Gross frequency.

## What it shows

The two theories part as $k\lambda_D$ grows, because the thermal correction stops being a
small expansion parameter: the kinetic root is 0.5 per cent above the fluid frequency at
$k\lambda_D = 0.2$, 2.9 per cent above it at 0.3 and 7.0 per cent above it at 0.5 (roots
from `docs/scripts/dispersion.py`). The measurement follows the kinetic root, to within a
quarter of a per cent at every wavenumber of the scan, where the fluid frequency would be
2.8 per cent low at 0.3: the code is not solving a fluid model with a pressure term, it
is solving the Vlasov equation. Panel (b) of the Landau figure in
{doc}`../numerics/verification` carries the comparison to $k\lambda_D = 0.5$.

Two details make that agreement visible. The grid shifts the frequency by a known amount:
depositing and gathering with the quadratic spline, $S(k) = \mathrm{sinc}^3(k\Delta x/2)$,
and the staggered Gauss law, which replaces $k$ by $K = (2/\Delta x)\sin(k\Delta x/2)$,
make a cold plasma oscillate at $\omega_{pe}S(k)\sqrt{k/K}$, 0.4 per cent low with 32
cells per wavelength ({doc}`../numerics/deposition`), and the script compares with the
theory times that factor. And the seed, $ak = 10^{-2}$ followed for 1600 steps, keeps the
wave far enough above the particle noise for the maxima to mark its period; with a tenth
of it, the noise moves them by a per cent or more at $k\lambda_D \ge 0.25$.

The numbers on this page are printed by the example itself; no figure script records
them.
