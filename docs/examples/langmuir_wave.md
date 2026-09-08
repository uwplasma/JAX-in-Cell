# Langmuir waves

`examples/langmuir_wave.py`

The simplest plasma oscillation, measured across a range of wavenumbers and compared
with the Bohm-Gross dispersion relation

```{math}
\omega^2 = \omega_{pe}^2\left(1 + 3k^2\lambda_D^2\right)
```

{cite}`bohm1949`, which is the fluid limit of the kinetic root.

## Running it

```bash
python examples/langmuir_wave.py
```

Six runs, one per $k\lambda_D$, each measuring the frequency from the spacing of the
maxima of $|E_k(t)|$.

## What it shows

Agreement is close at small $k\lambda_D$ and drifts at large: by $k\lambda_D = 0.3$ the
fluid formula is already a per cent off, and by 0.5 it is eight per cent, because the
thermal correction is no longer a small expansion parameter. Panel (b) of
{doc}`../numerics/verification`'s Landau figure carries the comparison further and
shows the measurement following the *kinetic* root rather than the fluid one, which is
the useful statement: the code is not solving a fluid model with a pressure term, it
is solving the Vlasov equation.
