# Langmuir wave

`examples/Langmuir_wave.py` sets up a cold electron oscillation: $k\lambda_D$ is small
(`grid_points_per_Debye_length = 3` with 33 cells gives $k\lambda_D \approx 0.06$),
so Landau damping is negligible and the wave oscillates at very nearly the plasma
frequency, $\omega^2 = \omega_{pe}^2(1 + 3k^2\lambda_D^2)$.

```{literalinclude} ../../examples/Langmuir_wave.py
:language: python
```

The script prints the dominant frequency of $E_x$ at the box centre and its relative
difference from $\omega_{pe}$. With 1000 steps at $\omega_{pe}\Delta t \approx 0.03$
the frequency resolution of the FFT is $2\pi/(1000 \times 0.03) \approx 0.2\,\omega_{pe}$,
so a difference of a few percent is the resolution of the estimate, not a physical
shift. A finer estimate comes from the zero crossings of the field, as in
{doc}`../numerics/diagnostics`.

The example is a good first test of a new installation: it runs in a few seconds and
the electric field panel of {func}`jaxincell.plot` should show a standing wave whose
amplitude stays constant.
