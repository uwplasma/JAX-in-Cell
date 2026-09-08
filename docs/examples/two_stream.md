# Two-stream instability

`examples/two_stream.py`

Two counter-streaming electron beams on a proton background. The seeded mode grows
exponentially, the beams trap each other, and the phase space rolls into the vortex
that ends the growth.

```{figure} ../_static/figures/two_stream.png
:width: 100%
:alt: Growth of the seeded two-stream mode and the electron phase space after saturation

(a) The seeded mode. (b) The electron phase space after saturation.
```

## Theory

For two cold beams of density $n/2$ drifting at $\pm v_0$, the dispersion relation is

```{math}
1 = \frac{\omega_{pe}^2}{2}\left[\frac{1}{(\omega - k v_0)^2} + \frac{1}{(\omega + k v_0)^2}\right],
```

which has a purely growing root for $kv_0 < \omega_{pe}$, fastest at
$kv_0/\omega_{pe} = \sqrt{3/8}$ where $\gamma = \omega_{pe}/2\sqrt2$
{cite}`buneman1959`. Warm beams are less unstable and the comparison in
{doc}`../numerics/verification` uses the full kinetic root, which the measured rates
match to {{ two_stream_scan_mean_deviation_percent }} per cent on average.

## Running it

```bash
python examples/two_stream.py
```

prints the energy drift and the growth of the electric energy, then shows the
animation. Uncomment the last line to write an MP4 instead.

## Things to try

* Change `drift` and watch the growth rate move along the curve of
  {doc}`../numerics/verification`.
* Set `quiet=True` on the electrons: the noise floor drops by orders of magnitude and
  the linear phase becomes long enough to fit properly.
* Switch to `Solver(algorithm="implicit")` and watch the energy error fall to
  round-off.
