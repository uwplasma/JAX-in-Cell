# Two-stream instability

Two counter-streaming electron beams on a proton background. The seeded mode grows
exponentially, the beams trap each other, and the phase space rolls into the vortex that
ends the growth.

```{figure} ../_static/figures/two_stream.png
:width: 100%
:alt: Growth of the seeded two-stream mode and the electron phase space after saturation

(a) $|E_{k=1}(t)|$, the fit through its linear phase (dashed) and the kinetic growth rate
drawn at the same intercept (dotted). (b) The electron phase space after saturation,
binned as pseudo-particles per bin: the vortex that ends the growth.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| growth rate $\gamma/\omega_{pe}$, seeded mode | {{ two_stream_gamma_measured }} | {{ two_stream_gamma_theory }} (kinetic root) | {{ two_stream_gamma_deviation_percent }} % |
| growth rate over seven drifts | scan below | kinetic root | {{ two_stream_scan_mean_deviation_percent }} % mean, {{ two_stream_scan_max_deviation_percent }} % worst |
| energy drift over the run | {{ two_stream_energy_error }} | zero | — |

The fit runs over $\omega_{pe}t$ = {{ two_stream_fit_window }} with $R^2$ = {{ two_stream_fit_r2 }}.

```{figure} ../_static/figures/two_stream_scan.png
:width: 100%
:alt: Measured two-stream growth rate against the kinetic root across the unstable range

The fitted growth rate of seven separate runs (circles) against the kinetic root (solid),
across the unstable range of $kv_0/\omega_{pe}$. The grey line marks the cold-beam cutoff
at $kv_0 = \omega_{pe}$.
```

## Theory

For two cold beams of density $n/2$ drifting at $\pm v_0$,

```{math}
1 = \frac{\omega_{pe}^2}{2}\left[\frac{1}{(\omega - k v_0)^2} + \frac{1}{(\omega + k v_0)^2}\right],
```

which has a purely growing root for $kv_0 < \omega_{pe}$, fastest at
$kv_0/\omega_{pe} = \sqrt{3/8}$ where $\gamma = \omega_{pe}/2\sqrt2$ {cite}`buneman1959`.
Warm beams are less unstable, and the comparison above uses the full kinetic root.

## The setup

| | |
|---|---|
| cells | {{ two_stream_cells }} |
| particles | {{ two_stream_particles }} |
| $\omega_{pe}$ | {{ two_stream_omega_pe }} rad/s |
| $\omega_{pe}\Delta t$ | {{ two_stream_omega_pe_dt }} |
| $\Delta x/\lambda_D$ | {{ two_stream_dx_over_debye }} |
| seed | $ak$ = {{ two_stream_seed_ak }} |
| $kv_0/\omega_{pe}$ | {{ two_stream_k_v0_over_wpe }} |

The figure comes from `docs/scripts/fig_two_stream.py`, which runs a quieter version of
this problem so that the growth rate can be fitted: a quiet start, a drift of
$5\times10^7$ m/s and no filter. The example itself keeps the noisier setup that
{doc}`../getting_started/first_simulation` takes apart parameter by parameter.

## How to run

```bash
python examples/1_basic/two_stream.py
jaxincell inputs/two_stream.toml
```

It prints the energy drift and the growth of the electric energy, then shows the
animation. Uncomment the last line to write an MP4 instead.

## Things to try

* Change `drift` and watch the growth rate move along the scan curve above.
* Set `sampling="quiet"` on the electrons: the noise floor drops by orders of magnitude
  and the linear phase becomes long enough to fit properly.
* Switch to `Solver(algorithm="implicit")` and watch the energy error fall to round-off
  ({doc}`conservation`).
