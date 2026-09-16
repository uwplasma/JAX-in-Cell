# Verification

Every claim on this page is a number produced by a script in `docs/scripts/`,
recorded in `measurements.json` and substituted into the text, so the prose and the
figures come from the same run. Reproduce them with

```bash
python docs/scripts/make_all.py
```

The reference values are not fits to other simulations. They are roots of the linear
kinetic dispersion relations, solved in `docs/scripts/dispersion.py` with the plasma
dispersion function $Z(\zeta) = i\sqrt\pi\,w(\zeta)$ evaluated through the Faddeeva
function, for exactly the densities, drifts and thermal speeds each run was
initialised with.

:::{note}
For a symmetric pair of counter-streaming beams, and for the Weibel instability, the
unstable root is purely growing, so the dispersion function is real on the imaginary
axis and its roots can be bracketed by a sign change. That is what
`purely_growing_roots` does, and it is worth the trouble: Newton iterations started
from a grid of complex guesses will converge onto a different Riemann sheet of $Z$
and report a growth rate where the system is provably stable — for two beams at
$kv_0 > \omega_{pe}$ it returns $\gamma = 0.106\,\omega_{pe}$ for a mode that cannot
grow at all.
:::

## Landau damping

A Langmuir wave at $k\lambda_D = 0.5$ decays because the electrons at the phase
velocity absorb it {cite}`landau1946`. The least damped root of

```{math}
1 + \frac{1}{(k\lambda_D)^2}\left[1 + \zeta Z(\zeta)\right] = 0, \qquad
\zeta = \frac{\omega}{k v_{th}},
```

is $\omega/\omega_{pe} = 1.4157 - 0.1533\,i$, tabulated by Canosa {cite}`canosa1972`.

```{figure} ../_static/figures/landau_damping.png
:width: 100%
:alt: Landau damping of the seeded mode and the measured dispersion relation

(a) The mode amplitude decays through three e-foldings before reaching the
discrete-particle noise floor. The rate comes from the maxima of $|E_k|$; successive
maxima of a modulus are half a period apart, which gives the frequency from the same
points. (b) The measured frequency against $k\lambda_D$, with the Bohm-Gross fluid
result and the exact kinetic root.
```

| | measured | theory | deviation |
|---|---|---|---|
| $\gamma/\omega_{pe}$ | {{ landau_gamma_measured }} | {{ landau_gamma_theory }} | {{ landau_gamma_deviation_percent }} % |
| $\omega/\omega_{pe}$ | {{ landau_omega_measured }} | {{ landau_omega_theory }} | {{ landau_omega_deviation_percent }} % |

with {{ landau_particles }} quiet-start electrons on {{ landau_cells }} cells,
$\omega_{pe}\Delta t = $ {{ landau_omega_pe_dt }}, a seed of
$ak = $ {{ landau_seed_ak }} and {{ landau_peaks_used }} maxima used for the fit.

Panel (b) is the sharper test. Across $k\lambda_D$ from 0.05 to 0.5 the measured
frequency stays within {{ landau_dispersion_max_deviation_percent }} % of the exact
kinetic root, and it follows the kinetic curve rather than the Bohm-Gross
approximation $\omega^2 = \omega_{pe}^2(1 + 3k^2\lambda_D^2)$ {cite}`bohm1949`, which
the kinetic root exceeds by 2.9 per cent at $k\lambda_D = 0.3$ and by 7.0 per cent at 0.5.
The code is reproducing kinetic physics, not a fluid limit.

## Two-stream instability

Two counter-streaming beams of density $n/2$ each are unstable for
$kv_0 < \omega_{pe}$, with the cold-beam maximum $\gamma = \omega_{pe}/2\sqrt2$ at
$kv_0/\omega_{pe} = \sqrt{3/8}$ {cite}`buneman1959`. The beams here are warm
($v_{th} = 0.05c$), so the comparison is against the full kinetic root.

```{figure} ../_static/figures/two_stream.png
:width: 100%
:alt: Growth of the seeded two-stream mode and the electron phase space after saturation

(a) The seeded mode grows exponentially and saturates when the beams trap each other.
(b) The electron phase space at the end of the run, showing the vortex that closes the
growth.
```

```{figure} ../_static/figures/two_stream_scan.png
:width: 80%
:alt: Measured two-stream growth rate against the kinetic dispersion relation

Growth rate against $kv_0/\omega_{pe}$, over the whole unstable range up to the
cold-beam cutoff at one. Points are measured, the line is the kinetic root.
```

At $kv_0/\omega_{pe} = $ {{ two_stream_k_v0_over_wpe }} the measured rate is
{{ two_stream_gamma_measured }} against {{ two_stream_gamma_theory }} from theory,
{{ two_stream_gamma_deviation_percent }} % away, fitted over
$t\omega_{pe} = $ {{ two_stream_fit_window }} with $R^2 = $ {{ two_stream_fit_r2 }}.
Over the seven drifts of the scan the mean deviation is
{{ two_stream_scan_mean_deviation_percent }} % and the largest is
{{ two_stream_scan_max_deviation_percent }} %.

The window is set by amplitude, not by time — from ten times the seed to a tenth of
saturation — so that the same part of the growth is fitted at every drift. The total
energy changes by {{ two_stream_energy_error }} across the run.

## Bump-on-tail instability

A weak beam on the tail makes $\partial f/\partial v > 0$, and every wave whose phase
velocity sits in that window grows {cite}`oneil1965`. The beam here carries
{{ bump_on_tail_beam_fraction }} of the density, drifts at
{{ bump_on_tail_beam_drift_over_vth }} $v_{th}$ and is
{{ bump_on_tail_beam_width_over_vth }} $v_{th}$ wide.

```{figure} ../_static/figures/bump_on_tail.png
:width: 100%
:alt: Growth of the resonant mode and the quasilinear plateau in the distribution

(a) The resonant mode. (b) The distribution before and after: the bump has flattened
into a plateau reaching from the bulk to the beam, the quasilinear end state.
```

Mode {{ bump_on_tail_mode }}, whose phase velocity is
{{ bump_on_tail_phase_velocity_over_vth }} $v_{th}$, grows at
{{ bump_on_tail_gamma_measured }} against {{ bump_on_tail_gamma_theory }} from the
kinetic root, {{ bump_on_tail_gamma_deviation_percent }} % away, with
{{ bump_on_tail_particles }} particles on {{ bump_on_tail_cells }} cells.

## Weibel instability

A plasma hotter across the simulation axis than along it drives purely growing
transverse magnetic modes {cite}`weibel1959`. Setting $\omega=0$ in the transverse
dispersion relation gives the marginal wavenumber in closed form,

```{math}
k_c c = \omega_{pe}\sqrt{\frac{T_z}{T_x} - 1},
```

which for the anisotropy used here, $T_z/T_x = $ {{ weibel_anisotropy }}, is
$k_c c/\omega_{pe} = $ {{ weibel_kc_c_over_wpe }}.

```{figure} ../_static/figures/weibel.png
:width: 100%
:alt: Weibel mode growth below and above the cutoff, and the growth rate against wavenumber

(a) Eight modes in one box with a random start. Every mode below the cutoff grows;
none above it does. (b) Growth rates from single-mode runs, each in a box one
wavelength long with a coherent transverse current seeded, against the kinetic root.
```

Panel (a) is a threshold test with no fitting: modes below $k_c$ gain at least
{{ weibel_gain_min_unstable }} in amplitude while the largest gain above it is
{{ weibel_gain_max_stable }}. Panel (b) is quantitative:
{{ weibel_modes_compared }} of the {{ weibel_modes_run }} seeded runs produce a clean
exponential ($R^2 > 0.85$), and those agree with theory to
{{ weibel_mean_deviation_percent }} % on average and
{{ weibel_max_deviation_percent }} % at worst. The runs that do not qualify are the
ones closest to the cutoff, where the growth rate vanishes and there are not enough
e-foldings to fit; they are excluded rather than fitted anyway, and the exclusion is
part of the recorded result.

The total energy changes by {{ weibel_energy_error }} over the run.

## Wall reflection

A reflection law returns its flux average from a Maxwellian, not its average over the
distribution: {{ reflection_returned_sigma }} for a Gaussian law of width $\sigma$, whose
flux average is one half and whose distribution average is
{{ reflection_distribution_average_sigma }}, and within {{ reflection_max_error }} of the
flux average at five widths. See {doc}`../examples/wall_reflection`.

## Sheath formation

The edge of a plasma, between a thermal source wall {cite}`schwager1990` and a floating
conductor. The ions reach the Bohm speed $c_s=\sqrt{T_e/m_i}$ {cite}`bohm1949sheath`
{{ sheath_edge_debye }} Debye lengths from the wall, and from there the potential falls
by $(T_e/e)\left[\tfrac12\ln(m_i/2\pi m_e) + \ln(1-R_{\rm eff})\right]$ {cite}`hobbs1967`:
{{ sheath_drop_absorbing }} against {{ sheath_drop_theory }} for a wall that collects
every electron, and {{ sheath_drop_half }} and {{ sheath_drop_slow }} against
{{ sheath_drop_theory_reflecting }} for a wall that returns half of each electron and one
that returns the slow electrons with the same flux average. The drop does not change
when the cells are refined; what is left of the difference lies mostly in placing the Bohm
point, where the potential still falls by about 0.1 $T_e/e$ per Debye length. See
{doc}`../examples/sheath_reflection`.

## Collisions

The Takizuka-Abe operator is checked against the Fokker-Planck relaxation rates in the
fast-beam limit, where they are closed-form and contain no adjustable constant. The
worst of the four measured ratios is {{ collisions_max_deviation_percent }} % from
theory; see {doc}`collisions` for the figure and the details.

## Conservation

Largest relative errors ({doc}`diagnostics`) over the two-stream run of
{doc}`../examples/conservation` and, at the walls, over a drifting plasma of 120 steps; the
explicit wall runs are filtered, the implicit ones cannot be.

| quantity | explicit | implicit (8 Picard) |
|---|---|---|
| energy | {{ energy_error_max_explicit }} | {{ energy_error_max_implicit_8 }} |
| momentum | {{ momentum_error_relative }} | {{ momentum_error_implicit }} |
| Gauss law, periodic | {{ gauss_residual_max_explicit }} | {{ gauss_residual_max_implicit }} |
| Gauss law, reflective walls | {{ gauss_residual_reflective_wall }} | {{ gauss_residual_reflective_wall_implicit }} |
| Gauss law, absorbing walls | {{ gauss_residual_absorbing_wall }} | {{ gauss_residual_absorbing_wall_implicit }} |
| Gauss law, walls returning half of each electron | {{ gauss_residual_reflecting_wall }} | {{ gauss_residual_reflecting_wall_implicit }} |
| Gauss law, thermal wall and floating conductor | {{ gauss_residual_thermal_wall }} | {{ gauss_residual_thermal_wall_implicit }} |
| charge, deposited against carried | {{ charge_error_relative }} | |

## Reproducibility

The whole suite runs on one CPU core in a few minutes. `pytest` repeats the physics
above at reduced resolution as part of the test suite, so a regression in any of the
rates fails CI rather than waiting to be noticed in a figure; see
{doc}`../development`.
