# Relativistic two-stream instability

The explicit scheme has two particle pushers, the standard Boris rotation and its
relativistic version, selected with `solver_parameters["relativistic"]`. This page runs
the same two-stream instability with beams at $0.8c$ twice, once with each pusher and
everything else identical, and checks two things against theory: the linear growth
rate and the conservation of the relativistic energy.

```python
from jaxincell import Simulation, diagnostics

for relativistic in (True, False):
    parameters["solver_parameters"]["relativistic"] = relativistic
    output = Simulation(parameters).run()
```

The full input is in `docs/scripts/fig_relativistic.py`; its values are listed below.

## Set-up

| quantity | value |
|---|---|
| beam drift $v_0/c$ | $\pm$ {{ relativistic_v0_over_c }}, Lorentz factor $\gamma_0 = $ {{ relativistic_gamma0 }} |
| beam thermal speed $v_{th}/c$ | {{ relativistic_vth_over_c }} |
| electron density (both beams) | {{ relativistic_density }} m$^{-3}$, protons at the same density, cold |
| Debye length | $\lambda_D = v_{th}/(\sqrt2\,\omega_{pe})$ = {{ relativistic_debye_c_over_wpe }} $c/\omega_{pe}$, with $v_{th} = \sqrt{2T/m}$ the beam thermal speed (the convention of `dx_over_Debye_length` and of the kinetic dispersion relation) |
| box length | {{ relativistic_length_c_over_wpe }} $c/\omega_{pe}$ = {{ relativistic_length_over_debye }} $\lambda_D$, one wavelength, $k v_0/\omega_{pe} = $ {{ relativistic_k_v0_over_wpe }} |
| grid | {{ relativistic_grid_points }} cells, $\Delta x = $ {{ relativistic_dx_wpe_over_c }} $c/\omega_{pe}$ = {{ relativistic_dx_over_debye }} $\lambda_D$ |
| time step | $c\,\Delta t/\Delta x = $ {{ relativistic_c_dt_over_dx }}, $\omega_{pe}\Delta t = $ {{ relativistic_omega_pe_dt }}, {{ relativistic_steps }} steps |
| pseudo-particles | {{ relativistic_particles }} electrons and {{ relativistic_particles }} protons, evenly spaced |
| seed | electron displacement of {{ relativistic_perturbation_over_L }} $L$ in the first mode |
| solver | explicit Boris leapfrog, `field_solver = 0`, default digital filter |

## Theory

For two cold beams of equal density drifting at $\pm v_0$, the electrostatic dispersion
relation is

```{math}
1 = \frac{\omega_{b}^2}{\gamma_0^3}\left[\frac{1}{(\omega - k v_0)^2} + \frac{1}{(\omega + k v_0)^2}\right],
\qquad \omega_b^2 = \frac{\omega_{pe}^2}{2},
```

where the factor $\gamma_0^3$ is the longitudinal mass of a relativistic particle:
along the drift, $dp/dv = \gamma^3 m$. Without it, the relation is the non-relativistic
one. The growing root is largest at $k v_0 = (\sqrt3/2)\,\omega_b\gamma_0^{-3/2}$,
with rate $\omega_b\gamma_0^{-3/2}/2$. The box is one wavelength of this mode, so for
the relativistic equations only the first box mode is unstable, with
$\gamma = $ {{ relativistic_gamma_cold_relativistic }} $\omega_{pe}$. With the
non-relativistic equations modes 1 to 3 are unstable (rates
{{ relativistic_gamma_modes_newtonian }} $\omega_{pe}$ for modes 1 to 5) and mode
{{ relativistic_fastest_mode_newtonian }} grows fastest. The script takes the roots
from the quartic numerically; the warm kinetic dielectric of the same beams with
$\omega_b^2 \to \omega_b^2/\gamma_0^3$ gives the same rates to four digits
({{ relativistic_gamma_warm_relativistic }} and {{ relativistic_gamma_warm_newtonian }}),
so the thermal spread does not matter here.

The pushers store the velocity $\mathbf v$, not the momentum, in both cases. The script
therefore forms $\gamma_p = (1 - |\mathbf v_p|^2/c^2)^{-1/2}$ for every pseudo-particle
and two total energies, each with the field energy:

```{math}
\mathcal E_{rel} = \sum_p (\gamma_p - 1)\, m_p c^2 + \mathcal E_{field}, \qquad
\mathcal E_{N} = \sum_p \tfrac12 m_p |\mathbf v_p|^2 + \mathcal E_{field},
```

with $m_p$ the pseudo-particle mass (the weight included). The relativistic equations
conserve $\mathcal E_{rel}$, the non-relativistic ones $\mathcal E_{N}$. The
`total_energy` returned by {func}`~jaxincell.diagnostics` is $\mathcal E_{N}$ (see
{doc}`../numerics/diagnostics`).

## Result

```{figure} ../_static/figures/relativistic_two_stream.png
:width: 100%
:alt: Relativistic two-stream instability with the relativistic and non-relativistic Boris pushers

Relativistic Boris pusher (vermillion) and non-relativistic Boris pusher (blue) on the
same input. (a) Electrostatic energy, with $e^{2\gamma t}$ (dashed, same colour as the
run) for the fastest box mode of each cold dispersion relation: mode 1 with
$\omega_b^2 \to \omega_b^2/\gamma_0^3$, mode {{ relativistic_fastest_mode_newtonian }}
without. (b) Relative change of $\mathcal E_{rel}$ (solid) and $\mathcal E_{N}$ (dotted).
The blue solid line stops at the dash-dotted line, when the first electron of the
non-relativistic run reaches $|v| \ge c$: from then on $\gamma_p$, and with it
$\mathcal E_{rel}$, is not defined. (c), (d) Electron phase space at the peak of the
electrostatic energy, position in Debye lengths; the shaded bands are $|v_x| > c$. Generated by `docs/scripts/fig_relativistic.py`.
```

| | relativistic pusher | non-relativistic pusher |
|---|---|---|
| fastest box mode (theory) | 1 | {{ relativistic_fastest_mode_newtonian }} |
| growth rate, cold theory ($\omega_{pe}$) | {{ relativistic_gamma_cold_relativistic }} | {{ relativistic_gamma_cold_newtonian }} |
| growth rate, simulation ($\omega_{pe}$) | {{ relativistic_gamma_fit_relativistic }} | {{ relativistic_gamma_fit_newtonian }} |
| deviation from theory | {{ relativistic_gamma_deviation_percent_relativistic }} % | {{ relativistic_gamma_deviation_percent_newtonian }} % |
| largest change of $\mathcal E_{rel}$ | {{ relativistic_error_rel_max_relativistic }} | {{ relativistic_error_rel_max_newtonian_before_superluminal }} before $\omega_{pe}t = $ {{ relativistic_t_superluminal_newtonian }}, undefined after |
| largest change of $\mathcal E_{N}$ | {{ relativistic_error_newton_max_relativistic }} | {{ relativistic_error_newton_max_newtonian }} |
| electrons with $\lvert v\rvert \ge c$ | none (largest $\gamma_p$: {{ relativistic_lorentz_max_relativistic }}) | up to {{ relativistic_superluminal_percent_newtonian }} % |

The growth rate is half the slope of the energy of the fastest mode, fitted from the
time that energy has grown by three decades above its initial level (by then the
three non-growing roots of the quartic excited by the seed no longer matter) to the
time it reaches a tenth of its first peak. Each pusher reproduces the growth rate of
its own equations to within 1 %; the relativistic beams grow more than twice as
slowly because of their larger longitudinal inertia.

Each pusher conserves the energy of its own equations, and only that one. With the
relativistic pusher $\mathcal E_{rel}$ changes by about $10^{-7}$ through the linear
phase and by at most {{ relativistic_error_rel_max_relativistic }} at saturation, no
worse than $\mathcal E_{N}$ with the non-relativistic pusher
({{ relativistic_error_newton_max_newtonian }}). Neither is exact: this is the usual
energy error of the explicit leapfrog, discussed in {doc}`energy_conservation`.
Measured in the other energy, each run is wrong by tens of per cent; in particular
the `total_energy` of a relativistic run is not a conservation check. The
non-relativistic pusher also accelerates trapped electrons past the speed of light,
as panel (d) shows, while the relativistic pusher keeps every electron below $c$ with
Lorentz factors up to {{ relativistic_lorentz_max_relativistic }}.

The two runs took {{ relativistic_seconds_relativistic }} s and
{{ relativistic_seconds_newtonian }} s, compilation included, on the shared CPU that
built this documentation.
