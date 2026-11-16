# Relativistic two-stream instability

The explicit scheme has two particle pushers, the Boris rotation and its relativistic
version, selected with `Solver(relativistic=...)`. This page runs the same two-stream
instability with beams at $0.8c$ twice, once with each pusher and everything else
identical, and checks each against its own theory: the linear growth rate, and the energy
its equations conserve.

```python
from jaxincell import Domain, Simulation, Solver, Species

beams = Species.electrons(n=10000, density=1e18, vth=(0.01 * c, 0, 0), drift=(0.8 * c, 0, 0),
                          plus_minus=True, perturbation_mode=1, perturbation_amplitude=1e-6 * L)
protons = Species.ions(n=10000, density=1e18, mass_ratio=1.0, vth=(1e-4 * c, 0, 0))
for relativistic in (True, False):
    out = Simulation(Domain(length=L, cells=128, dt_over_dx_c=0.9), [beams, protons],
                     Solver(relativistic=relativistic)).run(steps)
```

The full script is `examples/2_intermediate/relativistic_two_stream.py` (about a minute on a
CPU); `docs/scripts/fig_relativistic.py` runs it for this page, and its values are listed below.

## Set-up

| quantity | value |
|---|---|
| beam drift $v_0/c$ | $\pm$ {{ relativistic_v0_over_c }}, Lorentz factor $\gamma_0 = $ {{ relativistic_gamma0 }} |
| beam thermal speed $v_{th}/c$ | {{ relativistic_vth_over_c }}, with $v_{th} = \sqrt{2T/m}$ as in `Species.vth` |
| electron density (both beams) | {{ relativistic_density }} m$^{-3}$, protons at the same density, nearly cold |
| Debye length | $\lambda_D = v_{th}/(\sqrt2\,\omega_{pe})$ = {{ relativistic_debye_c_over_wpe }} $c/\omega_{pe}$ |
| box length | {{ relativistic_length_c_over_wpe }} $c/\omega_{pe}$ = {{ relativistic_length_over_debye }} $\lambda_D$, one wavelength, $k v_0/\omega_{pe} = $ {{ relativistic_k_v0_over_wpe }} |
| grid | {{ relativistic_grid_points }} cells, $\Delta x = $ {{ relativistic_dx_wpe_over_c }} $c/\omega_{pe}$ = {{ relativistic_dx_over_debye }} $\lambda_D$ |
| time step | $c\,\Delta t/\Delta x = $ {{ relativistic_c_dt_over_dx }}, $\omega_{pe}\Delta t = $ {{ relativistic_omega_pe_dt }}, {{ relativistic_steps }} steps |
| particles | {{ relativistic_particles }} electrons and {{ relativistic_particles }} protons on the same lattice of positions, velocities drawn |
| seed | electron displacement of {{ relativistic_perturbation_over_L }} $L$ in the first mode |
| solver | explicit Boris leapfrog, electromagnetic, `field_solver="ampere"`, no filter |

## Theory

For two cold beams of equal density drifting at $\pm v_0$ the electrostatic dispersion
relation is

```{math}
1 = \frac{\omega_{b}^2}{\gamma_0^3}\left[\frac{1}{(\omega - k v_0)^2} + \frac{1}{(\omega + k v_0)^2}\right],
\qquad \omega_b^2 = \frac{\omega_{pe}^2}{2},
```

where $\gamma_0^3$ is the longitudinal mass of a relativistic particle: along the drift,
$dp/dv = \gamma^3 m$. Without it the relation is the non-relativistic one. The growing root
is largest at $k v_0 = (\sqrt3/2)\,\omega_b\gamma_0^{-3/2}$, with rate
$\omega_b\gamma_0^{-3/2}/2$. The box is one wavelength of this mode, so for the relativistic
equations only the first box mode is unstable, at
$\gamma = $ {{ relativistic_gamma_cold_relativistic }} $\omega_{pe}$. With the
non-relativistic equations modes 1 to 3 are unstable and mode
{{ relativistic_fastest_mode_newtonian }} grows fastest. The script takes the roots of the
quartic numerically; the warm kinetic dielectric of the same beams, with
$\omega_b^2 \to \omega_b^2/\gamma_0^3$, gives the same rates to four digits
({{ relativistic_gamma_warm_relativistic }} and {{ relativistic_gamma_warm_newtonian }}),
so the thermal spread does not matter here.

Both runs store the velocity $\mathbf v$ ({doc}`../user_guide/output`), so the script forms
$\gamma_p = (1 - |\mathbf v_p|^2/c^2)^{-1/2}$ for every particle and two total energies,
each with the field energy:

```{math}
\mathcal E_{rel} = \sum_p (\gamma_p - 1)\, m_p c^2 + \mathcal E_{field}, \qquad
\mathcal E_{N} = \sum_p \tfrac12 m_p |\mathbf v_p|^2 + \mathcal E_{field},
```

with $m_p$ the mass of a particle times its weight. The relativistic equations conserve
$\mathcal E_{rel}$, the non-relativistic ones $\mathcal E_{N}$; the `energy_error` of
{func}`~jaxincell.diagnostics` is taken from whichever of the two the run's pusher conserves.

## Result

```{figure} ../_static/figures/relativistic_two_stream.png
:width: 100%
:alt: Relativistic two-stream instability with the relativistic and non-relativistic Boris pushers

Relativistic Boris pusher (vermillion) and non-relativistic Boris pusher (blue) on the same
input. (a) Electric energy, with $e^{2\gamma t}$ (dashed, the colour of the run) for the
fastest box mode of each cold dispersion relation: mode 1 with
$\omega_b^2 \to \omega_b^2/\gamma_0^3$, mode {{ relativistic_fastest_mode_newtonian }}
without. (b) Relative change of $\mathcal E_{rel}$ (solid) and $\mathcal E_{N}$ (dotted). The
blue solid line stops at the dash-dotted line, when the first electron of the
non-relativistic run reaches $|v| \ge c$: from then on $\gamma_p$, and with it
$\mathcal E_{rel}$, is not defined. (c), (d) Electron phase space at the peak of the electric
energy, position in Debye lengths; the shaded bands are $|v_x| > c$.
```

| | relativistic pusher | non-relativistic pusher |
|---|---|---|
| fastest box mode (theory) | 1 | {{ relativistic_fastest_mode_newtonian }} |
| growth rate, cold theory ($\omega_{pe}$) | {{ relativistic_gamma_cold_relativistic }} | {{ relativistic_gamma_cold_newtonian }} |
| growth rate, simulation ($\omega_{pe}$) | {{ relativistic_gamma_fit_relativistic }} | {{ relativistic_gamma_fit_newtonian }} |
| deviation from theory | {{ relativistic_gamma_deviation_percent_relativistic }} % | {{ relativistic_gamma_deviation_percent_newtonian }} % |
| largest change of $\mathcal E_{rel}$ | {{ relativistic_error_rel_max_relativistic }} | {{ relativistic_error_rel_max_newtonian_before_superluminal }} before $\omega_{pe}t = $ {{ relativistic_t_superluminal_newtonian }}, undefined after |
| largest change of $\mathcal E_{N}$ | {{ relativistic_error_newton_max_relativistic }} | {{ relativistic_error_newton_max_newtonian }} |
| electrons with $\lvert v\rvert \ge c$ | {{ relativistic_superluminal_percent_relativistic }} % (largest $\gamma_p$: {{ relativistic_lorentz_max_relativistic }}) | up to {{ relativistic_superluminal_percent_newtonian }} % |

The growth rate is half the slope of the energy of the fastest mode, fitted from the last
time that energy was within three decades of its initial level (by then the non-growing roots
of the quartic the seed excites no longer matter) to the last time it was below a tenth of its
largest value, both before the first saturation. Each pusher reproduces the growth rate of its
own equations to about 1 %; the relativistic beams grow more than twice as slowly because of
their larger longitudinal inertia.

Each pusher conserves the energy of its own equations, and only that one, to the usual
energy error of the explicit leapfrog ({doc}`conservation`). Measured in the other energy each
run is wrong by tens of per cent. The non-relativistic pusher also accelerates trapped
electrons past the speed of light, as panel (d) shows, while the relativistic pusher keeps
every electron below $c$ with Lorentz factors up to {{ relativistic_lorentz_max_relativistic }}.

The two runs took {{ relativistic_seconds_relativistic }} s and
{{ relativistic_seconds_newtonian }} s, compilation included, on the shared CPU that built
this documentation.


For a smaller demonstration, run
`python examples/2_intermediate/relativistic_two_stream.py --quick`.
The default remains the documented benchmark; the quick preset is a smoke run.
