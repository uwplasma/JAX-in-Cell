# Conservation laws

The quiet two-stream problem of {doc}`two_stream`, run with the explicit leapfrog and with
the implicit Crank-Nicolson scheme, in a periodic box and between two absorbing walls. The
three relative errors are those of {doc}`../numerics/diagnostics`: energy, momentum and
charge (the discrete Gauss law).

```{figure} ../_static/figures/conservation.png
:width: 100%
:alt: Energy, momentum and Gauss-law errors of both schemes

(a) Energy error, with the implicit scheme at one, two, four and eight Picard iterations.
(b) Momentum error. (c) Gauss-law residual, periodic (solid) and between absorbing walls
(dashed), where the walls take energy and momentum away and the charge is what is left to
check.
```

## What is measured against what

| largest error over the run | explicit | implicit, 8 Picard | reference |
|---|---|---|---|
| energy | {{ energy_error_max_explicit }} | {{ energy_error_max_implicit_8 }} | exact conservation |
| momentum | {{ momentum_error_relative }} | {{ momentum_error_implicit }} | exact conservation |
| charge (Gauss law) | {{ gauss_residual_max_explicit }} | {{ gauss_residual_max_implicit }} | discrete Gauss law, zero |

The implicit energy error falls geometrically with the iteration count:

| Picard iterations | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| largest energy error | {{ energy_error_max_implicit_1 }} | {{ energy_error_max_implicit_2 }} | {{ energy_error_max_implicit_4 }} | {{ energy_error_max_implicit_8 }} |

until it reaches the round-off of double precision — the signature of a scheme that
conserves energy exactly once its nonlinear system is solved {cite}`chen2011`. The explicit
error is *bounded* instead: it oscillates and does not accumulate, as a time-reversible,
volume-preserving integrator does {cite}`qin2013`.

The Gauss law sits at round-off for both schemes, periodic or not, because both take the
charge-conserving current ({doc}`../numerics/deposition`). No scheme here keeps all three:
the explicit one trades the energy for the momentum, the implicit one the momentum for the
energy ({doc}`../numerics/implicit`).

## The setup

| | |
|---|---|
| steps | {{ energy_steps }} |
| $\omega_{pe}\Delta t$ | {{ energy_omega_pe_dt }} |
| `dt_over_dx_c` | {{ energy_courant }} |
| drift | $5\times10^7$ m/s, quiet start |
| boundaries | periodic, and two absorbing walls |

## At a larger step

The implicit scheme is worth its cost when it takes a larger step than the explicit one
can. Here both schemes run two problems whose rates are known, Landau damping at
$k\lambda_D = 0.5$ and the quiet two-stream problem of {doc}`two_stream`, with the implicit
step {{ schemes_step_ratio }} times the explicit one.

```{figure} ../_static/figures/explicit_implicit.png
:width: 100%
:alt: Landau damping and two-stream growth with the explicit and the implicit scheme, and their energy errors

(a) Field energy of Landau damping, {{ schemes_landau_particles }} quiet-start electrons and
a seed of $ak = {{ schemes_landau_seed_ak }}$: explicit at
$\omega_{pe}\Delta t = {{ schemes_landau_omega_pe_dt_explicit }}$
($c\Delta t/\Delta x = {{ schemes_landau_courant_explicit }}$, the explicit light-wave
limit), implicit at $\omega_{pe}\Delta t = {{ schemes_landau_omega_pe_dt_implicit }}$
($c\Delta t/\Delta x = {{ schemes_landau_courant_implicit }}$). The dotted line is the
kinetic decay $e^{2\gamma t}$ from the first maximum; each scheme's rate in the legend comes
from the maxima of its field energy. (b) Field energy of the two-stream problem: explicit at
$\omega_{pe}\Delta t = {{ schemes_two_stream_omega_pe_dt_explicit }}$, implicit at
{{ schemes_two_stream_omega_pe_dt_implicit }}. Each rate is fitted to the seeded mode as in
{doc}`two_stream`, and the dotted line is the kinetic growth $e^{2\gamma t}$ drawn to where
the fit ends; early in the growth the field energy is mostly the noise of the other modes,
which is why the rate is fitted to the seeded mode. (c) and (d) The relative total-energy error of each run in (a) and (b).
```

| | explicit | implicit | kinetic root |
|---|---|---|---|
| Landau damping, $\gamma/\omega_{pe}$ | {{ schemes_landau_gamma_explicit }} ({{ schemes_landau_gamma_deviation_percent_explicit }} %) | {{ schemes_landau_gamma_implicit }} ({{ schemes_landau_gamma_deviation_percent_implicit }} %) | {{ schemes_landau_gamma_theory }} |
| two-stream, $\gamma/\omega_{pe}$ | {{ schemes_two_stream_gamma_explicit }} ({{ schemes_two_stream_gamma_deviation_percent_explicit }} %) | {{ schemes_two_stream_gamma_implicit }} ({{ schemes_two_stream_gamma_deviation_percent_implicit }} %) | {{ schemes_two_stream_gamma_theory }} |
| Landau damping, largest energy error | {{ schemes_landau_energy_error_explicit }} | {{ schemes_landau_energy_error_implicit }} | |
| two-stream, largest energy error | {{ schemes_two_stream_energy_error_explicit }} | {{ schemes_two_stream_energy_error_implicit }} | |

At {{ schemes_step_ratio }} times the step the implicit rates stay as close to the kinetic
roots as the explicit ones, and the energy error stays at round-off in the Landau run and at
{{ schemes_two_stream_energy_error_implicit }} in the two-stream run, with the default
{{ schemes_picard_iterations }} Picard iterations. There a beam electron crosses
{{ schemes_two_stream_beam_cells_per_step_implicit }} cells per step, against
{{ schemes_two_stream_beam_cells_per_step_explicit }} in the explicit run, past the one cell
the explicit scheme wants ({doc}`../numerics/stability`).
The figure and the numbers come from `docs/scripts/fig_explicit_implicit.py`.

## The cost

Eight Picard iterations with two sub-steps costs about
{{ scaling_implicit_over_explicit }} times an explicit step, so the implicit scheme pays
for itself when it lets you take a step at least that much larger — or when the energy
budget is the point. The script prints the wall-clock time of each run; see
{doc}`../user_guide/performance`.

## How to run

```bash
python examples/3_advanced/conservation.py
jaxincell inputs/conservation_implicit.toml
```

The figure and the numbers come from `docs/scripts/fig_conservation.py`, which runs this
setup and also sweeps the Picard iteration count.
