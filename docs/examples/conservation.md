# Conservation laws

`examples/conservation.py`

The quiet two-stream problem of {doc}`two_stream`, run with the explicit leapfrog and with
the implicit Crank-Nicolson scheme, in a periodic box and between two absorbing walls, with
the three relative errors of {doc}`../numerics/diagnostics`: energy, momentum and charge (the
discrete Gauss law). The figure and the table come from `docs/scripts/fig_conservation.py`,
which runs this setup, a drift of $5\times10^7$ m/s and 400 steps, and also sweeps the Picard
iteration count.

```{figure} ../_static/figures/conservation.png
:width: 100%
:alt: Energy, momentum and Gauss-law errors of both schemes

(a) Energy error, with the implicit scheme at one, two, four and eight Picard iterations.
(b) Momentum error. (c) Gauss-law residual, periodic (solid) and between absorbing walls
(dashed), where the walls take energy and momentum away and the charge is what is left to
check.
```

## What it shows

| largest error over the run | explicit | implicit, 8 Picard |
|---|---|---|
| energy | {{ energy_error_max_explicit }} | {{ energy_error_max_implicit_8 }} |
| momentum | {{ momentum_error_relative }} | {{ momentum_error_implicit }} |
| charge (Gauss law) | {{ gauss_residual_max_explicit }} | {{ gauss_residual_max_implicit }} |

The implicit energy error falls geometrically with the iteration count —
{{ energy_error_max_implicit_1 }}, {{ energy_error_max_implicit_2 }},
{{ energy_error_max_implicit_4 }} at one, two and four — until it reaches the round-off of
double precision, the signature of a scheme that conserves energy exactly once its nonlinear
system is solved {cite}`chen2011`. The explicit error is *bounded* instead: it oscillates and
does not accumulate, as a time-reversible, volume-preserving integrator does {cite}`qin2013`.

The Gauss law sits at round-off for both schemes, periodic or not, because both take the
charge-conserving current ({doc}`../numerics/deposition`). No scheme here keeps all three: the
explicit one trades the energy for the momentum, the implicit one the momentum for the energy
({doc}`../numerics/implicit`).

## The cost

The script prints the wall-clock time of each run. Eight Picard iterations with two
sub-steps costs about {{ scaling_implicit_over_explicit }} times an explicit step, so
the implicit scheme pays for itself when it lets you take a step at least that much
larger — or when the energy budget is the point.
