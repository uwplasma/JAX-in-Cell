# Energy conservation

`examples/energy_conservation.py`

The same two-stream problem run with the explicit leapfrog and with the implicit
Crank-Nicolson scheme at one, two, four and eight Picard iterations.

```{figure} ../_static/figures/conservation.png
:width: 100%
:alt: Energy error of both schemes and the Gauss-law residual

(a) Total energy error. (b) The Gauss-law residual, at round-off for both schemes.
```

## What it shows

| scheme | energy error |
|---|---|
| explicit | {{ energy_error_max_explicit }} |
| implicit, 1 Picard iteration | {{ energy_error_max_implicit_1 }} |
| implicit, 2 | {{ energy_error_max_implicit_2 }} |
| implicit, 4 | {{ energy_error_max_implicit_4 }} |
| implicit, 8 | {{ energy_error_max_implicit_8 }} |

Two separate points. First, the explicit scheme's error is *bounded*: it oscillates at
a few parts in $10^5$ and does not accumulate, which is what a time-reversible,
volume-preserving integrator does {cite}`qin2013`. Second, the implicit scheme's error
falls geometrically with the iteration count until it hits the round-off of double
precision, which is the signature of a scheme that conserves energy exactly once its
nonlinear system is solved {cite}`chen2011`.

The Gauss-law residual sits at round-off for both, because both use the
charge-conserving current deposit ({doc}`../numerics/deposition`).

## The cost

The script prints the wall-clock time of each run. Eight Picard iterations with two
sub-steps costs about {{ scaling_implicit_over_explicit }} times an explicit step, so
the implicit scheme pays for itself when it lets you take a step at least that much
larger — or when the energy budget is the point.
