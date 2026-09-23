# Bump-on-tail instability

A weak beam on the tail of a Maxwellian makes $\partial f/\partial v > 0$ there, and every
wave with a phase velocity in that window grows. The waves saturate by flattening the bump
into a plateau — the quasilinear end state {cite}`vedenov1961,oneil1965`.

```{figure} ../_static/figures/bump_on_tail.png
:width: 100%
:alt: Growth of the resonant mode and the quasilinear plateau

(a) $|E_{k=5}(t)|$, the fit through its linear phase (dashed) and the kinetic growth rate
at the same intercept (dotted). (b) The velocity distribution initially (dashed) and at
the end (solid), with the phase velocity of mode {{ bump_on_tail_mode }} marked: the
positive slope has gone.
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| growth rate $\gamma/\omega_{pe}$, mode {{ bump_on_tail_mode }} | {{ bump_on_tail_gamma_measured }} | {{ bump_on_tail_gamma_theory }} (kinetic root) | {{ bump_on_tail_gamma_deviation_percent }} % |
| frequency $\omega/\omega_{pe}$ | — | {{ bump_on_tail_omega_theory }} (kinetic root) | — |
| the positive slope, after saturation | panel (b) | flat, quasilinear theory | — |

## The setup

| | |
|---|---|
| beam fraction of the density | {{ bump_on_tail_beam_fraction }} |
| beam drift | {{ bump_on_tail_beam_drift_over_vth }} $v_{th}$ |
| beam width | {{ bump_on_tail_beam_width_over_vth }} $v_{th}$ |
| resonant mode | {{ bump_on_tail_mode }}, phase velocity {{ bump_on_tail_phase_velocity_over_vth }} $v_{th}$ |
| bulk $k\lambda_D$ | {{ bump_on_tail_bulk_k_lambda_D }} |
| cells | {{ bump_on_tail_cells }} |
| $\Delta x/\lambda_D$ | {{ bump_on_tail_dx_over_debye }} |
| particles | {{ bump_on_tail_particles }} |
| seed | $ak$ = {{ bump_on_tail_seed_ak }} |

The thermal speed is chosen from the resonance condition $\omega_{pe}/k = v_{\rm beam}$,
so that the fastest-growing wave fits an integer number of times in the box and is well
resolved by the grid. Without that the seeded mode can land outside the unstable band
entirely.

## How to run

```bash
python examples/2_intermediate/bump_on_tail.py
jaxincell inputs/bump_on_tail.toml
```

The figure and the numbers come from `docs/scripts/fig_bump_on_tail.py`, which runs this
setup.

## Things to try

* Widen the beam until it merges with the bulk and the instability disappears.
* Watch the plateau form: it fills the whole velocity range between the bulk and the beam,
  not just the resonant band of the seeded mode, because as the distribution flattens more
  waves come into resonance.
* Compare the energy that went into the field with the free energy in the bump.
