# Bump-on-tail instability

`examples/bump_on_tail.py`

A weak beam on the tail of a Maxwellian makes $\partial f/\partial v > 0$ there, and
every wave with a phase velocity in that window grows. The waves saturate by
flattening the bump into a plateau — the quasilinear end state
{cite}`vedenov1961,oneil1965`.

```{figure} ../_static/figures/bump_on_tail.png
:width: 100%
:alt: Growth of the resonant mode and the quasilinear plateau

(a) The resonant mode. (b) The distribution before and after: the positive slope has
gone.
```

## The setup

A beam carrying {{ bump_on_tail_beam_fraction }} of the density, drifting at
{{ bump_on_tail_beam_drift_over_vth }} $v_{th}$ and
{{ bump_on_tail_beam_width_over_vth }} $v_{th}$ wide. The thermal speed is chosen from
the resonance condition $\omega_{pe}/k = v_{\rm beam}$, so that the fastest-growing
wave fits an integer number of times in the box and is well resolved by the grid —
without that the seeded mode can land outside the unstable band entirely.

Mode {{ bump_on_tail_mode }} grows at {{ bump_on_tail_gamma_measured }} against
{{ bump_on_tail_gamma_theory }} from the kinetic root.

## Things to try

* Widen the beam until it merges with the bulk and the instability disappears.
* Watch the plateau form: it fills the whole velocity range between the bulk and the
  beam, not just the resonant band of the seeded mode, because as the distribution
  flattens more waves come into resonance.
* Compare the energy that went into the field with the free energy in the bump.
