# One problem, seven solvers

The same seeded two-stream problem run with one solver switch changed at a time: which
switches change the physics, and which only change the cost. Each growth rate is compared
with the purely growing root of the electrostatic kinetic dispersion relation of the same
populations ({func}`jaxincell.theory.two_stream_rate`), which knows nothing of the solver.

```{figure} ../_static/figures/compare_models.png
:width: 100%
:alt: Growth of the seeded two-stream mode, its rate against kinetic theory and the energy error for seven solver settings

(a) The seeded mode for every setting, with the kinetic root (dashed). (b) Each fitted rate
against its reference. (c) The relative error of the total energy.
```

## What is measured against what

| setting | $\gamma/\omega_{pe}$ | reference | deviation (%) | energy error | wall time (s) |
|---|---|---|---|---|---|
| explicit electromagnetic, no filter | {{ compare_reference_rate }} | {{ compare_reference_reference }} | {{ compare_reference_deviation }} | {{ compare_reference_energy }} | {{ compare_reference_seconds }} |
| `model="electrostatic"` | {{ compare_electrostatic_rate }} | {{ compare_electrostatic_reference }} | {{ compare_electrostatic_deviation }} | {{ compare_electrostatic_energy }} | {{ compare_electrostatic_seconds }} |
| `field_solver="gauss"` | {{ compare_gauss_rate }} | {{ compare_gauss_reference }} | {{ compare_gauss_deviation }} | {{ compare_gauss_energy }} | {{ compare_gauss_seconds }} |
| `algorithm="implicit"` | {{ compare_implicit_rate }} | {{ compare_implicit_reference }} | {{ compare_implicit_deviation }} | {{ compare_implicit_energy }} | {{ compare_implicit_seconds }} |
| `filter_passes=2` | {{ compare_filtered_rate }} | {{ compare_filtered_reference }} | {{ compare_filtered_deviation }} | {{ compare_filtered_energy }} | {{ compare_filtered_seconds }} |
| `relativistic=True` | {{ compare_relativistic_rate }} | {{ compare_relativistic_reference }} | {{ compare_relativistic_deviation }} | {{ compare_relativistic_energy }} | {{ compare_relativistic_seconds }} |
| `Collisions()`, electrostatic | {{ compare_collisional_rate }} | {{ compare_collisional_reference }} | {{ compare_collisional_deviation }} | {{ compare_collisional_energy }} | {{ compare_collisional_seconds }} |

{{ compare_particles }} electrons, {{ compare_steps }} steps. Wall times are from the
machine that made the figure and are for comparing rows, not for quoting.

What the table shows:

* **The field model does not change the rate.** Electrostatic, Gauss and Ampere agree to
  every printed digit: in one dimension with no transverse velocity they solve the same
  $E_x$.
* **The implicit scheme conserves energy to round-off**, where the explicit leapfrog drifts
  by about $10^{-4}$ through saturation, at the same step and the same rate.
* **Two filter passes leave mode 1 alone**: the binomial filter damps the short wavelengths
  only.
* **The relativistic push lowers the rate** by the longitudinal mass $\gamma_0^3$ of beams
  at $0.17c$; its reference is the kinetic root times the cold relativistic reduction at
  the same $k$.
* **Collisions change nothing**, as they must at a collision rate some ten million times
  below $\omega_{pe}$. They run with the electrostatic model: a collision turns a
  longitudinal beam isotropic, and transverse velocity at $c\,\Delta t/\Delta x = 4.5$
  seeds a light wave the explicit electromagnetic solver cannot hold. The `Simulation`
  warns when such a run is built ({doc}`../numerics/stability`).

The common 3 % below the kinetic root is the seeded start: the fit window opens while the
damped roots the seed also excites are still decaying.

## How to run

```bash
python examples/2_intermediate/compare_models.py
python examples/2_intermediate/compare_models.py --quick    # fewer particles, noisier
```

It writes `compare_models/run.json` (settings, every rate, energy error and timing,
provenance), `compare_models/curves.npz` and its figure; `docs/scripts/fig_compare_models.py`
runs it and reads it back for this page.
