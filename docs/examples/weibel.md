# Weibel instability

`examples/weibel.py`

A plasma hotter across the simulation axis than along it is unstable to purely growing
transverse magnetic modes {cite}`weibel1959`. It is the electromagnetic instability of
the set, and it converts anisotropy in the distribution into magnetic field.

```{figure} ../_static/figures/weibel.png
:width: 100%
:alt: Weibel mode growth below and above the cutoff and the growth rate against wavenumber

(a) Modes below the cutoff grow, modes above it do not. (b) Growth rates against the
kinetic root.
```

## The marginal wavenumber

Setting $\omega = 0$ in the transverse dispersion relation gives the boundary of the
unstable band in closed form,

```{math}
k_c c = \omega_{pe}\sqrt{\frac{T_z}{T_x} - 1},
```

which is a sharp prediction needing no fitting: put several wavelengths in one box and
every mode below $k_c$ grows while none above it does. The example prints the gain of
each mode and marks the cutoff.

## Two things this example needs

**A Courant number at or below one.** The instability lives in the transverse fields,
so the explicit field solve is subject to the light-wave limit. `dt_over_dx_c=0.5`.

**A custom initial condition.** A bi-Maxwellian with $T_z \ne T_x$ is
`vth=(v, 0, v * sqrt(ratio))`, which `Species` supports directly. Seeding one mode
coherently — the way {doc}`../numerics/verification` measures the rates — needs
`quiet_start` plus a transverse current, and that is what
{func}`~jaxincell.quiet_start` is for.

## Things to try

* Change the anisotropy and check that the cutoff moves as $\sqrt{T_z/T_x - 1}$.
* Let it run past saturation: the field feeds back on the particles, isotropising the
  distribution and shutting the instability off.
* Look at $B_y$ in real space rather than in $k$: the growing modes are current
  filaments, and their merging is what the late nonlinear stage is about.
