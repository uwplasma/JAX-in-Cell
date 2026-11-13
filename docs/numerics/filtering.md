# Digital filtering

Depositing a finite number of pseudo-particles onto a grid produces short-wavelength
noise that no physical process damps. Left alone it heats the plasma and, at
$\Delta x \gtrsim \lambda_D$, drives the finite-grid instability ({doc}`stability`).
The standard remedy is to smooth the deposited sources with a digital filter
{cite}`birdsall1991`.

## The filter

`Solver(filter_passes=p, filter_alpha=a, filter_strides=(s1, s2, ...))` applies, for
each stride $s$, `p` three-point passes

```{math}
f_i \to a f_i + \frac{1-a}{2}\left(f_{i-s} + f_{i+s}\right)
```

followed by one *compensation* pass with the weight $a_c = 1 + p(1-a)$, which is
greater than one and therefore sharpens rather than smooths. The response of one pass
at wavenumber $k$ is

```{math}
G_s(k) = a + (1-a)\cos(k s \Delta x),
```

so that a group of `p` passes plus its compensation has

```{math}
G(k) = G_s(k)^{p}\,\left[1 + p(1-a)\left(1 - \cos(k s \Delta x)\right)\right]^{\!*},
```

where the starred bracket is $G$ evaluated with $a_c$. Expanding for
$k s\Delta x \ll 1$, the leading $\mathcal{O}((ks\Delta x)^2)$ terms of the two
factors cancel by construction, so $G(k) = 1 + \mathcal{O}((ks\Delta x)^4)$: long
wavelengths pass essentially untouched while the grid scale is strongly damped.

```{figure} ../_static/figures/filter_response.png
:width: 100%
:alt: Transfer function of the compensated binomial filter for several pass counts and strides

Transfer function against $k\Delta x$. The compensated filter is flat at long
wavelength and cuts the grid scale; adding strides widens the cut without adding
passes at stride one.
```

:::{note}
`filter_passes=1` with the default `filter_alpha=0.5` is very nearly the identity: one
smoothing pass followed by its own compensation. Use `0` to turn filtering off and
`2` or more for it to do anything.
:::

## Strides

A stride $s>1$ applies the same stencil to cells $s$ apart, which puts the zero of the
response at $k\Delta x = \pi/s$ instead of $\pi$. Combining strides, for example
`filter_strides=(1, 2, 4)`, damps a wide band at a cost linear in the number of
strides, rather than needing many passes at stride one. The defaults `(1,)` with
`filter_passes=0` leave the sources alone; a common working choice is
`filter_passes=2, filter_strides=(1, 2, 4)`, which removes essentially everything
below about twenty cells.

## What to watch out for

Filtering is not free. It smooths the source that the field solve sees, so it damps
the physical field at short wavelength too. If the mode being studied is only a few
cells long, the filter will attenuate it: check the response at the wavenumber of
interest before turning passes on. The verification runs in {doc}`verification` use
`filter_passes=0` for exactly this reason, and the growth rates they report agree with
kinetic theory to a few per cent without any smoothing.

Charge conservation survives filtering, for a more direct reason than commuting
operators: the current is taken from the *filtered* density by the cumulative sum of
{eq}`cumsum-current`, so {eq}`discrete-continuity` holds for whatever the filter
produced. The Gauss residual with `filter_passes=2` is the same
{{ gauss_residual_max_explicit }} as without, at every wall type.

The filter is also conservative in its own right, which is a separate requirement:
smoothing should move source around, not create it. At a periodic wall that is
automatic. At a reflective wall the stencil is **mirrored** back into the box, so what
a cell would have sent through the wall stays on this side and the total is untouched
for any stride. Clamping to the boundary cell instead — the zero-gradient
extrapolation that is right for a *field* — invents charge at the wall, several per
cent of the total for a stride-two stencil, which shows up as a spurious sheath field
and an energy error an order of magnitude larger than it should be. Only an absorbing
wall drops the part of the stencil that falls outside, which is what letting charge
leave means.
