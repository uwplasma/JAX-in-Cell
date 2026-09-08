# Resolution and stability

Four conditions bound the time step and the cell size. Three are hard stability limits
of the explicit scheme; the fourth is an accuracy requirement that applies to both
schemes.

## 1. Plasma oscillations

The leapfrog integrates the harmonic oscillator $\ddot x = -\omega_p^2 x$ stably only
for $\omega_p\Delta t < 2$, and its frequency error is

```{math}
\frac{\omega_{\rm numerical}}{\omega_p} = \frac{2}{\omega_p\Delta t}\arcsin\!\left(\frac{\omega_p\Delta t}{2}\right)
= 1 + \frac{(\omega_p\Delta t)^2}{24} + \mathcal{O}\!\left((\omega_p\Delta t)^4\right).
```

Take $\omega_p\Delta t \le 0.2$ for a frequency error below $0.2$ per cent. The
verification runs use {{ energy_omega_pe_dt }} and {{ landau_omega_pe_dt }}.

Print the value before a long run:

```python
print(float(simulation.plasma_frequency() * simulation.domain.dt))
```

## 2. Light waves (explicit only)

The Yee update of the transverse fields is stable only for

```{math}
\frac{c\,\Delta t}{\Delta x} \le 1,
```

the Courant condition, which is `dt_over_dx_c` directly. At exactly one the scheme is
*exact* for a plane wave in vacuum — the "magic time step", at which the numerical
dispersion relation reduces to $\omega = ck$ — and a pulse is translated by a whole
cell per step with no error at all. Below one the scheme is stable but dispersive;
above one it blows up.

:::{warning}
Purely electrostatic problems never excite the transverse fields, so they are often
run at $c\Delta t/\Delta x \gg 1$ on purpose: the two-stream runs here use
{{ energy_courant }}. That is safe only while $E_y$, $E_z$, $B_y$ and $B_z$ stay
identically zero. Give the particles any transverse velocity — an isotropic
temperature, a magnetic field, collisions — and the light-wave branch is seeded and
the run diverges within a few steps. {class}`~jaxincell.Simulation` emits a
`UserWarning` when it sees that combination.
:::

## 3. Cell crossing

A particle should not cross more than about one cell per step, or the deposit and
gather no longer sample a smooth orbit and the charge-conserving current loses
accuracy:

```{math}
\frac{v_{\max}\Delta t}{\Delta x} \lesssim 1 .
```

Because $\Delta t$ is set through $c\Delta t/\Delta x$, this is automatic for
non-relativistic particles whenever the Courant condition holds, and is only a
constraint when the Courant condition is deliberately violated in electrostatic mode.

## 4. Debye length (explicit only)

If the cell is much larger than the Debye length the aliased short-wavelength modes
exchange energy with the particles and the plasma heats until $\lambda_D \sim \Delta x$
— the finite-grid instability {cite}`birdsall1991`. The threshold for the quadratic
shape function is around $\Delta x \lesssim 3\lambda_D$; staying below one is
comfortable. The verification runs use $\Delta x/\lambda_D = $
{{ two_stream_dx_over_debye }}.

Two things relax this. Digital {doc}`filtering` removes the aliased modes and pushes
the threshold out by a factor of a few. The {doc}`implicit` scheme does not suffer
from the instability at all, which is the main reason to use it when the Debye length
is impossible to resolve.

## Putting it together

For an electromagnetic problem, choose $\Delta x \le \lambda_D$ and
$c\Delta t/\Delta x \le 1$; check that $\omega_p\Delta t$ came out below $0.2$ and
lower the Courant number if not. For an electrostatic problem, choose
$\Delta x \le \lambda_D$ and then $\Delta t$ from $\omega_p\Delta t \le 0.2$, which
usually means a Courant number above one — allowed, as long as nothing transverse is
excited.

The single most useful check is the energy budget. Total energy that grows without
bound means a stability limit has been broken; the {doc}`diagnostics` page shows how
to watch it.
