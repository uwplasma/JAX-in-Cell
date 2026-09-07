# Diagnostics

{func}`jaxincell.diagnostics` post-processes an output dictionary outside the
compiled program. The quantities it adds are listed in {doc}`../user_guide/output`;
this page gives their definitions.

## Energies

With $\Delta x$ the cell size and the sums over the $N_x$ grid values,

```{math}
\mathcal E_E = \frac{\epsilon_0}{2}\sum_i |\mathbf E_i|^2\,\Delta x, \qquad
\mathcal E_B = \frac{1}{2\mu_0}\sum_i |\mathbf B_i|^2\,\Delta x,
```

and the same expressions for the external arrays. The kinetic energy sums over all
pseudo-particles with their pseudo-masses $m_p = m_s w_s$,

```{math}
\mathcal E_K = \sum_p \tfrac12 m_p |\mathbf v_p|^2 ,
```

split into electrons (negative charge) and ions (non-negative charge). The kinetic
energy is the non-relativistic one even when the relativistic pusher is used; for
$v \le 0.3c$ the difference is below 10 %. The total is the sum of all five terms.
All values are energies per unit area of the $y$-$z$ plane, in J/m².

The velocities stored in the output are the integer-time velocities and the fields
are the integer-time fields, so the energies are synchronous and their sum is the
right quantity to monitor for conservation.

## Dominant frequency

The time series of $E_x$ at the centre cell is centred, normalised and transformed
with the FFT; `dominant_frequency` is the angular frequency of the largest peak,
$2\pi f$. The resolution is $2\pi/(S\Delta t)$ with $S$ the number of steps, which is
0.2 $\omega_{pe}$ for a run of 30 plasma periods; the examples that print the ratio to
the plasma frequency use it only as a rough check.

## Species views

Particles are grouped by the sign of their charge into `*_electrons` and `*_ions`
arrays, and by exact (charge, mass) pairs into the `species` list. Absorbed particles
have zero charge and appear among the ions.

## Growth and damping rates

The code does not fit rates. The documentation figures use the following recipe,
implemented in `docs/scripts/common.py`:

1. take the Fourier transform of $E_x(x, t)$ (or $B_y$) in $x$ at every step and
   follow the amplitude of the mode of interest;
2. choose a window in which the mode energy is well above its initial noise level and
   well below its saturation value;
3. fit a straight line to the logarithm of the mode energy in that window; the
   growth rate of the amplitude is half the slope.

For damped waves the fit goes through the local maxima of the energy. Frequencies are
measured from the zero crossings of the mode amplitude. Both are compared with the
roots of the kinetic dispersion relation computed in `docs/scripts/dispersion.py`
with the plasma dispersion function {cite}`fried1961`.
