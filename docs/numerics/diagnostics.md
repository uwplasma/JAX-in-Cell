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

split into electrons (negative charge) and ions (non-negative charge). For the
relativistic pusher it uses $\sum_p(\gamma_p-1)m_pc^2$, evaluated as
$\sum_p m_p v_p^2/[r_p(1+r_p)]$ with $r_p=\sqrt{1-v_p^2/c^2}$ to retain precision
at low speed. The total is the sum of all five terms.
All values are energies per unit area of the $y$-$z$ plane, in J/m².

The velocities stored in the output are the integer-time velocities and the fields
are the integer-time fields, so the energies are synchronous and their sum is the
right quantity to monitor for conservation.

## Charge and momentum

Charge conservation is measured through Gauss's law with the backward difference
that the field solve uses (periodic wrap for periodic fields, $E_{x,-1} = 0$
otherwise),

```{math}
r_i = \frac{E_{x,i} - E_{x,i-1}}{\Delta x} - \frac{\rho_i}{\epsilon_0}, \qquad
\texttt{gauss\_error\_Linf\_rel} = \frac{\max_i |r_i|}{\max_i |\rho_i/\epsilon_0|},
```

with the mean of $r$ removed for periodic fields, which can only satisfy Gauss's law
up to the mean charge. The total momentum is $\mathbf P = \sum_p m_p\mathbf v_p$
or $\sum_p\gamma_p m_p\mathbf v_p$ for the relativistic pusher. `momentum_error_rel`
is $|\mathbf P(t)-\mathbf P(t_{\rm ref})|/\sum_p|\mathbf p_p(t_{\rm ref})|$,
where $t_{\rm ref}$ is the first stored row; the denominator is used because the
net momentum itself is often zero. Neither scheme conserves momentum exactly.

## Dominant frequency

The time series of $E_x$ at the centre cell is centred and transformed
with the FFT; `dominant_frequency` is the angular frequency of the largest peak,
$2\pi f$. The resolution is $2\pi/(S\Delta t)$ with $S$ the number of steps, which is
0.2 $\omega_{pe}$ for a run of 30 plasma periods; the examples that print the ratio to
the plasma frequency use it only as a rough check. The FFT uses the number of
stored rows and their uniform time spacing when `time_array` is supplied. A single
stored row or a constant signal has zero dominant frequency.

## Species views

Particles are grouped by the sign of their charge into `*_electrons` and `*_ions`
arrays. The `species` list uses the configured population IDs and labels, so two
electron populations with identical charge and mass remain separate. Without IDs
it falls back to exact (charge, mass) pairs. The raw arrays are retained for repeated
diagnostics and other analyses. Absorbed particles with zero charge appear among
the legacy ion arrays while their population IDs remain unchanged.

For each population the weights define its bulk velocity
$\mathbf u=\sum_p w_p\mathbf v_p/\sum_p w_p$. Its non-relativistic directional
temperatures in K are
$T_j=\sum_p m_p(v_{p,j}-u_j)^2/(k_B\sum_p w_p)$, with the pseudo-mass $m_p=m_s w_p$;
`temperature` is their mean. This Newtonian velocity-variance moment uses physical
particle mass as a temperature scale and excludes bulk-flow energy; it does not
define a relativistic thermodynamic temperature.

The output supplies one set of particle charges, masses and weights for the whole
history. It cannot reconstruct when particles were partly or wholly absorbed, so
these moments and energies do not fix historical wall accounting. Time-dependent
weights and a wall ledger are needed to diagnose that exchange correctly.

## Growth and damping rates

The code does not fit rates. The documentation figures use the following procedure,
implemented in `docs/scripts/common.py`:

1. Fourier transform $E_x(x, t)$ (or $B_y$) in $x$ at every step and follow the
   complex amplitude of the mode of interest.
2. Choose the fitting window with `robust_growth_fit`: among all windows inside the
   growth phase, keep the **longest** whose straight-line fit to the logarithm of the
   mode energy reaches $R^2 \ge 0.95$, requiring at least 15 $\omega_{pe}^{-1}$ and
   1.5 e-foldings. Length is the right thing to maximise: the steepest or
   best-correlated window tends to be a short one sitting on a noise excursion, and
   the first inverse plasma times of a seeded run are the perturbation settling onto
   the growing eigenmode.
3. The growth rate of the amplitude is half the slope of the log of the energy.
4. If no window qualifies, report the mode as unmeasured instead of fitting it.

For damped waves the fit goes through the local maxima of the energy. Frequencies are
measured from the zero crossings of the real part of the mode amplitude. Both are
compared with the roots of the kinetic dispersion relation computed in
`docs/scripts/dispersion.py` with the plasma dispersion function {cite}`fried1961`.

Measuring a rate at all requires the mode to grow over a usable range. A mode that
starts at the particle-noise floor and saturates two e-foldings later does not, which
is why the verification runs load the plasma quietly (equally spaced positions,
velocities at quantiles of a bit-reversed sequence) and seed a single mode; see
{doc}`verification`.
