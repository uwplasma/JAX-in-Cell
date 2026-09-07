# Initialisation

## Positions

For each population the $x$ coordinates are either equally spaced over the box
(`random_positions_x = false`, the default) or uniform random numbers. Then the
displacement

```{math}
x_p \to x_p + a\sin\!\left(\frac{2\pi m}{L}x_p\right)
```

is applied with $a$ = `perturbation_amplitude_x` and $m$ = `perturbation_wavenumber_x`.
To first order in $ak$ this produces the density perturbation
$\delta n/n = -ak\cos(kx)$ with $k = 2\pi m/L$, which seeds a standing wave of mode
number $m$. The same is done for $y$ and $z$ with their own parameters; those
coordinates default to random and have no effect on the fields.

## Velocities

Each component is drawn from a normal distribution of standard deviation
$v_{th}/\sqrt2$, where $v_{th}$ = `vth_over_c_*` times $c$, and the drift is added.
The result is a Maxwellian with $k_B T = m v_{th}^2/2$ in that component. If
`velocity_plus_minus_*` is set, the component of every second particle is negated,
which creates two counter-propagating beams from one population without changing its
density. Finally every component is clipped to $\pm0.99c$ so that the Lorentz factor
of the relativistic pusher is finite.

Random numbers come from `jax.random` with keys derived from `solver_parameters.seed`
as described in {doc}`../user_guide/species`; two runs with the same inputs give the
same particles.

A user-supplied phase space (`initial_positions`, `initial_velocities`) replaces the
generated one. The quiet start used for the Landau-damping check in
{doc}`verification` is built this way: equally spaced positions with the displacement
applied by hand, and velocities placed at the quantiles of the Maxwellian in
bit-reversed order, which lowers the initial noise by orders of magnitude compared
with random sampling.

## Weights

The weight of each population is computed from its `grid_points_per_Debye_length` and
the thermal speed and charge of the first electron population, as derived in
{doc}`../user_guide/species`:

```{math}
w_s = \frac{\epsilon_0 m_e c^2}{q_e^2}\,\frac{N_x^2 g_s^2}{2 L N_s}\left(\frac{v_{th,e}}{c}\right)^2 .
```

Charges and masses of the pseudo-particles are $q_s w_s$ and $m_s w_s$; the
charge-to-mass ratio used by the pusher is $q_s/m_s$. All populations are concatenated
into single arrays in input order, and an integer index per particle records the
population.

## Initial fields

The magnetic field starts at zero. The electric field starts from Gauss's law: the
charge density is deposited on the cell centres from the initial positions, filtered
with the solver's filter settings, and integrated from the left wall,
$E_{x,i+1/2} = (\Delta x/\epsilon_0)\sum_{j\le i}\rho_j$. For a neutral box the field is
periodic; for a box with net charge it grows linearly across the box. The transverse
components start at zero. External field arrays, if supplied, are stored separately
and never modified.

## Half-step positions

The explicit leapfrog needs positions at $t^{\pm1/2}$. They are created by
$x^{\pm1/2} = x^0 \pm \tfrac12\Delta t\,v_x^0$ followed by the particle boundary
condition, so the first current deposit uses the motion from $t^{-1/2}$ to $t^{1/2}$
and the first push sees fields at $t^{1/2}$. This is a first-order initialisation of a
second-order scheme, which is standard practice and affects only the first step.
