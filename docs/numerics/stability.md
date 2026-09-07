# Stability and resolution

The constraints below are the ones that decide whether a run is meaningful. Most of
them are stated in terms of the electron plasma frequency $\omega_{pe}$ and Debye
length $\lambda_D$ printed at the start of a run with `print_info = true`.

## Time step

Plasma oscillations
: The leapfrog integrator is stable for $\omega_{pe}\Delta t < 2$ and accurate to a
  few percent in the oscillation frequency for $\omega_{pe}\Delta t \lesssim 0.3$. The
  examples use $\omega_{pe}\Delta t \approx 0.08$ to $0.1$. The implicit scheme has no
  stability limit here but the same accuracy consideration, and its Picard iteration
  converges only for $\omega_{pe}\Delta t$ of order one or smaller.

Cell crossing
: The charge-conserving current deposit sweeps six cells around each particle and
  assumes that the particle moves by less than one cell per step,
  $|v_x|\Delta t < \Delta x$. Faster particles deposit a truncated current and Gauss's
  law is no longer preserved. Check the thermal tails: with $v_{th}/c = 0.05$ and
  $c\,\Delta t/\Delta x = 4.5$ the bulk moves $0.2$ cells per step but a drift of $0.2c$
  brings it to $0.9$ cells, which is why the examples with drifts of this size are at
  the limit.

Light waves
: The explicit field update is stable for $c\,\Delta t/\Delta x \le 1$. The constraint
  applies as soon as any transverse field can be excited: transverse thermal spread or
  drift, an external magnetic field, or the Weibel instability. Purely electrostatic
  problems (all velocities along $x$, no $\mathbf B$) never excite the transverse
  equations and can run with larger values, which the two-stream examples exploit.
  The implicit scheme removes this constraint.

Gyration
: With an external magnetic field $B$, resolve the cyclotron motion,
  $\Omega_c\Delta t \lesssim 0.3$ with $\Omega_c = |q|B/m$. The Boris rotation stays
  stable for any $\Omega_c\Delta t$ but the gyro-phase becomes inaccurate.

## Cell size

Debye length
: Explicit electrostatic schemes suffer from the finite-grid instability when
  $\Delta x \gtrsim 3\lambda_D$ with linear weighting {cite}`langdon1970`. The
  quadratic spline and the filter push the limit to larger cells, and the
  implicit scheme is not subject to it, but a resolved Debye length,
  `grid_points_per_Debye_length` $\gtrsim 0.5$, is the safe choice. Coarser grids
  heat the plasma until $\lambda_D$ grows to the cell size.

Wavelength
: A mode of wavenumber $k$ needs $k\Delta x \ll 1$ for the spline and the
  finite-difference curl to represent it, and it must survive the filter:
  $k\Delta x \lesssim 0.1\pi$ with the default filter settings, see
  {doc}`filtering`.

Skin depth
: Electromagnetic structures form on the scale $d_e = c/\omega_{pe}$; the Weibel
  example resolves it with about twelve cells.

## Particle number

Noise
: Fluctuations of the deposited density scale as $1/\sqrt{N_c}$ with $N_c$ particles
  per cell. Instabilities that grow from noise saturate after
  $\ln(\text{saturation}/\text{noise})$ e-foldings, so few particles means a short
  linear phase and a growth rate that is hard to measure; see the two-stream
  discussion in {doc}`verification`. The examples use 50 to 1000 particles per cell.

Collisionality
: The finite number of pseudo-particles introduces numerical collisions at a rate
  that decreases with the number of particles per Debye length. For runs longer than a
  few hundred plasma periods, watch the kinetic energy of the ions for spurious heating.

## Length of the run

Recurrence
: With equally spaced initial positions and random velocities there is no recurrence
  in the classical sense, but the noise floor is reached once a damped wave has decayed
  by $\ln\sqrt{N}$ e-foldings, which limits how long Landau damping can be observed.

Energy drift
: The explicit scheme's total energy drifts by $10^{-3}$ to $10^{-2}$ relative over the
  examples; if the drift matters, use the implicit scheme, whose error stays at
  round-off.

## Amplitudes

A displacement perturbation of relative amplitude $ak$ excites a wave with electric
field $E \approx a k\, n e/(\epsilon_0 k)$ and bounce frequency
$\omega_b = \sqrt{|q|kE/m} \approx \sqrt{ak}\,\omega_{pe}$. Linear theory applies while
$\omega_b \ll |\gamma|$; `examples/Landau_damping.py` uses $ak = 0.16$ and
$\omega_b = 0.4\,\omega_{pe}$, above the linear damping rate, and shows nonlinear
damping as a result. Small amplitudes need the low noise floor of a quiet start or
many particles.
