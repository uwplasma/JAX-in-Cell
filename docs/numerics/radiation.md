# Classical radiation reaction

`Radiation(model="ll_reduced")` adds the field-quadratic part of the classical
Landau–Lifshitz force to selected relativistic species. It represents recoil from
unresolved emission; it does not create photon particles or a radiation transport
field. This is the reduced model used in radiation-reaction PIC treatments such as
[Tamburini et al. (2010)](https://doi.org/10.1088/1367-2630/12/12/123005).
The field-derivative term of full Landau–Lifshitz is omitted. Small quantum effects
alone do not establish that this additional approximation is adequate for a rapidly
varying field.

```python
from jaxincell import Radiation, Simulation, Solver, Species

# u0 has shape (n, 3), in m/s: it is p/m = gamma*v, not p/(m*c).
electrons = Species.electrons(n=n, density=density, x=x0, u=u0)
sim = Simulation(domain, (electrons,), Solver(relativistic=True),
                 external_B=B0_on_grid,
                 radiation=Radiation(species="electrons", substeps=1))
out = sim.run(steps)
```

`species=None` selects all species; a name or tuple of names selects a subset.
`radiation=None` leaves the Lorentz/collision model unchanged. The model and the
species selection are static JAX configuration. Physical fields, initial momentum,
charge, mass, and prescribed-field parameters remain differentiable leaves.

## Force and units

The simulation carries momentum per unit physical mass,
$\boldsymbol u=\gamma\boldsymbol v$, with
$\gamma=\sqrt{1+u^2/c^2}$. Let

```{math}
\tau=\frac{q^2}{6\pi\epsilon_0 m c^3},\qquad
\boldsymbol F=\boldsymbol E+\boldsymbol v\times\boldsymbol B,\qquad
Q=F^2-\frac{(\boldsymbol v\cdot\boldsymbol E)^2}{c^2}.
```

Here $q,m$ are one physical particle's charge and mass. The implemented acceleration
and physical emitted power are

```{math}
\boldsymbol a_{\rm RR}
=\tau\left(\frac qm\right)^2
\left[\boldsymbol F\times\boldsymbol B
 +\frac{(\boldsymbol v\cdot\boldsymbol E)\boldsymbol E}{c^2}
 -\frac{\gamma^2 Q}{c^2}\boldsymbol v\right],
\qquad
P_{\rm rad}=\tau\frac{q^2}{m}\gamma^2 Q.
```

The momentum equation is $d\boldsymbol u/dt=(q/m)\boldsymbol F+
\boldsymbol a_{\rm RR}$. The kernel returns $\boldsymbol a_{\rm RR}$ in m/s² and
$P_{\rm rad}$ in W. The planar particle weight $w$ is in m⁻² and enters only the
energy sum $\sum_p w_p P_p\,dt$, in J/m². Using weighted macroparticle charge or mass
inside this force would change the cooling rate when the particle count changes.

The displayed $Q$ suffers cancellation near a force-free trajectory and at high
Lorentz factor. The implementation instead defines

```{math}
\boldsymbol\eta=\boldsymbol u/c,\quad
\boldsymbol\beta=\boldsymbol\eta/\gamma,\quad
\boldsymbol f=\boldsymbol E/c+\boldsymbol\beta\times\boldsymbol B,\quad
\boldsymbol h=\boldsymbol\eta\times\boldsymbol f,\quad
S=f^2+h^2=\gamma^2 Q/c^2.
```

It evaluates the algebraically equivalent form

```{math}
\boldsymbol a_{\rm RR}=\tau(q/m)^2c\left[
\boldsymbol f\times\boldsymbol B
-(\boldsymbol\beta\cdot\boldsymbol f)(\boldsymbol\beta\times\boldsymbol B)
+(\boldsymbol\beta\times\boldsymbol f)\times\boldsymbol f
-h^2\boldsymbol\beta\right],
\qquad P_{\rm rad}=\tau(q/m)^2mc^2 S.
```

This uses a sum of squares, no clipped invariant, and no division by $|u|$. It keeps
finite derivatives at zero momentum and cancels the reduced force for parallel
$\boldsymbol E,\boldsymbol v$ with $\boldsymbol B=0$. In that last case the Larmor
power is positive. Mechanical reaction work and emission are therefore **separate
quantities**; defining one as minus the other would fail this basic control.

## Integration and the energy ledger

The explicit mover composes a midpoint Runge–Kutta radiation half-step, a relativistic
Boris step, and another radiation half-step. `Radiation.substeps` subdivides only
the two radiation half-steps, with the gathered fields held fixed; there is still
one original Boris Lorentz kick. It refines the local cooling map, but **does not
refine the Lorentz rotation, particle positions, deposited current, or sampled RF
phase**. Resolve those by decreasing `Domain.time_step`.

The implicit mover includes the radiation force at the midpoint momentum in the
particle/field Picard iteration. Its particle substep count is
`Solver.substeps * Radiation.substeps`. A radiation update after a converged
collisionless solve would solve a different discrete problem. The Picard residual,
iteration count, and time-step convergence remain part of the accuracy check.

Radiation or a time-dependent prescribed-field callback currently requires periodic
particle boundaries with the implicit solver. That combination is rejected for
implicit reflecting, absorbing, or thermal walls: the current, work, and emission
would all need a consistent split at the event. The explicit solver supports these
wall paths. A source entry window must also end before any newly loaded trajectory
reaches a wall; an unresolved crossing during that window invalidates the output.
Call `out.validate()` and shorten `Source.every` or the time step if it reports this
condition. The guard prevents work or radiation from being assigned to an already
collected particle.

Mechanical work uses the discrete kinetic-energy gradient

```{math}
\overline{\boldsymbol v}=
\frac{\boldsymbol u_0+\boldsymbol u_1}{\gamma_0+\gamma_1},\qquad
K_1-K_0=m\,\overline{\boldsymbol v}\cdot(\boldsymbol u_1-\boldsymbol u_0).
```

The emitted power is integrated at the radiation stage separately. `State.work`
contains per-species arrays; `Output.work` adds the stored-time axis. All entries are
in J/m²:

| field | meaning |
|---|---|
| `kinetic` | current particle kinetic energy, from carried momentum |
| `prescribed_work` | accumulated signed work on particles by static and time-dependent external electric fields |
| `rr_work` | accumulated signed mechanical work of the radiation force |
| `radiated_energy` | accumulated nonnegative physical emission |
| `collision_work` | accumulated signed kinetic-energy exchange in collisions, per species |

These arrays remain available when particle histories are disabled. With histories
enabled, use `out.u` for relativistic energies and momenta; reconstructing momentum
from a velocity that has rounded to $c$ loses information. `out.v` remains the velocity.

In a pure constant magnetic field, reaction work is minus radiated energy, and
$K+U_{\rm rad}$ gives a direct convergence check. General fields also exchange
near-field energy, and the reduced model omits the derivative force: an arbitrary
$K+U_{\rm rad}$ residual is not by itself a valid conservation test. Account for the
self-consistent field energy, prescribed work, collisions, boundaries, and sources
when they are present. A prescribed-particle work ledger is not a wave-source power
ledger.

## Exact magnetic cooling and derivatives

For uniform $\boldsymbol B=B\hat{\boldsymbol x}$, put
$a_B=\tau(qB/m)^2$. The radiation drift obeys

```{math}
\dot\eta_\parallel=-a_B\eta_\parallel\eta_\perp^2/\gamma,\qquad
\dot{\boldsymbol\eta}_\perp=-a_B(1+\eta_\perp^2)
\boldsymbol\eta_\perp/\gamma,\qquad
\dot\gamma=-a_B\eta_\perp^2.
```

The Lorentz force rotates the perpendicular vector without changing its magnitude.
The constant parallel velocity defines
$\Gamma_\parallel=[1-(v_{\parallel,0}/c)^2]^{-1/2}$ and
$g_0=\gamma_0/\Gamma_\parallel$. Then

```{math}
z(t)=\frac{g_0-1}{g_0+1}\exp(-2a_B t/\Gamma_\parallel),\qquad
\gamma(t)=\Gamma_\parallel\frac{1+z}{1-z}.
```

For a perpendicular orbit, $\Gamma_\parallel=1$ and
$\partial\gamma(T)/\partial B=-2a_B T[\gamma(T)^2-1]/B$ when the initial momentum
and observation time are fixed. This is an analytic derivative, independent of the
AD implementation.

Run `examples/2_intermediate/electron_cooling.py` for one electron and a beam with
energy and pitch spread. It uses physical electron parameters, $B=1000$ T,
$\gamma_0\simeq1000$ and a 1 ns window, with negligible collective fields. The
strong field is a chosen benchmark parameter; the radiation coefficient is unscaled.
The script compares the energy and its derivative to the exact solution and reports
gyrophase error separately. It does not claim collective-plasma cooling.

The kernel tests also cover zero fields, force-free motion, parallel E, magnetic
pitch damping, marker splitting, float32/high-$\gamma$ cancellation, and derivatives
at zero momentum. Refine a full orbit as well as the radiation substep: cooling can
converge well before gyrophase does.

## Prescribed RF and a finite-time control example

Run `examples/3_advanced/runaway_rf.py` for a short control demonstration through the
actual `Simulation` time loop. The chosen setup is a dilute electron seed at
$\gamma_0=40$, $B_0=2$ T, with a counterpropagating mode at 476 MHz and
$k_\parallel=-20$ rad/m. These are demonstration inputs, not a reproduction of a
particular experiment. Resonant RF pitch scattering of relativistic electrons is
established experimentally; see
[Choudhury et al.](https://doi.org/10.1103/jz8f-ck22).

The callback uses a fixed periodic wave number and the vector potential

```{math}
A_y=\frac{E_*}{\omega}g(t)\sin(kx-\omega t),\qquad
A_z=\frac{E_*}{\omega}g(t)\cos(kx-\omega t).
```

It returns $\boldsymbol E_{\rm RF}=-\partial_t\boldsymbol A$ and
$\boldsymbol B_{\rm RF}=\nabla\times\boldsymbol A$, including the envelope derivative.
A time-dependent multiplier on the monochromatic relation
$\boldsymbol B=\boldsymbol k\times\boldsymbol E/\omega$ would miss that term.
The envelope and its first derivative vanish before the final RF-off interval.
No quasilinear diffusion is applied on top of this resolved mode.

Four bounded coefficients shape an 8 ns pulse, followed by 2 ns with RF off. The
smooth finite-time response is
$R(T)=\langle u_\perp^2/u^2\rangle$. SLSQP uses JAX derivatives to minimize

```{math}
J=\frac{1}{E_*^2 T_c}\int_0^{T_c}\langle E_{\rm RF}^2\rangle_x\,dt
=\frac{1}{T_c}\int_0^{T_c}\left[g^2+(\dot g/\omega)^2\right]dt,
```

subject to a target rms $\sin\theta$ and a fixed 0.5% design margin. The script checks
a directional derivative against central differences, then repeats the optimized
control with half the time step and twice the gyro-phase samples. Assertions enforce
the derivative check, the target in both runs, a declared refinement tolerance, and
a reduction in field exposure. This verifies a best-found control in a fixed finite parameterization; it is not a global optimum
or a long-time response derivative.

The dimensional field exposure is $J E_*^2 T_c$, in V² s/m². **It is not input energy.**
The script separately prints prescribed particle work, mechanical reaction work,
physical emission, and kinetic-energy change, in J/m², together with the instantaneous
magnetic cooling time after the RF pulse. The physical radiation loss during this
short window is tiny. A vacuum intensity conversion is not a power calibration for a
supplied plasma mode, and signed negative particle work would mean extraction, not a
minimum positive input-power cost.

A sustained runaway-control calculation additionally needs a specified accelerating
field/current source, a thermal bulk or explicit reservoir, validated relativistic
collisions, and primary/avalanche sources when they matter. A long-time reduced RF
response must be checked against resolved trajectories, including coherent trapping
and energy transfer; phase diffusion at fixed energy cannot simply be assumed.
Minimum **source power** requires an explicit source/Poynting balance, bulk absorption,
returned and retained wave energy, and suppression throughout a declared RF-off hold.
Those are subsequent physics and validation gates, not results of this example.

## Laser–beam cooling

`examples/3_advanced/beam_laser.py` follows a dilute beam with energy, pitch and
gyrophase spread through a counterpropagating 800 nm pulse. Its compact potential
is $A_y=mc\,a(\phi)/e$, with
$a=a_0\cos^2[\phi/(2N)]\sin\phi$ for $|\phi|<\pi N$ and zero outside.
For integer $N\geq2$, the potential and its first two derivatives vanish at the
edges. The callback evaluates $E_y=-mc\omega a'/e$ and $B_z=E_y/c$.

Writing $s=\gamma-\eta_x$ and $\rho=\tau\omega$, the reduced-LL phase equations are

```{math}
s'=-\rho(a')^2s^2,\qquad (\eta_y/s)'=a'/s,\qquad \eta_z/s=\text{constant}.
```

The odd potential gives $\int a(a')^2d\phi=0$. With
$I=\pi a_0^2(3N+1/N)/8$ and $h=1+\rho s_0 I$, the complete-pulse result is

```{math}
s_f=s_0/h,\qquad \eta_{\perp f}=\eta_{\perp0}/h,\qquad
\gamma_f=\frac{h^2+s_0^2+|\eta_{\perp0}|^2}{2hs_0}.
```

The script checks this endpoint, an independent phase ODE, no-RR canonical and
light-front invariants, work/emission, timestep convergence and an analytic
amplitude derivative. At 8,192 steps its maximum final-gamma error is $1.59\times
10^{-8}$ and the amplitude-derivative error is $1.44\times10^{-6}$ relative;
the mean gamma decreases from 80 to 78.42357. These are classical-model numerical
errors: the bound $\chi<0.0082$ does not bound quantum corrections at that level.
The full LL solution's extra endpoint terms vanish for this special complete
pulse, but its trajectory inside the pulse can differ. See
[Di Piazza's plane-wave solution](https://doi.org/10.1007/s11005-008-0228-9).
The incident fluence equals the initial pulse energy per area; it is distinct
from particle work and from boundary energy crossing this box during the run.

## A prescribed source-energy benchmark

`examples/3_advanced/rf_source_control.py` uses a cold neutral electron–proton
plasma with a four-cycle, 500 MHz drive. Its linear response has an independent
five-frequency solution. The simulation is spatially uniform; finite-wave-number
checks separately verify the analytic Maxwell identities. There is no radiation,
collision or runaway suppression in this source-accounting example.

For the prescribed vector potential, the external current and its supplied power
per area are

```{math}
\boldsymbol J_{\rm ext}=\epsilon_0(c^2\nabla\times\boldsymbol B_p-
\partial_t\boldsymbol E_p),\qquad
P_s=-\int\boldsymbol J_{\rm ext}\cdot(\boldsymbol E_s+\boldsymbol E_p)\,dx.
```

The plasma current already drives the self fields; subtracting it again would
double count it. The periodic energy balance includes
$U_\times=\epsilon_0\int(\boldsymbol E_s\cdot\boldsymbol E_p+
c^2\boldsymbol B_s\cdot\boldsymbol B_p)dx$:
$\Delta(K+U_s+U_p+U_\times)=\int P_sdt$. The example checks convergence of this
external diagnostic, including the large error from omitting cross energy.
It is not a new exact discrete production conservation law.

The declared ideal actuator supplies all positive instantaneous work and dumps
returned energy: $Q_+=\int\max(P_s,0)dt$, $Q_-=\int\max(-P_s,0)dt$.
Their difference equals signed source work. A physical port with reactive storage
or recovery needs that model; arbitrary cycle-average bins change the cost and
can change pulse rankings. This example does not estimate wall-plug power.

Four bounded Bernstein coefficients minimize $Q_+$ for a fixed terminal
plasma-energy target. Frozen controls are checked at 512, 1,024 and 2,048 steps.
The best-found pulse lowers supplied energy by 2.18% relative to flat amplitude;
minimizing field exposure instead raises supplied energy by about 0.98%.
Peak field, current and power are reported but have no independent constraint.
This is a finite control-family result, not a global minimum or a runaway result.

Source-power zero crossings matter for derivatives. The script integrates the
positive area of the linearly interpolated endpoint powers before rectification,
and uses the same endpoint trapezoid for signed work. It checks triangle areas,
isolated-zero sensitivity, AD/finite differences and quadratic homogeneity.
AD agreement with the same discrete quadrature alone cannot establish a continuum
gradient. The default displays a figure; `JIC_RF_OUTPUT` optionally names a results
directory outside the checkout. Keep generated reports and plots out of Git.

## Classical validity

For electrons, the local quantum parameter can be evaluated without a large-square
subtraction as $\chi=\sqrt S/B_S$, with $B_S\simeq4.4140\times10^9$ T.
The classical model requires $\chi\ll1$ and a justified neglect of the field-derivative
term. Neither `Radiation.substeps` nor a smaller time step changes that model range.
There is no quantum-correction switch, photon sampling, pair creation, or automatic
transition to another model. Implement and verify those separately after the
classical force, trajectories, energy exchanges, and derivatives are established.
