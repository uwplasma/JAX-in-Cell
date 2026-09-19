# A maintained sheath

`examples/1_basic/sheath_unmagnetized.py` — the figure and the numbers on this page come
from `docs/scripts/fig_sheath_source.py`, which runs this setup, and the convergence table
from `docs/scripts/sheath_convergence.py`.

Where a plasma touches a wall it does not stay neutral: electrons are
$\sqrt{m_i/m_e}$ times faster than ions, reach the wall first and charge it negative,
and its field then holds back all but as many electrons as ions arrive. What forms is a
thin positively charged layer, the sheath, and the potential across it is what the wall
sees.

To compare that with theory the plasma has to be held up. This example puts a
{class}`~jaxincell.Source` behind the left plane — a reservoir sending in electrons from
a Maxwellian at rest and ions as a cold beam, at fluxes that do not depend on what
leaves — and a floating conductor at the right. The sheath then reaches a steady state
instead of draining, and two closed-form results follow.

## What is checked

Electrons conserve $\tfrac12 v^2 - e\phi/m_e$, so only those launched faster than
$\sqrt{-2e\phi_w/m_e}$ reach the wall; equal particle currents at a floating collector
then fix the wall potential as the root of

$$\frac{\exp\phi_w}{1+\operatorname{erf}\sqrt{-\phi_w}} = \sqrt{\pi/2}\,v_0,$$

in $T_e/e$, with $v_0$ the beam speed in electron spreads. The same conservation law
gives the densities as functions of the potential alone,

$$n_e = \tfrac12 n_{e0}e^{\phi}\left[1+\operatorname{erf}\sqrt{\phi-\phi_w}\right],
\qquad n_i = \frac{v_0}{\sqrt{v_0^2 - 2\phi\,m_e/m_i}},$$

a **local** relation: it says what the densities are where the potential takes a value,
without knowing where in the box that is, so the whole profile can be tested against it
point by point. {mod}`jaxincell.sheath` has both, in plain NumPy, sharing nothing with
the deposit, the gather or the field solver.

The parameters are those of the sheath benchmark of the Vlasov code kobra
{cite}`konewko2026`: $m_i/m_e = 1836$, $v_0 = 0.2$ (Mach 8.6), a box ten Debye lengths
across. Their printed wall potential, $+0.739$, does not solve their printed equation;
the root of that equation at $v_0=0.2$ is $-0.79926$, which is what is used here, and
$n_{e0} = 1.11490$ follows from it.

```{figure} ../_static/figures/sheath_source.png
:width: 100%
:alt: The sheath potential against the kinetic reference, the densities against n(phi), and the gradient against finite differences

(a) The potential a reservoir and a floating collector hold, averaged over the second
half of the run, against the root of the current-balance equation. (b) The measured
densities against the local relation $n(\phi)$: the electrons are pushed out of the
sheath and the beam is not, which is the positive charge that holds the drop up.
(c) is explained in {doc}`sheath_optimization`.
```

## What comes out

At {{ source_sheath_cells }} cells, $\omega_{pe}\Delta t = 0.1$,
{{ source_sheath_capacity }} slots and {{ source_sheath_emit }} particles emitted per
step per species, over six ion transits ({{ source_sheath_steps }} steps):

| quantity | measured | reference |
|---|---|---|
| wall potential | {{ source_sheath_phi_wall }} $\pm$ {{ source_sheath_phi_wall_error }} $T_e/e$ | {{ source_sheath_phi_wall_reference }} $T_e/e$ |
| net collector current, late window | {{ source_sheath_net_current_percent }} % of the ion current | zero at a floating wall |
| $n_e(\phi)$, worst cell inside the box | {{ source_sheath_density_error_electrons }} $n_0$ | the kinetic relation |
| $n_i(\phi)$, worst cell inside the box | {{ source_sheath_density_error_ions }} $n_0$ | the kinetic relation |

The pool ends with {{ source_sheath_live_electrons }} electrons and
{{ source_sheath_live_ions }} ions live of {{ source_sheath_capacity }} slots, and
`Output.wall.overflow` stays at zero. Refining the time step or lengthening the box
without raising the capacity is the one way to get this wrong: both make particles live
longer, the pool fills, and the source starts overwriting live particles. The overflow
diagnostic says so.

## Where the relation and the measurement differ

The ions match the relation to {{ source_sheath_density_error_ions }} $n_0$ and the
electrons to {{ source_sheath_density_error_electrons }}. Almost all of that two per cent
sits on the monotonic fall from the presheath maximum to the collector rather than on the
maximum itself, and the example prints both numbers so that the distinction is visible.

The interior does float a little above the source plane — 63 of the 120 centres, the
highest by $+0.0137\,T_e/e$ — and across that hump the relation predicts a Boltzmann rise
of 1.4 per cent which the measurement does not show. That is not what the disagreement is,
though, and the electrons themselves say so. The relation counts every orbit energy
conservation allows, which above the source plane includes orbits with $\epsilon < \phi$:
bound to the hump, connected to neither the source nor the wall, and unpopulated in a
collisionless plasma fed only from the plane. Emptying them is a materially different
prediction — at the hump of the smaller preset the test uses, 0.77 $n_0$ against 1.08 —
and the measurement is neither: a histogram of the electrons finds those orbits about nine
tenths full, so the relation's assumption is the right one.

`tests/test_physics.py` holds that, and the other thing the relation is built on: fewer
than one per cent of the electrons anywhere move back faster than
$\sqrt{\phi - \phi_w}$, because a faster one had the energy to reach the collector and
was collected. Deep in the sheath, where there are no bound orbits at all, the measured
density and the relation agree to better than a per cent.

## What the flow does not do

The ion flow never crosses $c_s$, and {func}`~jaxincell.bohm_edge` reports that there is
no crossing rather than returning the first bin: at Mach 8.6 the beam enters far above
the Bohm speed and there is no sheath edge to find. A sheath edge is a measurement, and
a measurement that fails is information.

## Convergence

Changing one thing at a time from that baseline, with the pool scaled where a change
makes particles live longer:

| variation | $\phi_w$ ($T_e/e$) | s.e. | % from the reference |
|---|---|---|---|
| baseline | $-0.8043$ | 0.0068 | 0.63 |
| $\Delta x/\lambda_D = 0.167$ (60 cells) | $-0.8131$ | 0.0068 | 1.73 |
| $\Delta x/\lambda_D = 0.042$ (240 cells) | $-0.7994$ | 0.0067 | 0.01 |
| $\omega_{pe}\Delta t = 0.05$ | $-0.8074$ | 0.0049 | 1.01 |
| $\omega_{pe}\Delta t = 0.025$ | $-0.8141$ | 0.0064 | 1.85 |
| a quarter of the particles | $-0.8033$ | 0.0111 | 0.50 |
| four times the particles | $-0.8055$ | 0.0030 | 0.79 |
| 3 ion transits | $-0.8097$ | 0.0077 | 1.31 |
| 12 ion transits | $-0.8112$ | 0.0057 | 1.49 |
| box 20 $\lambda_D$ | $-0.8305$ | 0.0327 | 3.91 |
| seed 1 | $-0.8085$ | 0.0065 | 1.15 |
| seed 2 | $-0.8059$ | 0.0053 | 0.83 |

Everything lands within two per cent of the closed form except the twenty-Debye box,
whose own standard error is four. **The cell size is the one knob that moves it
systematically, and it moves it towards the reference**: 1.73 per cent at
$\Delta x = 0.167\lambda_D$, 0.63 at 0.083 and 0.01 at 0.042, halving with the cell each
time. Extrapolating that first-order trend to $\Delta x \to 0$ gives $-0.795$, which is
the reference to within the scatter between seeds, so what is left at the baseline is a
discretisation error and not a disagreement with the theory.

The other knobs say what it is not. Four times the particles and a quarter of them give
0.79 and 0.50 per cent, so it is not statistical; three transits and twelve give 1.31 and
1.49, so it is not the duration. Three seeds spread 0.002 $T_e/e$, about 0.3 percentage
points, against a single run's standard error of 0.007 — which is also why the time step,
worth 1.2 points over a factor of four and in the direction away from the reference, is at
the edge of what this many realisations can resolve rather than a measured trend.

## Running it

```bash
python examples/1_basic/sheath_unmagnetized.py            # about two minutes on a laptop
python examples/1_basic/sheath_unmagnetized.py --quick    # about ten seconds, noisier
```

Everything worth changing is at the top of the file: the mass ratio, the beam speed, the
box in Debye lengths, the cell count, the time step, how many ion transits to run for,
and the pool size. {doc}`sheath_magnetized` is the same setup with a magnetic field
oblique to the wall, and {doc}`sheath_optimization` differentiates it.
