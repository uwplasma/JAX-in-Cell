# A maintained sheath

`examples/1_basic/sheath_unmagnetized.py` — the figure and the numbers on this page come
from `docs/scripts/fig_sheath_source.py`, which runs this setup.

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

The ion flow never crosses $c_s$, and {func}`~jaxincell.bohm_edge` reports that there is
no crossing rather than returning the first bin: at Mach 8.6 the beam enters far above
the Bohm speed and there is no sheath edge to find. A sheath edge is a measurement, and
a measurement that fails is information.

## Convergence

Changing one thing at a time from that baseline, with the pool scaled where a change
makes particles live longer:

| variation | $\phi_w$ ($T_e/e$) | s.e. | % from the reference |
|---|---|---|---|
| baseline | $-0.7832$ | 0.0069 | 2.01 |
| $\Delta x/\lambda_D = 0.167$ (60 cells) | $-0.7993$ | 0.0074 | 0.00 |
| $\Delta x/\lambda_D = 0.042$ (240 cells) | $-0.7751$ | 0.0068 | 3.03 |
| $\omega_{pe}\Delta t = 0.05$ | $-0.8027$ | 0.0053 | 0.43 |
| $\omega_{pe}\Delta t = 0.025$ | $-0.8119$ | 0.0064 | 1.58 |
| a quarter of the particles | $-0.7850$ | 0.0111 | 1.79 |
| four times the particles | $-0.7857$ | 0.0031 | 1.69 |
| 3 ion transits | $-0.7913$ | 0.0081 | 0.99 |
| 12 ion transits | $-0.7907$ | 0.0060 | 1.07 |
| box 20 $\lambda_D$ | $-0.8050$ | 0.0378 | 0.72 |
| seed 1 | $-0.7889$ | 0.0065 | 1.30 |
| seed 2 | $-0.7854$ | 0.0053 | 1.73 |

Everything lands within three per cent, most of it within two, and no single knob drives
what is left. Four times the particles do not move it, so it is not statistical; three
transits and twelve give the same answer, so it is not the duration. The two that do move
it are the time step, worth 1.6 percentage points between 0.1 and 0.05, and the box,
worth 1.3 between ten Debye lengths and twenty — and the box is where the difference
should be, since the reference is the semi-infinite problem in which the plasma is exactly
neutral and field-free at the source plane, and ten Debye lengths is not that. The scatter
between seeds is about 0.4 percentage points and the standard error of a single run about
0.7, so the residual is a real systematic of about one and a half per cent and not noise.

## Running it

```bash
python examples/1_basic/sheath_unmagnetized.py            # about four minutes on a laptop
python examples/1_basic/sheath_unmagnetized.py --quick    # about forty seconds, noisier
```

Everything worth changing is at the top of the file: the mass ratio, the beam speed, the
box in Debye lengths, the cell count, the time step, how many ion transits to run for,
and the pool size. {doc}`sheath_magnetized` is the same setup with a magnetic field
oblique to the wall, and {doc}`sheath_optimization` differentiates it.
