# A maintained sheath

Where a plasma touches a wall it charges the wall negative, and a thin positively charged
layer forms. This example holds such a sheath up with a {class}`~jaxincell.Source` behind
the left plane and a floating conductor at the right, so that it reaches a steady state and
can be compared with two closed-form kinetic results.

```{figure} ../_static/figures/sheath_source.png
:width: 100%
:alt: The sheath potential against the kinetic reference, the densities against n(phi), and the gradient against finite differences

(a) The potential a reservoir and a floating collector hold, averaged over the second half
of the run (solid), against the root of the current-balance equation (dashed); the circle
is the wall. (b) The measured $n_e$ and $n_i$ against the local relation $n(\phi)$
(dashed): the electrons are pushed out of the sheath and the beam is not, which is the
positive charge that holds the drop up. (c) is the gradient check of
{doc}`sheath_optimization`.
```

## What is measured against what

At {{ source_sheath_cells }} cells, $\omega_{pe}\Delta t = 0.1$,
{{ source_sheath_capacity }} slots and {{ source_sheath_emit }} particles emitted per step
per species, over six ion transits ({{ source_sheath_steps }} steps):

| quantity | measured | reference | deviation |
|---|---|---|---|
| wall potential | {{ source_sheath_phi_wall }} $\pm$ {{ source_sheath_phi_wall_error }} $T_e/e$ | {{ source_sheath_phi_wall_reference }} $T_e/e$ | {{ source_sheath_phi_wall_deviation_percent }} % |
| net collector current, late window | {{ source_sheath_net_current_percent }} % of the ion current | zero at a floating wall | — |
| $n_e(\phi)$, worst cell inside the box | {{ source_sheath_density_error_electrons }} $n_0$ | the kinetic relation | — |
| $n_i(\phi)$, worst cell inside the box | {{ source_sheath_density_error_ions }} $n_0$ | the kinetic relation | — |

## The two closed forms

Electrons conserve $\tfrac12 v^2 - e\phi/m_e$, so only those launched faster than
$\sqrt{-2e\phi_w/m_e}$ reach the wall. Equal particle currents at a floating collector then
fix the wall potential as the root of

$$\frac{\exp\phi_w}{1+\operatorname{erf}\sqrt{-\phi_w}} = \sqrt{\pi/2}\,v_0,$$

in $T_e/e$, with $v_0$ the beam speed in electron spreads. The same conservation law gives

$$n_e = \tfrac12 n_{e0}e^{\phi}\left[1+\operatorname{erf}\sqrt{\phi-\phi_w}\right],
\qquad n_i = \frac{v_0}{\sqrt{v_0^2 - 2\phi\,m_e/m_i}},$$

a **local** relation: it says what the densities are where the potential takes a value,
without knowing where in the box that is, so the whole profile can be tested point by
point. {mod}`jaxincell.sheath` has both, in plain NumPy, sharing nothing with the deposit,
the gather or the field solver.

## The setup

| | |
|---|---|
| $m_i/m_e$ | {{ source_sheath_mass_ratio }} |
| beam speed $v_0$ | {{ source_sheath_beam_speed }} (Mach {{ source_sheath_mach }}) |
| box | {{ source_sheath_box_debye }} $\lambda_D$ |
| $n_{e0}$ | {{ source_sheath_amplitude }} |
| pool at the end | {{ source_sheath_live_electrons }} electrons, {{ source_sheath_live_ions }} ions of {{ source_sheath_capacity }} slots |

The parameters are those of the sheath benchmark of the Vlasov code kobra
{cite}`konewko2026`. Their printed wall potential, $+0.739$, does not solve their printed
equation; the root of that equation at $v_0$ = {{ source_sheath_beam_speed }} is
{{ source_sheath_phi_wall_reference }}, which is what is used here, and
$n_{e0}$ = {{ source_sheath_amplitude }} follows from it.

`Output.wall.overflow` stays at zero. Refining the time step or lengthening the box without
raising the capacity is the one way to get this wrong: both make particles live longer, the
pool fills, and the source starts overwriting live particles. The overflow diagnostic says
so.

## Where the relation and the measurement differ

* Almost all of the electrons' two per cent sits on the monotonic fall from the presheath
  maximum to the collector, not on the maximum itself; the example prints both numbers.
* The interior floats a little above the source plane — 63 of the 120 centres, the highest
  by $+0.0137\,T_e/e$ — and across that hump the relation predicts a Boltzmann rise of 1.4
  per cent which the measurement does not show.
* That hump is **not** the disagreement. The relation counts every orbit energy
  conservation allows, which above the source plane includes orbits with $\epsilon < \phi$:
  bound to the hump and fed by neither the source nor the wall. Emptying them is a
  materially different prediction — 0.77 $n_0$ against 1.08 at the hump of the smaller
  preset the test uses — and a histogram of the electrons finds those orbits about nine
  tenths full, so the relation's assumption is the right one.
* `tests/test_physics.py` holds that, and the other thing the relation is built on: fewer
  than one per cent of the electrons anywhere move back faster than
  $\sqrt{\phi - \phi_w}$, because a faster one had the energy to reach the collector and
  was collected. Deep in the sheath, where there are no bound orbits at all, the measured
  density and the relation agree to better than a per cent.

## What the flow does not do

The ion flow never crosses $c_s$, and {func}`~jaxincell.bohm_edge` reports that there is no
crossing rather than returning the first bin: at Mach {{ source_sheath_mach }} the beam
enters far above the Bohm speed and there is no sheath edge to find. A sheath edge is a
measurement, and a measurement that fails is information.

## Convergence

Changing one thing at a time from that baseline, with the pool scaled where a change makes
particles live longer:

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

Everything lands within two per cent of the closed form except the twenty-Debye box, whose
own standard error is four.

* **The cell size is the one knob that moves it systematically, and it moves it towards the
  reference**: 1.73 per cent at $\Delta x = 0.167\lambda_D$, 0.63 at 0.083 and 0.01 at
  0.042, halving with the cell each time. Extrapolating that first-order trend to
  $\Delta x \to 0$ gives $-0.795$, the reference to within the scatter between seeds, so
  what is left at the baseline is a discretisation error and not a disagreement.
* **Not statistical**: four times the particles and a quarter of them give 0.79 and 0.50
  per cent.
* **Not the duration**: three transits and twelve give 1.31 and 1.49.
* **The time step is unresolved**: three seeds spread 0.002 $T_e/e$, about 0.3 percentage
  points, against a single run's standard error of 0.007, so 1.2 points over a factor of
  four is at the edge of what this many realisations can resolve rather than a trend.

## How to run

```bash
python examples/1_basic/sheath_unmagnetized.py            # about two minutes on a laptop
python examples/1_basic/sheath_unmagnetized.py --quick    # about ten seconds, noisier
jaxincell inputs/sheath_unmagnetized.toml
```

Everything worth changing is at the top of the file: the mass ratio, the beam speed, the
box in Debye lengths, the cell count, the time step, how many ion transits to run for, and
the pool size. {doc}`sheath_magnetized` is the same setup with a magnetic field oblique to
the wall, and {doc}`sheath_optimization` differentiates it.

The figure and the numbers come from `docs/scripts/fig_sheath_source.py`, which runs this
setup, and the convergence table from `docs/scripts/sheath_convergence.py`.
