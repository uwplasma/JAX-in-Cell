# A wall that reflects electrons

A plasma edge in a box: a thermal wall on the left standing in for the plasma behind
{cite}`schwager1990`, and a floating conductor on the right. The run is repeated for three
conductors — one that collects every electron, one that returns half of each, and one that
returns the slow ones — and the sheath each holds is compared with Hobbs and Wesson.

```{figure} ../_static/figures/sheath.png
:width: 100%
:alt: Potential above the wall for three reflection laws, ion flow and charge density near the wall, and the sheath drop against Hobbs and Wesson

(a) The potential above the conductor, averaged over the second half of the run, for the
three walls; the circles mark where the ions reach the Bohm speed. (b) The ion flow
$v_i/c_s$ and the charge density $10\rho/en_0$ in front of the wall that collects
everything: the positive layer builds where the ions reach $c_s$. (c) The measured sheath
drop for each wall (bars) against Hobbs and Wesson (dashes).
```

## What is measured against what

| quantity | measured | reference | deviation |
|---|---|---|---|
| sheath drop, absorbing wall | {{ sheath_drop_absorbing }} | {{ sheath_drop_theory }} | within {{ sheath_drop_deviation_percent }} % |
| sheath drop, returns half | {{ sheath_drop_half }} | {{ sheath_drop_theory_reflecting }} | within {{ sheath_drop_deviation_percent }} % |
| sheath drop, returns the slow ones | {{ sheath_drop_slow }} | {{ sheath_drop_theory_reflecting }} | within {{ sheath_drop_deviation_percent }} % |
| $R_{\rm eff}$ of the two reflecting walls | 0.000 and {{ sheath_reff_measured }} | 0.0 and 0.5, what the laws are meant to have | — |
| sheath edge, where $v_i$ crosses $c_s$ | {{ sheath_edge_debye }} $\lambda_D$ from the wall | the Bohm criterion | — |

**The Bohm criterion.** Ions must enter the sheath at no less than $c_s = \sqrt{T_e/m_i}$
{cite}`bohm1949sheath,riemann1991`, and the pre-sheath field accelerates them to it. Where
they get there is where the positive layer begins, and where the drop below is measured
from.

**The sheath drop.** Equating the ion flux at the Bohm speed with the electron flux that
clears the barrier, of which the wall keeps the fraction $1 - R_{\rm eff}$, gives
{cite}`hobbs1967`

```{math}
\frac{e\,\Delta\phi}{T_e} = \frac12\ln\frac{m_i}{2\pi m_e} + \ln(1 - R_{\rm eff}).
```

**The same flux average gives the same sheath.** The two reflecting walls return quite
different electrons — one half of every electron, the other the slow ones through
$R(v) = e^{-v^2/2\sigma^2}$ — but both have $R_{\rm eff} = 1/2$, the flux average of
{doc}`wall_reflection`, and they hold the same sheath. That reflectivity is **measured**,
from the weight the collector kept against the weight that reached it, so what is compared
is what the wall did against what the theory says, and not two things that were both
assumed. For the velocity-dependent law the average is taken over the electrons that clear
the barrier, at the wall; in a collisionless sheath they still form a half-Maxwellian at the
plasma temperature, so the flux average of a Maxwellian is the one that counts
({doc}`../numerics/boundaries`).

## What the remainder is not

* **Not the grid**: with the cells halved twice, to an eighth of a Debye length, the drop
  stays where it is.
* **Mostly where the edge is put.** At the Bohm point the potential still falls by about
  0.1 $T_e/e$ per Debye length, so the edge is taken where the ion flow crosses $c_s$,
  interpolated between the bins on either side, not at the centre of the first bin to reach
  it — which would move the drop by several per cent from one bin width to the next.

## Why a thermal wall, and what it costs

The formula assumes the electrons arriving at the wall are Maxwellian. Between two
absorbing walls they soon are not: the walls take the fast electrons first, and within a few
transits the tail the flux balance is derived from is gone. A thermal wall redraws every
electron the sheath turns back from the Maxwellian, so the distribution arriving at the
conductor stays complete.

It cannot replace what the conductor takes, though, so the plasma drains, and the script
measures the draining rather than mentioning it: over the one ion transit the run lasts it
loses 41.5 per cent of its ions, against 4.8 per cent for the same box, resolution and
collector with a reservoir on the left. {{ sheath_ions_left_percent }} per cent of the ions
remain at the end. The sheath drops above are therefore measured on a plasma that is going
away, but its temperature holds.

{doc}`sheath_unmagnetized` is the other end of that trade: a reservoir holds the plasma up
indefinitely, does not hold the arriving electrons Maxwellian by construction, and is
compared against the closed form for the whole wall potential rather than the drop alone.
Two absorbing walls remain the right model for a plasma between two electrodes, which are
then short-circuited conductors at one potential ({doc}`../numerics/boundaries`).

## The setup

| | |
|---|---|
| $m_i/m_e$ | {{ sheath_mass_ratio }}, reduced because the ion transit sets the cost and grows as $\sqrt{m_i/m_e}$ |
| box | {{ sheath_box_debye }} $\lambda_D$ |
| cells | {{ sheath_cells }} |
| steps | {{ sheath_steps }}, roughly one ion transit |
| particles | {{ sheath_particles }} in the figure runs; 30 000 per species in the example |
| $\omega_{pe}\Delta t$ | 0.2, electrostatic |

The run is electrostatic, so the time step follows $\omega_{pe}\Delta t = 0.2$ rather than
the light-wave limit. That is safe only while nothing excites the transverse fields; see
{doc}`../numerics/stability`.

## How to run

```bash
python examples/2_intermediate/sheath_reflection.py
```

About four minutes for the four runs. It writes a `sheath_reflection/` folder beside it:
the settings, results and versions in `run.json`, the profiles behind the figure in
`profiles.npz`, and the figure.

The figure and the substituted numbers come from `docs/scripts/fig_sheath.py`, which runs
the same three walls at the same preset, so that the page and the example are one
experiment rather than two that look alike.

## Things to try

* `reflection=(0.0, 0.75)`: the drop falls by $\ln 4$, and vanishes altogether as
  $R_{\rm eff}$ approaches $1 - \sqrt{2\pi m_e/m_i}$.
* `Domain(..., restitution=(1.0, 0.5))`: the reflected electrons come back slower, but the
  sheath does not change, because the wall keeps as many as before.
* Change the mass ratio and check that the drop follows $\tfrac12\ln(m_i/m_e)$.
