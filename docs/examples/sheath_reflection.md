# A wall that reflects electrons

`examples/2_intermediate/sheath_reflection.py`

Where a plasma touches a wall it does not stay neutral. Electrons are $\sqrt{m_i/m_e}$
times faster than ions, so they reach the wall first and charge it negative, and the
wall's field then holds back all but as many electrons as ions arrive. What forms is the
structure at the edge of every bounded plasma: a quasi-neutral plasma, a pre-sheath that
accelerates the ions, and a thin positively charged sheath against the wall.

The box is that edge. Its right wall is a floating conductor that collects whatever
reaches it. Its left wall is thermal: what reaches it comes back with a fresh velocity
from a Maxwellian at the starting temperature, as it would from the plasma behind, which
is the source boundary of the classic source-collector simulations {cite}`schwager1990`.
The run is repeated for three conductors: one that collects every electron, one that
returns half of each electron, and one that returns the slow electrons.

A thermal wall keeps the electrons that reach the conductor Maxwellian, which is what the
Hobbs-Wesson formula needs, but it returns only what reaches it and so cannot replace
what the conductor takes: the plasma drains slowly, and the run is stopped after about
one ion transit for that reason. {doc}`sheath_unmagnetized` puts a reservoir there
instead, which holds the plasma up indefinitely and reaches a steady state that can be
compared with a closed-form wall potential rather than with a formula for the drop
across the sheath alone.

```{figure} ../_static/figures/sheath.png
:width: 100%
:alt: Potential above the wall for three reflection laws, ion flow and charge density near the wall, and the sheath drop against Hobbs and Wesson

(a) The potential above the conductor, averaged over the second half of the run, for
the three walls; the dots mark where the ions reach the Bohm speed. (b) The ion flow and
the charge density in front of the wall that collects everything: the positive layer
builds where the ions reach $c_s$. (c) The drop from that point to the wall, against
Hobbs and Wesson.
```

## What is being tested

**The Bohm criterion.** Ions must enter the sheath at no less than $c_s = \sqrt{T_e/m_i}$
{cite}`bohm1949sheath,riemann1991`, and the pre-sheath field accelerates them to it. They
get there {{ sheath_edge_debye }} Debye lengths from the wall, which is where the
positive layer begins, and where the drop below is measured from.

**The sheath drop.** Equating the ion flux at the Bohm speed with the electron flux that
clears the barrier, of which the wall keeps the fraction $1 - R_{\rm eff}$, gives
{cite}`hobbs1967`

```{math}
\frac{e\,\Delta\phi}{T_e} = \frac12\ln\frac{m_i}{2\pi m_e} + \ln(1 - R_{\rm eff}),
```

{{ sheath_drop_theory }} for a wall that keeps everything at $m_i/m_e = $
{{ sheath_mass_ratio }}, and {{ sheath_drop_theory_reflecting }} for $R_{\rm eff} = 1/2$.
The runs give {{ sheath_drop_absorbing }}, {{ sheath_drop_half }} and
{{ sheath_drop_slow }}, all within {{ sheath_drop_deviation_percent }} per cent.

That remainder is not the grid: with the cells halved twice, to an eighth of a Debye
length, the drop stays where it is. Most of it is where the edge is put. At the Bohm point
the potential still falls by about 0.1 $T_e/e$ per Debye length, so the edge is taken where
the ion flow crosses $c_s$, interpolated between the bins on either side of the crossing,
not at the centre of the first bin to reach it, which moves the drop by several per cent
from one bin width to the next.

**The same flux average gives the same sheath.** The two reflecting walls return quite
different electrons, one half of every electron and the other the slow ones through
$R(v) = e^{-v^2/2\sigma^2}$, but both have $R_{\rm eff} = 1/2$, the flux average of
{doc}`wall_reflection`, and they hold the same sheath. For the velocity-dependent law the
average is taken over the electrons that clear the barrier, at the wall. In a
collisionless sheath they still form a half-Maxwellian at the plasma temperature, so the
flux average of a Maxwellian is exactly the one that counts
({doc}`../numerics/boundaries`).

## Why a thermal wall

The formula assumes that the electrons arriving at the wall are Maxwellian. Between two
absorbing walls they soon are not: nothing sustains that plasma, the walls take the
fast electrons first, and within a few transits the tail the flux balance is derived
from is gone. A smaller barrier then suffices, the drop comes out low, and reflection no
longer shifts it by $\ln 2$, since the electrons left to reflect are the slow ones. The
thermal wall is the simplest remedy. Every electron the sheath turns back is redrawn at
the other end from the Maxwellian, so the distribution arriving at the conductor stays
complete. The plasma still drains, and {{ sheath_ions_left_percent }} per cent of the ions
remain at the end, but its temperature holds.

Two absorbing walls are still the right model for a plasma between two electrodes, which
are then short-circuited conductors at one potential ({doc}`../numerics/boundaries`).

## Running it

```bash
python examples/2_intermediate/sheath_reflection.py
```

About a minute for the three runs, each with 30 000 particles per species, on
{{ sheath_cells }} cells over
{{ sheath_box_debye }} Debye lengths for {{ sheath_steps }} steps, roughly one ion
transit. The mass ratio is reduced to {{ sheath_mass_ratio }} for exactly that reason:
the ion transit sets the cost, and it grows as $\sqrt{m_i/m_e}$.

The figure and the numbers on this page come from `docs/scripts/fig_sheath.py`, which runs
the same three walls with 40 000 particles per species, {{ sheath_particles }} in all, to
lower the noise in the averaged potential.

The run is electrostatic, so the time step follows $\omega_{pe}\Delta t = 0.2$ rather
than the light-wave limit. That is safe only while nothing excites the transverse
fields; see {doc}`../numerics/stability`.

## Things to try

* `reflection=(0.0, 0.75)`: the drop falls by $\ln 4$, and it vanishes altogether as
  $R_{\rm eff}$ approaches $1 - \sqrt{2\pi m_e/m_i}$.
* `Domain(..., restitution=(1.0, 0.5))`: the reflected electrons come back slower, but
  the sheath does not change, because the wall keeps as many as before.
* Change the mass ratio and check that the drop follows $\tfrac12\ln(m_i/m_e)$.
