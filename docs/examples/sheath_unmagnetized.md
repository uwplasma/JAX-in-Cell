# A maintained sheath

`examples/1_basic/sheath_unmagnetized.py`

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

## What comes out

At 120 cells, $\omega_{pe}\Delta t = 0.1$, 120000 slots and 120 particles emitted per
step per species, over six ion transits:

| quantity | measured | reference |
|---|---|---|
| wall potential | $-0.783 \pm 0.007\,T_e/e$ | $-0.79926\,T_e/e$ |
| net collector current, late window | $+0.01$ % of the ion current | zero at a floating wall |
| $n_e(\phi)$, worst cell inside the box | $0.052\,n_0$ | the kinetic relation |
| $n_i(\phi)$, worst cell inside the box | $0.034\,n_0$ | the kinetic relation |

The ion flow never crosses $c_s$, and {func}`~jaxincell.bohm_edge` reports that there is
no crossing rather than returning the first bin: at Mach 8.6 the beam enters far above
the Bohm speed and there is no sheath edge to find. A sheath edge is a measurement, and
a measurement that fails is information.

The remaining two per cent in the wall potential does not fall with the grid or the time
step; it is the finite box. The reference is the semi-infinite problem, in which the
plasma is exactly neutral and field-free at the source plane, and ten Debye lengths of
box is not that.

## Running it

```bash
python examples/1_basic/sheath_unmagnetized.py            # about four minutes on a laptop
python examples/1_basic/sheath_unmagnetized.py --quick    # about forty seconds, noisier
```

Everything worth changing is at the top of the file: the mass ratio, the beam speed, the
box in Debye lengths, the cell count, the time step, how many ion transits to run for,
and the pool size. {doc}`sheath_magnetized` is the same setup with a magnetic field
oblique to the wall, and {doc}`sheath_optimization` differentiates it.
