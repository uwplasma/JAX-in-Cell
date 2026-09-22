# A sheath in an oblique magnetic field

The same reservoir, the same floating collector and the same electrostatic solve as
{doc}`sheath_unmagnetized`, with one line added: a uniform external field
$\mathbf B_0 = B_0(\sin\alpha, 0, \cos\alpha)$, with $\alpha$ the angle to the **wall
plane**, so that normal incidence is $\alpha = 90^\circ$ and grazing incidence approaches
zero.

```{figure} ../_static/figures/sheath_magnetized.png
:width: 100%
:alt: Potential and normal ion flow at three field angles, and the ion incidence distribution at the collector

The example's own output at the full preset, written to `sheath_magnetized/figure.png`
beside `run.json` and `profiles.npz`. Left: the potential $e\phi/T_e$ against distance from
the collector, one curve per field angle. Middle: the normal ion flow $v_{i,x}/c_s$ over
the same distance, with $c_s$ dashed. Right: what the wall is struck by — the ion incidence
from the wall normal, as a fraction of the fluence per degree, one histogram per angle.
```

## What is measured against what

At $m_i/m_e = 400$, $T_i = T_e$, $\rho_s/\lambda_D = 8$, in a box of 60 Debye lengths, over
four sound transits at 240 cells:

| $\alpha$ to the wall | wall potential | ion fluence (m$^{-2}$) | mean impact energy | mean incidence from the normal | fluence ratio to $90^\circ$ | $\sin\alpha$ |
|---|---|---|---|---|---|---|
| $90^\circ$ | $-1.94\,T_e/e$ | $9.67\times10^{13}$ | 4.70 eV | $25^\circ$ | 1 | 1.00 |
| $30^\circ$ | $-2.11\,T_e/e$ | $4.93\times10^{13}$ | 4.83 eV | $39^\circ$ | 0.51 | 0.50 |
| $15^\circ$ | $-2.07\,T_e/e$ | $2.61\times10^{13}$ | 4.71 eV | $52^\circ$ | 0.27 | 0.26 |

* **The fluence is the entrance condition, arriving.** The ratios to normal incidence match
  the $\sin\alpha$ the ions enter with to within a few per cent, so what reaches the wall
  is set by the normal component of the sound speed.
* **The incidence follows the field around** — $25^\circ$, $39^\circ$, $52^\circ$ — while
  the mean impact energy barely moves, because it is the sheath drop that sets it and that
  is much the same at all three angles.
* **This run does not resolve a trend in the wall potential with angle.** The three differ
  by about one realisation scatter — 0.17 and 0.13 $T_e/e$ against the 0.137 that two seeds
  of the same physics give — and three angles of one realisation each cannot resolve more.
  The potentials are measured from the source plane, so they include the presheath and are
  not the drop across the Debye sheath alone.
* **The field-free control comes out as it should.** The $B = 0$ run and the $90^\circ$ one
  differ by at most $0.079\,T_e/e$ over a drop of $1.87$, which is 0.58 of the realisation
  scatter of $0.137$. It is not round-off: an external field array takes a different path
  through the gather than `None` does, so $E_x$ differs in its last bit and a plasma
  between absorbing walls amplifies that. Saying so needs the second seed, which is why the
  script runs it.

## Two layers, not one

Only the component along $x$ is resolved by the grid, so ions still reach the wall along
the normal; between collisions with nothing they follow the field. Where the ion gyro-radius
is large compared with the Debye length the plasma-wall transition has two layers rather
than one: a magnetic presheath a few gyro-radii deep, in which the ions turn from following
the field to crossing it, and inside it the Debye sheath {cite}`chodura1982`. The separation
of scales

$$\frac{\rho_s}{\lambda_D} = \sqrt{\frac{m_i}{m_e}}\,\frac{\omega_{pe}}{\Omega_e},
\qquad \frac{\rho_e}{\lambda_D} = \frac{\omega_{pe}}{\Omega_e}$$

is what the example is really about. Both are printed, with $L/\rho_s$, $\Delta x/\lambda_D$,
$\omega_{pe}\Delta t$ and $\Omega_e\Delta t$, and the script warns when the electron
gyro-phase is under-resolved.

## How the ions enter

The ions here are warm, unlike the cold beam of {doc}`sheath_unmagnetized`, and they enter
**along the field** at the sound speed. That is Chodura's picture: the presheath turns them
towards the wall, so the speed normal to it is $c_s\sin\alpha$ and not $c_s$ — 1.00 $c_s$ at
$90^\circ$, 0.50 at $30^\circ$ and 0.26 at $15^\circ$. The initial population and the
reservoir are drawn from the same distribution.

An ion that enters at $c_s\sin\alpha$ takes longer to cross the box the more the field
grazes, so the pool has to hold what a run emits over that residence. The slots per species
and the particles emitted per step are printed, and `Output.validate()` refuses to report a
run whose source overwrote live particles. At the grazing angle this example ends with most
of its ion pool occupied, which is the constraint that sets the preset.

## Scope

A collisionless electrostatic model with full electron orbits. It is not a comparison with
gyrokinetic magnetic-presheath theory, which assumes an ordering in $\alpha$ and adiabatic
electrons that this does not; at grazing incidence the classical Debye sheath is expected to
weaken and the entrance conditions to change {cite}`geraldini2018`, and nothing here forces
a Bohm crossing or fits two layers to every run. {doc}`grazing_sheath` is the case set up
against that theory.

## How to run

```bash
python examples/2_intermediate/sheath_magnetized.py            # about an hour
python examples/2_intermediate/sheath_magnetized.py --quick    # about two minutes
jaxincell inputs/sheath_magnetized.toml
```
