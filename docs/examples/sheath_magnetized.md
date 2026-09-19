# A sheath in an oblique magnetic field

`examples/2_intermediate/sheath_magnetized.py`

The same reservoir, the same floating collector and the same electrostatic solve as
{doc}`sheath_unmagnetized`, with one line added: a uniform external field

$$\mathbf B_0 = B_0(\sin\alpha, 0, \cos\alpha),$$

with $\alpha$ the angle to the **wall plane**, so that normal incidence is
$\alpha = 90^\circ$ and grazing incidence approaches zero. The angle at which a particle
arrives at the wall is a different quantity, measured from the outward normal, and is
reported separately.

## Two layers, not one

Only the component along $x$ is resolved by the grid, so ions still reach the wall along
the normal; between collisions with nothing they follow the field. Where the ion
gyro-radius is large compared with the Debye length the plasma-wall transition has two
layers rather than one: a magnetic presheath a few gyro-radii deep, in which the ions
turn from following the field to crossing it, and inside it the Debye sheath
{cite}`chodura1982`. Resolving both means a box many gyro-radii long and cells a
fraction of a Debye length wide, so the separation of scales

$$\frac{\rho_s}{\lambda_D} = \sqrt{\frac{m_i}{m_e}}\,\frac{\omega_{pe}}{\Omega_e},
\qquad \frac{\rho_e}{\lambda_D} = \frac{\omega_{pe}}{\Omega_e}$$

is what the example is really about. Both are printed, with $L/\rho_s$,
$\Delta x/\lambda_D$, $\omega_{pe}\Delta t$ and $\Omega_e\Delta t$, and the script warns
when the electron gyro-phase is under-resolved.

## How the ions enter

The ions here are warm, unlike the cold beam of {doc}`sheath_unmagnetized`, and they
enter **along the field** at the sound speed. That is Chodura's picture: the presheath
turns them towards the wall, so the speed they enter with normal to it is
$c_s\sin\alpha$ and not $c_s$ — 1.00 $c_s$ at $90^\circ$, 0.50 at $30^\circ$ and 0.26
at $15^\circ$. The initial population and the reservoir are drawn from the same
distribution; they were not, and the box was filled with one plasma and fed with another.

An ion that enters at $c_s\sin\alpha$ takes longer to cross the box the more the field
grazes, so the pool has to hold what a run emits over that residence: the slots per
species and the particles emitted per step are printed, and `Output.validate()` refuses
to report a run whose source overwrote live particles. At the grazing angle this example
ends with most of its ion pool occupied, which is the constraint that sets the preset.

## What is checked

There is no closed-form wall potential for this problem, so what is checked is what can
be, and measured rather than asserted:

* **normal incidence against a matched $B = 0$ control**, and against a second
  realisation of the same physics. With $\mathbf B_0$ along $x$ the Boris rotation leaves
  $v_x$ alone exactly — $\mathbf v\times\mathbf B$ has no $x$ component when
  $\mathbf B$ has only one — so the two runs begin identical. Over a whole run they do
  not stay so: an external array takes a different path through the gather than `None`
  does, `E_x` differs in its last bit, and a plasma between absorbing walls is chaotic.
  What the difference means is therefore only visible against the scatter between two
  seeds of the same physics, and that is what the script prints;
* **the impact distributions**, which are what a wall actually feels, binned at the
  crossing itself rather than read off a snapshot of which particles happen to be nearby.
  As the field grazes, the mean ion incidence moves away from the normal, which is the
  presheath turning the orbits.

```{figure} ../_static/figures/sheath_magnetized.png
:width: 100%
:alt: Potential and densities at three field angles, the ion impact energy and incidence distributions, and the field-free control

The example's own output at the full preset, written to `sheath_magnetized/figure.png`
beside `run.json` and `profiles.npz`: the potential and densities at the three angles, the
energy and incidence distributions of what strikes the collector, and the normal-incidence
run against its field-free control. Reproduce it with
`python examples/2_intermediate/sheath_magnetized.py`.
```

At $m_i/m_e = 400$, $T_i = T_e$, $\rho_s/\lambda_D = 8$, in a box of 60 Debye lengths,
over four sound transits at 240 cells:

| $\alpha$ to the wall | wall potential | ion fluence (m$^{-2}$) | mean impact energy | mean incidence from the normal |
|---|---|---|---|---|
| $90^\circ$ | $-1.94\,T_e/e$ | $9.67\times10^{13}$ | 4.70 eV | $25^\circ$ |
| $30^\circ$ | $-2.11\,T_e/e$ | $4.93\times10^{13}$ | 4.83 eV | $39^\circ$ |
| $15^\circ$ | $-2.07\,T_e/e$ | $2.61\times10^{13}$ | 4.71 eV | $52^\circ$ |

The potentials are measured from the source plane, so they include the presheath and are
not the drop across the Debye sheath alone, and they differ between the three angles by
about one realisation scatter — 0.17 and 0.13 $T_e/e$ against the 0.137 two seeds of the
same physics give below. **This run does not resolve a trend in the wall potential with
angle**, and three angles of one realisation each cannot; what it does resolve is the
fluence and the incidence, which move by factors rather than by per cent.

**The fluence is the entrance condition,
arriving**: the ratios to normal incidence are 0.51 and 0.27, against the $\sin\alpha$ of
0.50 and 0.26 that the ions enter with, so what reaches the wall is set by the normal
component of the sound speed to within a few per cent. The mean incidence follows the
field around — $25^\circ$, $39^\circ$, $52^\circ$ from the normal — while the mean impact
energy barely moves, because it is the sheath drop that sets it and that is much the same
at all three angles.

The normal-incidence control comes out as it should, and only a measurement of the
scatter can say so: the field-free run and the $90^\circ$ one differ by at most
$0.079\,T_e/e$ over a drop of $1.87$, and two seeds of the same physics differ by
$0.137$. The field-free difference is 0.58 of the realisation scatter. It is not
round-off — an external field array takes a different path through the gather than `None`
does, so $E_x$ differs in its last bit and a plasma between absorbing walls amplifies
that — and saying so needs the second seed, which is why the script runs it.

## Scope

This is a collisionless electrostatic model with full electron orbits. It is not a
comparison with gyrokinetic magnetic-presheath theory, which assumes an ordering in
$\alpha$ and adiabatic electrons that this does not; at grazing incidence the classical
Debye sheath is expected to weaken and the entrance conditions to change
{cite}`geraldini2018`, and nothing here forces a Bohm crossing or fits two layers to
every run.

```bash
python examples/2_intermediate/sheath_magnetized.py            # about an hour
python examples/2_intermediate/sheath_magnetized.py --quick    # about two minutes
```
