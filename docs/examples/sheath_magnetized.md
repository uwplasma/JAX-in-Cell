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

There is no closed-form wall potential for this problem. What is checked is what can be:

* **normal incidence** ($\alpha = 90^\circ$) must leave the motion along the normal
  alone, and does — its potential is the field-free one;
* **the impact distributions**, which are what a wall actually feels. As the field grazes,
  the mean ion incidence moves away from the normal, which is the presheath turning the
  orbits.

At $m_i/m_e = 400$, $T_i = T_e$, $\rho_s/\lambda_D = 8$, in a box of 60 Debye lengths:

| $\alpha$ to the wall | wall potential | mean ion impact energy | mean incidence from the normal |
|---|---|---|---|
| $90^\circ$ | $-3.65\,T_e/e$ | 2.56 eV | $5^\circ$ |
| $30^\circ$ | $-2.92\,T_e/e$ | 2.77 eV | $38^\circ$ |
| $15^\circ$ | $-2.97\,T_e/e$ | 2.67 eV | $56^\circ$ |

(the `--quick` preset; the potentials are measured from the source plane, so they
include the presheath and are not the drop across the Debye sheath alone).

## Scope

This is a collisionless electrostatic model with full electron orbits. It is not a
comparison with gyrokinetic magnetic-presheath theory, which assumes an ordering in
$\alpha$ and adiabatic electrons that this does not; at grazing incidence the classical
Debye sheath is expected to weaken and the entrance conditions to change
{cite}`geraldini2018`, and nothing here forces a Bohm crossing or fits two layers to
every run.

```bash
python examples/2_intermediate/sheath_magnetized.py            # about six minutes
python examples/2_intermediate/sheath_magnetized.py --quick    # about a minute
```
