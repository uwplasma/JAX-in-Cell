# Collisions

Passing a {class}`~jaxincell.Collisions` object adds binary Coulomb collisions after the
particle push.

```python
from jaxincell import Collisions, Simulation

simulation = Simulation(domain, [electrons, ions], solver,
                        collisions=Collisions(pairs=(("electrons", "ions"),
                                                     ("electrons", "electrons"))))
```

| argument | meaning | default |
|---|---|---|
| `pairs` | tuple of `(name_a, name_b)` species pairs to collide, including a species with itself; `None` collides every pair (static) | `None` |
| `coulomb_log` | the Coulomb logarithm; `None` takes the NRL formulary value from the electrons, the lightest negatively charged species | `None` |

Names refer to `Species.name`, so give the species meaningful names when collisions are
used.

## What it does

Particles are paired at random inside each cell and each pair is scattered by rotating its
relative velocity, which conserves the pair's momentum and energy exactly whatever the time
step. The scattering angle carries the collision frequency. The scheme, the variance of the
angle and the handling of unequal particle counts and weights are described in
{doc}`../numerics/collisions`; verified against the Fokker-Planck relaxation rates, it is
within {{ collisions_max_deviation_percent }} % of theory
({doc}`../examples/collisions`).

## Two things to get right

* **The Courant condition.** Collisions scatter velocity into $y$ and $z$, which excites
  the transverse electromagnetic fields. An electrostatic run stepping above
  $c\Delta t = \Delta x$ was safe only while those fields were exactly zero; switching
  collisions on makes it diverge. Use `dt_over_dx_c <= 1` whenever collisions are active.
  The constructor warns about this combination.
* **The timescales.** In a weakly coupled plasma the collision frequency is far below the
  plasma frequency, $\nu/\omega_{pe}\sim \ln\Lambda/(n\lambda_D^3)$, which is $10^{-5}$ or
  smaller in most laboratory conditions. Resolving both in one run is expensive. Either
  accept that a run covers many plasma periods and few collision times, or study a
  transport coefficient on a small patch where the collisional timescale is the only one
  that matters.

## Setting the Coulomb logarithm

Left at `None`, $\ln\Lambda$ comes from the NRL electron-ion formula evaluated at the start
of the run for the electrons, taken to be the lightest negatively charged species:

* its `density`, and the temperature $T = m v_{th}^2/2$ of the largest of its three `vth`
  components;
* a simulation with no negatively charged species needs `coulomb_log` given;
* the value is kept above 2, the customary floor (Lee and More 1984) where the formula
  would turn small or negative in a cold, dense plasma, so a cold species with `vth = 0`
  gives 2.

This takes no account of how the plasma evolves. Set it explicitly when a specific value is
wanted, when the plasma drifts far from its initial conditions, or when running a
controlled numerical experiment on the operator itself:

```python
Collisions(pairs=(("electrons", "ions"),), coulomb_log=15.0)
```

## Self-collisions

`("electrons", "electrons")` collides a species with itself, which is what relaxes an
anisotropic or non-Maxwellian distribution back towards a Maxwellian. Include it whenever
the shape of a distribution matters; without it, cross-species collisions alone will
distort the distributions they act on.
