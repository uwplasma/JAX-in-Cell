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
| `restore` | return the weighted momentum and energy of every cell to their values before each cross-species stage, by correcting the species of larger weight; see {doc}`../numerics/collisions` (static) | `False` |

Names refer to `Species.name`, so give the species meaningful names when collisions are
used.

## What it does

Particles are paired at random inside each cell. With equal weights, the pair's momentum
and kinetic energy are conserved to roundoff. Unequal weights use a stochastic acceptance
rule and conserve these quantities in expectation, with fluctuations in an individual run.
The scattering angle carries the collision frequency. The scheme, the variance of the
angle and the handling of unequal particle counts and weights are described in
{doc}`../numerics/collisions`; verified against the Fokker-Planck relaxation rates, it is
within {{ collisions_max_deviation_percent }} % of theory
for the nonrelativistic beam benchmark ({doc}`../examples/collisions`).

### Relativistic particles

`Solver(relativistic=True)` selects relativistic center-of-momentum collisions with
the Pérez rate and Nanbu cumulative angles. Give the Coulomb logarithm explicitly:

```python
from jaxincell import Collisions, Simulation, Solver

simulation = Simulation(domain, [electrons, ions], Solver(relativistic=True),
                        collisions=Collisions(coulomb_log=15.0))
```

The operator works directly on `State.u = gamma*v`, in m/s; it never converts a
relativistic momentum through a velocity close to the speed of light. A pair is
rotated in its center-of-momentum frame and transformed back. The existing cell
pairing and weight correction are shared with the nonrelativistic operator.

This is elastic Coulomb scattering in a fully ionized, weakly coupled,
nondegenerate plasma. It covers thermal particles and relativistic tails, with
timestep and particle-number convergence required. It does not include partial
screening, hard Møller knock-on events, bremsstrahlung or a complete runaway
avalanche model. The nonrelativistic NRL logarithm is not silently used as a
relativistic screening prescription.

Use float64 for reference calculations. The float32 kernel normalizes masses
before forming products, so physical electron masses do not underflow. Its
accuracy still needs checking for the chosen momentum range, nearly comoving
pairs and observables. Tiny deflections that round to zero have zero numerical
derivatives; derivatives at zero collision rate are not an ensemble-sensitivity
guarantee.

Fixed random keys allow replay and common-random-number derivative checks. They
do not differentiate changes in cell membership, partner identities or
weight-dependent acceptance probabilities. For an optimization, compare
ensemble gradients and independent random realizations. Very different macro
weights can require many samples to resolve the background response to a dilute
tail; conservation in expectation does not imply a small energy error in one run.

## Two things to get right

* **The Courant condition.** Explicit electromagnetic runs require
  `dt_over_dx_c <= 1`; collisions can excite transverse fields even when they start
  at zero. The electrostatic model does not advance light waves and has no such
  limit. Both models still require timestep refinement to resolve particle and
  collision timescales.
* **The timescales.** In a weakly coupled plasma the collision frequency is far below the
  plasma frequency, $\nu/\omega_{pe}\sim \ln\Lambda/(n\lambda_D^3)$, which is $10^{-5}$ or
  smaller in most laboratory conditions. Resolving both in one run is expensive. Either
  accept that a run covers many plasma periods and few collision times, or study a
  transport coefficient on a small patch where the collisional timescale is the only one
  that matters.

## Setting the Coulomb logarithm

For a nonrelativistic run, `None` takes $\ln\Lambda$ from the NRL electron-ion formula evaluated at the start
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
