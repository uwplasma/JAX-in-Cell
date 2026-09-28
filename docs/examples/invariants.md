# What the integrators conserve

Two properties of the step itself, checked against what the discrete scheme promises: where
in the explicit step collisions act, and the implicit electrostatic scheme, which holds the
Gauss law without a Poisson solve.

```{figure} ../_static/figures/invariants.png
:width: 100%
:alt: Energy error against step size with and without collisions; implicit energy error against Picard iterations; energy and Gauss-law error over an electrostatic two-stream run

(a) Largest energy error of electrons and a $25\,m_e$ species in one harmonic well over
sixteen periods, against the step: with collisions it follows the collisionless leapfrog at
second order. (b) The implicit electrostatic scheme's energy error against the number of
Picard iterations per step. (c) Over an electrostatic two-stream run: the explicit energy
error, and the implicit energy and Gauss-law errors, both at round-off.
```

## Collisions at the integer time

The explicit step kicks $u^n\to u^{n+1}$ in the field at $x^{n+1/2}$, so the velocity lives
at integer times. Collisions act at $x^{n+1} = x^{n+1/2} + \Delta t\,u^{n+1}/2$, between
the two half drifts ({doc}`../numerics/collisions`, "Where in the step"). The test is a
harmonic well holding electrons and a heavier species of the same charge, all in one cell so
that each collision conserves its pair's energy exactly, at $\nu/\omega\approx 0.1$:

| $\omega\Delta t$ | collisionless | with collisions |
|---|---|---|
| 0.4 | {{ centring_04_collisionless }} | {{ centring_04_collisional }} |
| 0.2 | {{ centring_02_collisionless }} | {{ centring_02_collisional }} |
| 0.1 | {{ centring_01_collisionless }} | {{ centring_01_collisional }} |

Both fall as $\Delta t^2$. When collisions acted at $x^{n+1/2}$, as they once did, the next
half drift moved the position with the scattered velocity, and the same run's energy error
was 22.1, 6.83 and 1.53 at these three steps: not an error that converges.
`tests/test_collisions.py::test_collisions_at_the_integer_time_keep_the_leapfrog_energy_error`
fails on that placement.

## The implicit electrostatic scheme

`Solver(algorithm="implicit", model="electrostatic")` advances $E_x$ by Ampère's law with the
continuity current inside the Picard loop and evolves no transverse field
({doc}`../numerics/implicit`, "The electrostatic model"). On a 64-cell two-stream run (2000
electrons and 2000 ions, $c\Delta t/\Delta x = 4.5$, 150 steps):

| Picard iterations | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|
| energy error | {{ implicit_es_picard_2 }} | {{ implicit_es_picard_3 }} | {{ implicit_es_picard_4 }} | {{ implicit_es_picard_6 }} | {{ implicit_es_picard_8 }} | {{ implicit_es_picard_12 }} |

At the default count the energy error is {{ implicit_es_energy }} and the Gauss residual
{{ implicit_es_gauss }}; the explicit electrostatic leapfrog on the same run has
{{ explicit_es_energy }} and {{ explicit_es_gauss }}. The same invariants for every model and
boundary are tabulated in {doc}`../numerics/verification`, "Discrete invariants, model by model".

## Running it

```bash
python examples/3_advanced/invariants.py            # about a minute on a CPU
python examples/3_advanced/invariants.py --quick    # two step sizes, three Picard counts
```
